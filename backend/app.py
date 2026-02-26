import os
import uuid
import sqlite3
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "database.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ----------------
model1 = load_model(os.path.join(MODEL_FOLDER, "model1_tumor_detection.keras"))
model2 = load_model(os.path.join(MODEL_FOLDER, "tumor_type_classifier.h5"))
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]

# ---------------- DATABASE INIT ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        age INTEGER,
        gender TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        tumor_type TEXT,
        confidence REAL,
        tumor_probability REAL,
        risk_level TEXT,
        grade INTEGER,
        trend Text,
        gradcam_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- RISK ----------------
def calculate_risk(tumor_type, confidence):
    if tumor_type == "Glioma":
        if confidence >= 0.90:
            return "HIGH"
        elif confidence >= 0.75:
            return "MEDIUM"
        else:
            return "LOW"
    else:
        if confidence >= 0.85:
            return "MEDIUM"
        else:
            return "LOW"

# ---------------- GRADE ----------------
def calculate_grade(tumor_type, confidence):
    if tumor_type == "Glioma":
        if confidence < 0.60:
            return 1
        elif confidence < 0.75:
            return 2
        elif confidence < 0.90:
            return 3
        else:
            return 4
    else:
        if confidence < 0.65:
            return 1
        elif confidence < 0.80:
            return 2
        elif confidence < 0.90:
            return 3
        else:
            return 4

# ---------------- IMAGE ----------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def simple_gradcam(img_path, output_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, blended)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return "PrecisionDx Backend Running 🚀"

# REGISTER (Doctor / Reception)
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data["username"]
    password = data["password"]
    role = data["role"]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    hashed = generate_password_hash(password)

    try:
        cursor.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                       (username, hashed, role))
        conn.commit()
    except:
        return jsonify({"error":"User already exists"}),400

    conn.close()
    return jsonify({"message":"Registered Successfully"})

# LOGIN
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data["username"]
    password = data["password"]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id,password,role FROM users WHERE username=?",(username,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user[1], password):
        return jsonify({"success":True,"role":user[2],"user_id":user[0]})
    return jsonify({"success":False}),401

# ADD PATIENT (Reception)
@app.route("/add_patient", methods=["POST"])
def add_patient():
    data = request.json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO patients (patient_name,age,gender) VALUES (?,?,?)",
                   (data["patient_name"],data["age"],data["gender"]))
    conn.commit()
    conn.close()
    return jsonify({"message":"Patient Added"})

# GET PATIENTS
@app.route("/patients")
def get_patients():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients")
    rows = cursor.fetchall()
    conn.close()

    return jsonify([
        {"id":r[0],"patient_name":r[1],"age":r[2],"gender":r[3]}
        for r in rows
    ])

# PREDICT
@app.route("/predict", methods=["POST"])
def predict():

    patient_id = request.form.get("patient_id")
    file = request.files["image"]

    unique_id = str(uuid.uuid4())
    img_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.jpg")
    gradcam_path = os.path.join(RESULT_FOLDER, f"{unique_id}.jpg")

    file.save(img_path)
    img = preprocess_image(img_path)

    prob = float(model1.predict(img)[0][0])

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---------------- NO TUMOR ----------------
    if prob < 0.5:

        cursor.execute("""
        INSERT INTO reports
        (patient_id, tumor_type, confidence, tumor_probability,
         risk_level, grade, gradcam_path, trend)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            "No Tumor",
            1 - prob,
            prob,
            "LOW",
            0,   # grade integer
            ""
        ))

        conn.commit()
        conn.close()

        return jsonify({
            "tumor_detected": False,
            "grade": 0,
            "risk_level": "LOW",
            "confidence": 1 - prob,
            "gradcam_image": ""
        })

    # ---------------- TUMOR DETECTED ----------------
    preds = model2.predict(img)
    idx = int(np.argmax(preds))
    confidence = float(preds[0][idx])
    tumor_type = CLASS_NAMES[idx]

    risk = calculate_risk(tumor_type, confidence)
    grade = calculate_grade(tumor_type, confidence)

    simple_gradcam(img_path, gradcam_path)

    # Previous record check
    cursor.execute("""
    SELECT confidence, risk_level FROM reports
    WHERE patient_id=?
    ORDER BY created_at DESC
    LIMIT 1
    """, (patient_id,))

    prev = cursor.fetchone()

    prev_conf = prev[0] if prev else None
    prev_risk = prev[1] if prev else None
    
    trend = None

    if prev_conf is not None:
        if confidence > prev_conf:
            trend = "WORSENING"
        elif confidence < prev_conf:
            trend = "IMPROVING"
        else:
            trend="stable"
    cursor.execute("""
    INSERT INTO reports
    (patient_id, tumor_type, confidence, tumor_probability,
     risk_level, grade, gradcam_path, trend  )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        tumor_type,
        confidence,
        prob,
        risk,
        grade,
        f"/static/results/{unique_id}.jpg",
        trend
    ))

    report_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return jsonify({
        "tumor_detected": True,
        "tumor_type": tumor_type,
        "confidence": confidence,
        "risk_level": risk,
        "grade": grade,
        "previous_confidence": prev_conf,
        "previous_risk": prev_risk,
        "trend":trend,
        "gradcam_image": f"/static/results/{unique_id}.jpg",
        "report_id": report_id
    })
@app.route("/generate_report/<int:report_id>")
def generate_report(report_id):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT p.patient_name, r.tumor_type, r.confidence,
           r.risk_level, r.grade, r.created_at
    FROM reports r
    JOIN patients p ON r.patient_id = p.id
    WHERE r.id=?
    """, (report_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return "Report not found", 404

    filename = f"report_{report_id}.pdf"
    filepath = os.path.join(BASE_DIR, filename)

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4

    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("PrecisionDx Hospital", styles['Title']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Patient: {row[0]}", styles['Normal']))
    elements.append(Paragraph(f"Tumor Type: {row[1]}", styles['Normal']))
    elements.append(Paragraph(f"Grade: {row[4]}", styles['Normal']))
    elements.append(Paragraph(f"Confidence: {row[2]:.2f}", styles['Normal']))
    elements.append(Paragraph(f"Risk Level: {row[3]}", styles['Normal']))
    elements.append(Paragraph(f"Date: {row[5]}", styles['Normal']))
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Doctor Signature: ___________________", styles['Normal']))

    doc.build(elements)

    return send_from_directory(BASE_DIR, filename, as_attachment=True)
@app.route("/patient_trend_all")
def patient_trend_all():

    filter_type = request.args.get("filter", "all")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    base_query = """
    SELECT r.id,
           p.patient_name,
           p.age,
           r.tumor_type,
           r.grade,
           r.confidence,
           r.risk_level,
           r.created_at,
           r.gradcam_path
    FROM reports r
    JOIN patients p ON r.patient_id = p.id
    """

    # ---- DAY FILTER ----
    if filter_type == "today":
        base_query += " WHERE DATE(r.created_at) = DATE('now') "

    elif filter_type == "week":
        base_query += " WHERE DATE(r.created_at) >= DATE('now','-7 days') "

    # ---- SORTING ----
    base_query += """
    ORDER BY 
        CASE r.risk_level
            WHEN 'HIGH' THEN 1
            WHEN 'MEDIUM' THEN 2
            WHEN 'LOW' THEN 3
            ELSE 4
        END,
        r.created_at DESC
    """

    cursor.execute(base_query)
    rows = cursor.fetchall()
    conn.close()

    data = []
    for r in rows:
        data.append({
            "report_id": r[0],
            "patient_name": r[1],
            "age": r[2],
            "tumor_type": r[3],
            "grade": r[4],
            "confidence": r[5],
            "risk_level": r[6],
            "created_at": r[7],
            "gradcam_path": r[8]
        })

    return jsonify(data)
@app.route("/model_metrics")
def model_metrics():
    import json
    try:
        with open("metrics_model2.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({"error": "Metrics not found"})
if __name__ == "__main__":
    app.run(debug=True)