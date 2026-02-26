import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# Paths
# =========================
MODEL1_PATH = "model1_tumor_detection.keras"        # Tumor / No Tumor
MODEL2_PATH = "tumor_type_classifier.h5"   # Glioma / Meningioma / Pituitary
IMAGE_PATH  = "test.jpg"

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# Load models
# =========================
model1 = load_model(MODEL1_PATH)
model2 = load_model(MODEL2_PATH)

class_names = ["Glioma", "Meningioma", "Pituitary"]

# =========================
# Load & preprocess image
# =========================
img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# =========================
# Model 1: Tumor Detection
# =========================
det = model1.predict(img_array)
prob = float(det[0][0])

print(f"Tumor probability: {prob}")

if prob < 0.5:
    print("❌ No Tumor Detected")

    with open(os.path.join(RESULT_DIR, "prediction_output.txt"), "w") as f:
        f.write(f"Tumor Probability: {prob}\n")
        f.write("Result: No Tumor Detected\n")

else:
    print("✅ Tumor Detected")

    # =========================
    # Model 2: Tumor Type
    # =========================
    cls_pred = model2.predict(img_array)
    idx = np.argmax(cls_pred)
    tumor_type = class_names[idx]
    confidence = float(cls_pred[0][idx])

    print(f"🧠 Tumor Type: {tumor_type}")
    print(f"📊 Confidence: {confidence}")

    # =========================
    # Save results
    # =========================
    with open(os.path.join(RESULT_DIR, "prediction_output.txt"), "w") as f:
        f.write(f"Tumor Probability: {prob}\n")
        f.write("Result: Tumor Detected\n")
        f.write(f"Tumor Type: {tumor_type}\n")
        f.write(f"Confidence: {confidence}\n")