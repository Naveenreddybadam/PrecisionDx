import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ✅ Correct model paths
model1 = load_model("model1_tumor_detection.keras")   # Model-1
model2 = load_model("tumor_type_classifier.h5")       # Model-2

class_names = ["glioma", "meningioma", "pituitary"]

# Load test image
img_path = "test.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Step 1: Tumor detection
tumor_prob = model1.predict(img)[0][0]
print("Tumor probability:", tumor_prob)

if tumor_prob > 0.5:
    preds = model2.predict(img)
    tumor_type = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    print("✅ Tumor Detected")
    print("🧠 Tumor Type:", tumor_type)
    print("📊 Confidence:", f"{confidence:.2f}%")
else:
    print("❌ No Tumor Detected")