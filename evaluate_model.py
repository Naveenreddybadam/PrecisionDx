from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import json

# Load model
model = load_model("backend/models/tumor_type_classifier.h5")

# Load validation data
datagen = ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    "dataset_model2/Testing",
    target_size=(224,224),
    class_mode="categorical",
    shuffle=False
)

# Predict
predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

# Metrics
report = classification_report(y_true, y_pred, output_dict=True)

print(report)

# Save to JSON
with open("metrics_model2.json", "w") as f:
    json.dump(report, f, indent=4)

print("Metrics saved successfully")