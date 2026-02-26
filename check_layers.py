from tensorflow.keras.models import load_model

model = load_model("tumor_type_classifier.h5")
model.summary()