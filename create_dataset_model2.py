import os
import shutil
import random

RAW_DATASET = "dataset"      # already existing dataset
OUTPUT = "dataset_model2"
train_ratio = 0.8

classes = ["glioma", "meningioma", "pituitary"]

for split in ["Training", "Testing"]:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT, split, cls), exist_ok=True)

for cls in classes:
    images = []
    src_train = os.path.join(RAW_DATASET, "Training", cls)
    src_test = os.path.join(RAW_DATASET, "Testing", cls)

    for folder in [src_train, src_test]:
        for img in os.listdir(folder):
            images.append(os.path.join(folder, img))

    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)

    for img in images[:split_idx]:
        shutil.copy(img, os.path.join(OUTPUT, "Training", cls))

    for img in images[split_idx:]:
        shutil.copy(img, os.path.join(OUTPUT, "Testing", cls))

print("✅ dataset_model2 created successfully")