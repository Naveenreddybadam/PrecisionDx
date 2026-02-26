import os
import shutil
import random

# ✅ Kaggle dataset structure
RAW_DATASET = "dataset/Training"   # <-- IMPORTANT
OUTPUT = "dataset_model1"

train_ratio = 0.8

classes = {
    "Tumor": ["glioma", "meningioma", "pituitary"],
    "No_Tumor": ["notumor"]
}

# Create output folders
for split in ["Training", "Testing"]:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT, split, cls), exist_ok=True)

# Process images
for target_class, source_folders in classes.items():
    images = []

    for folder in source_folders:
        folder_path = os.path.join(RAW_DATASET, folder)

        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue

        for img in os.listdir(folder_path):
            images.append(os.path.join(folder_path, img))

    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)

    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    for img in train_imgs:
        shutil.copy(img, os.path.join(OUTPUT, "Training", target_class))

    for img in test_imgs:
        shutil.copy(img, os.path.join(OUTPUT, "Testing", target_class))

print("✅ dataset_model1 created successfully")