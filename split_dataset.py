import os
import shutil
import random

DATASET_DIR = "dataset"
OUTPUT_DIR = "dataset"
split_ratio = (0.7, 0.15, 0.15)  # 70% train, 15% validation, 15% test

# Create folders
for folder in ['train', 'validation', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

for category in os.listdir(DATASET_DIR):
    category_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(category_path):
        continue

    images = os.listdir(category_path)
    random.shuffle(images)

    train_end = int(len(images) * split_ratio[0])
    val_end = train_end + int(len(images) * split_ratio[1])

    splits = {
        'train': images[:train_end],
        'validation': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split_name, split_files in splits.items():
        split_folder = os.path.join(OUTPUT_DIR, split_name, category)
        os.makedirs(split_folder, exist_ok=True)
        for file in split_files:
            src = os.path.join(category_path, file)
            dst = os.path.join(split_folder, file)
            shutil.copy2(src, dst)

print("✅ Dataset successfully split into train/validation/test folders!")
