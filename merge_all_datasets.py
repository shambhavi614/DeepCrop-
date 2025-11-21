import os
import shutil

# Paths to your 3 folders — update paths as per your PC
sources = [
    "C:/Users/Shambhavi/Downloads/archive (1)/plantvillage dataset/color",
    "C:/Users/Shambhavi/Downloads/archive (1)/plantvillage dataset/grayscale",
    "C:/Users/Shambhavi/Downloads/archive (1)/plantvillage dataset/segmented"
]


destination = "C:/Users/Shambhavi/Desktop/DeepCrop/dataset"

# Make sure destination exists
os.makedirs(destination, exist_ok=True)

# Go through all folders and merge data safely
for src in sources:
    print(f"🔄 Copying from: {src}")
    for category in os.listdir(src):
        src_path = os.path.join(src, category)
        dest_path = os.path.join(destination, category)
        os.makedirs(dest_path, exist_ok=True)

        # Copy images one by one
        for file in os.listdir(src_path):
            src_file = os.path.join(src_path, file)
            dest_file = os.path.join(dest_path, file)

            try:
                if not os.path.exists(dest_file):  # avoid duplicates
                    shutil.copy2(src_file, dest_file)
            except Exception as e:
                print(f"⚠️ Skipped {file} due to error: {e}")

print("✅ All color + grayscale + segmented datasets merged successfully!")
