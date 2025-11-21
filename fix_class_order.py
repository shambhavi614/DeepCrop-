import os

# Path to your training dataset folder
dataset_path = "dataset/train"

# Make sure the folder exists
if not os.path.exists(dataset_path):
    print(f"⚠️ Folder not found: {dataset_path}")
    print("Please check your dataset path.")
    exit()

# Get all subfolder (class) names in alphabetical order
class_names = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])

# Print them as a Python list for app.py
print("\n✅ Correct CLASS_NAMES for your app.py:\n")
print("CLASS_NAMES = [")
for name in class_names:
    print(f"    '{name}',")
print("]\n")

# Save to a file for reference
with open("class_names.txt", "w", encoding="utf-8") as f:
    f.write("CLASS_NAMES = [\n")
    for name in class_names:
        f.write(f"    '{name}',\n")
    f.write("]\n")

print("✅ Saved to class_names.txt (you can copy this into app.py)")
