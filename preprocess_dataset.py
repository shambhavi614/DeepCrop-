import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to merged dataset
data_dir = "C:/Users/Shambhavi/Desktop/DeepCrop/merged_dataset"

# Path to save preprocessed (augmented) images
output_dir = "C:/Users/Shambhavi/Desktop/DeepCrop/preprocessed_dataset"

# Create an ImageDataGenerator for preprocessing + augmentation
datagen = ImageDataGenerator(
    rescale=1./255,        # normalize pixel values
    rotation_range=20,     # random rotations
    width_shift_range=0.1, # horizontal shift
    height_shift_range=0.1,# vertical shift
    zoom_range=0.2,        # zoom in/out
    horizontal_flip=True,  # flip horizontally
    validation_split=0.2   # 80% train, 20% validation
)

# Create generators for training and validation sets
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    save_to_dir=output_dir + "/train",
    save_prefix='aug',
    save_format='jpg'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    save_to_dir=output_dir + "/val",
    save_prefix='aug',
    save_format='jpg'
)

print("✅ Dataset preprocessing completed successfully!")
