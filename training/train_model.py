import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

# IMPORTANT: Adjust these paths to match your folder structure
DATA_DIRECTORIES = [
    r'C:\Users\chara\OneDrive\Desktop\Age Prediction Training\DATA_SET\utkface_aligned_cropped\UTKFace']

def load_data_paths(data_paths):
    """Loads a list of image file paths and their corresponding ages."""
    all_image_paths = []
    all_ages = []
    
    for dataset_path in data_paths:
        print(f"Loading file paths from: {dataset_path}")
        for filename in tqdm(os.listdir(dataset_path)):
            try:
                age = int(filename.split('_')[0])
                img_path = os.path.join(dataset_path, filename)
                all_image_paths.append(img_path)
                all_ages.append(age)
            except (ValueError, IndexError):
                continue
    
    return np.array(all_image_paths), np.array(all_ages)

# Load data paths instead of entire images
print("Loading and preprocessing data paths...")
image_paths, ages = load_data_paths(DATA_DIRECTORIES)

# Split paths into training and testing sets
X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    image_paths, ages, test_size=0.2, random_state=42
)
print("Data loading complete.")

# Define the batch size and steps per epoch
BATCH_SIZE = 32
train_steps_per_epoch = math.ceil(len(X_train_paths) / BATCH_SIZE)
val_steps_per_epoch = math.ceil(len(X_test_paths) / BATCH_SIZE)

def data_generator(image_paths, ages, batch_size):
    """Generates batches of images and ages from file paths."""
    num_samples = len(image_paths)
    while True: # Loop indefinitely
        for offset in range(0, num_samples, batch_size):
            batch_paths = image_paths[offset:offset+batch_size]
            batch_ages = ages[offset:offset+batch_size]
            
            # Load and process images for the current batch
            batch_images = []
            for img_path in batch_paths:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                img = img.astype('float32') / 255.0
                batch_images.append(img)
            
            yield np.array(batch_images), np.array(batch_ages)

# Create data generators for training and validation
train_generator = data_generator(X_train_paths, y_train, BATCH_SIZE)
val_generator = data_generator(X_test_paths, y_test, BATCH_SIZE)

# Build and compile the model (same as before)
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

print("Building and training the model...")
model = build_model()

# Train the model using the generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=50,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch
)

# Save the model
model.save('age_prediction_model.h5')
print("Model trained and saved as age_prediction_model.h5")