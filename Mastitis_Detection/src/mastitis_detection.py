import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set the dataset path
dataset_path = r'C:\Users\spoor\OneDrive\Desktop\Major\Dataset\Dataset\Mastitis_Cattle'
infected_path = os.path.join(dataset_path, 'Infected')
non_infected_path = os.path.join(dataset_path, 'Non Infected')

# Function to load and preprocess images
def load_images(folder_path, label, target_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is not None:
                # Resize and normalize
                img = cv2.resize(img, target_size)
                img = img / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return images, labels

# Load infected and non-infected images
infected_images, infected_labels = load_images(infected_path, 1)
non_infected_images, non_infected_labels = load_images(non_infected_path, 0)

# Combine datasets
X = np.array(infected_images + non_infected_images)
y = np.array(infected_labels + non_infected_labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=20,
                    validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Function to predict new images
def predict_mastitis(image_path, model, threshold=0.5):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image"
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img)
    infected_prob = prediction[0][1]
    
    if infected_prob > threshold:
        return f"Mastitis Detected (Confidence: {infected_prob:.2%})"
    else:
        return f"No Mastitis Detected (Confidence: {(1-infected_prob):.2%})"

# Example usage
sample_image_path = os.path.join(infected_path, os.listdir(infected_path)[0])
print(predict_mastitis(sample_image_path, model))

# Save the model
model.save('udder_mastitis_detector.h5')