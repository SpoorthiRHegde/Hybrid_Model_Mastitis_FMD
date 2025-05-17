#foot and mouth image model

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# === Image Preprocessing (Sharpening) ===
def preprocess_and_sharpen(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # Sharpening kernel
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5,-1],
                             [0, -1, 0]])
    image_sharp = cv2.filter2D(image, -1, kernel_sharp)
    image_sharp = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2RGB)
    return image_sharp

# === Load Dataset ===
def load_dataset(path):
    X, y = [], []
    for label in ['healthy', 'infected']:
        label_path = os.path.join(path, label)
        for img in os.listdir(label_path):
            try:
                img_path = os.path.join(label_path, img)
                img_array = preprocess_and_sharpen(img_path)
                X.append(img_array)
                y.append(0 if label == 'healthy' else 1)
            except:
                continue
    return np.array(X), np.array(y)

# === Build CNN Model ===
def build_model():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Train and Evaluate ===
def process_train_evaluate(data_path):
    X, y = load_dataset(data_path)
    X = X / 255.0

    # Fixed shuffle: ensure reproducibility
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model()
    model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred_bin)
    return model, acc, (X_test, y_test, y_pred_bin)

# === Train Both Models ===
foot_model, foot_acc, (foot_X, foot_y, foot_pred) = process_train_evaluate('../Dataset/foot')
foot_model.save('../models/foot_model.h5')
mouth_model, mouth_acc, (mouth_X, mouth_y, mouth_pred) = process_train_evaluate('../Dataset/mouth')
mouth_model.save('../models/mouth_model.h5')

# === Combine FMD Predictions ===
min_len = min(len(foot_pred), len(mouth_pred))
combined_pred = np.maximum(foot_pred[:min_len], mouth_pred[:min_len])
combined_true = np.maximum(foot_y[:min_len], mouth_y[:min_len])
combined_acc = accuracy_score(combined_true, combined_pred)

# === Show Results ===
print("\n--- Model Accuracies ---")
print(f"Foot model accuracy: {foot_acc:.2%}")
print(f"Mouth model accuracy: {mouth_acc:.2%}")
print(f"Combined FMD detection accuracy: {combined_acc:.2%}")