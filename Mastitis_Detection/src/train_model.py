import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === CLAHE Preprocessing ===
def preprocess_clahe(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

# === Dataset Loader ===
def load_mastitis_dataset(path):
    X, y = [], []
    for label in ['Non_infected', 'Infected']:
        label_path = os.path.join(path, label)
        for img in os.listdir(label_path):
            try:
                img_path = os.path.join(label_path, img)
                img_array = preprocess_clahe(img_path)
                X.append(img_array)
                y.append(0 if label == 'Non_infected' else 1)
            except:
                continue
    return np.array(X), np.array(y)

# === Build Model ===
def build_mastitis_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True 

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# === Train Function ===
def train_mastitis_model(data_path):
    X, y = load_mastitis_dataset(data_path)
    X = X / 255.0

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=42)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    model = build_mastitis_model()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    ]

    model.fit(datagen.flow(X_train, y_train, batch_size=16),
              epochs=10,
              validation_data=(X_val, y_val),
              callbacks=callbacks,
              verbose=1)

    model_save_path = '../models/mastitis_model.h5'
    model.save(model_save_path)

    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred_bin)

    return model, acc, (X_test, y_test, y_pred_bin)

# === Main ===
if __name__ == "__main__":
    model, acc, (X_test, y_test, y_pred) = train_mastitis_model('../Dataset/mastitis_cattle')
    print(f"\n✅ Final Mastitis Model Accuracy: {acc:.2%}")
