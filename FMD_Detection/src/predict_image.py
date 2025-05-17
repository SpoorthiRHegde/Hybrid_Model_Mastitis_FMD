#foot and mouth image (input)

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Load Trained Models ===
foot_model = load_model("../models/foot_model.h5")
mouth_model = load_model("../models/mouth_model.h5")

# === Preprocessing Function ===
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# === Take Input Paths ===
foot_path = input("Enter path to foot image: ")
mouth_path = input("Enter path to mouth image: ")

foot_img = preprocess_image(foot_path)
mouth_img = preprocess_image(mouth_path)

# === Predict Probabilities ===
foot_prob = float(foot_model.predict(foot_img)[0])
mouth_prob = float(mouth_model.predict(mouth_img)[0])

# === Combine Results ===
combined_label = int((foot_prob > 0.5) or (mouth_prob > 0.5))
final_diagnosis = "Infected" if combined_label else "Healthy"

# === Display Output ===
print("\n--- Prediction Results ---")
print(f"Foot Image Prediction Probability: {foot_prob:.4f}")
print(f"Mouth Image Prediction Probability: {mouth_prob:.4f}")
print(f"Final FMD Diagnosis: {final_diagnosis}")