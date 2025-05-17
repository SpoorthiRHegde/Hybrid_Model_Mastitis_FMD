import os
import sys
import numpy as np
import pandas as pd
import joblib
import cv2
from tensorflow.keras.models import load_model

# === Load models ===

foot_cnn = load_model(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\foot_model.h5")
mouth_cnn = load_model(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\mouth_model.h5")

foot_clf = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\foot_text_model.pkl")
foot_scaler = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\foot_text_scaler.pkl")

mouth_clf = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\mouth_text_model.pkl")
mouth_scaler = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\mouth_text_scaler.pkl")

# === Feature columns ===
foot_features = ['temperature', 'milk_production', 'lethargy', 'difficulty_in_walking',
                 'foot_blister', 'foot_swelling', 'hoof_detachment']

mouth_features = ['temperature', 'milk_production', 'lethargy', 'mouth_ulcers',
                  'mouth_blister', 'salivation', 'nasal_discharge']

# === Image Preprocessing ===
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# === Text input collection ===
def validate_input(prompt, min_val, max_val, input_type='float'):
    while True:
        try:
            val = input(prompt)
            val = float(val) if input_type == 'float' else int(val)
            if min_val <= val <= max_val:
                return val
            print(f"Value should be between {min_val} and {max_val}")
        except:
            print("Invalid input. Try again.")

def get_common_input():
    t = validate_input("Temperature (37.5–41.0): ", 37.5, 41.0)
    m = validate_input("Milk production (0–5): ", 0, 5, 'int')
    l = validate_input("Lethargy (0=no, 1=yes): ", 0, 1, 'int')
    return [t, m, l]

def get_foot_text_input(common):
    d = validate_input("Difficulty walking (0/1): ", 0, 1, 'int')
    b = validate_input("Foot blister (0–5): ", 0, 5, 'int')
    s = validate_input("Foot swelling (0–5): ", 0, 5, 'int')
    h = validate_input("Hoof detachment (0–5): ", 0, 5, 'int')
    return pd.DataFrame([common + [d, b, s, h]], columns=foot_features)

def get_mouth_text_input(common):
    u = validate_input("Mouth ulcers (0–5): ", 0, 5, 'int')
    b = validate_input("Mouth blister (0–5): ", 0, 5, 'int')
    s = validate_input("Salivation (0/1): ", 0, 1, 'int')
    n = validate_input("Nasal discharge (0/1): ", 0, 1, 'int')
    return pd.DataFrame([common + [u, b, s, n]], columns=mouth_features)

# === Prediction helpers ===
def get_image_prob(model, image_path):
    img = preprocess_image(image_path)
    return float(model.predict(img)[0])

def get_text_prob(model, scaler, df):
    scaled = scaler.transform(df)
    prob = model.predict_proba(scaled)[0]
    return prob[1]  # probability of 'Infected'

# === Combine and predict ===
def predict_combined(foot_probs, mouth_probs):
    # average infected probability if any value present
    combined_probs = foot_probs + mouth_probs
    avg_prob = sum(combined_probs) / len(combined_probs)
    return ("Infected" if avg_prob > 0.5 else "Healthy", avg_prob)

# === Main CLI ===
def main():
    print("Choose input options (Y for Yes, N for No):")
    ft = input("Foot Text Data? (Y/N): ").strip().upper() == 'Y'
    mt = input("Mouth Text Data? (Y/N): ").strip().upper() == 'Y'
    fi = input("Foot Image? (Y/N): ").strip().upper() == 'Y'
    mi = input("Mouth Image? (Y/N): ").strip().upper() == 'Y'

    foot_probs = []
    mouth_probs = []

    common_input = get_common_input() if (ft or mt) else None

    if ft:
        foot_text_df = get_foot_text_input(common_input)
        prob = get_text_prob(foot_clf, foot_scaler, foot_text_df)
        foot_probs.append(prob)
        print(f"Foot Text Prediction: {prob:.4f}")

    if mt:
        mouth_text_df = get_mouth_text_input(common_input)
        prob = get_text_prob(mouth_clf, mouth_scaler, mouth_text_df)
        mouth_probs.append(prob)
        print(f"Mouth Text Prediction: {prob:.4f}")

    if fi:
        path = input("Enter Foot Image Path: ").strip()
        prob = get_image_prob(foot_cnn, path)
        foot_probs.append(prob)
        print(f"Foot Image Prediction: {prob:.4f}")

    if mi:
        path = input("Enter Mouth Image Path: ").strip()
        prob = get_image_prob(mouth_cnn, path)
        mouth_probs.append(prob)
        print(f"Mouth Image Prediction: {prob:.4f}")

    if not (foot_probs or mouth_probs):
        print("No inputs selected. Exiting.")
        return

    result, final_prob = predict_combined(foot_probs, mouth_probs)
    print("\n--- Final FMD Prediction ---")
    print(f"Diagnosis: {result}")
    print(f"Combined Probability: {final_prob:.4f}")

if __name__ == "__main__":
    main()
