from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import cv2
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)
CORS(app)

# === Load Mastitis Models ===
mastitis_text_model = joblib.load('../Mastitis_Detection/Models/mastitis_text_model.pkl')
mastitis_scaler = joblib.load('../Mastitis_Detection/Models/mastitis_scaler.pkl')
mastitis_image_model = load_model('../Mastitis_Detection/Models/mastitis_model.h5')

# === Load FMD Models ===
foot_cnn = load_model('../FMD_Detection/models/foot_model.h5')
mouth_cnn = load_model('../FMD_Detection/models/mouth_model.h5')
foot_clf = joblib.load('../FMD_Detection/models/foot_text_model.pkl')
foot_scaler = joblib.load('../FMD_Detection/models/foot_text_scaler.pkl')
mouth_clf = joblib.load('../FMD_Detection/models/mouth_text_model.pkl')
mouth_scaler = joblib.load('../FMD_Detection/models/mouth_text_scaler.pkl')

IMG_SIZE = (224, 224)

# === Image Preprocessing ===
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img / 255.0, axis=0)

@app.route('/predict/mastitis', methods=['POST'])
def predict_mastitis():
    input_types = request.form.getlist('inputTypes[]')
    response = {}

    if 'text' in input_types:
        try:
            features = [
                float(request.form['mastitis_temperature']),
                float(request.form['mastitis_hardness']),
                float(request.form['mastitis_pain']),
                float(request.form['mastitis_milk_yield']),
                float(request.form['mastitis_milk_color'])
            ]
            scaled = mastitis_scaler.transform([features])
            text_result = mastitis_text_model.predict(scaled)[0]
            response['text_result'] = 'Mastitis Detected' if text_result == 1 else 'No Mastitis'
            response['text_confidence'] = float(max(mastitis_text_model.predict_proba(scaled)[0]))
        except Exception as e:
            response['text_error'] = str(e)

    if 'image' in input_types and 'udderImage' in request.files:
        image_file = request.files['udderImage']
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        try:
            image_file.save(temp.name)
            temp.close()
            processed = preprocess_image(temp.name)
            if processed is not None:
                prediction = mastitis_image_model.predict(processed)[0][0]
                response['image_result'] = 'Infected' if prediction > 0.5 else 'Non-infected'
                response['image_confidence'] = float(prediction if prediction > 0.5 else 1 - prediction)
        finally:
            os.unlink(temp.name)

    return jsonify(response)

@app.route('/predict/fmd', methods=['POST'])
def predict_fmd():
    input_types = request.form.getlist('inputTypes[]')
    response = {}
    foot_probs = []
    mouth_probs = []

    # Process text inputs
    if 'text_foot' in input_types:
        try:
            foot_features = [
                float(request.form['foot_text_temperature']),
                float(request.form['foot_text_milk_production']),
                float(request.form['foot_text_lethargy']),
                float(request.form['foot_text_difficulty_walking']),
                float(request.form['foot_text_foot_blister']),
                float(request.form['foot_text_foot_swelling']),
                float(request.form['foot_text_hoof_detachment'])
            ]
            scaled = foot_scaler.transform([foot_features])
            prob = foot_clf.predict_proba(scaled)[0][1]
            foot_probs.append(prob)
            response['foot_text_result'] = 'Infected' if prob > 0.5 else 'Healthy'
            response['foot_text_confidence'] = float(prob if prob > 0.5 else 1 - prob)
        except Exception as e:
            response['foot_text_error'] = str(e)

    if 'text_mouth' in input_types:
        try:
            mouth_features = [
                float(request.form['mouth_text_temperature']),
                float(request.form['mouth_text_milk_production']),
                float(request.form['mouth_text_lethargy']),
                float(request.form['mouth_text_mouth_ulcers']),
                float(request.form['mouth_text_mouth_blister']),
                float(request.form['mouth_text_salivation']),
                float(request.form['mouth_text_nasal_discharge'])
            ]
            scaled = mouth_scaler.transform([mouth_features])
            prob = mouth_clf.predict_proba(scaled)[0][1]
            mouth_probs.append(prob)
            response['mouth_text_result'] = 'Infected' if prob > 0.5 else 'Healthy'
            response['mouth_text_confidence'] = float(prob if prob > 0.5 else 1 - prob)
        except Exception as e:
            response['mouth_text_error'] = str(e)

    # Process image inputs
    if 'image_foot' in input_types and 'footImage' in request.files:
        image_file = request.files['footImage']
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        try:
            image_file.save(temp.name)
            temp.close()
            processed = preprocess_image(temp.name)
            if processed is not None:
                prob = float(foot_cnn.predict(processed)[0][0])
                foot_probs.append(prob)
                response['foot_image_result'] = 'Infected' if prob > 0.5 else 'Healthy'
                response['foot_image_confidence'] = float(prob if prob > 0.5 else 1 - prob)
        finally:
            os.unlink(temp.name)

    if 'image_mouth' in input_types and 'mouthImage' in request.files:
        image_file = request.files['mouthImage']
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        try:
            image_file.save(temp.name)
            temp.close()
            processed = preprocess_image(temp.name)
            if processed is not None:
                prob = float(mouth_cnn.predict(processed)[0][0])
                mouth_probs.append(prob)
                response['mouth_image_result'] = 'Infected' if prob > 0.5 else 'Healthy'
                response['mouth_image_confidence'] = float(prob if prob > 0.5 else 1 - prob)
        finally:
            os.unlink(temp.name)

    # Calculate combined prediction
    if foot_probs or mouth_probs:
        combined_probs = foot_probs + mouth_probs
        avg_prob = sum(combined_probs) / len(combined_probs)
        response['combined_result'] = 'Infected' if avg_prob > 0.5 else 'Healthy'
        response['combined_confidence'] = float(avg_prob if avg_prob > 0.5 else 1 - avg_prob)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)