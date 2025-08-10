from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
import joblib
import cv2
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
import pandas as pd
import google.generativeai as genai
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Configure Gemini API
#GOOGLE_API_KEY = "AIzaSyBxLKlJGn9K-sHeBMdxjJ5MEycNXpd3jYc"
genai.configure(api_key="AIzaSyC35sQNpirJ-i0TumT3mk2EFoHdCZLpSU0")  # <-- Replace with your API key

chat_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"temperature": 0.7, "top_p": 1, "top_k": 1},
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
)

system_instructions = """
You are a veterinary assistant chatbot answering all queries of farmers related to Mastitis and Foot and Mouth Disease (FMD) in dairy cows.
- Provide care suggestions (cleaning, isolation, contacting a vet).
- Answer general questions also regarding cow health but focus mainly on fmd and mastitis.
- Respond in the user's language if you can. Default to English.
- Never diagnose definitively.
- Also answer:Normal health metrics (temperature, pulse, udder checks) and basic signs of illness relevant to Mastitis/FMD detection in cows.
- Recommend consulting a vet when fmd or mastitis symptoms are high .
- when the user says thanks, respond politely and ask if they have more questions
- Only greet if the user says hello/hi first
- Never repeat greetings in conversation
- If no greeting, go straight to the point
- If asked for more information or details on specific topic: Share *new* extra info only. Never repeat.
- For irrelevant queries: 
  "Sorry, I only help with Mastitis and FMD in cows."
Use short, simple answers suitable for rural farmers.
"""


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

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print("Received data:", data)  # debug print

        message = data.get('message', '').strip()
        print("Message:", message)  # debug print
        lat = data.get('latitude',None)
        lng = data.get('longitude',None)
        vet_keywords = ['vet', 'veterinarian', 'doctor', 'clinic', 'suggest vet', 'nearby doctor']
        if any(word in message.lower() for word in vet_keywords):
            if lat and lng:
                # Use Google Maps Places API
                maps_url = f"https://www.google.com/maps/search/vet+clinic/@{lat},{lng},14z"
                reply = f"📍 You can find nearby vets using Google Maps:\n{maps_url}"
            else:
                reply = "📍 Please allow location access so I can suggest nearby vets."
            return jsonify({"response": reply})

        # if not message:
        #     return jsonify({"response": "❌ Please enter a message."})

        prompt = f""" {system_instructions} You are a veterinary assistant. Answer briefly (5-6 sentences). Help dairy farmers by explaining or advising about Mastitis or Foot and Mouth Disease (FMD) in cows.\n\nUser: {message}\nAssistant:"""

        # DEBUG: print prompt
        print("Prompt:", prompt)

        gemini_response = chat_model.generate_content(prompt)
        print("Gemini response object:", gemini_response)

        if hasattr(gemini_response, "text") and gemini_response.text:
            reply = gemini_response.text.strip()
        else:
            reply = "❌ I couldn't understand that. Try asking in another way."

        return jsonify({"response": reply})

    except Exception as e:
        print("🔥 ERROR in /chat:", str(e))  # <-- Important
        return jsonify({
            "response": "⚠️ Something went wrong while generating a response.",
            "error": str(e)
        }), 500



# @app.route('/run_chatbot', methods=['POST'])
# def run_chatbot():
#     try:
#         # Execute the chatbot.py script and capture output
#         result = subprocess.run(['python', 'chatbot.py'], capture_output=True, text=True, check=True)
#         chatbot_output = result.stdout
#         return jsonify({'chatbot_output': chatbot_output})
#     except subprocess.CalledProcessError as e:
#         return jsonify({'error': f'Error running chatbot script: {e.stderr}'}), 500
#     except Exception as e:
#         return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)