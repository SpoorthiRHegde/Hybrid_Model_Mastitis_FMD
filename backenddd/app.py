import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load all models at startup
try:
    # Mastitis models
    mastitis_cnn = load_model(r"C:\Users\spoor\OneDrive\Desktop\Major\Mastitis_Detection\models\mastitis_model.h5")
    mastitis_scaler = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\Mastitis_Detection\models\mastitis_scaler.pkl")
    mastitis_clf = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\Mastitis_Detection\models\mastitis_text_model.pkl")
    
    # FMD models
    foot_cnn = load_model(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\foot_model.h5")
    mouth_cnn = load_model(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\mouth_model.h5")
    foot_clf = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\foot_text_model.pkl")
    foot_scaler = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\foot_text_scaler.pkl")
    mouth_clf = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\mouth_text_model.pkl")
    mouth_scaler = joblib.load(r"C:\Users\spoor\OneDrive\Desktop\Major\FMD_Detection\models\mouth_text_scaler.pkl")
    
    print("All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise e

def predict_mastitis_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        prediction = mastitis_cnn.predict(img_array)
        return "Infected" if prediction[0][0] > 0.5 else "Non-infected"
    except Exception as e:
        print(f"Error in mastitis image prediction: {str(e)}")
        return f"Error: {str(e)}"

def predict_mastitis_text(features):
    try:
        features_array = np.array(features).reshape(1, -1)
        scaled_features = mastitis_scaler.transform(features_array)
        prediction = mastitis_clf.predict(scaled_features)
        return "Mastitis Detected" if prediction[0] == 1 else "No Mastitis"
    except Exception as e:
        print(f"Error in mastitis text prediction: {str(e)}")
        return f"Error: {str(e)}"

def predict_image_fmd(image_path, image_type):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        model = foot_cnn if image_type == "foot" else mouth_cnn
        prediction = model.predict(img_array)
        return "Infected" if prediction[0][0] > 0.5 else "Non-infected"
    except Exception as e:
        print(f"Error in FMD {image_type} image prediction: {str(e)}")
        return f"Error: {str(e)}"

def predict_text_fmd(features, input_type):
    try:
        scaler = foot_scaler if input_type == "foot" else mouth_scaler
        clf = foot_clf if input_type == "foot" else mouth_clf
        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        prediction = clf.predict(scaled_features)
        return "FMD Detected" if prediction[0] == 1 else "No FMD"
    except Exception as e:
        print(f"Error in FMD {input_type} text prediction: {str(e)}")
        return f"Error: {str(e)}"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure we're getting multipart form data
        if not request.content_type.startswith('multipart/form-data'):
            return jsonify({"status": "error", "message": "Content-Type must be multipart/form-data"}), 400

        # Get form data and files
        data = request.form
        files = request.files
        
        disease = data.get("disease")
        if not disease:
            return jsonify({"status": "error", "message": "No disease specified"}), 400

        if disease == "mastitis":
            return handle_mastitis(data, files)
        elif disease == "fmd":
            return handle_fmd(data, files)
        else:
            return jsonify({"status": "error", "message": "Unknown disease"}), 400
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def handle_mastitis(data, files):
    input_type = data.get("inputType")
    if not input_type:
        return jsonify({"status": "error", "message": "No input type specified"}), 400

    result = {"status": "success", "details": {}}
    
    # Text processing
    if input_type in ["text", "both"]:
        try:
            features = [
                float(data.get("temperature", 0)),
                float(data.get("hardness", 0)),
                float(data.get("pain", 0)),
                float(data.get("milk_yield", 0)),
                float(data.get("milk_color", 0)),
            ]
            text_result = predict_mastitis_text(features)
            result["details"]["text"] = {
                "result": text_result,
                "features": features
            }
        except Exception as e:
            result["details"]["text"] = {"error": str(e)}

    # Image processing
    if input_type in ["image", "both"]:
        if "image" in files:
            file = files["image"]
            if file and file.filename:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    image_result = predict_mastitis_image(filepath)
                    result["details"]["image"] = {
                        "result": image_result,
                        "filename": filename
                    }
                except Exception as e:
                    result["details"]["image"] = {"error": str(e)}
            else:
                result["details"]["image"] = {"error": "No image provided"}
        else:
            result["details"]["image"] = {"error": "No image uploaded"}

    # Format final result
    if input_type == "both":
        text_res = result["details"].get("text", {}).get("result", "")
        img_res = result["details"].get("image", {}).get("result", "")
        result["result"] = f"Text: {text_res}, Image: {img_res}"
    elif input_type == "text":
        result["result"] = result["details"].get("text", {}).get("result", "No text result")
    else:
        result["result"] = result["details"].get("image", {}).get("result", "No image result")

    return jsonify(result)

def handle_fmd(data, files):
    result = {"status": "success", "details": {}}
    
    # Process foot text
    if data.get("foot_text") == "true":
        try:
            features = [
                float(data.get("ft_temp", 0)),
                float(data.get("ft_milk", 0)),
                float(data.get("ft_lethargy", 0)),
                float(data.get("ft_walk", 0)),
                float(data.get("ft_blister", 0)),
                float(data.get("ft_swelling", 0)),
                float(data.get("ft_hoof", 0)),
            ]
            result["details"]["foot_text"] = {
                "result": predict_text_fmd(features, "foot"),
                "features": features
            }
        except Exception as e:
            result["details"]["foot_text"] = {"error": str(e)}

    # Process mouth text
    if data.get("mouth_text") == "true":
        try:
            features = [
                float(data.get("mt_temp", 0)),
                float(data.get("mt_milk", 0)),
                float(data.get("mt_lethargy", 0)),
                float(data.get("mt_ulcers", 0)),
                float(data.get("mt_blister", 0)),
                float(data.get("mt_salivation", 0)),
                float(data.get("mt_discharge", 0)),
            ]
            result["details"]["mouth_text"] = {
                "result": predict_text_fmd(features, "mouth"),
                "features": features
            }
        except Exception as e:
            result["details"]["mouth_text"] = {"error": str(e)}

    # Process foot image
    if data.get("foot_image") == "true":
        file_key = "foot_image_file"
        if file_key in files:
            file = files[file_key]
            if file and file.filename:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    image_result = predict_image_fmd(filepath, "foot")
                    result["details"]["foot_image"] = {
                        "result": image_result,
                        "filename": filename
                    }
                except Exception as e:
                    result["details"]["foot_image"] = {"error": str(e)}
            else:
                result["details"]["foot_image"] = {"error": "No foot image provided"}
        else:
            result["details"]["foot_image"] = {"error": "No foot image uploaded"}

    # Process mouth image
    if data.get("mouth_image") == "true":
        file_key = "mouth_image_file"
        if file_key in files:
            file = files[file_key]
            if file and file.filename:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    image_result = predict_image_fmd(filepath, "mouth")
                    result["details"]["mouth_image"] = {
                        "result": image_result,
                        "filename": filename
                    }
                except Exception as e:
                    result["details"]["mouth_image"] = {"error": str(e)}
            else:
                result["details"]["mouth_image"] = {"error": "No mouth image provided"}
        else:
            result["details"]["mouth_image"] = {"error": "No mouth image uploaded"}

    # Format final result
    detected = []
    for input_type, detail in result["details"].items():
        if "result" in detail and ("Detected" in detail["result"] or "Infected" in detail["result"]):
            detected.append(input_type.replace("_", " "))

    result["result"] = f"FMD Detected in: {', '.join(detected)}" if detected else "No FMD Detected"
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)