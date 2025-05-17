import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = 'C:/Users/spoor/OneDrive/Desktop/Major/Mastitis_Detection/models/mastitis_model.h5'
IMG_SIZE = (224, 224)

# Load the pre-trained model
model = load_model(MODEL_PATH)

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid image path."
    img = enhance_image(img)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img / 255.0, axis=0)
    prediction = model.predict(img)[0][0]
    return "Infected" if prediction > 0.5 else "Non-infected"
