# foot and mouth text input
import joblib
import pandas as pd

# Load the foot and mouth models and scalers
foot_model = joblib.load('C:/Users/spoor/OneDrive/Desktop/Major/FMD_Detection/models/foot_text_model.pkl')
foot_scaler = joblib.load('C:/Users/spoor/OneDrive/Desktop/Major/FMD_Detection/models/foot_text_scaler.pkl')

mouth_model = joblib.load('C:/Users/spoor/OneDrive/Desktop/Major/FMD_Detection/models/mouth_text_model.pkl')
mouth_scaler = joblib.load('C:/Users/spoor/OneDrive/Desktop/Major/FMD_Detection/models/mouth_text_scaler.pkl')

# Features for foot and mouth models
foot_features = ['temperature', 'milk_production', 'lethargy', 'difficulty_in_walking', 
                 'foot_blister', 'foot_swelling', 'hoof_detachment']

mouth_features = ['temperature', 'milk_production', 'lethargy', 'mouth_ulcers', 
                  'mouth_blister', 'salivation', 'nasal_discharge']

# Function to validate input within range
def validate_input(value, min_value, max_value, input_type="float"):
    try:
        if input_type == "float":
            value = float(value)
            if not min_value <= value <= max_value:
                raise ValueError(f"Value should be between {min_value} and {max_value}.")
        elif input_type == "int":
            value = int(value)
            if not min_value <= value <= max_value:
                raise ValueError(f"Value should be between {min_value} and {max_value}.")
        return value
    except ValueError as e:
        print(f"Invalid input: {e}")
        return None

# Function to take user input for common features (temperature, milk production, lethargy)
def get_common_input():
    print("\nEnter the following common features for both Foot and Mouth FMD prediction:")
    
    # Validate common data inputs
    temperature = None
    while temperature is None:
        temperature_input = input("Temperature (in °C): ")
        temperature = validate_input(temperature_input, 37.5, 41.0, "float")
    
    milk_production = None
    while milk_production is None:
        milk_production_input = input("Milk production (in liters, scale 0-5): ")
        milk_production = validate_input(milk_production_input, 0, 5, "int")
    
    lethargy = None
    while lethargy is None:
        lethargy_input = input("Lethargy (0 for no, 1 for yes): ")
        lethargy = validate_input(lethargy_input, 0, 1, "int")
    
    return [temperature, milk_production, lethargy]

# Function to take user input for foot-specific features
def get_foot_input(common_data):
    print("\nEnter the following features for Foot FMD prediction:")
    foot_data = common_data.copy()  # Copy common data for foot model
    
    # Validate foot data inputs
    difficulty_in_walking = None
    while difficulty_in_walking is None:
        walking_input = input("Difficulty in walking (0 for no, 1 for yes): ")
        difficulty_in_walking = validate_input(walking_input, 0, 1, "int")
    
    foot_blister = None
    while foot_blister is None:
        blister_input = input("Foot blister (0-5): ")
        foot_blister = validate_input(blister_input, 0, 5, "int")
    
    foot_swelling = None
    while foot_swelling is None:
        swelling_input = input("Foot swelling (0-5): ")
        foot_swelling = validate_input(swelling_input, 0, 5, "int")
    
    hoof_detachment = None
    while hoof_detachment is None:
        hoof_input = input("Hoof detachment (0-5): ")
        hoof_detachment = validate_input(hoof_input, 0, 5, "int")
    
    foot_data.extend([difficulty_in_walking, foot_blister, foot_swelling, hoof_detachment])
    
    return pd.DataFrame([foot_data], columns=foot_features)

# Function to take user input for mouth-specific features
def get_mouth_input(common_data):
    print("\nEnter the following features for Mouth FMD prediction:")
    mouth_data = common_data.copy()  # Copy common data for mouth model
    
    # Validate mouth data inputs
    mouth_ulcers = None
    while mouth_ulcers is None:
        ulcers_input = input("Mouth ulcers (0-5): ")
        mouth_ulcers = validate_input(ulcers_input, 0, 5, "int")
    
    mouth_blister = None
    while mouth_blister is None:
        blister_input = input("Mouth blister (0-5): ")
        mouth_blister = validate_input(blister_input, 0, 5, "int")
    
    salivation = None
    while salivation is None:
        salivation_input = input("Salivation (0 for normal, 1 for excessive): ")
        salivation = validate_input(salivation_input, 0, 1, "int")
    
    nasal_discharge = None
    while nasal_discharge is None:
        discharge_input = input("Nasal discharge (0 for no, 1 for yes): ")
        nasal_discharge = validate_input(discharge_input, 0, 1, "int")
    
    mouth_data.extend([mouth_ulcers, mouth_blister, salivation, nasal_discharge])
    
    return pd.DataFrame([mouth_data], columns=mouth_features)

# Function to predict combined FMD
def predict_fmd():
    # Get common input (temperature, milk production, lethargy)
    common_input = get_common_input()
    
    # Get foot and mouth-specific input
    foot_input = get_foot_input(common_input)
    mouth_input = get_mouth_input(common_input)
    
    # Scale the inputs using the scalers
    foot_input_scaled = foot_scaler.transform(foot_input)
    mouth_input_scaled = mouth_scaler.transform(mouth_input)
    
    # Make predictions for foot and mouth FMD
    foot_prediction = foot_model.predict(foot_input_scaled)[0]
    foot_probabilities = foot_model.predict_proba(foot_input_scaled)[0]
    
    mouth_prediction = mouth_model.predict(mouth_input_scaled)[0]
    mouth_probabilities = mouth_model.predict_proba(mouth_input_scaled)[0]

    # Display results for foot FMD
    print("\nFoot FMD Prediction Result:")
    print(f"FMD Status: {'Infected' if foot_prediction == 1 else 'Healthy'}")
    print("Prediction Probabilities:")
    for i, cls in enumerate(foot_model.classes_):
        print(f"{cls}: {foot_probabilities[i]:.4f}")
    
    # Display results for mouth FMD
    print("\nMouth FMD Prediction Result:")
    print(f"FMD Status: {'Infected' if mouth_prediction == 1 else 'Healthy'}")
    print("Prediction Probabilities:")
    for i, cls in enumerate(mouth_model.classes_):
        print(f"{cls}: {mouth_probabilities[i]:.4f}")

    # Compute combined infected probability
    combined_infected_prob = (
        foot_probabilities[1] * mouth_probabilities[1] +
        foot_probabilities[1] * mouth_probabilities[0] +
        foot_probabilities[0] * mouth_probabilities[1]
    )/3
    
    combined_healthy_prob = 1 - combined_infected_prob
    
    # Final prediction based on higher combined probability
    if combined_infected_prob > combined_healthy_prob:
        final_prediction = 'Infected'
    else:
        final_prediction = 'Healthy'
    
    final_probabilities = {
        'Infected': combined_infected_prob,
        'Healthy': combined_healthy_prob
    }


    # Display combined FMD prediction
    print("\nCombined FMD Prediction Result:")
    print(f"FMD Status: {final_prediction}")
    print("Prediction Probabilities:")
    for status, prob in final_probabilities.items():
        print(f"{status}: {prob:.4f}")

# Main function to run the prediction
predict_fmd()