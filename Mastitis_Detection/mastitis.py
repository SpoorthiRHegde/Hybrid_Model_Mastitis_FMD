from src.text_mastitis import predict_text
from src.mastitis_detection import predict_image

def get_valid_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if 0 <= value <= 5:
                return value
            else:
                print("Please enter a number between 0 and 5.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def get_text_input():
    temperature = input("Enter temperature (in Celsius): ")
    hardness = get_valid_input("Enter hardness (0–5): ")
    pain = get_valid_input("Enter pain level (0–5): ")
    milk_yield = get_valid_input("Enter milk yield (0–5): ")
    milk_color = get_valid_input("Enter milk color (0–5): ")
    return [temperature, hardness, pain, milk_yield, milk_color]


def main():
    input_type = input("Enter input type (1) Text 2) Image 3) Image and Text: ")

    if input_type == '1':
        # Handle Text Input
        features = get_text_input()
        result = predict_text(features)
        print(f"Text prediction: {result}")

    elif input_type == '2':
        # Handle Image Input
        image_path = input("Enter image path: ")
        result = predict_image(image_path)
        print(f"Image prediction: {result}")

    elif input_type == '3':
        # Handle Both Text and Image Input
        features = get_text_input()
        text_result = predict_text(features)
        image_path = input("Enter image path: ")
        image_result = predict_image(image_path)

        # Combine predictions with equal weight
        if text_result == "Mastitis Detected" and image_result == "Infected":
            final_result = "Mastitis Detected"
        elif text_result == "No Mastitis" and image_result == "Non-infected":
            final_result = "No Mastitis"
        else:
    # When results conflict, give more weight to text_result
            if text_result == "Mastitis Detected":
                final_result = "Mastitis Detected "
            elif text_result == "No Mastitis":
                final_result = "No Mastitis "
            else:
                final_result = "Uncertain"


        print(f"Combined prediction: {final_result}")

if __name__ == "__main__":
    main()
