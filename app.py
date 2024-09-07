from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import google.generativeai as gemini
from flask_cors import CORS

# Configure the Gemma API
gemini.configure(api_key="AIzaSyD1FPKl0lENNaIw8JGtMBzPXopVDIqcab8")
gemma_model = gemini.GenerativeModel("gemini-1.5-flash")  # Use a separate variable for the Gemma model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the saved Alzheimer's disease prediction model
prediction_model = load_model('prediction_model2.h5')  # Use a separate variable for the prediction model

# Define image processing constants
IMG_HEIGHT = 72
IMG_WIDTH = 72

# Directory paths for training dataset to get class names
train_dir = "Alzheimer_s Dataset/train/"

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Convert single image to a batch.
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict the class of the image and calculate risk factor
def predict_image_with_risk(img_path):
    img_array = preprocess_image(img_path)
    predictions = prediction_model.predict(img_array)  # Use the correct variable here

    # Get class names from the training directory
    class_names = [name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))]

    # Get the predicted class and probabilities
    predicted_class = class_names[np.argmax(predictions)]
    predicted_probability = predictions[0][np.argmax(predictions)]

    # Calculate risk factor only for the predicted class
    risk_factor = f"{predicted_probability * 100:.2f}%"

    return predicted_class, risk_factor

# Function to generate a prevention report using the Gemma API
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
       - Purpose of the report
       - Context of general health and wellness

    2. **Risk Description**
       - General description of the identified risk
       - Common factors associated with the risk

    3. **Stage of Risk**
       - General information about the risk stage
       - Any typical considerations

    4. **Risk Assessment**
       - General overview of the risk's impact on health

    5. **Findings**
       - General wellness observations
       - Supporting information

    6. **Recommendations**
       - General wellness tips and lifestyle changes
       - Actions to promote well-being

    7. **Way Forward**
       - Suggested next steps for maintaining health
       - Advanced follow-up actions for this risk, like how we can overcome it.

    8. **Conclusion**
       - Summary of overall wellness advice
       - General support resources

    9. **Contact Information**
       - Information for general inquiries

    10. **References**
        - Simplified wellness resources (if applicable)

    **Details:**
    Risk: {risk}%
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """

    
    try:
        response = gemma_model.generate_content(prompt)  # Use the correct variable for Gemma API
        return response.text if response and hasattr(response, 'text') else "No content generated."
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return None

# API route to upload an image and get a prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded."}), 400
        
        # Read the image file
        image_file = request.files['image']
        image_path = 'uploaded_image.png'
        image_file.save(image_path)
        
        # Predict the class and risk level
        predicted_class, risk_factors = predict_image_with_risk(image_path)
        
        # Generate the prevention report
        age = request.form.get('age', 30)  # Default age is 30, can be changed with dynamic input
        risk_name = "Alzheimer's"
        disease = predicted_class
        report = generate_prevention_report(risk_name, disease, age)
        
        # Return the results as a JSON response
        return jsonify({
            "risk": risk_name,
            "predicted_class": predicted_class,
            "risk_factors": risk_factors,
            "wellness_report": report
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
