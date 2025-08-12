from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and label encoder
model = None
label_encoder = None

def load_models():
    global model, label_encoder
    try:
        model = load_model('CNN_Covid19_Xray_Version.h5')
        with open('Label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("Model and label encoder loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models when the app starts
load_models()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
            
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize
        image_resized = cv2.resize(image, (150, 150))
        image_normalized = image_resized / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)
        
        return image_input, image_resized
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def get_class_probabilities(predictions, label_encoder):
    """Convert model predictions to class probabilities"""
    # Get class names from label encoder
    class_names = label_encoder.classes_
    # Convert predictions to probabilities using softmax
    exp_scores = np.exp(predictions[0] - np.max(predictions[0]))  # For numerical stability
    probabilities = exp_scores / np.sum(exp_scores)
    
    # Create dictionary of class: probability
    class_probabilities = {}
    for i, class_name in enumerate(class_names):
        class_probabilities[class_name] = float(probabilities[i]) * 100  # Convert to percentage
    return class_probabilities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            image_input, _ = preprocess_image(filepath)
            predictions = model.predict(image_input, verbose=0)
            
            # Get prediction results
            predicted_index = np.argmax(predictions)
            confidence = float(predictions[0][predicted_index]) * 100  # Convert to percentage
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]
            
            # Get class probabilities
            class_probabilities = get_class_probabilities(predictions, label_encoder)
            
            # Get emoji for prediction
            if predicted_label == "Covid-19":
                emoji = "🦠"
            elif predicted_label == "Normal":
                emoji = "✅"
            else:  # Pneumonia
                emoji = "🫁"
            
            # Model performance metrics
            model_metrics = {
                'overall_accuracy': 95.38,
                'covid': {'precision': 92.08, 'recall': 91.70},
                'normal': {'precision': 96.35, 'recall': 97.16},
                'pneumonia': {'precision': 96.86, 'recall': 91.82},
                'trained_on': 15153,
                'tested_on': 3031
            }
            
            # Clean up the temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'prediction': predicted_label,
                'emoji': emoji,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'model_metrics': model_metrics,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
