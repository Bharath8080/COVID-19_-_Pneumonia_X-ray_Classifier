import streamlit as st
import cv2
import numpy as np
import pickle
import os
import requests
from tensorflow.keras.models import load_model
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="COVID-19 X-ray Classifier",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .covid-prediction {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .normal-prediction {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .pneumonia-prediction {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        color: #f57c00;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoder():
    """Load the trained model and label encoder"""
    model_path = "CNN_Covid19_Xray_Version.h5"
    encoder_path = "Label_encoder.pkl"
    
    def download_file(url, filename):
        """Download file silently"""
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except:
            return False
    
    # Check if files exist and are valid
    if not (os.path.exists(model_path) and os.path.exists(encoder_path) and 
            os.path.getsize(model_path) > 1024*1024):  # Model should be > 1MB
        
        # Download URLs
        model_url = "https://github.com/Bharath8080/Covid-19-Pneumonia_X-ray_Classifier/raw/main/CNN_Covid19_Xray_Version.h5"
        encoder_url = "https://github.com/Bharath8080/Covid-19-Pneumonia_X-ray_Classifier/raw/main/Label_encoder.pkl"
        
        # Try downloading
        if not (download_file(model_url, model_path) and download_file(encoder_url, encoder_path)):
            st.error("❌ Could not download model files. Please check your internet connection.")
            return None, None
    
    # Load files
    try:
        model = load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, image_size=150):
    """Preprocess the uploaded image for prediction"""
    image_array = np.array(image)
    
    # Handle different image formats
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    elif len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    
    # Resize and normalize
    image_resized = cv2.resize(image_array, (image_size, image_size))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    
    return image_input, image_resized

def predict_image(model, label_encoder, image_input):
    """Make prediction on the preprocessed image"""
    predictions = model.predict(image_input, verbose=0)
    predicted_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_index]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    
    # Get all class probabilities
    class_probabilities = {}
    for i, class_name in enumerate(label_encoder.classes_):
        class_probabilities[class_name] = predictions[0][i] * 100
    
    return predicted_label, confidence_score * 100, class_probabilities

def display_prediction_result(predicted_label, confidence_score, class_probabilities):
    """Display the prediction result with appropriate styling"""
    # Determine styling based on prediction
    if predicted_label == "Covid-19":
        css_class, emoji = "covid-prediction", "🦠"
    elif predicted_label == "Normal":
        css_class, emoji = "normal-prediction", "✅"
    else:
        css_class, emoji = "pneumonia-prediction", "🫁"
    
    # Display main prediction
    st.markdown(f"""
    <div class="prediction-box {css_class}">
        <h2>{emoji} Prediction: {predicted_label}</h2>
        <p class="confidence-text">Confidence: {confidence_score:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display class probabilities
    st.subheader("📊 Class Probabilities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("COVID-19", f"{class_probabilities['Covid-19']:.1f}%")
    with col2:
        st.metric("Normal", f"{class_probabilities['Normal']:.1f}%")
    with col3:
        st.metric("Pneumonia", f"{class_probabilities['Pneumonia']:.1f}%")

def main():
    # Sidebar
    st.sidebar.image(
        "https://static.vecteezy.com/system/resources/thumbnails/060/046/568/small_2x/melancholic-gorgeous-doctor-examining-x-ray-no-background-with-transparent-background-luxury-free-png.png",
        use_container_width=True,
        caption="AI-Powered X-ray Analysis"
    )
    
    with st.sidebar.expander("📈 Model Performance"):
        st.markdown("""
        *Test Results:*
        - Overall Accuracy: 95.38%
        - COVID-19: 92.08% Precision
        - Normal: 96.35% Precision
        - Pneumonia: 96.86% Precision
        """)
    
    # Main header
    st.markdown('<h1 class="main-header">🩺 COVID-19 & Pneumonia X-ray Classifier</h1>', unsafe_allow_html=True)
    
    # Load model
    model, label_encoder = load_model_and_encoder()
    
    if model is None or label_encoder is None:
        st.error("❌ Could not load model files. Please try refreshing the page.")
        return
    
    # File uploader
    st.markdown("<h3 style='text-align: center;'>📤 Upload X-ray Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=['png', 'jpg', 'jpeg'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("Analyzing..."):
                try:
                    image_input, _ = preprocess_image(image)
                    predicted_label, confidence_score, class_probabilities = predict_image(
                        model, label_encoder, image_input
                    )
                    display_prediction_result(predicted_label, confidence_score, class_probabilities)
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
        
        # Disclaimer
        st.warning("""
        *Medical Disclaimer:* This tool is for educational purposes only. 
        Not a substitute for professional medical advice. Always consult healthcare professionals.
        """)

if _name_ == "_main_":
    main()
