import streamlit as st
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="COVID-19 X-ray Classifier",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better styling
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
    try:
        model = load_model('CNN_Covid19_Xray_Version.h5')
        with open('Label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please ensure 'CNN_Covid19_Xray_Version.h5' and 'Label_encoder.pkl' are in the same directory as this script.")
        return None, None

def preprocess_image(image, image_size=150):
    """Preprocess the uploaded image for prediction"""
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # If image has 4 channels (RGBA), convert to RGB
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Already RGB
        pass
    else:
        # Convert grayscale to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    
    # Resize the image
    image_resized = cv2.resize(image_array, (image_size, image_size))
    
    # Normalize pixel values
    image_normalized = image_resized / 255.0
    
    # Add batch dimension
    image_input = np.expand_dims(image_normalized, axis=0)
    
    return image_input, image_resized

def predict_image(model, label_encoder, image_input):
    """Make prediction on the preprocessed image"""
    # Predict
    predictions = model.predict(image_input, verbose=0)
    
    # Get predicted class and confidence
    predicted_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_index]
    
    # Decode the prediction
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    
    # Get all class probabilities
    class_probabilities = {}
    for i, class_name in enumerate(label_encoder.classes_):
        class_probabilities[class_name] = predictions[0][i] * 100
    
    return predicted_label, confidence_score * 100, class_probabilities

def display_prediction_result(predicted_label, confidence_score, class_probabilities):
    """Display the prediction result with appropriate styling"""
    # Determine the CSS class based on prediction
    if predicted_label == "Covid-19":
        css_class = "covid-prediction"
        emoji = "🦠"
    elif predicted_label == "Normal":
        css_class = "normal-prediction"
        emoji = "✅"
    else:  # Pneumonia
        css_class = "pneumonia-prediction"
        emoji = "🫁"
    
    # Display main prediction
    st.markdown(f"""
    <div class="prediction-box {css_class}">
        <h2>{emoji} Prediction: {predicted_label}</h2>
        <p class="confidence-text">Confidence: {confidence_score:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display all class probabilities with colored metrics
    st.subheader("📊 Class Probabilities")
    col1, col2, col3 = st.columns(3)
    
    # Custom CSS for metrics
    st.markdown("""
    <style>
        .covid-metric { color: #c62828 !important; font-weight: bold !important; }
        .normal-metric { color: #2e7d32 !important; font-weight: bold !important; }
        .pneumonia-metric { color: #f57c00 !important; font-weight: bold !important; }
        .metric-value { font-size: 1.2rem !important; }
    </style>
    """, unsafe_allow_html=True)
    
    with col1:
        st.markdown(f'<div class="covid-metric">COVID-19</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{class_probabilities["Covid-19"]:.2f}%</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="normal-metric">Normal</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{class_probabilities["Normal"]:.2f}%</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="pneumonia-metric">Pneumonia</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{class_probabilities["Pneumonia"]:.2f}%</div>', unsafe_allow_html=True)

def main():
    # Add image to sidebar
    st.sidebar.image(
        "https://static.vecteezy.com/system/resources/thumbnails/060/046/568/small_2x/melancholic-gorgeous-doctor-examining-x-ray-no-background-with-transparent-background-luxury-free-png.png",
        use_container_width=True,
        caption="AI-Powered X-ray Analysis"
    )
    
    # Main header
    st.markdown('<h1 class="main-header">🩺 COVID-19 X-ray Classifier🫁🩻</h1>', unsafe_allow_html=True)
    
    # Load model and encoder
    model, label_encoder = load_model_and_encoder()
    
    if model is None or label_encoder is None:
        st.error("❌ Failed to load model files. Please check if the model files are present.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.header("📤 Upload X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear chest X-ray image for analysis"
    )
    
    if uploaded_file is not None:
        # Create two columns for image display and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display uploaded image with fixed height
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            # Resize image for preview (maintaining aspect ratio)
            max_height = 300
            width_percent = (max_height / float(image.size[1]))
            new_width = int((float(image.size[0]) * float(width_percent)))
            image = image.resize((new_width, max_height), Image.Resampling.LANCZOS)
            st.image(image, caption="Uploaded X-ray", use_container_width=False, width=350)
        
        with col2:
            # Make prediction
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("Analyzing X-ray image..."):
                try:
                    # Preprocess image
                    image_input, processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predicted_label, confidence_score, class_probabilities = predict_image(
                        model, label_encoder, image_input
                    )
                    
                    # Display results
                    display_prediction_result(predicted_label, confidence_score, class_probabilities)
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
        
        # Additional information
        st.header("⚠ Important Disclaimer")
        st.warning("""
        *Medical Disclaimer:* This AI tool is for educational and research purposes only. 
        It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with qualified healthcare professionals for medical decisions.
        """)
        
        # Model performance metrics
        with st.expander("📈 Model Performance Metrics"):
            st.markdown("""
            *Test Set Results:*
            - *Overall Accuracy:* 95.38%
            - *COVID-19:* Precision: 92.08%, Recall: 91.70%
            - *Normal:* Precision: 96.35%, Recall: 97.16%
            - *Pneumonia:* Precision: 96.86%, Recall: 91.82%
            
            The model was trained on 15,153 images and tested on 3,031 images.
            """)

if __name__ == "__main__":
    main()
