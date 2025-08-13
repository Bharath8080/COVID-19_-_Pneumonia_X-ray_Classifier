import streamlit as st
import cv2
import numpy as np
import pickle
import os
import requests
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
    """Load the trained model and label encoder with proper LFS handling"""
    model_path = "CNN_Covid19_Xray_Version.h5"
    encoder_path = "Label_encoder.pkl"
    
    # Try multiple URL strategies for GitHub LFS files
    urls_to_try = [
        # Strategy 1: GitHub LFS direct download (if properly configured)
        {
            "model": "https://github.com/Bharath8080/Covid-19-Pneumonia_X-ray_Classifier/raw/main/CNN_Covid19_Xray_Version.h5",
            "encoder": "https://github.com/Bharath8080/Covid-19-Pneumonia_X-ray_Classifier/raw/main/Label_encoder.pkl"
        },
        # Strategy 2: GitHub releases (if you upload there)
        {
            "model": "https://github.com/Bharath8080/Covid-19-Pneumonia_X-ray_Classifier/releases/download/v1.0/CNN_Covid19_Xray_Version.h5",
            "encoder": "https://github.com/Bharath8080/Covid-19-Pneumonia_X-ray_Classifier/releases/download/v1.0/Label_encoder.pkl"
        }
    ]
    
    def download_file_with_validation(url, filename, min_size_mb=1):
        """Download file and validate it's not empty/corrupted"""
        try:
            st.info(f"🔄 Downloading {filename} from GitHub...")
            
            # Setup session with proper headers
            session = requests.Session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/octet-stream',
                'Accept-Encoding': 'identity'  # Prevent compression issues
            }
            
            # Download with streaming
            response = session.get(url, headers=headers, stream=True, timeout=300)
            response.raise_for_status()
            
            # Check if we got an LFS pointer file instead of the actual file
            content_type = response.headers.get('content-type', '')
            if 'text/plain' in content_type:
                # This might be an LFS pointer file
                first_chunk = next(response.iter_content(chunk_size=1024))
                if b'version https://git-lfs.github.com/spec/v1' in first_chunk:
                    st.warning(f"⚠️ Received LFS pointer for {filename}. File is stored with Git LFS.")
                    return False
            
            # Write file
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            downloaded = 0
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
            
            progress_bar.empty()
            
            # Validate file size
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            if filename.endswith('.h5') and file_size_mb < min_size_mb:
                st.error(f"❌ {filename} is too small ({file_size_mb:.2f} MB). Expected larger model file.")
                return False
            
            st.success(f"✅ {filename} downloaded successfully! ({file_size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to download {filename}: {str(e)}")
            return False
    
    # Check if files already exist and are valid
    files_exist = os.path.exists(model_path) and os.path.exists(encoder_path)
    if files_exist:
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        if model_size > 1:  # Model should be larger than 1MB
            st.info("📁 Using existing model files...")
        else:
            files_exist = False
    
    # Download files if needed
    if not files_exist:
        success = False
        
        for i, urls in enumerate(urls_to_try):
            st.info(f"🔄 Trying download strategy {i+1}...")
            
            # Try to download model
            if download_file_with_validation(urls["model"], model_path):
                # Try to download encoder
                if download_file_with_validation(urls["encoder"], encoder_path, min_size_mb=0):
                    success = True
                    break
            
            # Clean up partial downloads
            for file in [model_path, encoder_path]:
                if os.path.exists(file):
                    os.remove(file)
        
        if not success:
            st.error("❌ All download strategies failed!")
            st.info("🔧 **Solutions:**")
            st.info("1. **Use GitHub Releases**: Upload your files as release assets")
            st.info("2. **Use external hosting**: Google Drive, Dropbox, etc.")
            st.info("3. **Contact repository owner** to ensure files are properly uploaded")
            
            # Show some debugging info
            st.info("🐛 **Debug Info:**")
            st.code(f"Current directory: {os.getcwd()}")
            st.code(f"Directory contents: {os.listdir('.')}")
            
            return None, None
    
    # Load the model and encoder
    try:
        st.info("🔄 Loading model and encoder...")
        
        # Verify file sizes before loading
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        encoder_size = os.path.getsize(encoder_path) / 1024
        
        st.info(f"📊 Model file: {model_size:.2f} MB")
        st.info(f"📊 Encoder file: {encoder_size:.2f} KB")
        
        if model_size < 1:
            st.error("❌ Model file is too small - likely corrupted or incomplete")
            return None, None
        
        # Load model
        model = load_model(model_path)
        
        # Load encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        st.success("✅ Model and encoder loaded successfully!")
        return model, label_encoder
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        
        # Provide specific error guidance
        if "file signature not found" in str(e):
            st.error("🔍 **Diagnosis**: The .h5 file is corrupted or not a valid Keras model")
            st.info("💡 **Solution**: Re-upload the model file to GitHub properly")
        elif "No module named" in str(e):
            st.error("🔍 **Diagnosis**: Missing required dependencies")
        else:
            st.error("🔍 **Diagnosis**: Unknown error during model loading")
        
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
    
    # Add model info to sidebar
    with st.sidebar.expander("📈 Model Performance", expanded=True):
        st.markdown("""
        **Test Set Results:**
        - **Overall Accuracy:** 95.38%
        - **COVID-19:** Precision: 92.08%, Recall: 91.70%
        - **Normal:** Precision: 96.35%, Recall: 97.16%
        - **Pneumonia:** Precision: 96.86%, Recall: 91.82%
        
        Trained on 15,153 images, tested on 3,031 images.
        """)
    
    # Main header
    st.markdown('<h1 class="main-header">🩺 COVID-19 & Pneumonia X-ray Classifier 🫁🩻</h1>', unsafe_allow_html=True)
    
    # Add file upload instructions
    st.info("🚀 **First time setup**: The app will automatically download model files from GitHub. This may take a few minutes.")
    
    # Load model and encoder
    model, label_encoder = load_model_and_encoder()
    
    if model is None or label_encoder is None:
        st.error("❌ Could not load model files.")
        st.markdown("""
        ### 🔧 **Immediate Solutions:**
        
        1. **GitHub Releases Approach** (Recommended):
           - Go to your GitHub repo
           - Click "Releases" → "Create a new release"
           - Upload your .h5 and .pkl files as assets
           - Update the code with release URLs
        
        2. **Alternative Hosting**:
           - Upload to Google Drive (make public)
           - Use Dropbox public links
           - Try Hugging Face Model Hub
        
        3. **Local Files**:
           - Download files manually from Kaggle
           - Add directly to your GitHub repository (if <100MB)
        """)
        return
    
    # File uploader - centered
    st.markdown("<h3 style='text-align: center;'>📤 Upload X-ray Image</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear chest X-ray image for analysis",
            label_visibility="collapsed"
        )
    
    if uploaded_file is not None:
        # Create two columns for image display and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display uploaded image
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
        
        # Additional information - centered
        st.markdown("<h2 style='text-align: center;'>⚠ Important Disclaimer</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.warning("""
            **Medical Disclaimer:** This AI tool is for educational and research purposes only. 
            It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult with qualified healthcare professionals for medical decisions.
            """)

if __name__ == "__main__":
    main()
