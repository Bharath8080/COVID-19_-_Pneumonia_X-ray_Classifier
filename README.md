# COVID-19 & Pneumonia X-ray Classifier

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow/keras-FF6F00?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

![Image](https://github.com/user-attachments/assets/ccbc968e-7f83-448a-aab8-a476fba96726)

A deep learning-based web application that classifies chest X-ray images into three categories: COVID-19, Normal, and Viral Pneumonia. The application uses a custom CNN model built with TensorFlow and Keras, and provides a user-friendly interface using Streamlit.

## 🚀 Demo

Try the live demo: [https://covid19x.streamlit.app/](https://covid19x.streamlit.app/)

## 📊 Model Performance

- **Overall Accuracy**: 95.38%
- **COVID-19**: 
  - Precision: 92.08%
  - Recall: 91.70%
- **Normal**: 
  - Precision: 96.35%
  - Recall: 97.16%
- **Pneumonia**: 
  - Precision: 96.86%
  - Recall: 91.82%

*Trained on 15,153 images, tested on 3,031 images*

## 🛠️ Features

- 🔍 Classify chest X-ray images into three categories
- 📊 View detailed prediction probabilities
- 🖼️ Image preview before analysis
- 🎨 Clean and intuitive user interface
- ⚡ Fast and accurate predictions

## 📚 Dataset

The model was trained on the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle, which contains chest X-ray images for:
- COVID-19 positive cases
- Normal (healthy) cases
- Viral Pneumonia cases

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bharath8080/Covid-19-Pneumonia_X-ray_Classifier.git
   cd Covid-19-Pneumonia_X-ray_Classifier
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload a chest X-ray image using the file uploader

4. View the prediction results and probability distribution

## 🧠 Model Architecture

The model is based on a Convolutional Neural Network (CNN) architecture with the following key components:
- Multiple Convolutional and MaxPooling layers
- Batch Normalization
- Dropout for regularization
- Dense layers for classification

## 📂 Project Structure

```
Covid-19-Pneumonia_X-ray_Classifier/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── CNN_Covid19_Xray_Version.h5  # Trained model
├── Label_encoder.pkl       # Label encoder for class names
├── testimages/             # Sample test images
└── assets/                 # Static assets
```

## 📝 Requirements

- Python 3.9
- TensorFlow
- Streamlit
- OpenCV
- NumPy
- scikit-learn
- Pillow
- Matplotlib

## ⚠️ Important Note

This application is intended for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- Streamlit for the amazing web framework
- TensorFlow and Keras for deep learning capabilities
