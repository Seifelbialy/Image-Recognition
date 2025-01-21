import streamlit as st
import numpy as np
import joblib
from PIL import Image
from sklearn.svm import SVC

# Load pre-trained models
scaler = joblib.load('scaler.pkl')  # Replace with correct path
svm_model = joblib.load('svm_model.pkl')  # Replace with correct path

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocessing function
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image).reshape(1, -1)  # Flatten to 1D array
    image = image / 255.0  # Normalize pixel values
    image = scaler.transform(image)  # Scale features
    return image

# Streamlit UI
st.title("👚 Image Classification with SVM")
st.write("Upload an image to predict its class.")

# Add gradient background and style to the UI
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #FF00FF, #00FFFF);
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader>div>div>div>div {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 5px;
        padding: 10px;
    }
    .stFileUploader>div>div>div>div>div>div>div {
        display: none; /* Hide the file name */
    }
    .prediction-box {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-size: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display uploaded image with a smaller width
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)  # Set width to 300 pixels

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = svm_model.predict(processed_image)

    # Output prediction in a stylish rectangle
    predicted_class = class_names[int(prediction[0])]
    st.markdown(f'<div class="prediction-box">This is a {predicted_class}</div>', unsafe_allow_html=True)