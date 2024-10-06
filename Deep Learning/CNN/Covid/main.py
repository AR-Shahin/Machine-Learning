import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import time  # For adding delay

# Define the model path and categories
model_path = "/Users/anisurrahman/Documents/ML/Deep Learning/CNN/Covid/covid_normal_pneumoia_cnn.h5"
CATEGORIES = ["Covid", "Normal", "Viral Pneumonia"]
IMAGE_SIZE = 224

st.set_page_config(
    page_title="CNN",
    page_icon="🌟",
    layout="centered",
)


# Load the pre-trained model
def load_cnn_model():
    return load_model(model_path)


# Prediction function
def predict_image(image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize the image
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size

    # Simulate loading time and show progress bar
    progress_bar = st.progress(0)
    for percent_complete in range(0, 101, 20):
        time.sleep(0.4)  # Sleep for 0.4 seconds
        progress_bar.progress(percent_complete)

    # Predict
    model = load_cnn_model()
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Return the predicted label and the prediction result (confidence scores)
    return CATEGORIES[predicted_class], prediction


# Streamlit UI
st.subheader("AI Diagnosis: COVID, Normal, Pneumonia")
# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=250)

    # Predict the image category
    st.write("Classifying the image... Please wait!")
    label, confidence = predict_image(img)

    # Display prediction result
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{np.max(confidence):.2f}%**")
