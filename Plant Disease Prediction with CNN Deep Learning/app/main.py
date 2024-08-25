import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import webbrowser

# Define the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'trained_model', 'plant_disease_prediction_model.h5')

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices_path = os.path.join(working_dir, 'class_indices.json')
with open(class_indices_path) as f:
    class_indices = json.load(f)

# Load disease information
disease_info_path = os.path.join(working_dir, 'disease_info.json')
with open(disease_info_path) as f:
    disease_info = json.load(f)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to Open a Search Browser for Disease Control
def search_disease_control(disease_name):
    query = f"how to control {disease_name} plant disease"
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open_new_tab(url)

# Streamlit App
st.set_page_config(page_title='Plant Disease Classifier', page_icon='🌱', layout='wide')

# Custom CSS for styling and footer
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f0f0; /* Light background for main area */
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    .css-1d391kg {
        color: black;
    }
    .css-1v3fvcr {
        border: 2px solid #000;
        border-radius: 10px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.9);
    }
    .css-1v0mb4d {
        color: #000;
    }
    .css-16cvfj6 {
        color: #000;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('🌿 Plant Disease Classifier')
st.write("Upload an image of a plant leaf to classify the disease and get more information about it.")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.markdown(f"### Prediction: **{prediction}**")
            if prediction in disease_info:
                info = disease_info[prediction]
                st.markdown(f"#### Disease Information:")
                st.write(info)
            else:
                st.write("No information available for this disease.")

            # Button to search for control measures
            st.markdown(
                f'<a href="https://www.google.com/search?q=how+to+control+{prediction}+plant+disease" target="_blank" '
                f'class="btn">How can {prediction} be controlled?</a>',
                unsafe_allow_html=True
            )

# Footer
st.markdown('<div class="footer">Developed by Nithilan</div>', unsafe_allow_html=True)
