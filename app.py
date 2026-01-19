import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('pneumonia_cnn_model.h5')


# Image size your model expects
IMG_SIZE = (150, 150)

st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image and get prediction:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image for model
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 150, 150, 3)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.markdown("<h3 style='color:red'>Prediction: Pneumonia</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:green'>Prediction: Normal</h3>", unsafe_allow_html=True)
