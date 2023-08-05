import streamlit as st
import cv2
import numpy as np
import os
import tensorflow as tf
from keras.applications.xception import preprocess_input, decode_predictions
from keras.applications.xception import Xception
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Load the model 
model = load_model('model.h5')  

# Function to treshold 
def Predict(value):
    if value >= 0.5:
        return "Car"
    else:
        return "Bike"

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = load_img(image, target_size=(200, 200))
    img_array = img_to_array(img) / 255.0
    img_resized = tf.image.resize(img_array, (200, 200))
    img_batch = tf.expand_dims(img_resized, axis=0)
    return img_batch

# Function to predict the image
def predict_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    return predictions[0][0]

def main():
    st.title("Image Classification")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        
        # Save the uploaded image temporarily
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, 'wb') as f:
            f.write(uploaded_image.read())
        
        # Perform prediction
        prediction = predict_image(temp_image_path)
        prediction_text = Predict(prediction)
        st.write(f"Prediction: {prediction_text}")

        # Remove the temporary image file
        os.remove(temp_image_path)

if __name__ == '__main__':
    main()
