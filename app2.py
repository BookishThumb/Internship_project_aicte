import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import sys

sys.setrecursionlimit(15000)


def load_cifar10_model():
    return tf.keras.models.load_model("TechSaksham_project\cifar10_model.h5")


def classify_with_cifar10(image):
    model = load_cifar10_model()
    img = image.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence


st.title("Image Classifier")

st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose a Model", ("CIFAR-10 Classifier", "MobileNetV2 Classifier"))

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    if option == "CIFAR-10 Classifier":
        predicted_class, confidence = classify_with_cifar10(image)
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
    elif option == "MobileNetV2 Classifier":
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        st.write(f"Predicted Class: {decoded_predictions[0][1]}")
        st.write(f"Confidence: {decoded_predictions[0][2] * 100:.2f}%")
