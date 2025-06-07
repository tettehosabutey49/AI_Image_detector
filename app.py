# app.py
import streamlit as st
from PIL import Image
from image_processor import ImageProcessor
from image_processor import AIDetector #as detector

st.title("🔍 AI Image Detector")
uploaded_file = st.file_uploader("Upload an image:")

if uploaded_file:
    image = Image.open(uploaded_file)
    image.save("image2.jpg")
    processor = ImageProcessor("image2.jpg")
    preprocessed = processor.preprocess()
    detector = AIDetector()

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image")
    with col2:
        st.write("### Analysis")
        st.write(f"ML Prediction: **{detector.predict(preprocessed)}**")
        st.write(f"Artifact Check: **{'AI-like' if processor.is_ai(detector) else 'Real'}**")