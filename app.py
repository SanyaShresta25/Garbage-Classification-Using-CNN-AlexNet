# pastel_garbage_classifier.py

# --- Constants and Imports --- #
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Parameters --- #
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMAGE_SIZE = (224, 224)  # Model input size
BATCH_SIZE = 8
EPOCHS = 35

# --- Page Configuration --- #
st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️", layout="centered")

# --- Custom CSS --- #
st.markdown("""
    <style>
    .stApp {
        background-color: #ffeef8;
    }
    h1, h2, h3 {
        color: #d63384;
    }
    .stButton button, .stFileUploader label {
        background-color: #f8bbd0 !important;
        color: #d63384 !important;
        border: 1px solid #d63384 !important;
        border-radius: 12px !important;
    }
    .stProgress > div > div > div > div {
        background-color: #d63384;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title --- #
st.title("🩷 Garbage Classifier")
st.caption("Classify your trash in style! Upload an image to see its class.")

# --- Load Model --- #
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "final_alexnet_garbage_classifier.h5")
    if not os.path.exists(model_path):
        st.error(f"🚫 Model file not found at {model_path}")
        st.stop()
    return tf.keras.models.load_model(model_path)

# --- Class Map --- #
class_map = dict(enumerate(labels))

# --- Image Preprocessing --- #
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

# --- File Upload --- #
uploaded_file = st.file_uploader("📷 Upload an image of garbage", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img_input, display_img = preprocess_image(uploaded_file)
        st.image(display_img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Classifying... 🧠"):
            model = load_model()
            prediction = model.predict(img_input)[0]
            predicted_class_index = int(np.argmax(prediction))
            predicted_class = class_map[predicted_class_index]
            confidence = float(prediction[predicted_class_index])

        st.success(f"### 🌸 Predicted Class: {predicted_class.upper()} (Confidence: {confidence:.2%})")

        # --- Class Confidences --- #
        st.write("### Confidence for Each Class:")
        for i, class_name in class_map.items():
            conf = float(prediction[i])
            st.progress(conf)
            st.write(f"**{class_name.capitalize()}**: {conf:.2%}")

        # --- Recycling Tips --- #
        recycling_tips = {
            "glass": "Rinse before recycling. Remove caps and lids.",
            "paper": "Keep dry and clean. Remove plastic windows or coatings.",
            "cardboard": "Break down boxes. Keep clean and dry.",
            "plastic": "Check for recycling numbers. Rinse containers.",
            "metal": "Rinse cans. Remove paper labels when possible.",
            "trash": "Cannot be recycled. Dispose in general waste."
        }

        st.info(f"♻️ **Recycling Tip**: {recycling_tips.get(predicted_class, 'No tip available.')}")

        st.balloons()

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# --- About Section --- #
with st.expander("ℹ️ About the Model"):
    st.write("""
    This app uses an AlexNet-based convolutional neural network to classify garbage into 6 categories:
    - Cardboard
    - Glass
    - Metal
    - Paper
    - Plastic
    - Trash

    **Training Specs**:
    - Image Size: 224x224
    - Batch Size: 8
    - Epochs: 35
    """)

# --- Footer --- #
st.markdown("---")
st.markdown("### About This App")
st.write("""
This application encourages responsible recycling by helping users correctly classify their waste.
Just snap a pic of your trash, and let the AI do the rest! 🌍
""")

st.markdown("""
<div style="text-align: center; padding: 20px; font-size: 16px; color: #777;">
    Made with 🤍 by Sanya
</div>
""", unsafe_allow_html=True)
