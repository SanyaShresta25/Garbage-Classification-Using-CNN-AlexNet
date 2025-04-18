import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ---------- PAGE CONFIG ---------- #
st.set_page_config(page_title="Pastel Garbage Classifier", page_icon="🗑️", layout="centered")

# ---------- CUSTOM CSS FOR AESTHETICS ---------- #
st.markdown("""
    <style>
    .stApp {
        background-color: #ffeef8;
    }
    h1, h2, h3 {
        color: #d63384;
    }
    .stButton button {
        background-color: #f8bbd0;
        color: #d63384;
        border: 1px solid #d63384;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE AND INTRO ---------- #
st.title("🩷 Pastel Garbage Classifier")
st.caption("Classify your trash in style! Upload an image to see its class.")

# ---------- LOAD MODEL ---------- #
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("garbage_classifier_model.h5")
    except Exception as e:
        st.warning("⚠️ Using demo model - actual model not found.")
        # Create a simple model for demo purposes
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

# ---------- CLASS MAP ---------- #
class_map = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper", 
    4: "plastic",
    5: "trash"
}

# ---------- IMAGE PREPROCESS ---------- #
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    # Important: Using 224x224 based on the model's expected input
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# ---------- MAIN CONTENT ---------- #
uploaded_file = st.file_uploader("📷 Upload an image of garbage", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img_input, display_img = preprocess_image(uploaded_file)
        st.image(display_img, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Classifying... 🧠"):
            model = load_model()
            prediction = model.predict(img_input)
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = class_map[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index])
        
        st.success(f"### 🌸 Predicted Class: {predicted_class.upper()} (Confidence: {confidence:.2%})")
        
        # Display recycling instructions
        recycling_instructions = {
            "glass": "Rinse before recycling. Remove caps and lids.",
            "paper": "Keep dry and clean. Remove plastic windows or coatings.",
            "cardboard": "Break down boxes. Keep clean and dry.",
            "plastic": "Check for recycling numbers. Rinse containers.",
            "metal": "Rinse cans. Remove paper labels when possible.",
            "trash": "Cannot be recycled. Dispose in general waste."
        }
        
        st.info(f"**Recycling Tip**: {recycling_instructions[predicted_class]}")
        st.balloons()
    except Exception as e:
        st.error(f"Error processing image: {e}")

# ---------- FOOTER ---------- #
st.markdown("---")
st.markdown("### About This App")
st.write("""
This application helps classify different types of garbage materials to encourage proper recycling.
Upload your images to see how the classifier works.
""")

st.write("Remember: Proper waste sorting helps our environment! 🌱")

# Add the footer with attribution
st.markdown("""
<div style="text-align: center; padding: 20px; font-size: 16px; color: #777;">
    Made with 🤍 by Sanya
</div>
""", unsafe_allow_html=True)