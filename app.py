
import streamlit as st
import tensorflow as tf
import json
import numpy as np
import os
from PIL import Image

# Get app folder location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model_path = os.path.join(BASE_DIR, "Pothole_Detector.keras")
model = tf.keras.models.load_model(model_path)

# Load class names
class_path = os.path.join(BASE_DIR, "classes.json")

with open(class_path, "r") as f:
    class_names = json.load(f)

# Streamlit app
st.title("🛣️ Pothole Detection System")

st.write("Upload a road image to detect potholes.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(img, caption="Uploaded Road Image",
             use_container_width=True)

    # Resize image
    img = img.resize((64, 64))

    # Convert to array
    x = np.array(img)

    # Add batch dimension
    x = np.expand_dims(x, axis=0)

    # Prediction
    p = model.predict(x)

    # Get predicted class
    index = np.argmax(p)
    prediction = class_names[index]

    # Confidence
    confidence = float(p[0][index]) * 100

    # Display result
    st.subheader("Detection Result")

    if "pothole" in prediction.lower():
        st.error("⚠️ Pothole Detected!")
    else:
        st.success("✅ Normal Road")

    st.write("Prediction:", prediction)
    st.write("Confidence:", round(confidence, 2), "%")

    # Show probabilities
    st.subheader("Prediction Details")

    for i in range(len(class_names)):
        st.write(
            class_names[i] + ": " +
            str(round(float(p[0][i]) * 100, 2)) + "%"
        )
