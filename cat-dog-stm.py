import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("CAT.keras")

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Streamlit title
st.title("🐶🐱 Dog vs Cat Classifier")

st.write("Upload an image of a dog or cat to get the prediction.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open uploaded image
    img = Image.open(uploaded_file)

    # Display uploaded image
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image to model input size
    img = img.resize((64, 64))

    # Convert image to NumPy array
    x = np.array(img)

    # Add batch dimension
    x = np.expand_dims(x, axis=0)

    # Make prediction
    prediction = model.predict(x)

    # Get predicted class
    predicted_class = class_names[np.argmax(prediction)]

    # Get confidence
    confidence = np.max(prediction) * 100

    # Display result
    st.success("Prediction: " + predicted_class)

    st.write(
        "Confidence: " + str(round(confidence, 2)) + "%"
    )
