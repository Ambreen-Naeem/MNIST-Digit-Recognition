# Import required libraries
import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the trained model
with open("mnist_model.pkl", "rb") as file:
    model = pickle.load(file)

# Set page title
st.title("MNIST Digit Recognition")

st.write("Upload a handwritten digit image (0-9) and predict the digit.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("L")

    # Display image
    st.image(image, caption="Uploaded Image", width=200)

    # Resize image to 28x28
    image = image.resize((28, 28))

    # Convert image to array
    img_array = np.array(image)

    # Auto-detect background color
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize pixel values
    img_array = img_array / 255.0

    # Reshape for prediction
    img_array = img_array.reshape(1, 784)

    # Predict digit
    prediction = model.predict(img_array)

    # Display result
    st.success(f"Predicted Digit: {prediction[0]}")



