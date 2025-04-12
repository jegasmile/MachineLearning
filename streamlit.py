import streamlit as st
import pickle
import numpy as np
from google.colab import drive


# Mount Google Drive
drive.mount('/content/drive')

model_path = "model.pkl"
scaler_path = "scaler.pkl"

# Load trained model and scaler
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

st.title("Salary Prediction App")

# Example inputs
education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
experience = st.slider("Years of Experience", 0, 30, 5)

# Encode and scale input
education_encoded = {"Bachelor": 0, "Master": 1, "PhD": 2}[education]
features = np.array([[education_encoded, experience]])
features_scaled = scaler.transform(features)

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(features_scaled)[0]
    st.success(f"Estimated Salary: ₹{prediction:,.2f}")
