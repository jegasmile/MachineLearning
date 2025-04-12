import streamlit as st
import pickle
import numpy as np


model_path = "model.pkl"
scaler_path = "scaler.pkl"

# Load trained model and scaler
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

st.title("Salary Prediction App")

# Example inputs
# Streamlit Inputs for all 5 features
education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
experience = st.slider("Years of Experience", 0, 30, 5)
gender = st.selectbox("Gender", ["Male", "Female"])
role = st.selectbox("Role", ["Developer", "Manager", "Analyst"])
age = st.slider("Age", 20, 60, 25)

# Encode inputs (matching training feature encoding)
education_encoded = {"Bachelor": 0, "Master": 1, "PhD": 2}[education]
gender_encoded = {"Male": 0, "Female": 1}[location]
role_encoded = {"Developer": 0, "Manager": 1, "Analyst": 2}[role]

# Combine all inputs in a single array (make sure the order matches training)
features = np.array([[education_encoded, experience, gender_encoded, role_encoded, age]])

# Scale the features (must match the number of features used during training)
features_scaled = scaler.transform(features)

# Predict salary
if st.button("Predict Salary"):
    prediction = model.predict(features_scaled)[0]
    st.success(f"Predicted Salary: ₹{prediction:,.2f}")
