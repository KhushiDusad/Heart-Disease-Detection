import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("Heart Disease Prediction App")

# User inputs
st.header("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Encoding the categorical variables 
sex = 1 if sex == "Male" else 0
cp_types = {"ATA": [0, 0, 0], "NAP": [1, 0, 0], "ASY": [0, 1, 0], "TA": [0, 0, 1]}
cp_encoded = cp_types[cp]
resting_ecg_types = {"Normal": [0, 0], "ST": [1, 0], "LVH": [0, 1]}
resting_ecg_encoded = resting_ecg_types[resting_ecg]
exercise_angina = 1 if exercise_angina == "Yes" else 0
st_slope_types = {"Up": [0, 0], "Flat": [1, 0], "Down": [0, 1]}
st_slope_encoded = st_slope_types[st_slope]

# Combine all features into one array
features = [age, resting_bp, cholesterol, fasting_bs, max_hr, oldpeak, sex,
            *cp_encoded, *resting_ecg_encoded, exercise_angina, *st_slope_encoded]

# Scale numerical data
features_scaled = scaler.transform([features])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    if prediction == 1:
        st.error("The model predicts that the person is likely to have heart disease.")
    else:
        st.success("The model predicts that the person is unlikely to have heart disease.")