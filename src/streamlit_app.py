import streamlit as st
import pickle
import os
import numpy as np

MODEL_DIR = 'data/models/'

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def list_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]

st.title("Diabetes Prediction Model")

models = list_models()
selected_model = st.selectbox("Choose a model", models)

model_path = os.path.join(MODEL_DIR, selected_model)
model = load_model(model_path)

st.header("Enter the patient's data")

Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=0)
BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=0)
SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=0)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.0)
Age = st.number_input("Age", min_value=0, max_value=120, value=0)

input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
prediction = model.predict(input_data)

if st.button("Predict"):
    st.write(f"The selected model is: {selected_model}")
    st.write(f"The model predicts: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")