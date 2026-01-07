import os
import joblib
import streamlit as st

# اسم الموديل
model_path = "logistic_model.joblib"

if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' not found! Please upload it.")
else:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
