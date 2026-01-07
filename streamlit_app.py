import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logistic_model.joblib')  # الموديل لازم joblib

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🛳 Titanic Survival Predictor")
st.write("Predict whether a passenger would survive based on their features.")

# Main input area
st.subheader("Enter Passenger Details:")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Pclass", [1,2,3])
    sex = st.selectbox("Sex (0=female, 1=male)", [0,1])
    age = st.slider("Age", 0, 100, 30)
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)

with col2:
    parch = st.number_input("Parents/Children aboard", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert Embarked to one-hot encoding
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Prepare input dataframe
input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked_Q': embarked_q,
    'Embarked_S': embarked_s
}])

# Prediction button
st.write("")  # Space
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.markdown(f"""
            <div style="background-color:#d4edda; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:#155724;">✅ The passenger is likely to survive!</h2>
                <p>Probability: {probability:.2f}</p>
                <img src="https://cdn-icons-png.flaticon.com/512/616/616408.png" width="100">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background-color:#f8d7da; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:#721c24;">❌ The passenger is unlikely to survive.</h2>
                <p>Probability: {probability:.2f}</p>
                <img src="https://cdn-icons-png.flaticon.com/512/1828/1828843.png" width="100">
            </div>
        """, unsafe_allow_html=True)
