import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logistic_model.joblib')  # الموديل لازم joblib

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
    embarked_q = st.selectbox("Embarked_Q", [0,1])
    embarked_s = st.selectbox("Embarked_S", [0,1])

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
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ The passenger is likely to survive! (Probability: {probability:.2f})")
    else:
        st.error(f"❌ The passenger is unlikely to survive. (Probability: {probability:.2f})")
