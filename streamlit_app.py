import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🛳 Titanic Survival Predictor")
st.write("Predict whether a passenger would survive based on their features.")

# Model path
model_path = "logistic_model.joblib"

# Check if model exists
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' not found! Please upload it to the project folder.")
    st.stop()  # Stop app if model not found
else:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")

# Input area
st.subheader("Enter Passenger Details:")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Pclass", [1,2,3], 
        help="Passenger class: 1 = 1st class, 2 = 2nd class, 3 = 3rd class"
    )
    sex = st.selectbox(
        "Sex (0=female, 1=male)", [0,1], 
        help="0 = Female, 1 = Male"
    )
    age = st.slider(
        "Age", 0, 100, 30, 
        help="Age of the passenger in years"
    )
    sibsp = st.number_input(
        "Siblings/Spouses aboard", 0, 10, 0, 
        help="Number of siblings or spouses aboard the Titanic"
    )

with col2:
    parch = st.number_input(
        "Parents/Children aboard", 0, 10, 0, 
        help="Number of parents or children aboard the Titanic"
    )
    fare = st.number_input(
        "Fare", 0.0, 600.0, 32.0, 
        help="Ticket fare in British Pounds"
    )
    embarked = st.selectbox(
        "Embarked", ["C", "Q", "S"], 
        help="Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton"
    )

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
