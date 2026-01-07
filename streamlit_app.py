import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('logistic_model.joblib')  # احفظ الموديل المحلي بـ joblib

st.title("Titanic Survival Analysis & Prediction")

uploaded_file = st.file_uploader("Upload your Titanic CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']
    st.subheader("Feature Visualizations")
    for feature in features:
        st.write(f"### {feature}")
        st.bar_chart(df[feature].value_counts())

    # Prediction section
    st.subheader("Make Predictions")
    input_data = {}
    input_data['Pclass'] = st.selectbox("Pclass", [1,2,3])
    input_data['Sex'] = st.selectbox("Sex (0=female, 1=male)", [0,1])
    input_data['Age'] = st.number_input("Age", 0, 100, 30)
    input_data['SibSp'] = st.number_input("SibSp", 0, 10, 0)
    input_data['Parch'] = st.number_input("Parch", 0, 10, 0)
    input_data['Fare'] = st.number_input("Fare", 0.0, 600.0, 32.0)
    input_data['Embarked_Q'] = st.selectbox("Embarked_Q", [0,1])
    input_data['Embarked_S'] = st.selectbox("Embarked_S", [0,1])

    input_df = pd.DataFrame([input_data])
    if st.button("Predict Survival"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        if pred == 1:
            st.success(f"Likely to survive! Probability: {prob:.2f}")
        else:
            st.error(f"Unlikely to survive. Probability: {prob:.2f}")
