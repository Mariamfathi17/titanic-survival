import streamlit as st
import pandas as pd
import pickle

# Load model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Titanic Survival Analysis & Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload your Titanic CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Features to visualize
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']

    st.subheader("Feature Visualizations")

    for feature in features:
        st.write(f"### {feature}")
        counts = df[feature].value_counts()
        st.bar_chart(counts)

    # Prediction Section
    st.subheader("Make Predictions")
    input_data = {}
    input_data['Pclass'] = st.selectbox("Pclass", [1,2,3])
    input_data['Sex'] = st.selectbox("Sex (0=female, 1=male)", [0,1])
    input_data['Age'] = st.number_input("Age", min_value=0, max_value=100, value=30)
    input_data['SibSp'] = st.number_input("SibSp", min_value=0, max_value=10, value=0)
    input_data['Parch'] = st.number_input("Parch", min_value=0, max_value=10, value=0)
    input_data['Fare'] = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
    input_data['Embarked_Q'] = st.selectbox("Embarked_Q", [0,1])
    input_data['Embarked_S'] = st.selectbox("Embarked_S", [0,1])

    input_df = pd.DataFrame([input_data])

    if st.button("Predict Survival"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.success(f"The passenger is likely to survive! (Probability: {probability:.2f})")
        else:
            st.error(f"The passenger is unlikely to survive. (Probability: {probability:.2f})")
