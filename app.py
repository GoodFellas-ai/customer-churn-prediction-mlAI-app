
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(layout="wide")
st.title("📱 Telco Customer Churn Prediction (Logistic Regression)")
st.markdown("Enter customer details to predict churn likelihood using a Logistic Regression model.")

# 1. Load the unified model pipeline
def load_model_pipeline():
    with open('model.pkl', 'rb') as f:
        model_pipeline = pickle.load(f)
    return model_pipeline

try:
    model_pipeline = load_model_pipeline()
    st.success("Logistic Regression Model Pipeline loaded successfully!")
except Exception as e:
    st.error(f"Error loading model pipeline file: {e}")
    st.stop()

# 2. User Interface (Inputs)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
charges_m = st.number_input("Monthly Charges", value=50.0, min_value=0.0)
charges_t = st.number_input("Total Charges", value=float(tenure * charges_m), min_value=0.0)

# 3. Prediction Logic
if st.button("Predict Churn"):
    input_dict = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple,
        'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': backup,
        'DeviceProtection': protection, 'TechSupport': support, 'StreamingTV': tv,
        'StreamingMovies': movies, 'Contract': contract, 'PaperlessBilling': billing,
        'PaymentMethod': payment, 'MonthlyCharges': charges_m, 'TotalCharges': charges_t
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model_pipeline.predict(input_df)
    probability = model_pipeline.predict_proba(input_df)[0][1]

    st.divider()
    if prediction[0] == 1:
        st.error(f"🚨 Customer is likely to Churn! (Probability: {probability*100:.2f}%) 🔥")
    else:
        st.success(f"✅ Customer is likely to Stay. (Probability: {probability*100:.2f}%) 🎉")
