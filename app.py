app_code = """
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load saved assets
def load_assets():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

try:
    model, preprocessor = load_assets()
    st.success("Model and Preprocessor loaded successfully!")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.title("📱 Telco Customer Churn Prediction")
st.markdown("Enter customer details to calculate the probability of churn.")

# 2. User Interface (Inputs)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen?", [0, 1])
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
charges_m = st.number_input("Monthly Charges", value=50.0)
charges_t = st.number_input("Total Charges", value=tenure * charges_m)

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
    
    # Preprocessing
    input_processed = preprocessor.transform(input_df)
    
    # Prediction
    prediction = model.predict(input_processed)
    probability = model.predict_proba(input_processed)[0][1]
    
    st.divider()
    if prediction[0] == 1 or prediction[0] == 'Yes':
        st.error(f"🚨 Customer is likely to Churn! (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Customer is likely to Stay. (Probability: {probability*100:.2f}%)")
"""

with open('app.py', 'w') as f:
    f.write(app_code)

print("app.py has been updated with English text. Place model.pkl, preprocessor.pkl, and app.py in the same folder.")
