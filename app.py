import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("📊 Customer Churn Prediction")

tenure = st.slider("Tenure", 0, 72)
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", ["Electronic check", "Credit card", "Bank transfer"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "PaymentMethod": payment,
    "InternetService": internet
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    result = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    if result[0] == 1:
        st.error(f"⚠️ Churn Risk: {prob:.2%}")
    else:
        st.success(f"✅ Stay Probability: {(1-prob):.2%}")
