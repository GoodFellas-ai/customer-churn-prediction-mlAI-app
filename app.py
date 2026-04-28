import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -----------------------------
# Load artifacts
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -----------------------------
# UI
# -----------------------------
st.title("📊 Customer Churn Prediction App")

st.write("Enter customer information below:")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total = st.number_input("Total Charges", min_value=0.0, value=500.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]
)
internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# -----------------------------
# Build input
# -----------------------------
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "PaymentMethod": payment,
    "InternetService": internet
}

input_df = pd.DataFrame([input_dict])

# -----------------------------
# Preprocessing (CRITICAL FIX)
# -----------------------------
input_df = pd.get_dummies(input_df)

# Align with training columns (MOST IMPORTANT STEP)
input_df = input_df.reindex(columns=columns, fill_value=0)

# -----------------------------
# Scale + Predict
# -----------------------------
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk ({probability:.2%})")
    else:
        st.success(f"✅ Low Churn Risk ({(1-probability):.2%})")

    st.progress(float(probability))
