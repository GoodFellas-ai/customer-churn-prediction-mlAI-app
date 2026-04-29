import streamlit as st
import pandas as pd
import pickle

st.title("Churn App (Minimal Test)")

# Modeli yükle
model = pickle.load(open("model.pkl", "rb"))

# Dataset kolonlarını birebir gir (küçük/büyük harf dahil!)
tenure = st.number_input("tenure", 0, 72, 12)
MonthlyCharges = st.number_input("MonthlyCharges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("TotalCharges", 0.0, 10000.0, 500.0)
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

# DataFrame (KOLON İSİMLERİ %100 AYNI OLMALI)
input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "Contract": Contract,
    "PaymentMethod": PaymentMethod,
    "InternetService": InternetService
}])

if st.button("Predict"):
    st.write("Input:", input_df)
    pred = model.predict(input_df)
    st.write("Prediction:", pred)
