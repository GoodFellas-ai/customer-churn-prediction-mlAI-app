import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os



MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


try:
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    model_pipeline = joblib.load(model_path)
except FileNotFoundError:
    st.error("The model.pkl file could not be found. Please make sure that the model has been trained and saved..")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model.: {e}")
    st.stop()

st.title('📉 Customer Churn Prediction System')
st.caption("Machine Learning - AI powered retention analysis tool")
st.write('Enter customer information to predict churn probability.')

gender = st.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.selectbox('Senior Citizen', [0, 1])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['No', 'Yes'])
tenure = st.slider('Tenure (Month)', 0, 72, 1)
phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input('Monthly Charges', 0.0, 120.0, 50.0)
total_charges = st.number_input('Total Charges', 0.0, 9000.0, 1000.0)

if st.button('Please make a prediction'):
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    prediction = model_pipeline.predict(input_data)[0]
    prediction_proba = model_pipeline.predict_proba(input_data)[0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.write(f"The customer is likely to churn. (Probability: {prediction_proba[1]:.2f})")
        st.markdown("<p style='color:red;'><b>Churn Risk: High</b></p>", unsafe_allow_html=True)
    else:
        st.write(f"The customer's likelihood of churning is low. (Probability: {prediction_proba[0]:.2f})")
        st.markdown("<p style='color:green;'><b>Churn Risk: Low</b></p>", unsafe_allow_html=True)



# ===============================
# Customer Churn Prediction App
# Startup-Style UI (Production Upgrade)
# ===============================
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Churn AI", layout="wide")

st.title("Customer Churn Intelligence")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

st.sidebar.header("Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure", 0, 100, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])

streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly = st.sidebar.number_input("Monthly Charges", 50.0)
total = st.sidebar.number_input("Total Charges", 600.0)

# =========================
# RAW DATA (IMPORTANT FIX)
# =========================
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error("High churn risk")
        else:
            st.success("Low churn risk")

        st.metric("Churn Probability", f"{proba:.2%}")
        st.progress(float(proba))

    except Exception as e:
        st.error(f"Prediction error: {e}")
        
st.subheader("Insights")

st.info("""
• Higher tenure → lower churn risk  
• Month-to-month contracts → higher churn  
• Higher monthly charges → higher churn probability
""")

