import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os


st.write("📁 Current directory files:")
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

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ======================
# HEADER
# ======================
st.title("Customer Churn Intelligence")
st.caption("AI-powered customer retention prediction system")

st.markdown("---")

# ======================
# LOAD MODEL + COLUMNS
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
    return model, columns

model, feature_columns = load_artifacts()

# ======================
# SIDEBAR INPUTS
# ======================
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 100, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", value=50.0)
total_charges = st.sidebar.number_input("Total Charges", value=600.0)
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

# ======================
# BUILD INPUT (SAFE FIX)
# ======================
input_dict = dict.fromkeys(feature_columns, 0)

# only overwrite known numeric fields (safe)
if "tenure" in input_dict:
    input_dict["tenure"] = tenure
if "MonthlyCharges" in input_dict:
    input_dict["MonthlyCharges"] = monthly_charges
if "TotalCharges" in input_dict:
    input_dict["TotalCharges"] = total_charges

# contract encoding (if exists in model columns)
contract_col = f"Contract_{contract}"
if contract_col in input_dict:
    input_dict[contract_col] = 1

input_data = pd.DataFrame([input_dict])

# ======================
# PREDICTION
# ======================
st.subheader("Prediction Engine")

if st.button("Run Analysis 🚀"):
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("⚠️ High Risk: Customer likely to churn")
        else:
            st.success("✅ Low Risk: Customer likely to stay")

        st.metric("Churn Probability", f"{proba:.2%}")
        st.progress(float(proba))

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("Built with Streamlit • Clean ML Deployment Version")
st.markdown("---")
st.caption("Built with Streamlit • ML Churn Intelligence System")

