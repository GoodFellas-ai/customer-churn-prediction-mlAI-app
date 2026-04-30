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
import numpy as np
import joblib
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Churn AI",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# SIMPLE CUSTOM UI STYLE
# -------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 Customer Churn Intelligence")
st.caption("AI-powered customer retention prediction system")

st.markdown("---")

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
feature_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# -------------------------------
# SIDEBAR INPUTS (STARTUP STYLE)
# -------------------------------
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 100, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", value=50.0)
total_charges = st.sidebar.number_input("Total Charges", value=600.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

input_data = pd.DataFrame([dict(zip(feature_columns, [0]*len(feature_columns)))])

input_data.update({
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
})

# -------------------------------
# MAIN ACTION
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction Engine")

    if st.button("Run Churn Analysis 🚀"):
        if model:
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0][1]

                st.markdown("### Result")

                if prediction == 1:
                    st.error("⚠️ High Risk: Customer likely to churn")
                else:
                    st.success("✅ Low Risk: Customer likely to stay")

                st.progress(float(proba))
                st.metric("Churn Probability", f"{proba:.2%}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("Model not loaded")

with col2:
    st.subheader("Insights")
    st.info("- Higher tenure reduces churn risk\n- Month-to-month contracts increase churn\n- Higher charges may increase churn probability")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit • ML Churn Intelligence System")

