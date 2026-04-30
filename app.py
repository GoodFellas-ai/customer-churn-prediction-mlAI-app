import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.title("📊 Customer Churn Intelligence")
st.caption("AI-powered customer retention prediction system")

st.markdown("---")

# -----------------------------
# MODEL PATH (ROBUST)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 100, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", value=50.0)
total_charges = st.sidebar.number_input("Total Charges", value=600.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# -----------------------------
# INPUT DATAFRAME
# -----------------------------
input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service
}])

# -----------------------------
# MAIN UI
# -----------------------------
st.subheader("Prediction Engine")

if st.button("Run Analysis 🚀"):

    if model is None:
        st.warning("Model is not loaded.")
    else:
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

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit • Customer Churn ML System")

