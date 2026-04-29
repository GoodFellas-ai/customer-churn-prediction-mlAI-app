app_code_final = """
import streamlit as st
import pandas as pd
import pickle

st.title("Churn App (Minimal Test)")

# Load the model pipeline (preprocessor is now part of the pipeline)
def load_model_pipeline():
    with open("model.pkl", "rb") as f:
        model_pipeline = pickle.load(f)
    return model_pipeline

try:
    model_pipeline = load_model_pipeline()
    st.success("Model Pipeline loaded successfully!")
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# Enter dataset columns exactly (including case!)
tenure = st.number_input("tenure", 0, 72, 12)
MonthlyCharges = st.number_input("MonthlyCharges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("TotalCharges", 0.0, 10000.0, 500.0)
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

# You must add all inputs expected by the model here.
# These are the columns from the X dataframe obtained from df.drop("Churn", axis=1).
# If your model's trained X dataframe has more columns (e.g., gender, SeniorCitizen, etc.),
# you should add them to the Streamlit interface or assign default values.
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "Contract": Contract,
    "PaymentMethod": PaymentMethod,
    "InternetService": InternetService,
    'gender': 'Male', # Example value, you may need to add this for all columns expected by the model.
    'SeniorCitizen': 0, # Example value
    'Partner': 'No', # Example value
    'Dependents': 'No', # Example value
    'PhoneService': 'No', # Example value
    'MultipleLines': 'No phone service', # Example value
    'OnlineSecurity': 'No internet service', # Example value
    'OnlineBackup': 'No internet service', # Example value
    'DeviceProtection': 'No internet service', # Example value
    'TechSupport': 'No internet service', # Example value
    'StreamingTV': 'No internet service', # Example value
    'StreamingMovies': 'No internet service', # Example value
    'PaperlessBilling': 'No' # Example value
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict"):
    st.write("Input:", input_df)
    # Make predictions using the pipeline. Preprocessing is automatically applied.
    pred = model_pipeline.predict(input_df)
    probability = model_pipeline.predict_proba(input_df)[0]

    st.write("Prediction:", pred)
    st.write("Probability (No Churn, Churn):", probability)
    if pred[0] == 1:
        st.error(f"🚨 Customer is likely to churn! (Probability: {probability[1]*100:.2f}%) ")
    else:
        st.success(f"✅ Customer is likely to stay. (Probability: {probability[0]*100:.2f}%) ")
"""

with open('app.py', 'w') as f:
    f.write(app_code_final)

print("✅ 'app.py' file has been updated. You should now only download the 'model.pkl' file and use it with your Streamlit application.")
