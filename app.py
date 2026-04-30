import streamlit as st
import pandas as pd
import joblib
import numpy as np

try:
    model_pipeline = joblib.load('/content/model.pkl')
except FileNotFoundError:
    st.error("model.pkl dosyası bulunamadı. Lütfen modelin eğitilip kaydedildiğinden emin olun.")
    st.stop()
except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")
    st.stop()

st.title('Müşteri Churn Tahmin Uygulaması')
st.write('Müşteri verilerini girerek churn tahmininde bulunun.')

gender = st.selectbox('Cinsiyet', ['Female', 'Male'])
senior_citizen = st.selectbox('Senior Citizen', [0, 1])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['No', 'Yes'])
tenure = st.slider('Tenure (Ay)', 0, 72, 1)
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

if st.button('Tahmin Yap'):
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

    st.subheader('Tahmin Sonucu:')
    if prediction == 1:
        st.write(f"Müşterinin churn etme olasılığı yüksektir. (Olasılık: {prediction_proba[1]:.2f})")
        st.markdown("<p style='color:red;'><b>Churn Riski: Yüksek</b></p>", unsafe_allow_html=True)
    else:
        st.write(f"Müşterinin churn etme olasılığı düşüktür. (Olasılık: {prediction_proba[0]:.2f})")
        st.markdown("<p style='color:green;'><b>Churn Riski: Düşük</b></p>", unsafe_allow_html=True)
