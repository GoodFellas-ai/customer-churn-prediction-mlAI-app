# 📊 Customer Churn Prediction App

## 🚀 Overview

This project predicts whether a customer is likely to churn using machine learning. It includes an end-to-end pipeline from data preprocessing to deployment.

## 🌐 Live Demo

👉 Link : (https://customer-churn-prediction-mlai-app-arw3du8bwkjz4xormvpbrw.streamlit.app/)

---

## 🧠 Problem Statement

Customer churn is a critical business problem. This model helps identify customers at some kind of risk of leaving.

---

## ⚙️ Features

* Data preprocessing & feature engineering
* Categorical encoding with consistency
* Model training using Random Forest
* Probability-based predictions
* Interactive Streamlit web app

---Ensured feature alignment between training and inference pipeline using persisted column schema.

## 📊 Input Features

* Tenure
* Monthly Charges
* Total Charges
* Contract Type
* Payment Method
* Internet Service

---

## 🛠 Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Streamlit

---

## 🧪 Model Details

* Algorithm: Random Forest Classifier
* Preprocessing: StandardScaler + One-Hot Encoding
* Evaluation: Accuracy, Precision, Recall, F1-score

---

## 📂 Project Structure

* `app.py` → Streamlit app
* `model.pkl` → trained model
* `columns.pkl` → feature alignment
* `notebooks/` → EDA & training

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Key Insights

* Month-to-month contracts have higher churn
* High monthly charges increase churn risk
* Long-term customers are more stable

---

## 👤 Author

ERDAL ERDOĞAN aka RUPERT PUMPKIN 
