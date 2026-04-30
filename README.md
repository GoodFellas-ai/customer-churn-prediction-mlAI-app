#### 📊 Customer Churn Prediction App

## 🚀 Overview

This project predicts whether a customer is likely to churn using machine learning. It includes an end-to-end pipeline from data preprocessing to deployment.

🌐 Live Demo

👉 Link : (https://customer-churn-prediction-mlai-app-arw3du8bwkjz4xormvpbrw.streamlit.app/)

---

🧠 Problem Statement

Customer churn is a critical business problem. This model helps identify customers at some kind of risk of leaving.

---

⚙️ Features

* Data preprocessing & feature engineering
* Categorical encoding with consistency
* Model training using Random Forest
* Probability-based predictions
* Interactive Streamlit web app

---Ensured feature alignment between training and inference pipeline using persisted column schema.

📊 Input Features

* Tenure
* Monthly Charges
* Total Charges
* Contract Type
* Payment Method
* Internet Service

---

🛠 Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Streamlit

---

🧪 Model Details

* Algorithm: Random Forest Classifier
* Preprocessing: StandardScaler + One-Hot Encoding
* Evaluation: Accuracy, Precision, Recall, F1-score

---

📂 Project Structure

* `app.py` → Streamlit app
* `model.pkl` → trained model
* `columns.pkl` → feature alignment
* `notebooks/` → EDA & training

---

▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

🎯 Key Insights

* Month-to-month contracts have higher churn
* High monthly charges increase churn risk
* Long-term customers are more stable

---

📉 Customer Churn Intelligence

An end-to-end machine learning system that predicts customer churn and helps identify at-risk customers using behavioral and contract-based data.

---

🚀 Live Demo

👉 https://customer-churn-prediction-mlai-app-arw3du8bwkjz4xormvpbrw.streamlit.app/

---


🧠 Problem Statement

Customer churn is one of the most critical challenges in subscription-based businesses.  
The goal of this project is to predict the probability of churn and support proactive customer retention strategies.

---

⚙️ Solution Overview

This project implements a full ML pipeline:

- Data preprocessing & feature engineering
- Categorical encoding with consistent inference alignment
- Trained Random Forest classifier
- Probability-based churn scoring
- Deployed interactive Streamlit dashboard

A persistent feature schema (`columns.pkl`) ensures training-inference consistency.

---

📊 Key Input Features

- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Payment Method
- Internet Service
- Customer service usage indicators

---

🧠 Model Architecture

- Algorithm: Random Forest Classifier  
- Preprocessing: One-Hot Encoding + Scaling  
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score  
- Output: Churn probability (0–100%)

---

📈 Key Business Insights

- Month-to-month contracts have significantly higher churn risk  
- Higher monthly charges correlate with increased churn probability  
- Long-tenure customers are more likely to stay  
- Service usage patterns strongly influence retention

---

🖥️ Application Features

- Interactive Streamlit dashboard
- Real-time churn prediction
- Probability-based risk scoring
- Clean UI with business insights panel

---

📂 Project Structure

app.py # Streamlit application (production UI)
model.pkl # Trained ML pipeline
columns.pkl # Feature schema alignment
notebooks/ # EDA and model training
requirements.txt

▶️ Run Locally

git clone https://github.com/your-username/customer-churn-intelligence.git
cd customer-churn-intelligence
pip install -r requirements.txt
streamlit run app.py
---

🎯 Project Impact

This project demonstrates:

End-to-end machine learning workflow
Production-ready model deployment
Feature consistency handling in inference pipelines
Business-oriented interpretation of ML outputs
Practical AI productization using Streamlit

👤 Author
Erdal Erdoğan aka RUPERT PUMPKIN

