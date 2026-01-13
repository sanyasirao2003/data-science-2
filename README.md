# 💳 Financial Fraud Detection System (End-to-End ML Project)

An end-to-end **Financial Fraud Detection System** built using Machine Learning, designed to detect fraudulent transactions with **high recall** and deployed as an interactive **Streamlit web application**.

This project focuses on real-world fraud detection challenges such as extreme class imbalance, domain-driven feature engineering, probability threshold tuning, and deployment consistency.

---

## 🚀 Project Highlights

- 📊 Worked with a **large-scale dataset (6.3M+ transactions)**
- ⚠️ Handled **severe class imbalance** (~0.13% fraud cases)
- 🧠 Engineered **domain-specific fraud features**
- 🤖 Built a **high-recall XGBoost model (~99% fraud recall)**
- 🎯 Applied **probability threshold tuning** to balance recall and precision
- 🌐 Deployed a **production-style Streamlit application**
- 🛠️ Resolved **training vs deployment feature mismatch**
- ✅ Tested with fraud, legitimate, and borderline scenarios

---

## 🧠 Why High Recall?

In fraud detection, missing a fraudulent transaction is far more costly than raising a false alert.  
Therefore, the model is optimized for **high recall**, ensuring that nearly all fraud cases are detected.

---

## 🧩 Tech Stack

- **Programming Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, XGBoost, Streamlit, joblib  

---

## 📁 Project Structure

Fraud-Detection-ML/
│
├── app.py # Streamlit application
├── model.pkl # Trained XGBoost model
├── feature_columns.pkl # Saved feature schema for deployment consistency
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── notebooks/
└── fraud_detection.ipynb # Model training & EDA (optional)


## ▶️ How to Run the Application

### 1️⃣ Install dependencies

pip install -r requirements.txt
2️⃣ Run the Streamlit app

streamlit run app.py
🧪 Sample Test Case (Fraud)

Transaction Type: TRANSFER
Amount: 181
Sender Old Balance: 181
Sender New Balance: 0
Receiver Old Balance: 0
Receiver New Balance: 0
Expected Output:
🚨 Fraud Detected (High Probability)

📌 Dataset Information
The dataset contains simulated financial transactions.

Due to its large size (6.3M+ rows), it is not included in this repository.

The dataset is required only for training, not for deployment.

Dataset source and details are documented in the project report.

⚠️ Limitations
Uses simulated transaction data

No user-level transaction history modeling

No real-time streaming integration

These limitations are acknowledged and discussed in the project report.

📈 Future Enhancements
SHAP-based explainability

Real-time fraud detection pipeline

User behavior profiling

API-based deployment

Batch transaction analysis (CSV upload)

🎓 Academic Context
This project was developed as part of an MCA (Master of Computer Applications) curriculum and demonstrates applied skills in:

Data Science

Machine Learning

Model Evaluation

ML Deployment

👤 Author
Mangipudi Sanyasi Rao
MCA (Data Science)
Aspiring Data Analyst / Data Scientist
