import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# Load model and feature schema
# -------------------------------------------------
model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Financial Fraud Detection System")
st.write(
    "This application predicts whether a financial transaction is "
    "**Fraudulent** or **Legitimate** using a high-recall XGBoost model."
)

st.divider()

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
st.subheader("Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)

oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, step=1.0)
newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, step=1.0)

oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, step=1.0)
newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, step=1.0)

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]
)

st.divider()

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Detect Fraud"):

    # ---------------- Feature Engineering ----------------
    input_data = {
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balanceDiffOrig": oldbalanceOrg - newbalanceOrig,
        "balanceDiffDest": newbalanceDest - oldbalanceDest,
        "zeroBalanceOrig": int(newbalanceOrig == 0),
        "zeroBalanceDest": int(newbalanceDest == 0),
        "highRiskType": int(transaction_type in ["TRANSFER", "CASH_OUT"]),
        "type": transaction_type
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode transaction type
    input_df = pd.get_dummies(input_df, columns=["type"])

    # Add missing columns (if any)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[feature_columns]

    # ---------------- Prediction ----------------
    fraud_probability = model.predict_proba(input_df)[0][1]
    threshold = 0.33  # tuned for high recall
    prediction = int(fraud_probability >= threshold)

    # ---------------- Output ----------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"🚨 **FRAUD DETECTED**\n\n"
            f"**Fraud Probability:** {fraud_probability:.2%}"
        )
    else:
        st.success(
            f"✅ **Legitimate Transaction**\n\n"
            f"**Fraud Probability:** {fraud_probability:.2%}"
        )

st.divider()

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.caption(
    "Model: XGBoost | Optimized for High Recall | "
    "Use case: Financial Fraud Detection"
)
