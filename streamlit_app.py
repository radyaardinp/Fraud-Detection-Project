import streamlit as st
import pandas as pd
import numpy as np
import joblib

from feature_engineering import preprocess_form_input
from normalize import normalize_data
from predict_pipeline import predict_fraud
from lime_explainer import explain_instance
from selected_features import FEATURES

st.set_page_config(page_title="Online Payment Transaction Fraud Detection", layout="centered")

st.title("💳 Deteksi Fraud Transaksi Online")

st.markdown("Isi detail transaksi di bawah ini untuk mengetahui kemungkinan fraud.")

# --- Form Input ---
with st.form(key="fraud_form"):
    amount = st.number_input("Nominal Transaksi (amount)", min_value=0.0)
    inquiryAmount = st.number_input("Nominal Inquiry", min_value=0.0)
    merchant = st.text_input("Merchant ID")
    settlementAmount = st.number_input("Nominal Settlement", min_value=0.0)
    feeAmount = st.number_input("Fee Amount", min_value=0.0)
    discountAmount = st.number_input("Discount Amount", min_value=0.0)
    paymentSource = st.text_input("Sumber Pembayaran (paymentSource)")
    status = st.text_input("Status Transaksi")
    statusCode = st.text_input("Kode Status Transaksi")
    createdTime = st.datetime_input("Waktu Transaksi Dimulai (createdTime)")
    updatedTime = st.datetime_input("Waktu Transaksi Selesai (updatedTime)")

    submit = st.form_submit_button("Prediksi Fraud")

if submit:
    with st.spinner("Sedang memproses..."):
        # 1. Kumpulkan input ke dictionary
        input_dict = {
            'amount': amount,
            'inquiryAmount': inquiryAmount,
            'merchant': merchant,
            'settlementAmount': settlementAmount,
            'feeAmount': feeAmount,
            'discountAmount': discountAmount,
            'paymentSource': paymentSource,
            'status': status,
            'statusCode': statusCode,
            'createdTime': createdTime,
            'updatedTime': updatedTime
        }

        # 2. Preprocessing (feature engineering)
        df_features = preprocess_form_input(input_dict)

        # 3. Normalisasi
        df_scaled = normalize_data(df_features)

        # 4. Prediksi
        prediction, probability = predict_fraud(df_scaled)

        label = "🚨 FRAUDULEN" if prediction == 1 else "✅ AMAN"
        st.subheader(f"Hasil Prediksi: {label}")
        st.write(f"Probabilitas Fraud: `{probability*100:.2f}%`")

        # 5. XAI - LIME
        st.markdown("---")
        st.markdown("### 🔍 Penjelasan Model (LIME)")
        lime_html = explain_instance(df_scaled)
        st.components.v1.html(lime_html, height=400, scrolling=True)
