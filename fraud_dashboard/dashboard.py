import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from joblib import load
from lime.lime_tabular import LimeTabularExplainer
from preprocessing_pipeline import preprocess_for_prediction
from normalize import normalize_data
from predict_pipeline import activation_function


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("📊 Fraud Detection Dashboard")
st.markdown("Unggah data historis transaksi untuk analisis otomatis dan deteksi fraud berdasarkan model ELM yang telah dilatih.")

# Upload file CSV
uploaded_file = st.file_uploader("📁 Upload file CSV transaksi", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def load_model_components():
    model = load("fraud_dashboard/hyperparameter_ELM.joblib")
    scaler = load("fraud_dashboard/scaler.joblib")
    selected_features = load("fraud_dashboard/selected_features.joblib")
    return model, scaler, selected_features

if uploaded_file:
    raw_df = load_data(uploaded_file)
    st.success("✅ Data berhasil dimuat!")

    # Layout dua kolom
    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.subheader("📥 Input Data")
        st.dataframe(raw_df.head(10))

        # Jalankan preprocessing
        df = preprocess_for_prediction(raw_df)

        # Load model dan komponen
        model, scaler, selected_features = load_model_components()

        missing = [f for f in selected_features if f not in df.columns]
        if missing:
            st.error(f"❌ Data tidak memiliki kolom-kolom berikut yang dibutuhkan untuk prediksi: {missing}")
        else:
            X_input = df.loc[:, selected_features]
            X_scaled = scaler.transform(X_input)
            y_pred = model.predict(X_scaled)

            df['predicted_fraud'] = y_pred
            detected = df[df['predicted_fraud'] == 1]

            st.success(f"✅ Terdeteksi {len(detected)} transaksi yang dicurigai sebagai fraud.")

            idx_to_explain = st.number_input("Index transaksi untuk dijelaskan", min_value=0, max_value=len(df)-1, step=1)
            explainer = LimeTabularExplainer(
                training_data=X_scaled,
                feature_names=selected_features,
                class_names=["Non-Fraud", "Fraud"],
                mode="classification"
            )
            exp = explainer.explain_instance(
                data_row=X_scaled[idx_to_explain],
                predict_fn=lambda x: model.predict_proba(x)
            )
            st.write(f"🧠 Penjelasan Transaksi ke-{idx_to_explain}:")
            st.pyplot(exp.as_pyplot_figure())

    with col_output:
        st.subheader("📊 Ringkasan & Visualisasi")
        total_transaksi = len(df)
        total_fraud = df['predicted_fraud'].sum()
        total_nonfraud = total_transaksi - total_fraud
        fraud_rate = round((total_fraud / total_transaksi) * 100, 2)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Transaksi", f"{total_transaksi:,}")
        k2.metric("Jumlah Fraud", f"{total_fraud:,}")
        k3.metric("Jumlah Non-Fraud", f"{total_nonfraud:,}")
        k4.metric("Fraud Rate", f"{fraud_rate}%")

        # Distribusi Pie Chart
        fraud_counts = df['predicted_fraud'].value_counts().rename({0: 'Non-Fraud', 1: 'Fraud'})
        fig1, ax1 = plt.subplots()
        ax1.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=["#00cc96", "#ff6361"])
        ax1.axis('equal')
        st.pyplot(fig1)

        # Distribusi jam jika ada
        if 'trx_hour' in df.columns:
            trx_hour_df = df.groupby(['trx_hour', 'predicted_fraud']).size().unstack().fillna(0)
            trx_hour_df.columns = ['Non-Fraud', 'Fraud']
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            trx_hour_df.plot(kind='bar', stacked=True, ax=ax2, color=["#00cc96", "#ff6361"])
            ax2.set_xlabel("Jam Transaksi")
            ax2.set_ylabel("Jumlah")
            st.pyplot(fig2)

        # Top merchant fraud
        if 'merchantId' in df.columns:
            top_fraud_merchant = df[df['predicted_fraud'] == 1].groupby('merchantId').size().sort_values(ascending=False).head(10)
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            top_fraud_merchant.plot(kind='bar', ax=ax3, color="#ff6361")
            ax3.set_ylabel("Jumlah Fraud")
            st.pyplot(fig3)
else:
    st.info("⬆️ Silakan upload file .csv terlebih dahulu.")
