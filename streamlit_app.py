import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

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
    model = pickle.load(open("ELM_hyperparameter.joblib", "rb"))
    scaler = pickle.load(open("scaler.joblib", "rb"))
    selected_features = pickle.load(open("selected_features.joblib", "rb"))
    return model, scaler, selected_features

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ Data berhasil dimuat!")

    # Data mentah
    with st.expander("🔍 Lihat Data"):
        st.dataframe(df.head())

    if 'fraud' not in df.columns:
        st.error("❌ Kolom 'fraud' tidak ditemukan.")
    else:
        # Convert string label jadi angka 
        if df['fraud'].dtype == object:
            df['fraud'] = df['fraud']
            df['fraud'] = df['fraud'].map({'Not Fraud': 0, 'Fraud': 1})
    
        # Cek kalau ada nilai NaN setelah map
        if df['fraud'].isnull().any():
            st.warning("⚠️ Ada label yang tidak dikenali di kolom fraud. Baris tersebut akan dihapus.")
            df = df.dropna(subset=['fraud'])

        # Ubah ke integer biar aman diproses
        df['fraud'] = df['fraud'].astype(int)


        # ────────── VISUALISASI SUMMARY ────────── #
        st.subheader("📈 Ringkasan Transaksi")
        total_transaksi = len(df)
        total_fraud = df['fraud'].sum()
        total_nonfraud = total_transaksi - total_fraud
        fraud_rate = round((total_fraud / total_transaksi) * 100, 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transaksi", f"{total_transaksi:,}")
        col2.metric("Jumlah Fraud", f"{total_fraud:,}")
        col3.metric("Jumlah Non-Fraud", f"{total_nonfraud:,}")
        col4.metric("Fraud Rate", f"{fraud_rate}%")

        st.divider()

        # ────────── PIE CHART ────────── #
        st.subheader("📌 Distribusi Fraud vs Non-Fraud")
        fraud_counts = df['fraud'].value_counts().rename({0: 'Non-Fraud', 1: 'Fraud'})
        fig, ax = plt.subplots()
        ax.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=["#00cc96", "#ff6361"])
        ax.axis('equal')
        st.pyplot(fig)

        st.divider()

        # ────────── WAKTU: PER JAM ────────── #
        if 'trx_hour' in df.columns:
            st.subheader("🕒 Distribusi Transaksi per Jam")
            trx_hour_df = df.groupby(['trx_hour', 'fraud']).size().unstack().fillna(0)
            trx_hour_df.columns = ['Non-Fraud', 'Fraud']

            fig, ax = plt.subplots(figsize=(10, 4))
            trx_hour_df.plot(kind='bar', stacked=True, ax=ax, color=["#00cc96", "#ff6361"])
            ax.set_xlabel("Jam Transaksi (0-23)")
            ax.set_ylabel("Jumlah Transaksi")
            st.pyplot(fig)

        # ────────── MERCHANT ────────── #
        if 'merchantId' in df.columns:
            st.subheader("🏬 Top Merchant dengan Fraud Tertinggi")
            top_fraud_merchant = df[df['fraud'] == 1].groupby('merchantId').size().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 4))
            top_fraud_merchant.plot(kind='bar', ax=ax, color="#ff6361")
            ax.set_ylabel("Jumlah Fraud")
            st.pyplot(fig)

        st.divider()

        # ────────── DETEKSI FRAUD DENGAN ELM ────────── #
        st.subheader("🔍 Deteksi Otomatis Transaksi Mencurigakan")

        model, scaler, selected_features = load_model_components()

        if all(f in df.columns for f in selected_features):
            X_input = df[selected_features]
            X_scaled = scaler.transform(X_input)
            y_pred = model.predict(X_scaled)

            df['predicted_fraud'] = y_pred
            detected = df[df['predicted_fraud'] == 1]

            st.success(f"✅ Terdeteksi {len(detected)} transaksi yang dicurigai sebagai fraud.")

            with st.expander("📄 Lihat Transaksi Mencurigakan"):
                st.dataframe(detected[['id', 'amount', 'merchantId', 'trx_hour', 'feeAmount', 'predicted_fraud']].head(15))

            # ────────── LIME EXPLAINER ────────── #
            st.subheader("🧠 Penjelasan Model (XAI)")
            idx_to_explain = st.number_input("Masukkan index transaksi untuk dijelaskan (0 - {})".format(len(df)-1), min_value=0, max_value=len(df)-1, step=1)

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

            st.write(f"💬 Penjelasan Transaksi ke-{idx_to_explain}:")
            st.pyplot(exp.as_pyplot_figure())
        else:
            st.error("❌ Data tidak memiliki semua fitur yang dibutuhkan untuk prediksi.")
else:
    st.info("⬆️ Silakan upload file .csv terlebih dahulu.")
