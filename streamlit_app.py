import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Atur tampilan halaman
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Title
st.title("📊 Fraud Detection Dashboard")
st.markdown("Upload data historis transaksi untuk analisis otomatis dan visualisasi fraud detection.")

# Upload file CSV
uploaded_file = st.file_uploader("📁 Upload file CSV transaksi", type=["csv"])

# Load data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Main block
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ Data berhasil dimuat!")
    
    # Tampilkan 5 baris pertama
    with st.expander("🔍 Lihat data mentah"):
        st.dataframe(df.head())

    st.divider()

    # Cek apakah kolom 'fraud_label' ada
    if 'fraud_label' not in df.columns:
        st.error("❌ Kolom 'fraud_label' tidak ditemukan! Harap pastikan file memiliki kolom tersebut.")
    else:
        # Summary angka
        total_transaksi = len(df)
        total_fraud = df['fraud_label'].sum()
        total_nonfraud = total_transaksi - total_fraud
        fraud_rate = round((total_fraud / total_transaksi) * 100, 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transaksi", f"{total_transaksi:,}")
        col2.metric("Jumlah Fraud", f"{total_fraud:,}")
        col3.metric("Jumlah Non-Fraud", f"{total_nonfraud:,}")
        col4.metric("Fraud Rate", f"{fraud_rate}%")

        st.divider()

        # Grafik pie chart fraud vs non-fraud
        st.subheader("📌 Distribusi Fraud vs Non-Fraud")
        fraud_counts = df['fraud_label'].value_counts().rename({0: 'Non-Fraud', 1: 'Fraud'})

        fig, ax = plt.subplots()
        ax.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=["#00cc96", "#ff6361"])
        ax.axis('equal')
        st.pyplot(fig)

        st.divider()

        # (Tambahan visualisasi lanjutan bisa dimasukkan di bawah sini...)

else:
    st.info("⬆️ Silakan upload file .csv terlebih dahulu untuk memulai analisis.")
