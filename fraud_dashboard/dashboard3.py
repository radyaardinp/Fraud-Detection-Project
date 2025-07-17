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

# ====== SETUP DASHBOARD ======
st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")

# ====== CSS CUSTOM (Tema Biru-Putih) ======
st.markdown("""
<style>
/* Background */
.main {
    background-color: #f9f9f9;
    padding: 20px;
}

/* Judul */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    color: #007bff;
    margin-bottom: 5px;
}

/* Sub Judul */
.subtitle {
    text-align: center;
    font-size: 22px;
    font-weight: 500;
    color: #444;
    margin-bottom: 10px;
}

/* Deskripsi */
.description {
    text-align: center;
    font-size: 16px;
    color: #555;
    margin-bottom: 30px;
    padding: 0 15%;
    line-height: 1.6;
}

/* Upload Box */
.uploadedFile {
    border: 2px dashed #007bff !important;
    padding: 20px;
    border-radius: 10px;
    background-color: #fff;
}

/* Tombol */
div.stButton > button {
    background-color: #007bff;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 8px;
    width: 100%;
    height: 50px;
    margin-top: 20px;
}
div.stButton > button:hover {
    background-color: #0056b3;
}
</style>
""", unsafe_allow_html=True)

# ====== SESSION STATE ======
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
    st.session_state.df = None

# ====== HALAMAN 1 ======
def landing_page():
    # Judul & Deskripsi
    st.markdown('<p class="title">Fraud Detection System Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Online Payment Transaction</p>', unsafe_allow_html=True)

    # Deskripsi Singkat
    st.markdown("""
    <p class="description">
    Dashboard ini dirancang untuk mendeteksi potensi transaksi fraud pada pembayaran online 
    menggunakan metode <b>Extreme Learning Machine (ELM)</b>. 
    Sistem ini mempermudah analisis pola transaksi mencurigakan dan 
    memberikan interpretasi model menggunakan Explainable AI (LIME).
    </p>
    """, unsafe_allow_html=True)

    # Upload file
    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"], label_visibility="collapsed")

    # Kalau sudah upload
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # Simpan data

        # Success message
        st.success("✅ File berhasil diupload!")

        # Preview Data
        with st.expander("👀 Lihat 5 baris pertama data"):
            st.dataframe(df.head())

        # Tombol lanjut
        if st.button("➡ Lanjut ke Analisis"):
            st.session_state.file_uploaded = True

# Jalankan halaman
landing_page()
