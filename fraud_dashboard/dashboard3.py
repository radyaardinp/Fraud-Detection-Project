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


def landing_page():
    st.markdown("# 🛡️ Fraud Detection Dashboard")
    st.markdown("Deteksi fraud otomatis menggunakan model ELM")

    st.markdown("### 📁 Upload Data CSV untuk memulai analisis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    # Kalau file sudah di-upload, tampilkan preview + tombol Next
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # simpan data

        st.success("✅ File berhasil diupload!")
        
        # Preview 5 baris pertama
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(15))

        # Tombol Next untuk pindah ke halaman analisis
        if st.button("➡ Lanjut ke Analisis"):
            st.session_state.file_uploaded = True
