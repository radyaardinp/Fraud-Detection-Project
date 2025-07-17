import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st


def landing_page():
    st.markdown("# 🛡️ Fraud Detection Dashboard")
    st.markdown("Deteksi fraud otomatis menggunakan model ELM")

    st.markdown("### 📁 Upload Data CSV untuk memulai analisis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        # Simpan data ke session_state
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.file_uploaded = True
        st.success("✅ File berhasil diupload!")
        st.experimental_rerun()  # Refresh untuk pindah halaman
