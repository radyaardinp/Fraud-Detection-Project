
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

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'upload'
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'show_lime' not in st.session_state:
    st.session_state.show_lime = False
if 'lime_idx' not in st.session_state:
    st.session_state.lime_idx = None
    
#===== HALAMAN 1 ========
# Page configuration
st.set_page_config(
    page_title="🛡️Fraud Detection System Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)
def page_upload():
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.2rem;
        padding: 0.5 rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #666;
        text-align: center;
        margin-bottom: 0.5 rem auto;
    }
    
    .description-text {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        line-height: 1.6;
        margin: 1rem auto;
        max-width: 800px;
        padding: 0.2rem;
    }
    
    .upload-icon {
        font-size: 4rem;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    
    .upload-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        font-size: 0.8 rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        text-align: center;
        color: #666;
        border-top: 1px solid #eee;
    }
    
    /* Custom file uploader styling */
    .stFileUploader {
        background: transparent;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
        }
        
        .description-text {
            font-size: 1rem;
            padding: 0 1rem;
        }
        
        .upload-section {
            padding: 2rem 1rem;
            margin: 1rem;
        }
        
        .upload-icon {
            font-size: 2rem;
        }
        
        .upload-title {
            font-size: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">Fraud Detection System Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Online Payment Transaction</div>', unsafe_allow_html=True)

# Description
st.markdown("""
<div class="description-text">
Dashboard ini dirancang untuk mendeteksi potensi transaksi fraud pada 
pembayaran online menggunakan metode <span class="highlight-text">Extreme Learning Machine 
(ELM)</span>. Sistem ini mempermudah analisis pola transaksi mencurigakan 
dan memberikan interpretasi model menggunakan <span class="highlight-text">Explainable AI (LIME)</span>.
</div>
""", unsafe_allow_html=True)

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "",
    type=['csv'],
    help="Upload your transaction data in CSV format",
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# Show file info if uploaded
if uploaded_file is not None:
    st.markdown("---")
    st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")
    try:
        df=pd.read_csv(uploaded_file)
        # File details
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Rows:** {len(df):,}")
            st.info(f"**Columns:** {df.shape[1]}")
        with col2:
            st.info(f"**File Size:** {uploaded_file.size / (1024*1024):.2f} MB")
            st.info(f"**File Type:** {uploaded_file.type}")

        # Show sample of uploaded data
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Start Analysis button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 Start Analysis", key="analysis_btn", use_container_width=True):
                st.session_state.current_page = 'analysis'
                st.session_state.uploaded_data = df  # Store data in session
                st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted.")


#==================== HALAMAN 2 ==============================

def page_analysis():
    from joblib import load

    # Fungsi aktivasi
    def activation_function(x, func_type='sigmoid'):
        if func_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func_type == 'tanh':
            return np.tanh(x)
        elif func_type == 'relu':
            return np.maximum(0, x)
        else:
            return x

    # Prediksi ELM
    def elm_predict(X, model_dict):
        input_weights = model_dict['input_weights']
        biases = model_dict['biases']
        output_weights = model_dict['output_weights']
        activation_type = model_dict['activation_type']
        threshold = model_dict['threshold']

        H = activation_function(np.dot(X, input_weights) + biases, func_type=activation_type)
        y_raw = np.dot(H, output_weights)
        y_prob = 1 / (1 + np.exp(-y_raw))
        y_pred = (y_prob >= threshold).astype(int)
        return y_pred, y_prob

    def elm_predict_proba(X, model_dict):
        H = activation_function(np.dot(X, model_dict['input_weights']) + model_dict['biases'],
                                func_type=model_dict['activation_type'])
        y_raw = np.dot(H, model_dict['output_weights'])
        y_prob = 1 / (1 + np.exp(-y_raw))
        return np.column_stack([1 - y_prob, y_prob])

    # Load komponen model
    model = load("fraud_dashboard/hyperparameter_ELM.joblib")
    scaler = load("fraud_dashboard/scaler.joblib")
    selected_features = load("fraud_dashboard/selected_features.joblib")

    # Ambil data dari halaman 1
    if st.session_state.uploaded_data is None:
        st.error("❌ No data uploaded. Please go back to Upload Page.")
        if st.button("⬅ Back to Upload Page"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return

    df = st.session_state.uploaded_data

    st.markdown("# 📊 Fraud Detection Analysis")
    st.write("Berikut hasil deteksi fraud menggunakan model ELM:")

    # Preprocessing
    df_processed = preprocess_for_prediction(df)

    # Validasi fitur
    if not all(f in df_processed.columns for f in selected_features):
        st.error("❌ Missing required features. Pastikan preprocessing sudah benar.")
        return

    # Prediksi
    X_input = df_processed[selected_features]
    X_scaled = scaler.transform(X_input)
    y_pred, y_prob = elm_predict(X_scaled, model)
    df['predicted_fraud'] = y_pred
    df['fraud_probability'] = y_prob

    # Metrics
    total_tx = len(df)
    fraud_tx = df['predicted_fraud'].sum()
    fraud_rate = (fraud_tx / total_tx) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total_tx:,}")
    col2.metric("Fraud Detected", f"{fraud_tx:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    # Visualisasi
    st.markdown("### 🔍 Fraud Distribution")
    fig, ax = plt.subplots()
    df['predicted_fraud'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Non-Fraud', 'Fraud'], colors=['#28a745', '#dc3545'], ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    # Tabel hasil fraud
    st.markdown("### 🚨 Fraud Transactions")
    fraud_data = df[df['predicted_fraud'] == 1]
    st.dataframe(fraud_data.head(10), use_container_width=True)

    # Tombol download
    csv = fraud_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Fraud Data",
        data=csv,
        file_name="fraud_transactions.csv",
        mime="text/csv"
    )

    # Tombol kembali
    st.markdown("---")
    if st.button("⬅ Back to Upload Page"):
        st.session_state.current_page = 'upload'
        st.rerun()

# =============================
# ROUTING
# =============================
if st.session_state.current_page == 'upload':
    page_upload()
elif st.session_state.current_page == 'analysis':
    page_analysis()

# Footer
st.markdown("""
<div class="footer">
    <p>🔐 <strong>Fraud Detection System</strong> - Powered by Extreme Learning Machine & LIME</p>
</div>
""", unsafe_allow_html=True)
