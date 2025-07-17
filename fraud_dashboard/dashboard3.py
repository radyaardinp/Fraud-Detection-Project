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

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .description-text {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        line-height: 1.6;
        margin: 2rem auto;
        max-width: 800px;
        padding: 0 2rem;
    }
    
    .upload-section:hover {
        border-color: #2E86AB;
        background: #f0f8ff;
    }
    
    .upload-icon {
        font-size: 4rem;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    
    .upload-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
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
    
    /* Hide Streamlit default elements */
    .stFileUploader > div > div > div {
        display: none;
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
            font-size: 3rem;
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
st.markdown('<div class="upload-icon">☁️</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-title">Drag and drop file here</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-subtitle">Limit 200MB per file • CSV</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "",
    type=['csv'],
    help="Upload your transaction data in CSV format",
    label_visibility="collapsed"
)

# Browse files button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("📁 Browse files", key="browse_btn"):
        st.info("💡 Please use the drag & drop area above to select your CSV file")

st.markdown('</div>', unsafe_allow_html=True)

# Show file info if uploaded
if uploaded_file is not None:
    st.markdown("---")
    st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")
    
    # File details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
    with col2:
        st.metric("File Type", uploaded_file.type)
    with col3:
        st.metric("Status", "Ready for Analysis")
    
    # Show sample of uploaded data
    try:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Basic info
        st.markdown("### 📊 Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Rows:** {len(df):,}")
            st.info(f"**Columns:** {df.shape[1]}")
        with col2:
            st.info(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            st.info(f"**Data Types:** {df.dtypes.value_counts().to_dict()}")
        
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

# Example of how to handle different pages
if st.session_state.current_page == 'analysis':
    st.markdown("---")
    st.markdown("## 🔍 Analysis Page")
    st.success("✅ Data berhasil diupload dan siap untuk dianalisis!")
    
    # Show uploaded data info
    if 'uploaded_data' in st.session_state:
        df = st.session_state.uploaded_data
        st.markdown("### 📊 Data yang akan dianalisis:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", f"{df.shape[1]}")
        with col3:
            st.metric("File Size", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    st.info("🔄 Halaman analisis sedang dalam pengembangan...")
    
    # Back to upload button
    if st.button("⬅️ Back to Upload"):
        st.session_state.current_page = 'upload'
        st.rerun()

elif st.session_state.current_page == 'dashboard':
    st.markdown("---")
    st.markdown("## 📊 Dashboard Page")
    st.info("This is where the dashboard functionality would go")
    
    # Example dashboard content
    if 'uploaded_data' in st.session_state:
        st.markdown("### Data Summary")
        st.dataframe(st.session_state.uploaded_data.describe())
    
    # Back to upload button
    if st.button("⬅️ Back to Upload"):
        st.session_state.current_page = 'upload'
        st.rerun()

# Footer
st.markdown("""
<div class="footer">
    <p>🔐 <strong>Fraud Detection System</strong> - Powered by Extreme Learning Machine & LIME</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
