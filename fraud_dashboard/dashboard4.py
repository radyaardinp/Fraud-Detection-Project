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

# Page configuration
st.set_page_config(
    page_title="🛡️ Fraud Detection System Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Aktivasi fungsi
def activation_function(x, func_type='sigmoid'):
    if func_type == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif func_type == 'tanh':
        return np.tanh(x)
    elif func_type == 'relu':
        return np.maximum(0, x)
    else:
        return x

# Fungsi prediksi untuk model dict
def elm_predict(X, model_dict):
    input_weights = model_dict['input_weights']
    biases = model_dict['biases']
    output_weights = model_dict['output_weights']
    activation_type = model_dict['activation_type']
    threshold = model_dict['threshold']
    
    # Hidden layer
    H = activation_function(np.dot(X, input_weights) + biases, func_type=activation_type)
    # Output layer
    y_pred_raw = np.dot(H, output_weights)
    # Probabilitas fraud (gunakan sigmoid)
    y_prob = 1 / (1 + np.exp(-y_pred_raw))
    # Threshold untuk klasifikasi
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_pred, y_prob
    
def elm_predict_proba(X, model_dict):
    input_weights = model_dict['input_weights']
    biases = model_dict['biases']
    output_weights = model_dict['output_weights']
    activation_type = model_dict['activation_type']

    # Hitung hidden layer
    H = activation_function(np.dot(X, input_weights) + biases, func_type=activation_type)
    y_raw = np.dot(H, output_weights)

    # Konversi ke probabilitas
    y_prob = 1 / (1 + np.exp(-y_raw))

    # LIME butuh bentuk (n_samples, 2): [non-fraud, fraud]
    return np.column_stack([1 - y_prob, y_prob])

def page_upload():
    """Halaman 1: Upload dan Preview Data"""
    
    # Custom CSS untuk styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 0.2rem;
            padding: 0.5rem;
        }
        
        .sub-header {
            font-size: 1.3rem;
            font-weight: 500;
            color: #666;
            text-align: center;
            margin-bottom: 0.5rem;
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
        
        .highlight-text {
            color: #2E86AB;
            font-weight: 600;
        }
        
        .upload-section {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            margin: 2rem auto;
            max-width: 600px;
            border: 2px dashed #dee2e6;
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
        
        .stButton > button:hover {
            background-color: #1e5f7a;
        }
        
        .footer {
            margin-top: 3rem;
            padding: 2rem;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
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
                padding: 1.5rem;
                margin: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<div class="main-header">🛡️ Fraud Detection System Dashboard</div>', unsafe_allow_html=True)
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
    
    st.markdown("### 📁 Upload Data Transaksi")
    st.markdown("Silakan upload file CSV berisi data transaksi untuk dianalisis")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your transaction data in CSV format"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Show file info if uploaded
    if uploaded_file is not None:
        st.markdown("---")
        st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")
        
        try:
            df = pd.read_csv(uploaded_file)
            
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

            # Column information
            st.markdown("### 📊 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)

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

    # Footer
    st.markdown("""
    <div class="footer">
        <p>📊 Fraud Detection System | Powered by Extreme Learning Machine & LIME</p>
        <p>Upload your transaction data to begin fraud detection analysis</p>
    </div>
    """, unsafe_allow_html=True)

class FraudDetectionDashboard:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        
    @st.cache_resource
    def load_model_components(_self):
        """Load model, scaler, dan selected features"""
        try:
            model = load("fraud_dashboard/hyperparameter_ELM.joblib")
            scaler = load("fraud_dashboard/scaler.joblib")
            selected_features = load("fraud_dashboard/selected_features.joblib")
            return model, scaler, selected_features
        except Exception as e:
            st.error(f"Error loading model components: {str(e)}")
            return None, None, None
    
    def validate_data(self, df, selected_features):
        """Validasi apakah data memiliki kolom yang diperlukan"""
        missing = [f for f in selected_features if f not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
            st.error("Please ensure your preprocessing pipeline generates all required features.")
            return False
        return True
    
    def perform_prediction(self, df, model, scaler, selected_features):
        """Melakukan prediksi fraud"""
        try:
            # Debug info
            st.write(f"🔍 **Debug Info:**")
            st.write(f"- Data columns: {len(df.columns)} → {list(df.columns)}")
            st.write(f"- Selected features: {len(selected_features)} → {list(selected_features)}")
            
            # Check missing features
            missing_features = [f for f in selected_features if f not in df.columns]
            if missing_features:
                st.error(f"❌ Missing features: {missing_features}")
                return None, None
            
            # Select only the required features
            X_input = df.loc[:, selected_features]
            st.write(f"- Input shape: {X_input.shape}")
            st.write(f"- Scaler expects: {scaler.n_features_in_} features")

            # Penyesuaian urutan kolom
            if list(X_input.columns) != selected_features:
                st.warning("⚠️ Urutan fitur tidak sama, akan disesuaikan.")
                X_input = X_input.reindex(columns=selected_features)
            
            # Check if shapes match
            if X_input.shape[1] != scaler.n_features_in_:
                st.error(f"❌ Feature count mismatch:")
                st.error(f"   - Data has: {X_input.shape[1]} features")
                st.error(f"   - Scaler expects: {scaler.n_features_in_} features")
                return None, None
            
            # Perform scaling and prediction
            X_scaled = scaler.transform(X_input.values)
            y_pred, y_prob = elm_predict(X_scaled, model)
            df['predicted_fraud'] = y_pred
            df['fraud_probability'] = y_prob
            
            st.success(f"✅ Prediction successful! Shape: {X_scaled.shape}")
            return df, X_scaled
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None

    def create_compact_metrics(self, df):
        """Membuat metrics yang compact"""
        total_transaksi = len(df)
        total_fraud = df['predicted_fraud'].sum()
        total_nonfraud = total_transaksi - total_fraud
        fraud_rate = round((total_fraud / total_transaksi) * 100, 2)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transaksi", f"{total_transaksi:,}")
        with col2:
            st.metric("Fraud", f"{total_fraud:,}", delta=f"{total_fraud}")
        with col3:
            st.metric("Non-Fraud", f"{total_nonfraud:,}")
        with col4:
            st.metric("Fraud Rate", f"{fraud_rate}%", delta=f"{fraud_rate-5:.1f}%")
        
        return total_fraud, fraud_rate
    
    def create_fraud_pie_chart(self, df):
        """Membuat pie chart"""
        fraud_counts = df['predicted_fraud'].value_counts().rename({0: 'Non-Fraud', 1: 'Fraud'})
        
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#28a745', '#dc3545']
        wedges, texts, autotexts = ax.pie(fraud_counts, labels=fraud_counts.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Distribusi Fraud vs Non-Fraud', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_hourly_chart(self, df):
        """Membuat chart distribusi per jam"""
        if 'trx_hour' in df.columns:
            hourly_fraud = df.groupby('trx_hour')['predicted_fraud'].agg(['count', 'sum']).reset_index()
            hourly_fraud['fraud_rate'] = (hourly_fraud['sum'] / hourly_fraud['count'] * 100).round(2)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(hourly_fraud['trx_hour'], hourly_fraud['fraud_rate'], 
                         color='#dc3545', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Jam Transaksi', fontsize=12)
            ax.set_ylabel('Fraud Rate (%)', fontsize=12)
            ax.set_title('Fraud Rate per Jam', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Highlight highest fraud rate
            max_idx = hourly_fraud['fraud_rate'].idxmax()
            bars[max_idx].set_color('#a71d2a')
            
            plt.tight_layout()
            return fig
        return None
    
    def create_merchant_chart(self, df):
        """Membuat chart merchant"""
        if 'merchantId' in df.columns:
            merchant_fraud = df[df['predicted_fraud'] == 1]['merchantId'].value_counts().head(10)
            
            if len(merchant_fraud) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bars = ax.barh(range(len(merchant_fraud)), merchant_fraud.values, 
                              color='#dc3545', alpha=0.7)
                
                ax.set_yticks(range(len(merchant_fraud)))
                ax.set_yticklabels([f"Merchant {mid}" for mid in merchant_fraud.index])
                ax.set_xlabel('Jumlah Fraud', fontsize=12)
                ax.set_title('Top 10 Merchant dengan Fraud Terbanyak', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                return fig
        return None
    
    def create_lime_explanation(self, X_scaled, selected_features, model, idx_to_explain):
        """Membuat LIME explanation"""
        try:
            # Check NaN & inf
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                st.error("❌ X_scaled contains NaN or infinite values. Cannot generate LIME explanation.")
                return None

            # Remove constant columns
            variance = X_scaled.var(axis=0)
            non_constant_idx = variance > 0
            X_scaled_filtered = X_scaled[:, non_constant_idx]
            
            if X_scaled_filtered.shape[1] == 0:
                st.error("❌ No features with variance > 0 for LIME.")
                return None
                
            explainer = LimeTabularExplainer(
                training_data=X_scaled,
                feature_names=selected_features,
                class_names=["Non-Fraud", "Fraud"],
                mode="classification", 
                discretize_continuous=False
            )
            
            exp = explainer.explain_instance(
                data_row=X_scaled[idx_to_explain],
                predict_fn=lambda x: elm_predict_proba(x, model)
            )
            
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(12, 8)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"LIME error: {str(e)}")
            return None

def page_analysis():
    """Halaman 2: Analisis dan Visualisasi"""
    
    # Custom CSS untuk halaman analisis
    st.markdown("""
    <style>
        .analysis-header {
            font-size: 2rem;
            font-weight: 700;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2E86AB;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0px 20px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header dengan tombol back
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Upload", key="back_btn"):
            st.session_state.current_page = 'upload'
            st.rerun()
    
    with col2:
        st.markdown('<div class="analysis-header">🔍 Fraud Analysis Results</div>', unsafe_allow_html=True)
    
    # Check if data exists
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ No data found. Please upload data first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return
    
    # Initialize dashboard
    dashboard = FraudDetectionDashboard()
    
    # Load model components
    model, scaler, selected_features = dashboard.load_model_components()
    
    if model is None:
        st.error("❌ Failed to load model components")
        return
    
    # Get data from session
    raw_df = st.session_state.uploaded_data
    
    # Preprocessing
    with st.spinner("🔄 Processing data..."):
        try:
            df = preprocess_for_prediction(raw_df)
            
            # Validate and predict
            if not dashboard.validate_data(df, selected_features):
                st.error("❌ Data validation failed")
                return
            
            df_with_pred, X_scaled = dashboard.perform_prediction(df, model, scaler, selected_features)
            
            if df_with_pred is None:
                st.error("❌ Prediction failed")
                return
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            return
    
    # Show results
    st.markdown("---")
    
    # Metrics
    total_fraud, fraud_rate = dashboard.create_compact_metrics(df_with_pred)
    
    # Tabs untuk visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Details", "🧠 AI Explanation"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Fraud Distribution")
            fig_pie = dashboard.create_fraud_pie_chart(df_with_pred)
            st.pyplot(fig_pie, use_container_width=True)
            plt.close(fig_pie)
        
        with col2:
            st.markdown("### 🏪 Top Fraud Merchants")
            fig_merchant = dashboard.create_merchant_chart(df_with_pred)
            if fig_merchant:
                st.pyplot(fig_merchant, use_container_width=True)
                plt.close(fig_merchant)
            else:
                st.info("No merchant data available for visualization")
    
    with tab2:
        st.markdown("### ⏰ Hourly Fraud Pattern")
        fig_hourly = dashboard.create_hourly_chart(df_with_pred)
        if fig_hourly:
            st.pyplot(fig_hourly, use_container_width=True)
            plt.close(fig_hourly)
        else:
            st.info("No hourly data available for visualization")
    
    with tab3:
        # Fraud details
        detected = df_with_pred[df_with_pred['predicted_fraud'] == 1]
        
        if len(detected) > 0:
            st.markdown(f"### 🚨 Fraud Transactions ({len(detected)} detected)")
            
            # Show fraud data
            st.dataframe(detected, use_container_width=True)
            
            # Download button
            csv = detected.to_csv(index=False)
            st.download_button(
                label="📥 Download Fraud Data",
                data=csv,
                file_name='fraud_transactions.csv',
                mime='text/csv',
                use_container_width=True
            )
            
            # Show probability distribution
            st.markdown("### 📊 Fraud Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_with_pred['fraud_probability'], bins=30, alpha=0.7, color='#2E86AB')
            ax.set_xlabel('Fraud Probability')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Fraud Probabilities')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        else:
            st.success("🎉 No fraud transactions detected!")
    
    with tab4:
        st.markdown("### 🧠 AI Explanation (LIME)")
        st.markdown("Select a transaction to see why the AI made its prediction:")
        
        if len(df_with_pred) > 0:
            # Select transaction
            max_display = min(50, len(df_with_pred))
            idx_to_explain = st.selectbox(
                "Choose transaction to explain:",
                range(max_display),
                format_func=lambda x: f"Transaction {x} - {'FRAUD' if df_with_pred.iloc[x]['predicted_fraud'] == 1 else 'NON-FRAUD'} (Prob: {df_with_pred.iloc[x]['fraud_probability']:.3f})"
            )
            
            # Show transaction details
            st.markdown("#### Transaction Details:")
            selected_transaction = df_with_pred.iloc[idx_to_explain]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Prediction:** {'🚨 FRAUD' if selected_transaction['predicted_fraud'] == 1 else '✅ NON-FRAUD'}")
                st.write(f"**Probability:** {selected_transaction['fraud_probability']:.4f}")
            with col2:
                if 'merchantId' in selected_transaction:
                    st.write(f"**Merchant ID:** {selected_transaction['merchantId']}")
                if 'trx_hour' in selected_transaction:
                    st.write(f"**Hour:** {selected_transaction['trx_hour']}")
            
            # Generate LIME explanation
            if st.button("🔍 Generate AI Explanation", key="lime_btn", use_container_width=True):
                with st.spinner("Generating AI explanation..."):
                    lime_fig = dashboard.create_lime_explanation(
                        X_scaled, selected_features, model, idx_to_explain
                    )
                    if lime_fig:
                        st.pyplot(lime_fig, use_container_width=True)
                        plt.close(lime_fig)
                        
                        st.info("""
                        **How to read this explanation:**
                        - 🟢 Green bars: Features that contribute to NON-FRAUD prediction
                        - 🔴 Red bars: Features that contribute to FRAUD prediction
                        - Longer bars = stronger influence on the prediction
                        """)
        else:
            st.info("No transactions available for explanation")

# Main navigation logic
def main():
    """Main function untuk navigasi antar halaman"""
    
    if st.session_state.current_page == 'upload':
        page_upload()
    elif st.session_state.current_page == 'analysis':
        page_analysis()

if __name__ == "__main__":
    main()
