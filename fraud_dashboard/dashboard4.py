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
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'selected_features_data' not in st.session_state:
    st.session_state.selected_features_data = None
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
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip untuk stabilitas
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
    y_prob = 1 / (1 + np.exp(-np.clip(y_pred_raw, -500, 500)))  # Clip untuk stabilitas
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

    # Konversi ke probabilitas dengan clipping untuk stabilitas
    y_prob = 1 / (1 + np.exp(-np.clip(y_raw, -500, 500)))

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
                'Data Type': df.dtypes})
            # Buat keterangan manual untuk setiap kolom
            manual_descriptions = {
                'id': 'Identitas unik transaksi',
                'createdTime': 'Waktu ketika transaksi dibuat',
                'updateTime': 'Waktu ketika transaksi diperbarui',
                'currency': 'Mata uang yang digunakan',
                'amount': 'Jumlah nominal transaksi',
                'inquiryId': 'Identitas unik dari proses inquiry',
                'merchantId': 'Identitas unik dari merchant',
                'type': 'Tipe transaksi',
                'paymentSource': 'Sumber pembayaran',
                'status': 'Status akhir transaksi',
                'statusCode': 'Kode status numerik',
                'networkReferenceId': 'Identitas rujukan jaringan pembayaran',
                'settlementAmount': 'Jumlah nominal transaksi yang dikirimkan ke merchant',
                'inquiryId': 'Jumlah nominal yang direquest pada tahap inquiry',
                'discountAmount': 'Jumlah nominal diskon',
                'feeAmount': 'Biaya Transaksi',
                'typeToken': 'Jenis tokenisasi'
            }

            # Fungsi untuk mendapatkan keterangan
            def get_description(col_name):
                return manual_descriptions.get(col_name, 'Tidak ada deskripsi')

            # Menambahkan kolom keterangan
            col_info['Keterangan'] = col_info['Column'].apply(get_description)

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
            
            # Buat copy untuk hasil prediksi
            result_df = df.copy()
            
            result_df['predicted_fraud'] = y_pred
            result_df['fraud_probability'] = y_prob
            
            st.success(f"✅ Prediction successful! Shape: {X_scaled.shape}")
            return result_df, X_scaled
            
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
        fraud_counts = df['predicted_fraud'].value_counts().rename({1: 'Non-Fraud', 0: 'Fraud'})
        
        fig, ax = plt.subplots(figsize=(4, 3))
        colors = ['#28a745', '#dc3545']
        wedges, texts, autotexts = ax.pie(fraud_counts, labels=fraud_counts.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Distribusi Fraud vs Non-Fraud', fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_payment_source_chart(self, preprocessed_df):
        """Membuat chart distribusi payment source dari data preprocessing"""
        if 'paymentSourceCode' in preprocessed_df.columns:
            payment_counts = preprocessed_df['paymentSourceCode'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(payment_counts.index, payment_counts.values, 
                         color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Payment Source', fontsize=10)
            ax.set_ylabel('Jumlah Transaksi', fontsize=10)
            ax.set_title('Distribusi Transaksi per Payment Source', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            return fig
        return None
    
    def create_merchant_fraud_chart(self, preprocessed_df, processed_df):
        """Membuat chart merchant fraud dari data preprocessing"""
        if 'merchantId' in preprocessed_df.columns:
            # Gabungkan preprocessing data dengan hasil prediksi
            merchant_fraud_data = preprocessed_df.copy()
            merchant_fraud_data['predicted_fraud'] = processed_df['predicted_fraud']
            
            # Hitung fraud per merchant
            merchant_fraud = merchant_fraud_data[merchant_fraud_data['predicted_fraud'] == 1]['merchantId'].value_counts().head(10)
            
            if len(merchant_fraud) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                
                bars = ax.barh(range(len(merchant_fraud)), merchant_fraud.values, 
                              color='#dc3545', alpha=0.7)
                
                ax.set_yticks(range(len(merchant_fraud)))
                ax.set_yticklabels([f"Merchant {mid}" for mid in merchant_fraud.index])
                ax.set_xlabel('Jumlah Fraud', fontsize=10)
                ax.set_title('Top 10 Merchant dengan Fraud Terbanyak', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                return fig
        return None
    
    def create_lime_explanation(self, X_scaled, selected_features, model, idx_to_explain):
        """Membuat LIME explanation - FIXED VERSION"""
        try:
            st.info("🔄 Generating LIME explanation...")
            
            # Validasi input
            if X_scaled is None or len(X_scaled) == 0:
                st.error("❌ X_scaled is empty")
                return None
                
            if idx_to_explain >= len(X_scaled):
                st.error(f"❌ Index {idx_to_explain} out of range (max: {len(X_scaled)-1})")
                return None
            
            # Check for NaN & inf values
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                st.error("❌ X_scaled contains NaN or infinite values")
                return None
            
            # Pastikan data training dan instance yang akan dijelaskan valid
            training_data = X_scaled.copy()
            instance_to_explain = X_scaled[idx_to_explain].copy()
            
            # Debug info
            st.write(f"Training data shape: {training_data.shape}")
            st.write(f"Instance shape: {instance_to_explain.shape}")
            st.write(f"Features: {len(selected_features)}")
            
            # Buat explainer
            explainer = LimeTabularExplainer(
                training_data=training_data,
                feature_names=selected_features,
                class_names=["Non-Fraud", "Fraud"],
                mode="classification",
                discretize_continuous=True,  # Ubah ke True untuk stabilitas
                random_state=42
            )
            
            # Wrapper function untuk prediksi
            def predict_fn(X):
                try:
                    return elm_predict_proba(X, model)
                except Exception as e:
                    st.error(f"Prediction error in LIME: {str(e)}")
                    # Return default probabilities
                    return np.array([[0.5, 0.5]] * len(X))
            
            # Test prediction function
            test_pred = predict_fn(instance_to_explain.reshape(1, -1))
            st.write(f"Test prediction shape: {test_pred.shape}")
            st.write(f"Test prediction: {test_pred}")
            
            # Generate explanation
            exp = explainer.explain_instance(
                data_row=instance_to_explain,
                predict_fn=predict_fn,
                num_features=min(10, len(selected_features)),  # Batasi jumlah fitur
                num_samples=1000  # Reduce samples for faster computation
            )
            
            # Create plot
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(10, 6)
            plt.tight_layout()
            
            st.success("✅ LIME explanation generated successfully!")
            return fig
            
        except Exception as e:
            st.error(f"❌ LIME error: {str(e)}")
            st.error("Please try with a different transaction or check your data.")
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
    original_df = st.session_state.uploaded_data
    
    # Preprocessing
    with st.spinner("🔄 Processing data..."):
        try:
            # Preprocess data
            preprocessed_df = preprocess_for_prediction(original_df)
            
            # Store preprocessed data in session (untuk visualisasi)
            st.session_state.preprocessed_data = preprocessed_df
            
            # Validate and predict
            if not dashboard.validate_data(preprocessed_df, selected_features):
                st.error("❌ Data validation failed")
                return
            
            # Perform prediction
            df_with_pred, X_scaled = dashboard.perform_prediction(preprocessed_df, model, scaler, selected_features)
            
            if df_with_pred is None:
                st.error("❌ Prediction failed")
                return
                
            # Store selected features data (untuk preview)
            selected_features_df = df_with_pred[selected_features + ['predicted_fraud', 'fraud_probability']]
            st.session_state.selected_features_data = selected_features_df
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            return
    
    # Show results
    st.markdown("---")
    
    # Metrics
    total_fraud, fraud_rate = dashboard.create_compact_metrics(df_with_pred)
    
    # Tabs untuk visualisasi - HANYA 2 TABS
    tab1, tab2 = st.tabs(["🔍 Details", "🧠 AI Explanation"])
    
    with tab1:
        st.markdown("### 🔍 Data Analysis Details")
        
        # 1. TAMPILKAN DATAFRAME UTAMA DULU (preprocessing + feature selection + prediction)
        st.markdown("#### 📊 Complete Analysis Results")
        
        # Gabungkan data original dengan hasil prediksi untuk tampilan yang lebih lengkap
        complete_df = selected_features_df.copy()
        
        # Tampilkan dataframe utama
        st.dataframe(complete_df, use_container_width=True)
        
        # Download button untuk complete data
        csv_complete = complete_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Results",
            data=csv_complete,
            file_name='complete_fraud_analysis.csv',
            mime='text/csv',
            use_container_width=True
        )
        
        st.markdown("---")
        
        # 2. VISUALISASI
        st.markdown("### 📊 Fraud Analysis Visualizations")
        
        # Visualisasi dalam grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Fraud Distribution")
            fig_pie = dashboard.create_fraud_pie_chart(df_with_pred)
            st.pyplot(fig_pie, use_container_width=True)
            plt.close(fig_pie)
        
        with col2:
            st.markdown("#### 💳 Payment Source Distribution")
            fig_payment = dashboard.create_payment_source_chart(st.session_state.preprocessed_data)
            if fig_payment:
                st.pyplot(fig_payment, use_container_width=True)
                plt.close(fig_payment)
            else:
                st.info("No payment source data available")
        
        # Merchant fraud chart (full width)
        st.markdown("#### 🏪 Top Fraud Merchants")
        fig_merchant = dashboard.create_merchant_fraud_chart(st.session_state.preprocessed_data, df_with_pred)
        if fig_merchant:
            st.pyplot(fig_merchant, use_container_width=True)
            plt.close(fig_merchant)
        else:
            st.info("No merchant fraud data available")
        
        # Fraud details section
        st.markdown("---")
        detected_fraud = df_with_pred[df_with_pred['predicted_fraud'] == 1]
        
    with tab2:
        st.markdown("### 🧠 AI Explanation (LIME)")
        st.markdown("Select a transaction to see why the AI made its prediction:")
        
        if len(df_with_pred) > 0:
            # Select transaction dengan lebih banyak opsi
            max_display = min(100, len(df_with_pred))
            
            # Filter options berdasarkan prediction
            fraud_indices = df_with_pred[df_with_pred['predicted_fraud'] == 1].index.tolist()
            non_fraud_indices = df_with_pred[df_with_pred['predicted_fraud'] == 0].index.tolist()
            
            # Selection method
            explanation_method = st.radio(
                "Choose explanation method:",
                ["Show All", "Show Only Fraud", "Show Only Non-Fraud"]
            )
            
            if explanation_method == "Show Only Fraud":
                available_indices = fraud_indices[:max_display]
            elif explanation_method == "Show Only Non-Fraud":
                available_indices = non_fraud_indices[:max_display]
            else:
                available_indices = list(range(min(max_display, len(df_with_pred))))
            
            if not available_indices:
                st.warning("⚠️ No transactions available for the selected filter")
                return
            
            # Select transaction
            idx_to_explain = st.selectbox(
                "Choose transaction to explain:",
                available_indices,
                format_func=lambda x: f"Transaction {x} - {'🚨 FRAUD' if df_with_pred.iloc[x]['predicted_fraud'] == 1 else '✅ NON-FRAUD'} (Prob: {df_with_pred.iloc[x]['fraud_probability']:.4f})"
            )
            
            # Show transaction details
            st.markdown("#### Transaction Details:")
            selected_transaction = df_with_pred.iloc[idx_to_explain]
            selected_original = original_df.iloc[idx_to_explain]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Prediction:** {'🚨 FRAUD' if selected_transaction['predicted_fraud'] == 1 else '✅ NON-FRAUD'}")
                st.write(f"**Probability:** {selected_transaction['fraud_probability']:.4f}")
            with col2:
                if 'merchantId' in selected_original:
                    st.write(f"**Merchant ID:** {selected_original['merchantId']}")
                if 'paymentSourceCode' in selected_original:
                    st.write(f"**Payment Source:** {selected_original['paymentSourceCode']}")
            
            # Show some feature values
            st.markdown("#### Key Features:")
            feature_sample = selected_features[:5]  # Show first 5 features
            for feature in feature_sample:
                if feature in selected_transaction:
                    st.write(f"**{feature}:** {selected_transaction[feature]}")
            
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
                        - Numbers show the actual impact values
                        """)
                        
                        # Additional explanation text
                        prediction_text = "FRAUD" if selected_transaction['predicted_fraud'] == 1 else "NON-FRAUD"
                        confidence = selected_transaction['fraud_probability']
                        
                        st.markdown(f"""
                        **Summary:**
                        - **Final Prediction:** {prediction_text}
                        - **Confidence Score:** {confidence:.4f}
                        - **Interpretation:** The model predicted this transaction as {prediction_text} 
                          with {confidence*100:.2f}% confidence based on the feature contributions shown above.
                        """)
                    else:
                        st.error("❌ Failed to generate explanation. Please try another transaction.")
        else:
            st.info("No transactions available for explanation")

# Main navigation logic
def main():
    """Main function untuk navigasi antar halaman"""
    
    # Sidebar untuk debug info (optional)
    with st.sidebar:
        st.markdown("### Debug Info")
        st.write(f"Current page: {st.session_state.current_page}")
        st.write(f"Data uploaded: {st.session_state.uploaded_data is not None}")
        st.write(f"Preprocessed data: {st.session_state.preprocessed_data is not None}")
        st.write(f"Selected features data: {st.session_state.selected_features_data is not None}")
        
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    if st.session_state.current_page == 'upload':
        page_upload()
    elif st.session_state.current_page == 'analysis':
        page_analysis()

if __name__ == "__main__":
    main()
