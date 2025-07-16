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


class FraudDetectionDashboard:
    def __init__(self):
        self.setup_page_config()
        self.model = None
        self.scaler = None
        self.selected_features = None
        
    def setup_page_config(self):
        """Setup konfigurasi halaman Streamlit"""
        st.set_page_config(
            page_title="Fraud Detection Dashboard", 
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Custom CSS untuk compact layout
        st.markdown("""
        <style>
        .main > div {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #007bff;
        }
        .metric-container {
            display: flex;
            justify-content: space-around;
            margin-bottom: 1rem;
        }
        .chart-container {
            height: 300px;
            margin-bottom: 1rem;
        }
        .sidebar-content {
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
        }
        .main-content {
            height: 100vh;
            overflow-y: auto;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            padding: 0px 12px;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def load_data(self, file):
        """Load data dari file CSV yang diupload"""
        try:
            return pd.read_csv(file)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
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
            
            # Check extra features in data
            extra_features = [f for f in df.columns if f not in selected_features and f != 'predicted_fraud']
            if extra_features:
                st.warning(f"⚠️ Extra features in data (will be ignored): {extra_features}")
            
            # Select only the required features
            X_input = df.loc[:, selected_features]
            st.write(f"- Input shape: {X_input.shape}")
            st.write(f"- Scaler expects: {scaler.n_features_in_} features")
            
            # Check if shapes match
            if X_input.shape[1] != scaler.n_features_in_:
                st.error(f"❌ Feature count mismatch:")
                st.error(f"   - Data has: {X_input.shape[1]} features")
                st.error(f"   - Scaler expects: {scaler.n_features_in_} features")
                st.error(f"   - Missing: {scaler.n_features_in_ - X_input.shape[1]} features")
                return None, None
            
            # Perform scaling and prediction
            X_scaled = scaler.transform(X_input.values)
            y_pred = model.predict(X_scaled)
            df['predicted_fraud'] = y_pred
            
            st.success(f"✅ Prediction successful! Shape: {X_scaled.shape}")
            return df, X_scaled
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error("Please check your preprocessing pipeline and ensure all required features are generated.")
            return None, None
    
    def create_compact_metrics(self, df):
        """Membuat metrics yang compact"""
        total_transaksi = len(df)
        total_fraud = df['predicted_fraud'].sum()
        total_nonfraud = total_transaksi - total_fraud
        fraud_rate = round((total_fraud / total_transaksi) * 100, 2)
        
        # Metrics dalam format compact
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Transaksi", f"{total_transaksi:,}")
            st.metric("Fraud Rate", f"{fraud_rate}%", delta=f"{fraud_rate-5:.1f}%")
        with col2:
            st.metric("Fraud", f"{total_fraud:,}", delta=f"{total_fraud}")
            st.metric("Non-Fraud", f"{total_nonfraud:,}")
        
        return total_fraud, fraud_rate
    
    def create_fraud_pie_chart(self, df):
        """Membuat pie chart yang compact"""
        fraud_counts = df['predicted_fraud'].value_counts().rename({0: 'Non-Fraud', 1: 'Fraud'})
        
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = ['#28a745', '#dc3545']
        wedges, texts, autotexts = ax.pie(fraud_counts, labels=fraud_counts.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        # Styling
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Distribusi Fraud vs Non-Fraud', fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_hourly_chart(self, df):
        """Membuat chart distribusi per jam yang compact"""
        if 'trx_hour' in df.columns:
            hourly_fraud = df.groupby('trx_hour')['predicted_fraud'].agg(['count', 'sum']).reset_index()
            hourly_fraud['fraud_rate'] = (hourly_fraud['sum'] / hourly_fraud['count'] * 100).round(2)
            
            fig, ax = plt.subplots(figsize=(6, 3))
            
            # Bar chart untuk fraud rate
            bars = ax.bar(hourly_fraud['trx_hour'], hourly_fraud['fraud_rate'], 
                         color='#dc3545', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Jam Transaksi', fontsize=10)
            ax.set_ylabel('Fraud Rate (%)', fontsize=10)
            ax.set_title('Fraud Rate per Jam', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Highlight highest fraud rate
            max_idx = hourly_fraud['fraud_rate'].idxmax()
            bars[max_idx].set_color('#a71d2a')
            
            plt.tight_layout()
            return fig
        return None
    
    def create_merchant_chart(self, df):
        """Membuat chart merchant yang compact"""
        if 'merchantId' in df.columns:
            merchant_fraud = df[df['predicted_fraud'] == 1]['merchantId'].value_counts().head(8)
            
            if len(merchant_fraud) > 0:
                fig, ax = plt.subplots(figsize=(6, 3))
                
                bars = ax.barh(range(len(merchant_fraud)), merchant_fraud.values, 
                              color='#dc3545', alpha=0.7)
                
                ax.set_yticks(range(len(merchant_fraud)))
                ax.set_yticklabels([f"Merchant {mid}" for mid in merchant_fraud.index])
                ax.set_xlabel('Jumlah Fraud', fontsize=10)
                ax.set_title('Top Merchant Fraud', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                return fig
        return None
    
    def create_lime_explanation(self, X_scaled, selected_features, model, idx_to_explain):
        """Membuat penjelasan LIME yang compact"""
        try:
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
            
            # Buat figure yang compact
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(6, 4)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"LIME error: {str(e)}")
            return None
    
    def render_sidebar(self):
        """Render sidebar untuk input data"""
        with st.container():
            st.markdown("### 📁 Input Data & Prediksi")
            
            # File uploader
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="file_upload")
            
            if uploaded_file:
                # Load data
                raw_df = self.load_data(uploaded_file)
                if raw_df is None:
                    return None, None, None
                
                st.success(f"✅ Data loaded: {len(raw_df)} rows")
                
                # Data preview
                with st.expander("👀 Data Preview", expanded=False):
                    st.dataframe(raw_df.head(), use_container_width=True)
                
                # Preprocessing
                with st.spinner("Processing..."):
                    df = preprocess_for_prediction(raw_df)
                    
                    # Show preprocessing results
                    st.write("**📊 Preprocessing Results:**")
                    st.write(f"- Original shape: {raw_df.shape}")
                    st.write(f"- Processed shape: {df.shape}")
                    st.write(f"- Generated features: {list(df.columns)}")
                
                # Load model
                self.model, self.scaler, self.selected_features = self.load_model_components()
                
                if self.model is None:
                    return None, None, None
                
                # Show model info
                st.write("**🤖 Model Info:**")
                st.write(f"- Expected features: {len(self.selected_features)}")
                st.write(f"- Feature names: {self.selected_features}")
                
                # Validasi dan prediksi
                if not self.validate_data(df, self.selected_features):
                    return None, None, None
                
                df, X_scaled = self.perform_prediction(df, self.model, self.scaler, self.selected_features)
                
                if df is None:
                    return None, None, None
                
                # Status prediksi
                total_fraud = df['predicted_fraud'].sum()
                st.info(f"🔍 Detected {total_fraud} fraud transactions")
                
                # LIME explanation controls
                st.markdown("### 🧠 AI Explanation")
                if len(df) > 0:
                    idx_to_explain = st.selectbox(
                        "Select transaction:", 
                        range(min(20, len(df))), 
                        format_func=lambda x: f"Transaction {x}"
                    )
                    
                    if st.button("🔍 Explain", use_container_width=True):
                        return df, X_scaled, idx_to_explain
                
                return df, X_scaled, None
            
            else:
                st.info("⬆️ Please upload a CSV file")
                st.markdown("""
                **Required columns:**
                - merchantId
                - trx_hour
                - Other features for prediction
                """)
                return None, None, None
    
    def render_main_content(self, df, X_scaled, idx_to_explain):
        """Render konten utama dengan hasil analisis"""
        if df is None:
            st.markdown("### 📊 Fraud Detection Results")
            st.info("Upload data to see analysis results")
            return
        
        # Header
        st.markdown("### 📊 Fraud Detection Analysis")
        
        # Metrics
        total_fraud, fraud_rate = self.create_compact_metrics(df)
        
        # Tabs untuk organisasi konten
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Details"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig_pie = self.create_fraud_pie_chart(df)
                st.pyplot(fig_pie, use_container_width=True)
                plt.close(fig_pie)
            
            with col2:
                # Merchant chart
                fig_merchant = self.create_merchant_chart(df)
                if fig_merchant:
                    st.pyplot(fig_merchant, use_container_width=True)
                    plt.close(fig_merchant)
                else:
                    st.info("No merchant data available")
        
        with tab2:
            # Hourly trend
            fig_hourly = self.create_hourly_chart(df)
            if fig_hourly:
                st.pyplot(fig_hourly, use_container_width=True)
                plt.close(fig_hourly)
            else:
                st.info("No hourly data available")
        
        with tab3:
            # Fraud details
            detected = df[df['predicted_fraud'] == 1]
            if len(detected) > 0:
                st.markdown("**🚨 Fraud Transactions**")
                st.dataframe(detected.head(10), use_container_width=True)
                
                # Download button
                csv = detected.to_csv(index=False)
                st.download_button(
                    label="📥 Download Fraud Data",
                    data=csv,
                    file_name='fraud_transactions.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            else:
                st.success("🎉 No fraud transactions detected!")
            
            # LIME explanation
            if idx_to_explain is not None:
                st.markdown("**🧠 AI Explanation**")
                with st.spinner("Generating explanation..."):
                    lime_fig = self.create_lime_explanation(
                        X_scaled, self.selected_features, self.model, idx_to_explain
                    )
                    if lime_fig:
                        st.pyplot(lime_fig, use_container_width=True)
                        plt.close(lime_fig)
    
    def run_dashboard(self):
        """Fungsi utama untuk menjalankan dashboard"""
        # Header
        st.markdown("# 🛡️ Fraud Detection Dashboard")
        st.markdown("Deteksi fraud otomatis menggunakan model ELM")
        
        # Layout utama: sidebar (1/3) dan main content (2/3)
        col_sidebar, col_main = st.columns([1, 2])
        
        with col_sidebar:
            df, X_scaled, idx_to_explain = self.render_sidebar()
        
        with col_main:
            self.render_main_content(df, X_scaled, idx_to_explain)


# Run dashboard
if __name__ == "__main__":
    dashboard = FraudDetectionDashboard()
    dashboard.run_dashboard()
