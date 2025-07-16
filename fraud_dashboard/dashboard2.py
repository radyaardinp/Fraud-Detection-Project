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
            initial_sidebar_state="expanded"
        )
        
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
            st.error(f"❌ Data tidak memiliki kolom-kolom berikut yang dibutuhkan untuk prediksi: {missing}")
            return False
        return True
    
    def perform_prediction(self, df, model, scaler, selected_features):
        """Melakukan prediksi fraud"""
        try:
            X_input = df.loc[:, selected_features]
            X_scaled = scaler.transform(X_input.values)
            y_pred = model.predict(X_scaled)
            df['predicted_fraud'] = y_pred
            return df, X_scaled
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None, None
    
    def create_lime_explanation(self, X_scaled, selected_features, model, idx_to_explain):
        """Membuat penjelasan LIME untuk transaksi tertentu"""
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
            return exp
        except Exception as e:
            st.error(f"Error creating LIME explanation: {str(e)}")
            return None
    
    def display_metrics(self, df):
        """Menampilkan metrics ringkasan"""
        total_transaksi = len(df)
        total_fraud = df['predicted_fraud'].sum()
        total_nonfraud = total_transaksi - total_fraud
        fraud_rate = round((total_fraud / total_transaksi) * 100, 2)
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Transaksi", f"{total_transaksi:,}")
        k2.metric("Jumlah Fraud", f"{total_fraud:,}")
        k3.metric("Jumlah Non-Fraud", f"{total_nonfraud:,}")
        k4.metric("Fraud Rate", f"{fraud_rate}%")
        
        return total_fraud
    
    def create_fraud_distribution_chart(self, df):
        """Membuat pie chart distribusi fraud"""
        fraud_counts = df['predicted_fraud'].value_counts().rename({0: 'Non-Fraud', 1: 'Fraud'})
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=["#00cc96", "#ff6361"])
        ax1.set_title("Distribusi Fraud vs Non-Fraud")
        ax1.axis('equal')
        st.pyplot(fig1)
        plt.close()
    
    def create_hourly_distribution_chart(self, df):
        """Membuat bar chart distribusi fraud per jam"""
        if 'trx_hour' in df.columns:
            st.subheader("📈 Distribusi Fraud per Jam")
            trx_hour_df = df.groupby(['trx_hour', 'predicted_fraud']).size().unstack().fillna(0)
            trx_hour_df.columns = ['Non-Fraud', 'Fraud']
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            trx_hour_df.plot(kind='bar', stacked=True, ax=ax2, color=["#00cc96", "#ff6361"])
            ax2.set_xlabel("Jam Transaksi")
            ax2.set_ylabel("Jumlah Transaksi")
            ax2.set_title("Distribusi Transaksi per Jam")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
            plt.close()
    
    def create_merchant_fraud_chart(self, df):
        """Membuat bar chart top merchant dengan fraud tertinggi"""
        if 'merchantId' in df.columns:
            st.subheader("🏪 Top 10 Merchant dengan Fraud Tertinggi")
            top_fraud_merchant = df[df['predicted_fraud'] == 1].groupby('merchantId').size().sort_values(ascending=False).head(10)
            
            if len(top_fraud_merchant) > 0:
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                top_fraud_merchant.plot(kind='bar', ax=ax3, color="#ff6361")
                ax3.set_ylabel("Jumlah Fraud")
                ax3.set_title("Top 10 Merchant dengan Fraud Tertinggi")
                plt.xticks(rotation=45)
                st.pyplot(fig3)
                plt.close()
            else:
                st.info("Tidak ada transaksi fraud yang terdeteksi pada merchant manapun.")
    
    def display_fraud_details(self, df):
        """Menampilkan detail transaksi fraud"""
        detected = df[df['predicted_fraud'] == 1]
        if len(detected) > 0:
            st.subheader("🚨 Detail Transaksi Fraud")
            st.dataframe(detected.head(20))
            
            # Download button untuk transaksi fraud
            csv = detected.to_csv(index=False)
            st.download_button(
                label="📥 Download Data Fraud",
                data=csv,
                file_name='fraud_transactions.csv',
                mime='text/csv'
            )
        else:
            st.info("Tidak ada transaksi fraud yang terdeteksi.")
    
    def run_dashboard(self):
        """Fungsi utama untuk menjalankan dashboard"""
        st.title("📊 Fraud Detection Dashboard")
        st.markdown("Unggah data historis transaksi untuk analisis otomatis dan deteksi fraud berdasarkan model ELM yang telah dilatih.")
        
        # Sidebar untuk kontrol
        st.sidebar.header("⚙️ Kontrol Dashboard")
        
        # Upload file CSV
        uploaded_file = st.file_uploader("📁 Upload file CSV transaksi", type=["csv"])
        
        if uploaded_file:
            # Load data
            raw_df = self.load_data(uploaded_file)
            if raw_df is None:
                return
            
            st.success("✅ Data berhasil dimuat!")
            
            # Sidebar info
            st.sidebar.metric("Total Rows", len(raw_df))
            st.sidebar.metric("Total Columns", len(raw_df.columns))
            
            # Layout dua kolom
            col_input, col_output = st.columns([1, 2])
            
            with col_input:
                st.subheader("📥 Input Data")
                
                # Show sample data
                show_rows = st.slider("Tampilkan baris", 5, 50, 10)
                st.dataframe(raw_df.head(show_rows))
                
                # Preprocessing
                with st.spinner("Memproses data..."):
                    df = preprocess_for_prediction(raw_df)
                
                # Load model
                self.model, self.scaler, self.selected_features = self.load_model_components()
                
                if self.model is None:
                    st.error("❌ Gagal memuat model!")
                    return
                
                # Validasi data
                if not self.validate_data(df, self.selected_features):
                    return
                
                # Prediksi
                with st.spinner("Melakukan prediksi..."):
                    df, X_scaled = self.perform_prediction(df, self.model, self.scaler, self.selected_features)
                
                if df is None:
                    return
                
                detected = df[df['predicted_fraud'] == 1]
                st.success(f"✅ Terdeteksi {len(detected)} transaksi yang dicurigai sebagai fraud.")
                
                # LIME Explanation
                if len(df) > 0:
                    st.subheader("🧠 Penjelasan AI (LIME)")
                    idx_to_explain = st.number_input(
                        "Index transaksi untuk dijelaskan", 
                        min_value=0, 
                        max_value=len(df)-1, 
                        step=1
                    )
                    
                    if st.button("Buat Penjelasan"):
                        with st.spinner("Membuat penjelasan..."):
                            exp = self.create_lime_explanation(X_scaled, self.selected_features, self.model, idx_to_explain)
                            if exp:
                                st.write(f"🧠 Penjelasan Transaksi ke-{idx_to_explain}:")
                                st.pyplot(exp.as_pyplot_figure())
                                plt.close()
            
            with col_output:
                st.subheader("📊 Ringkasan & Visualisasi")
                
                # Metrics
                total_fraud = self.display_metrics(df)
                
                # Visualisasi
                if total_fraud > 0:
                    # Pie chart
                    st.subheader("📊 Distribusi Fraud")
                    self.create_fraud_distribution_chart(df)
                    
                    # Hourly distribution
                    self.create_hourly_distribution_chart(df)
                    
                    # Merchant fraud
                    self.create_merchant_fraud_chart(df)
                    
                    # Detail fraud
                    self.display_fraud_details(df)
                else:
                    st.info("🎉 Tidak ada transaksi fraud yang terdeteksi dalam dataset ini!")
        else:
            st.info("⬆️ Silakan upload file .csv terlebih dahulu.")
            
            # Contoh format data yang diperlukan
            st.subheader("📋 Format Data yang Diperlukan")
            st.markdown("""
            File CSV harus memiliki kolom-kolom berikut:
            - `merchantId`: ID merchant
            - `trx_hour`: Jam transaksi (0-23)
            - Dan kolom-kolom lain yang diperlukan untuk prediksi
            
            Pastikan data sudah dalam format yang benar sebelum diupload.
            """)


# Inisialisasi dan jalankan dashboard
if __name__ == "__main__":
    dashboard = FraudDetectionDashboard()
    dashboard.run_dashboard()
