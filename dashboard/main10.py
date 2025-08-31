import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import lime
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Fraud Detection System Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
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
    .success-metric {
        border-left-color: #10b981;
    }
    .warning-metric {
        border-left-color: #f59e0b;
    }  
    .error-metric {
        border-left-color: #ef4444;
    }
        .footer-section {
        padding: 1rem;
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
    st.session_state.data = None
    st.session_state.processed_data = None
    st.session_state.missing_handled = False
    st.session_state.model_trained = False
    st.session_state.training_results = {}
    st.session_state.selected_resampling = 'none'
    st.session_state.feature_importance = None

# Main header
st.markdown('<div class="main-header">🛡️ Fraud Detection System Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced AI-Powered Transaction Analysis</div>', unsafe_allow_html=True)

if st.session_state.current_step == 1:
    # Description
    st.markdown("""
    <div class="description-text">
    Dashboard ini menggunakan Algoritma Machine Learning <span class="highlight-text">Extreme Learning Machine (ELM)</span> 
    yang telah terintegrasi dengan <span class="highlight-text">LIME (Local Interpretable Model-agnostic Explanations)</span> 
    untuk mendeteksi fraud dengan akurasi tinggi dan memberikan penjelasan yang dapat dipahami.
    </div>
    """, unsafe_allow_html=True)

#navigation button
st.markdown("<br><br>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📤 Upload Data", key="nav1", use_container_width=True):
        st.session_state.current_step = 1
        st.rerun()
with col2:
    if st.button("🔧 Preprocessing", key="nav2", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()
with col3:
    if st.button("📊 Analisis Data", key="nav3", use_container_width=True):
        st.session_state.current_step = 3
        st.rerun()
with col4:
    if st.button("🔍 Interpretasi LIME", key="nav4", use_container_width=True):
        st.session_state.current_step = 4
        st.rerun()

# ====== PARAMETER INISIALISASI KATEGORIK & NUMERIK ======
CAT_COLS = [
    'merchantId', 'paymentSource', 'status', 'statusCode',
    'currency', 'type', 'Type Token'
]

NUM_COLS = [
    'amount', 'settlementAmount', 'feeAmount', 'discountAmount',
    'inquiryAmount', 'discount_ratio', 'fee_ratio', 'selisih_waktu_sec'
]

# ======= Fungsi =======
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

#mengubah tipe data
def convert_data_types(df):
    df = df.copy()
    # Convert to datetime
    if 'updatedTime' in df.columns:
        df['updatedTime'] = pd.to_datetime(df['updatedTime'], dayfirst=True, format='mixed', errors='coerce')
    if 'createdTime' in df.columns:
        df['createdTime'] = pd.to_datetime(df['createdTime'], errors='coerce')

    # Convert to float
    float_cols = ['amount', 'settlementAmount', 'feeAmount', 'discountAmount', 'inquiryAmount']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert to categorical
    categorical_cols = ['merchantId','paymentSource', 'status','statusCode']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

#mengatasi missing value    
def handle_missing_values(df):
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype.name in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
            elif df[col].dtype.name in ['object', 'category']:
                df[col] = df[col].fillna('nan')

    return df

#Deteksi outlier
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers, lower_bound, upper_bound
    
#Handling Outlier   
def handle_outliers_iqr(df, columns):
    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound,
                           np.where(df[col] < lower_bound, lower_bound, df[col]))
    return df

#Membuat fitur baru
def feature_eng(df):
    df = df.copy()
    epsilon = 1e-6

    if 'amount' in df.columns and df['amount'].notna().any():
        df['discount_ratio'] = df['discountAmount'] / (df['amount'] + epsilon)
        df['fee_ratio'] = df['feeAmount'] / (df['amount'] + epsilon)
    else:
        df['discount_ratio'] = 0
        df['fee_ratio'] = 0

    df['fee_ratio'] = df['feeAmount'] / (df['amount'] + epsilon)
    df['hour_of_day'] = df['createdTime'].dt.hour
    mask = df['updatedTime'].notna() & df['createdTime'].notna()
    df['selisih_waktu_sec'] = np.where(
        mask, 
        (df['updatedTime'] - df['createdTime']).dt.total_seconds(),
        np.nan
    )

    if 'fraud' in df.columns:
        df['fraud'] = df['fraud'].apply(lambda x: 1 if x == 'Fraud' else 0)

    if 'merchantId' in df.columns and 'fraud' in df.columns:
        merchant_stats = df.groupby('merchantId')['fraud'].agg(['sum', 'count'])
        merchant_stats['fraud_rate'] = merchant_stats['sum'] / merchant_stats['count']
        df = df.merge(merchant_stats[['fraud_rate']], on='merchantId', how='left')

    return df

# Fungsi untuk Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

#Fraud Rule-Based Labeling
class FraudDetectionLabeler:
    def __init__(self):
        self.config = {
            'outlier_threshold': 0.95,
            'fraud_rules': {
                'failed_multiplier': 2.0,
                'fail_ratio_high': 0.7,
                'fail_ratio_medium': 0.4,
                'fail_interval_threshold': 60,  # seconds
                'mismatch_ratio_threshold': 1.0
            }
        }

    def ensure_columns(self, df):
        defaults = {
            'amount': 0, 'inquiryAmount': 0, 'settlementAmount': 0,
            'merchantId': 'merch1', 'createdTime': pd.Timestamp.now(),
            'status': 'captured'
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
        return df

    def add_metrics(self, df):
        df = df.copy()
        df['createdTime'] = pd.to_datetime(df['createdTime'], errors='coerce')
        df['createdDate'] = df['createdTime'].dt.date
        df['is_declined'] = df['status'].str.lower() == 'declined'

        # Daily frequency & failure count
        daily = df.groupby(['merchantId', 'createdDate'])
        df = df.merge(daily.size().reset_index(name='daily_freq'), on=['merchantId', 'createdDate'])
        df = df.merge(daily['is_declined'].sum().reset_index(name='failed_count'), on=['merchantId', 'createdDate'])

        # Avg failed per merchant & ratio
        avg_failed = df.groupby('merchantId')['failed_count'].transform('mean').fillna(0)
        df['avg_failed'] = avg_failed
        df['fail_ratio'] = df['failed_count'] / df['daily_freq'].clip(lower=1)
        return df

    def add_fail_intervals(self, df):
        failed = df[df['is_declined']].copy()
        if not failed.empty:
            failed.sort_values(by=['merchantId', 'createdTime'], inplace=True)
            failed['prev_time'] = failed.groupby('merchantId')['createdTime'].shift(1)
            failed['avg_fail_interval'] = (failed['createdTime'] - failed['prev_time']).dt.total_seconds()
            failed['createdDate'] = failed['createdTime'].dt.date

            interval = failed.groupby(['merchantId', 'createdDate'])['avg_fail_interval'].mean().reset_index()
            interval['avg_fail_interval'] = interval['avg_fail_interval'].fillna(0)
            df = df.merge(interval, on=['merchantId', 'createdDate'], how='left')
        df['avg_fail_interval'] = df['avg_fail_interval'].fillna(0)
        return df

    def calculate_thresholds(self, df):
        q = self.config['outlier_threshold']
        return {
            'daily_freq': df['daily_freq'].quantile(q),
            'amount': df['amount'].quantile(q),
            'failed_count': df['failed_count'].quantile(q),
            'mismatch': df['mismatch'].quantile(q)
        }

    def rule_1(self, row, t):
        if row['daily_freq'] > t['daily_freq']:
            if row['amount'] > t['amount']:
                if row['failed_count'] > t['failed_count']:
                    return 'Fraud'
                elif row['mismatch'] > t['mismatch']:
                    return 'Fraud' if not (row['mismatch_ratio'] < 0.01 and row['failed_count'] == 0) else 'Not Fraud'
            elif row['failed_count'] > t['failed_count'] and row['mismatch'] > t['mismatch']:
                return 'Fraud'
        elif row['amount'] > t['amount']:
            if row['failed_count'] > t['failed_count'] or row['mismatch'] > t['mismatch']:
                return 'Fraud'
        return 'Not Fraud'

    def rule_2(self, row):
        c = self.config['fraud_rules']
        if row['failed_count'] > c['failed_multiplier'] * row['avg_failed']:
            if row['fail_ratio'] > c['fail_ratio_high'] or row['avg_fail_interval'] < c['fail_interval_threshold']:
                return 'Fraud'
        elif row['fail_ratio'] > c['fail_ratio_medium']:
            return 'Fraud'
        return 'Not Fraud'

    def rule_3(self, row):
        return 'Fraud' if row['mismatch_ratio'] > self.config['fraud_rules']['mismatch_ratio_threshold'] else 'Not Fraud'

    def apply_rule_based_labeling(self, df):
        df = self.ensure_columns(df)

            # ⛔ Bersihkan sisa kolom labeling sebelumnya, kalau ada
        drop_cols = [
            'createdDate', 'is_declined', 'daily_freq', 'failed_count', 'avg_failed', 'fail_ratio', 
            'avg_fail_interval', 'mismatch', 'mismatch_ratio', 'is_nominal_tinggi', 'label1', 'label2', 'label3'
    ]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        df = self.add_metrics(df)
        df = self.add_fail_intervals(df)

        df['mismatch'] = abs(df['inquiryAmount'] - df['settlementAmount'])
        df['mismatch_ratio'] = np.where(df['inquiryAmount'] == 0, 0, df['mismatch'] / df['inquiryAmount'])
        df['is_nominal_tinggi'] = df['amount'] > 8_000_000

        thresholds = self.calculate_thresholds(df)

        df['label1'] = df.apply(lambda r: self.rule_1(r, thresholds), axis=1)
        df['label2'] = df.apply(self.rule_2, axis=1)
        df['label3'] = df.apply(self.rule_3, axis=1)
        df['fraud'] = df[['label1', 'label2', 'label3']].apply(lambda r: 'Fraud' if 'Fraud' in r.values else 'Not Fraud', axis=1)

        # Hapus fitur tambahan
        original_cols = [col for col in df.columns if col not in drop_cols]
        df = df[original_cols + ['fraud']]
        df = df.loc[:, ~df.columns.duplicated()]

        return df
        
#Membuat plot visualisasi confussion matrix
def confusion_matrix_plot(cm):
    fig = px.imshow(cm, 
                    text_auto=True, 
                    aspect="auto",
                    color_continuous_scale='RdYlBu_r',
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Normal', 'Fraud'],
                    y=['Normal', 'Fraud'])
    fig.update_layout(title="Confusion Matrix", width=400, height=400)
    return fig

def encode_full_dataset(df):
    """Encode semua kolom kategorikal & konversi datetime ke epoch detik."""
    df_encoded = df.copy()
    label_encoders = {}

    # Encode categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].astype(str)
        le.fit(df_encoded[col])
        df_encoded[col] = le.transform(df_encoded[col])
        label_encoders[col] = le

    # Convert datetime to epoch
    for col in df_encoded.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns:
        df_encoded[col] = df_encoded[col].astype(np.int64) // 10**9

    # Ensure all numeric
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_encoded, label_encoders

# Fungsi Aktivasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Fungsi Training ELM
def train_elm(X_train, y_train, hidden_neurons, activation=sigmoid, seed=42):
    np.random.seed(seed)
    input_dim = X_train.shape[1]
    W = np.random.randn(input_dim, hidden_neurons)
    b = np.random.randn(hidden_neurons)
    H = activation(np.dot(X_train, W) + b)
    H_pinv = np.linalg.pinv(H)
    beta = np.dot(H_pinv, y_train)
    return W, b, beta

# Fungsi Prediksi ELM 
def predict_elm(X_test, W, b, beta, activation=sigmoid):
    H_test = activation(np.dot(X_test, W) + b)
    y_pred_raw = np.dot(H_test, beta)
    y_pred = (y_pred_raw >= 0.5).astype(int)
    return y_pred
    

# ========== Halaman UI Dashboard ==========
if st.session_state.current_step == 1:
    # Step 1: Upload Data
    st.header("📤 Upload Data Transaksi")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=['csv'],
        help="Format: CSV (Max: 10MB)"
    )
    
    if uploaded_file is not None:
        df= pd.read_csv(uploaded_file)
        df=convert_data_types(df)
        st.session_state.data = df
        st.success("✅ File berhasil diupload!")
        
        # Dataset preview
        st.markdown("### Preview Dataset")
                    
        # File details
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Jumlah Baris:** {len(st.session_state.data):,}")
            st.info(f"**Jumlah Kolom:** {st.session_state.data.shape[1]}")
        with col2:
           st.info(f"**Ukuran Berkas:** {uploaded_file.size / (1024*1024):.2f} MB")
           st.info(f"**Tipe Berkas:** {uploaded_file.type}")

        # Data preview table
        st.dataframe(st.session_state.data.head(15), use_container_width=True)
        
        # Column information
        st.markdown("### 📊 Informasi Kolom")
        col_info = pd.DataFrame({
            'Kolom': st.session_state.data.columns,
            'Tipe Data': st.session_state.data.dtypes})
        
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
            'inquiryAmount': 'Jumlah nominal yang direquest pada tahap inquiry',
            'discountAmount': 'Jumlah nominal diskon',
            'feeAmount': 'Biaya Transaksi',
            'typeToken': 'Jenis tokenisasi'}

        # Fungsi untuk mendapatkan keterangan
        def get_description(col_name):
            return manual_descriptions.get(col_name, 'Tidak ada deskripsi')

        # Menambahkan kolom keterangan
        col_info['Keterangan'] = col_info['Kolom'].apply(get_description)

        st.dataframe(col_info, use_container_width=True)
        
        # tombol selanjutnya
        col1, col2, col3 = st.columns([6, 2, 2])  
        with col3:
            if st.button("➡️ Lanjut ke Preprocessing", type="primary"):
                st.session_state.current_step = 2
                st.rerun()

elif st.session_state.current_step == 2:
    # Step 2: Preprocessing
    st.header("🔧 Preprocessing Data")
    
    if st.session_state.data is None:
        st.warning("⚠️ Silakan upload data terlebih dahulu!")
        if st.button("⬅️ Kembali ke Upload"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        # 🔍 Identifikasi Missing Values
        st.subheader("🔍 Identifikasi Missing Values")

        # Hitung missing sebelum penanganan
        missing_before = st.session_state.data.isnull().sum()
        missing_df = missing_before[missing_before > 0].to_frame(name="Jumlah Missing Value")
        
        if not missing_df.empty and not st.session_state.get("missing_handled", False):
            st.warning("⚠️ Terdapat missing values pada dataset:")
            st.dataframe(missing_df, use_container_width=True)
        
            if st.button("🔧 Terapkan Penanganan Missing Values"):
                st.session_state.data = handle_missing_values(st.session_state.data)

                # Simpan raw_data setelah misval ditangani
                st.session_state.raw_data = copy.deepcopy(st.session_state.data)
        
                # Hitung missing setelah penanganan
                missing_after = st.session_state.data.isnull().sum()
                after_df = missing_after.to_frame(name="Jumlah Missing Value Setelah Penanganan")
        
                # Gabungkan before-after
                compare_df = missing_df.join(after_df)
                st.session_state.missing_comparison = compare_df
                st.session_state.missing_handled = True
                st.rerun()
        
        elif st.session_state.get("missing_handled", False):
            st.success("✅ Missing values telah berhasil ditangani.")
            st.markdown("### 📊 Perbandingan Jumlah Missing Value Sebelum & Sesudah Penanganan")
            st.dataframe(st.session_state.missing_comparison, use_container_width=True)
        
        else:
            st.success("✅ Tidak ada missing values dalam dataset!")
        
        st.markdown("---")
        
        # Rule-Based Labelling Section
        st.subheader("🏷️ Rule-Based Labelling")

        # Aturan Fraud Detection Labeling
        st.info("""
        **3 Aturan Deteksi Fraud:**
        1. **Rule 1**: Parameter untuk memantau transaksi berdasarkan jumlah frekuensi, nilai nominal, rasio transaksi gagal, dan perbedaan nilai antara tahapan otorisasi dan penyelesaian (Settlement) 
        2. **Rule 2**: Parameter yang difokuskan untuk mengindentifikasi adanya transaksi yang gagal  
        3. **Rule 3**: Parameter untuk mendeteksi transaksi yang mengalami peningkatan lebih dari 100%  
        
        **Threshold**: 95th percentile untuk outlier detection
        """)

        # Dataframe sebelum labelling
        st.markdown("### 📋 Data Sebelum Diberikan Label")
        if 'raw_data' in st.session_state:
            st.dataframe(st.session_state.raw_data.head(10), use_container_width=True)
        else:
            st.info("Data belum siap ditampilkan. Silakan tangani missing value terlebih dahulu.")

        # Button
        if 'fraud' not in st.session_state.data.columns:
            if st.button("🚀 Terapkan Rule-Based Labelling", key="apply_rules"):
                with st.spinner("Menerapkan rule-based labeling..."):
                    labeler = FraudDetectionLabeler()
                    st.session_state.data = labeler.apply_rule_based_labeling(st.session_state.data)
                    st.session_state.processed_data = st.session_state.data.copy()
                st.success("✅ Labelling berhasil diterapkan!")
                st.rerun()

        # 4. Dataframe setelah labelling
        if 'fraud' in st.session_state.data.columns:
            st.markdown("### ✅ Data Setelah Diberi Label")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)

        st.markdown("---")
        
        # Visualisasi Section
        st.subheader("📊 Visualisasi Data Fraud Detection")

        col1, col2 = st.columns(2)
        # 1. Distribusi Payment Source
        with col1:
            if 'paymentSource' in st.session_state.data.columns:
                payment_dist = st.session_state.data['paymentSource'].value_counts()
                fig = px.bar(
                    x=payment_dist.index, y=payment_dist.values,
                    labels={'x': 'Payment Source', 'y': 'Jumlah Transaksi'},
                    title="Distribusi Payment Source")
                st.plotly_chart(fig, use_container_width=True)

            merchant_col = 'merchantId'
            if 'fraud' in st.session_state.data.columns and merchant_col in st.session_state.data.columns:
                fraud_per_merchant = st.session_state.data[st.session_state.data['fraud'] == 'Fraud']
                top_merchant_fraud = fraud_per_merchant[merchant_col].value_counts().head(10)
                fig = px.bar(
                    x=top_merchant_fraud.values, y=top_merchant_fraud.index,
                    orientation='h',
                    labels={'x': 'Jumlah Fraud', 'y': 'Merchant'},
                    title="Top 10 Merchant dengan Fraud Terbanyak")
                st.plotly_chart(fig, use_container_width=True)

            df = feature_eng(st.session_state.data.copy())
            # Pastikan fraud berbentuk numerik (0/1)
            if df['fraud'].dtype == 'object':
                le = LabelEncoder()
                df['fraud'] = le.fit_transform(df['fraud'].astype(str))
        
            # --- 1. Korelasi Numerik vs Fraud (Pearson)
            num_cols = ['amount', 'settlementAmount', 'feeAmount', 'inquiryAmount', 'fee_ratio', 'selisih_waktu_sec', 'fraud_rate']
            available_num_cols = [c for c in num_cols if c in df.columns]
        
            if available_num_cols:
                corr_matrix = df[available_num_cols + ['fraud']].corr()
        
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title('Matriks Korelasi Fitur Numerik dan Fraud')
                st.pyplot(fig)
        
        # 2. Distribusi Status Transaksi
        with col2:
            if 'status' in st.session_state.data.columns:
                status_dist = st.session_state.data['status'].value_counts()
                fig = px.bar(
                    x=status_dist.index, y=status_dist.values,
                    labels={'x': 'Status Transaksi', 'y': 'Jumlah Transaksi'},
                    title="Distribusi Status Transaksi")
                st.plotly_chart(fig, use_container_width=True)
        
            fraud_label_col = 'fraud'
            if fraud_label_col in st.session_state.data.columns:
                fraud_dist = st.session_state.data[fraud_label_col].value_counts()
                fig = px.pie(
                    values=fraud_dist.values, names=fraud_dist.index,
                    title="Distribusi Fraud vs Not Fraud")
                st.plotly_chart(fig, use_container_width=True)

            # --- 2. Korelasi Kategorikal vs Fraud (Cramér’s V)
            categorical_features = ['paymentSource', 'status', 'statusCode', 'currency', 'type', 'Type Token']
            available_cat_cols = [c for c in categorical_features if c in df.columns]
        
            cramers_results = {}
            for col in available_cat_cols:
                try:
                    cramers_results[col] = cramers_v(df[col].astype(str).fillna("Missing"), df['fraud'])
                except Exception as e:
                    cramers_results[col] = np.nan
        
            if cramers_results:
                cramers_series = pd.Series(cramers_results).sort_values()
        
                # Bar chart interaktif pakai Plotly
                fig = px.bar(
                    cramers_series,
                    x=cramers_series.values,
                    y=cramers_series.index,
                    orientation='h',
                    text=cramers_series.round(3),
                    labels={'x': "Cramér's V", 'y': "Fitur Kategorikal"},
                    title="Korelasi Fitur Kategorikal dengan Fraud"
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ====== FEATURE SELECTION ======
        st.subheader("🧩 Feature Selection")
        
        # Fitur yang dipilih untuk pemodelan
        selected_features = ['amount', 'inquiryAmount', 'feeAmount', 'paymentSource', 'fraud_rate']
        
        # Simpan hasil selection
        df_selected_raw = df[selected_features + ['fraud']]
        df_selected, label_encoders = encode_full_dataset(df_selected_raw)
        
        #Menyimpan hasil ke session state
        st.session_state["df_selected"] = df_selected 
        st.session_state["selected_features_list"] = selected_features
        
        # Notes untuk menjelaskan
        st.info("""
        Pada tahap ini dilakukan **feature selection** untuk menentukan fitur yang akan digunakan dalam proses pemodelan.
        Fitur dipilih berdasarkan relevansi terhadap deteksi fraud, meliputi:
        - `amount`
        - `inquiryAmount`
        - `feeAmount`
        - `paymentSource`
        - `fraud_rate`
        
        Hanya fitur-fitur tersebut yang digunakan bersama dengan label `fraud` sebagai target.
        """)
        
        # Tampilkan dataframe hasil feature selection
        st.dataframe(df_selected.head())

        st.markdown("---")
        
        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Kembali"):
                st.session_state.current_step = 1
                st.rerun()
        with col3:
            if st.button("➡️ Lanjut ke Analisis", type="primary"):
                st.session_state.current_step = 3
                st.rerun()
                
elif st.session_state.current_step == 3:
    # Step 3: Analysis
    st.header("📊 Analisis Data")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Silakan selesaikan preprocessing terlebih dahulu!")
        if st.button("⬅️ Kembali ke Preprocessing"):
            st.session_state.current_step = 2
            st.rerun()
    else:
        # ======== Data Splitting =======
        st.subheader("✂️ Pembagian Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            train_size = st.slider("Training Size (%):", 60, 90, 70)
            test_size = 100 - train_size
        
        with col2:
            if st.button("🔄 Split Dataset") or 'X_train' not in st.session_state:
                selected_features = st.session_state.get('selected_features_list', [])
                if not selected_features:
                    st.error("❌ Belum ada fitur yang dipilih dari tahap Feature Selection!")
                    st.stop()

                 # Ambil dataframe hasil feature selection yang sudah encoded
                if "df_selected" not in st.session_state:
                    st.error("❌ Data hasil feature selection belum tersedia. Jalankan step Preprocessing & Feature Selection dulu!")
                    st.stop()
            
                df_selected = st.session_state["df_selected"]

                available_features = [f for f in selected_features if f in df_selected.columns]
                if not available_features:
                    st.error("❌ Tidak ada fitur valid!")
                    st.stop()
                
                X = df_selected.drop(columns=["fraud"])
                y = df_selected["fraud"]
                test_ratio = test_size / 100
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_ratio,
                    stratify=y,
                    random_state=42)

                # Simpan ke nromsession_state
                st.session_state.X_train = X_train.copy()
                st.session_state.X_test = X_test.copy()
                st.session_state.y_train = y_train.copy()
                st.session_state.y_test = y_test.copy()
                st.session_state.selected_features_used = available_features
                st.session_state.data_split = True
                st.session_state.outlier_handled = False
                st.session_state.data_normalized = False
        
        # Show dataset info after splitting
        st.session_state.get('data_split', False)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training", len(st.session_state.X_train))
        with col2:
            st.metric("Testing", len(st.session_state.X_test))
        with col3:
            st.metric("Total Features", st.session_state.X_train.shape[1])
        
        # === OUTLIER HANDLING ===
        st.subheader("📉 Penanganan Outlier pada Data Training")
        numeric_cols = st.session_state.X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # --- Tombol cek outlier ---
        if st.button("🔍 Periksa Outlier"):
            outlier_info = {}
            for col in numeric_cols:
                Q1, Q3 = st.session_state.X_train[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                outliers = st.session_state.X_train[
                    (st.session_state.X_train[col] < lower) |
                    (st.session_state.X_train[col] > upper)
                ]
                outlier_info[col] = len(outliers)
        
            # Simpan hasil perhitungan ke session_state
            st.session_state.outlier_info = outlier_info
            st.session_state.outlier_checked = True
        
        # --- Selalu tampilkan kalau sudah dicek ---
        if st.session_state.get("outlier_checked", False):
            st.write("**Jumlah Outlier per Kolom:**")
            st.dataframe(pd.DataFrame(
                list(st.session_state.outlier_info.items()), 
                columns=["Kolom", "Jumlah Outlier"]
            ))
        
            st.write("**Boxplot Sebelum Handling:**")
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 5), squeeze=False)
            for i, col in enumerate(numeric_cols):
                sns.boxplot(x=st.session_state.X_train[col], ax=axes[0][i])
                axes[0][i].set_title(f"{col} (Before)")
            plt.tight_layout()
            st.pyplot(fig)
        
        
        # --- Tombol handle outlier ---
        if st.button("🚨 Terapkan Penanganan Outlier"):
            X_train_processed = st.session_state.X_train.copy()
            for col in numeric_cols:
                Q1, Q3 = X_train_processed[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                X_train_processed[col] = X_train_processed[col].clip(lower, upper)
        
            # simpan hasil
            st.session_state.X_train = X_train_processed
            st.session_state.X_train_after_outlier = X_train_processed.copy()  # 👈 backup untuk plot
            st.session_state.outlier_handled = True
            st.success("✅ Outlier berhasil ditangani!")
        
        
        # --- Selalu tampilkan boxplot after kalau sudah handle ---
        if st.session_state.get("outlier_handled", False):
            st.write("**Boxplot Setelah Outlier Handling:**")
            X_train_for_plot = st.session_state.X_train_after_outlier  # 👈 pakai backup, biar ga ketimpa normalisasi
            numeric_cols = X_train_for_plot.select_dtypes(include=[np.number]).columns.tolist()
        
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 5), squeeze=False)
            for i, col in enumerate(numeric_cols):
                sns.boxplot(x=X_train_for_plot[col], ax=axes[0][i])
                axes[0][i].set_title(f"{col} (After)")
            plt.tight_layout()
            st.pyplot(fig)

                
            # ====== STANDARISASI ======
            st.subheader("📐 Standarisasi Data (MinMax Scaler)")
            numeric_cols_for_scaling = [
                col for col in st.session_state.selected_features_used
                if col in st.session_state.X_train.columns                    
                and pd.api.types.is_numeric_dtype(st.session_state.X_train[col])
            ]
            # Pilih kolom kategorikal yang sudah di-encode
            categorical_encoded_cols = [
                col for col in st.session_state.selected_features_used
                if col in st.session_state.X_train.columns 
                and not pd.api.types.is_numeric_dtype(st.session_state.X_train[col])
            ]
            # Gabungkan untuk preview
            cols_to_preview = numeric_cols_for_scaling + categorical_encoded_cols
            # Preview sebelum scaling
            st.write("**Data Sebelum Normalisasi (Training):**")
            st.dataframe(st.session_state.X_train[cols_to_preview].head())


            if st.button("⚡ Terapkan Normalisasi"):
                try:
                    # Backup sebelum scaling
                    st.session_state.X_train_before_norm = st.session_state.X_train.copy()
                    st.session_state.X_test_before_norm = st.session_state.X_test.copy()
            
                    # Init scaler
                    scaler = MinMaxScaler()
            
                    # Apply scaler hanya ke kolom numerik terpilih
                    X_train_scaled = st.session_state.X_train.copy()
                    X_test_scaled = st.session_state.X_test.copy()
            
                    X_train_scaled[numeric_cols_for_scaling] = scaler.fit_transform(X_train_scaled[numeric_cols_for_scaling])
                    X_test_scaled[numeric_cols_for_scaling] = scaler.transform(X_test_scaled[numeric_cols_for_scaling])
            
                    # Simpan hasil scaling
                    st.session_state.X_train = X_train_scaled
                    st.session_state.X_test = X_test_scaled
                    st.session_state.scaler = scaler
                    st.session_state.numeric_cols_scaled = numeric_cols_for_scaling
                    st.session_state.data_normalized = True
            
                    st.success("✅ Data berhasil dinormalisasi!")
            
                    # Preview sesudah scaling
                    st.write("**Data Setelah Normalisasi (Training):**")
                    st.dataframe(st.session_state.X_train[cols_to_preview].head())
            
                except Exception as e:
                    st.error(f"❌ Error saat normalisasi: {str(e)}")

        elif st.session_state.get("data_normalized", False):
            st.success("✅ Data sudah dinormalisasi, siap untuk modelling!")
        
        # Training Section (only show after normalization)
        if st.session_state.get('data_normalized', False):
            # Resampling method selection
            st.subheader("⚖️ Pilih Metode Resampling")
            
            resampling_options = [
                ("Tanpa Resampling", "Tanpa resampling"),
                ("SMOTE", "SMOTE"),
                ("ENN", "ENN")
            ]
            
            selected_resampling = st.selectbox(
                "Metode Resampling:",
                options=[option[0] for option in resampling_options],
                format_func=lambda x: next(option[1] for option in resampling_options if option[0] == x),
                index=0
            )
            st.session_state.selected_resampling = selected_resampling

            # Fungsi apply resampling
            def apply_resampling(method, X, y):
                if method == "SMOTE":
                    return SMOTE(random_state=42).fit_resample(X, y)
                elif method == "ENN":        
                    return EditedNearestNeighbours().fit_resample(X, y)
                return X, y
    
            # ELM Parameters
            if st.session_state.get('data_normalized', False):
                st.subheader("🧠 Parameter ELM")
                col1, col2 = st.columns(2)
                with col1:
                    hidden_neurons = st.slider("Jumlah Hidden Neuron:", min_value=50, max_value=500, value=100, step=10)
                with col2:
                    activation_functions = {"sigmoid": sigmoid, "tanh": tanh, "relu": relu}
                    activation_function = st.selectbox("Fungsi Aktivasi:", list(activation_functions.keys()))
            
                if st.button("🚀 Mulai Training ELM", type="primary"):
                    # Data split
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test
            
                    # Jalankan training untuk semua metode resampling
                    results_all = []
                    X_res, y_res = apply_resampling(selected_resampling, X_train, y_train)
                    act_func = activation_functions[activation_function]
            
                    # Train & Predict ELM
                    np.random.seed(42)
                    W, b, beta = train_elm(X_res, y_res, hidden_neurons, activation=act_func)
                    y_pred = predict_elm(X_test, W, b, beta, activation=act_func)

                    st.session_state.elm_params = {
                    "W": W, "b": b, "beta": beta, "activation": act_func,
                    "feature_names": list(st.session_state.X_train.columns)}
            
                    # Metrics
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    cm = confusion_matrix(y_test, y_pred)
            
                    results_all.append({
                        "method": selected_resampling,
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "cm": cm,
                        "y_pred": y_pred
                    })
            
                    st.session_state.training_results = results_all
                    st.session_state.y_test = y_test
                    st.session_state.model_trained = True
                    st.success("✅ Training selesai! Hasil evaluasi ditampilkan di bawah.")
                    st.rerun()
            
            # Evaluasi sesuai pilihan user
            if st.session_state.get('model_trained', False):
                results_all = st.session_state.training_results
                chosen = results_all[0]   # hanya 1 hasil
                y_test = st.session_state.y_test

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"🔄 Confusion Matrix ({chosen['method'].upper()})")
                    cm = chosen["cm"]
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=["Pred:0", "Pred:1"],
                        y=["True:0", "True:1"],
                        colorscale="Blues",
                        text=cm,
                        texttemplate="%{text}"
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Classification Report")
                    cr = classification_report(y_test, chosen["y_pred"], target_names=["Not Fraud", "Fraud"], output_dict=True)
                    st.dataframe(pd.DataFrame(cr).T, use_container_width=True)
        
                # Tombol lihat perbandingan
                if st.button("📊 Lihat Perbandingan Semua Metode"):
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    act_func = activation_functions[activation_function]
        
                    all_results = []
                    for method in ["Tanpa Resampling", "SMOTE", "ENN"]:
                        X_res, y_res = apply_resampling(method, X_train, y_train)
                        np.random.seed(42)
                        W, b, beta = train_elm(X_res, y_res, hidden_neurons, activation=act_func)
                        y_pred = predict_elm(X_test, W, b, beta, activation=act_func)
        
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, zero_division=0)
                        rec = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        cm = confusion_matrix(y_test, y_pred)
        
                        all_results.append({
                            "method": method,
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1": f1,
                            "cm": cm,
                            "y_pred": y_pred,
                            "W": W,
                            "b": b,
                            "beta": beta
                        })
        
                    comp_df = pd.DataFrame([{
                        "Method": r["method"],
                        "Accuracy": r["accuracy"],
                        "Precision": r["precision"],
                        "Recall": r["recall"],
                        "F1-Score": r["f1"]
                    } for r in all_results])
        
                    st.subheader("🔍 Perbandingan Semua Metode")
                    cm_summary = []
                    for r in all_results:
                        cm = r["cm"].ravel()  # urutannya [TN, FP, FN, TP]
                        tn, fp, fn, tp = cm
                        cm_summary.append({
                            "Percobaan": r["method"],
                            "True Negative (TN)": tn,
                            "False Positive (FP)": fp,
                            "False Negative (FN)": fn,
                            "True Positive (TP)": tp
                        })
                    
                    cm_df = pd.DataFrame(cm_summary)
                    
                    st.subheader("📊 Ringkasan Confusion Matrices ELM + Resampling")
                    st.dataframe(cm_df, use_container_width=True)
                    
                    # Hitung selisih precision & recall
                    comp_df["diff"] = abs(comp_df["Precision"] - comp_df["Recall"])
                    comp_df["balance_score"] = comp_df["F1-Score"] - comp_df["diff"]
                    
                    # Cari metode dengan score terbaik
                    best_idx = comp_df["balance_score"].idxmax()
                    best_row = comp_df.loc[best_idx]

                    #Menyimpan metode yang paling baik
                    st.session_state.best_model = {
                        "method": best_row["Method"],
                        "W": all_results[best_idx]["W"],
                        "b": all_results[best_idx]["b"],
                        "beta": all_results[best_idx]["beta"],
                        "activation": act_func,
                        "feature_names": list(X_train.columns)
                    }
                    
                    # Buat style: highlight cuma precision, recall, dan f1 di baris terbaik
                    def highlight_best(row):
                        if row.name == best_idx:
                            return ["background-color: yellow" if col in ["Precision", "Recall", "F1-Score"] else "" for col in row.index]
                        return ["" for _ in row.index]
                    
                    # Tampilkan tanpa kolom diff & balance_score
                    comp_df_display = comp_df.drop(columns=["diff", "balance_score"])
                    st.dataframe(comp_df_display.style.apply(highlight_best, axis=1), use_container_width=True)

                    # === Penjelasan otomatis ===
                    st.subheader("📝 Interpretasi Otomatis")
                    st.write(
                        f"Berdasarkan perbandingan evaluasi di atas, metode ELM **{best_row['Method'].upper()}** "
                        f"memiliki kombinasi Precision ({best_row['Precision']:.3f}) dan Recall ({best_row['Recall']:.3f}) "
                        f"yang paling seimbang, dengan F1-Score sebesar {best_row['F1-Score']:.3f}. "
                        "Hal ini menunjukkan metode ini mampu menjaga keseimbangan antara mengurangi false positives "
                        "dan meningkatkan kemampuan mendeteksi fraud."
                    )
                
                # Navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⬅️ Kembali"):
                        st.session_state.current_step = 2
                        st.rerun()
                with col2:
                    if st.session_state.get('model_trained', False) and st.button("➡️ Lanjut ke Visualisasi LIME", type="primary"):
                        st.session_state.current_step = 4
                        st.rerun()
        
elif st.session_state.current_step == 4:
    # Step 4: interpretasi LIME
    st.header("🔍 Interpretasi LIME")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Silakan lakukan training model terlebih dahulu!")
        if st.button("⬅️ Kembali ke Pemodelan"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        # LIME Introduction
        st.info("""
        **Tentang LIME (Local Interpretable Model-agnostic Explanations)**
        
        LIME menjelaskan prediksi model dengan mengidentifikasi fitur-fitur yang paling berpengaruh 
        terhadap keputusan klasifikasi untuk setiap instance secara individual. Ini membantu memahami 
        mengapa model menganggap suatu transaksi sebagai fraud atau tidak.
        """)
        
        # Ambil data & model
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        params = st.session_state.get("best_model", st.session_state.get("elm_params", None))
        if params is None:
            st.error("⚠️ Model belum tersedia. Silakan lakukan training atau bandingkan metode terlebih dahulu!")
            st.stop()
    
        W, b, beta = params["W"], params["b"], params["beta"]
        act_func = params["activation"]
        feature_names = params["feature_names"]
    
        # Definisi fungsi prediksi probabilitas (fixed function name)
        def predict_proba_elm(X, W, b, beta, activation):
            """
            Menghitung probabilitas kelas (Not Fraud, Fraud) dari model ELM.
            Output: array shape (n_samples, 2)
            """
            H = activation(np.dot(X, W) + b)
            y_scores = H.dot(beta)
            # Ubah ke probabilitas dengan sigmoid
            probs = 1 / (1 + np.exp(-y_scores))
            return np.column_stack([1 - probs, probs])

        # Buat explainer LIME dengan error handling
        try:
            explainer = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=feature_names,
                class_names=["Not Fraud", "Fraud"],
                discretize_continuous=True,
                mode="classification"
            )
        except Exception as e:
            st.error(f"Error creating LIME explainer: {str(e)}")
            st.stop()
    
        # Layout dengan kolom untuk kontrol
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pilih transaksi dari test set
            idx = st.number_input(
                "Pilih indeks transaksi uji:",
                min_value=0, 
                max_value=len(X_test)-1, 
                value=0, 
                step=1,
                help="Pilih transaksi yang ingin dianalisis menggunakan LIME"
            )
        
        with col2:
            # Informasi target aktual
            y_test =  np.array(st.session_state.y_test)
            actual_label = y_test[idx]
            st.metric(
                "Label Aktual", 
                "Fraud" if actual_label == 1 else "Not Fraud",
                delta=None
            )
        
        # Tampilkan data transaksi yang dipilih
        idx = st.number_input("Pilih indeks transaksi uji:", min_value=0, max_value=len(X_test)-1, value=0, step=1)
        x = X_test.iloc[idx].valuess
        print(x.shape, len(feature_names))
        
        with st.expander("📋 Data Transaksi yang Dipilih"):
            transaction_df = pd.DataFrame({
                'Fitur': feature_names,
                'Nilai': x
            })
            st.dataframe(transaction_df, use_container_width=True)
        
        # Button untuk generate explanation
        if st.button("🧠 Generate LIME Explanation", type="primary"):
            # Progress indicator
            with st.spinner("Menganalisis dengan LIME..."):
                try:
                    # Generate LIME explanation
                    exp = explainer.explain_instance(
                        data_row=x,
                        predict_fn=predict_proba,  # Fixed function name
                        num_features=min(10, len(feature_names)),
                        num_samples=1000  # Increased for better stability
                    )
    
                    # Probabilitas model
                    proba = predict_proba(x.reshape(1, -1))[0, 1]
                    predicted_class = "Fraud" if proba >= 0.5 else "Not Fraud"
                    
                    # Display prediction results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Probabilitas Fraud", f"{proba:.4f}")
                    with col2:
                        st.metric("Prediksi Model", predicted_class)
                    with col3:
                        # Accuracy indicator
                        is_correct = (proba >= 0.5) == (actual_label == 1)
                        st.metric("Status Prediksi", 
                                "✅ Benar" if is_correct else "❌ Salah")
    
                    # Kontribusi fitur (tabel)
                    lime_list = exp.as_list()
                    lime_df = pd.DataFrame(lime_list, columns=["Fitur", "Kontribusi"])
                    
                    # Sort by absolute contribution for better visualization
                    lime_df['abs_kontribusi'] = abs(lime_df['Kontribusi'])
                    lime_df = lime_df.sort_values('abs_kontribusi', ascending=True)
                    
                    st.subheader("📊 Kontribusi Fitur terhadap Prediksi")
                    
                    # Enhanced dataframe display
                    styled_df = lime_df[['Fitur', 'Kontribusi']].copy()
                    styled_df['Kontribusi'] = styled_df['Kontribusi'].round(4)
                    st.dataframe(styled_df, use_container_width=True)
    
                    # Enhanced visualization with better colors and formatting
                    colors = ['#e74c3c' if c > 0 else '#27ae60' for c in lime_df["Kontribusi"]]
                    
                    fig = go.Figure(go.Bar(
                        x=lime_df["Kontribusi"],
                        y=lime_df["Fitur"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{c:+.4f}" for c in lime_df["Kontribusi"]],
                        textposition="auto",
                        hovertemplate="<b>%{y}</b><br>" +
                                    "Kontribusi: %{x:.4f}<br>" +
                                    "<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        title={
                            'text': "LIME Feature Contributions",
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        xaxis_title="Kontribusi terhadap Prediksi Fraud",
                        yaxis_title="Fitur",
                        height=max(400, len(lime_df) * 30),  # Dynamic height
                        template="plotly_white",
                        showlegend=False
                    )
                    
                    # Add vertical line at x=0
                    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation guide
                    st.subheader("📖 Panduan Interpretasi")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **🔴 Kontribusi Positif (Merah)**
                        - Mendorong prediksi ke arah **Fraud**
                        - Semakin besar nilai, semakin kuat indikasi fraud
                        """)
                    
                    with col2:
                        st.markdown("""
                        **🟢 Kontribusi Negatif (Hijau)**
                        - Mendorong prediksi ke arah **Not Fraud**
                        - Semakin kecil (negatif), semakin kuat indikasi legitimate
                        """)
                    
                    # Summary insights
                    positive_features = lime_df[lime_df['Kontribusi'] > 0]['Fitur'].tolist()
                    negative_features = lime_df[lime_df['Kontribusi'] < 0]['Fitur'].tolist()
                    
                    if positive_features or negative_features:
                        st.subheader("💡 Insight Utama")
                        if positive_features:
                            st.write(f"**Fitur yang mendukung prediksi Fraud:** {', '.join(positive_features[:3])}")
                        if negative_features:
                            st.write(f"**Fitur yang mendukung prediksi Not Fraud:** {', '.join(negative_features[:3])}")
                    
                    # Optional: HTML interaktif dengan improved styling
                    with st.expander("🌐 Lihat Visual LIME Interaktif (HTML)"):
                        try:
                            html_content = exp.as_html()
                            components.html(html_content, height=800, scrolling=True)
                        except Exception as e:
                            st.error(f"Error generating HTML visualization: {str(e)}")
                    
                    # Export explanation option
                    with st.expander("💾 Export Hasil Analisis"):
                        # Create summary text
                        summary_text = f"""
LIME Analysis Summary
=====================
Transaction Index: {idx}
Actual Label: {'Fraud' if actual_label == 1 else 'Not Fraud'}
Predicted Label: {predicted_class}
Fraud Probability: {proba:.4f}
Prediction Accuracy: {'Correct' if is_correct else 'Incorrect'}

Feature Contributions:
{lime_df[['Fitur', 'Kontribusi']].to_string(index=False)}
                        """
                except Exception as e:
                    st.error(f"Error generating LIME explanation: {str(e)}")
                    st.write("Debug info:")
                    st.write(f"- X shape: {x.shape}")
                    st.write(f"- Feature names length: {len(feature_names)}")
                    st.write(f"- Model parameters available: {list(params.keys())}")
        
        # Navigation and Reset
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬅️ Kembali"):
                st.session_state.current_step = 4
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Dashboard", type="secondary"):
                # Reset all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state.current_step = 1
                st.rerun()
        
        with col3:
            st.write("")  # Placeholder for alignment

# Footer
st.markdown("---")
st.markdown("""
<div style="margin-top: 3rem; padding: 2rem; text-align: center; color: #666; border-top: 1px solid #eee;">
<p>🛡️ Fraud Detection System | Powered by ELM + LIME Integration</p>
<p><small>Built with Streamlit • Machine Learning • Explainable AI</small></p>
</div>
""", unsafe_allow_html=True)
