import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
import subprocess

# Cek apakah matplotlib sudah terinstall, jika tidak maka install dulu
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Bank Fraud Detection Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .fraud-section {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        margin: 2rem 0;
    }
    .safe-transaction {
        background: #dcfce7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #16a34a;
        margin: 1rem 0;
    }
    .fraud-transaction {
        background: #fef2f2;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
    }
    .warning-transaction {
        background: #fefce8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ca8a04;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []

# Generate sample data for dashboard
@st.cache_data
def generate_sample_data():
    # Generate time series data
    times = pd.date_range(start='2024-01-01 07:00', end='2024-01-01 08:00', freq='5min')
    
    # Transaction volume data
    volume_data = {
        'time': times,
        'transaction_volume': np.random.randint(250, 400, len(times)),
        'higher_threshold': [380] * len(times),
        'lower_threshold': [280] * len(times)
    }
    
    # Vendor data
    vendors = ['Vendor A', 'Vendor B', 'Vendor C', 'Vendor D', 'Vendor E', 'Vendor F', 'Vendor G', 'Vendor H']
    vendor_values = [25, 20, 15, 12, 10, 8, 6, 4]
    
    # Card type data
    card_types = ['Maestro', 'Visa', 'Other', 'Disc']
    card_values = [40, 35, 15, 10]
    
    # Declined transactions by type
    decline_data = {
        'time': times,
        'ATM': np.random.uniform(0, 4, len(times)),
        'Debit': np.random.uniform(0, 3, len(times)),
        'Credit': np.random.uniform(0, 2, len(times)),
        'Check': np.random.uniform(0, 1, len(times))
    }
    
    return volume_data, vendors, vendor_values, card_types, card_values, decline_data

# Fraud detection function (simplified)
def detect_fraud(amount, transaction_type, time_hour, merchant_category):
    """
    Simplified fraud detection algorithm
    In real implementation, this would use ML model
    """
    risk_score = 0
    reasons = []
    
    # Amount-based rules
    if amount > 10000:
        risk_score += 30
        reasons.append("Transaksi nominal tinggi")
    elif amount > 5000:
        risk_score += 15
        reasons.append("Transaksi nominal sedang-tinggi")
    
    # Time-based rules
    if time_hour < 6 or time_hour > 22:
        risk_score += 20
        reasons.append("Transaksi di luar jam normal")
    
    # Transaction type rules
    if transaction_type == "ATM" and amount > 2000:
        risk_score += 25
        reasons.append("Penarikan ATM nominal tinggi")
    
    # Merchant category rules
    high_risk_merchants = ["Online Gaming", "Cryptocurrency", "Money Transfer"]
    if merchant_category in high_risk_merchants:
        risk_score += 35
        reasons.append(f"Merchant kategori berisiko tinggi: {merchant_category}")
    
    # Determine fraud status
    if risk_score >= 50:
        return "FRAUD", risk_score, reasons
    elif risk_score >= 30:
        return "SUSPICIOUS", risk_score, reasons
    else:
        return "SAFE", risk_score, reasons

# Main dashboard
def main():
    st.markdown('<h1 class="main-header">🏦 Bank Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Dashboard untuk monitoring transaksi bank dan deteksi fraud**")
    
    # Generate sample data
    volume_data, vendors, vendor_values, card_types, card_values, decline_data = generate_sample_data()
    
    # Top metrics row
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Credit Transaction",
            value="240",
            delta="0.45% Decline rate"
        )
    
    with col2:
        st.metric(
            label="Check Transaction", 
            value="20",
            delta="0.0% Decline rate"
        )
    
    with col3:
        st.metric(
            label="ATM Transaction",
            value="25", 
            delta="9.45% Decline rate"
        )
    
    with col4:
        st.metric(
            label="Debit Card Transaction",
            value="30",
            delta="5.45% Decline rate"
        )
    
    # Charts section
    st.markdown("### 📈 Transaction Analytics")
    
    # Transaction volume chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Transaction Volume Network and Point**")
        fig_vendor = px.pie(
            values=vendor_values,
            names=vendors,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_vendor.update_traces(textposition='inside', textinfo='percent+label')
        fig_vendor.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_vendor, use_container_width=True)
    
    with col2:
        st.markdown("**Transaction Volume by Card Type**")
        fig_card = px.pie(
            values=card_values,
            names=card_types,
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_card.update_traces(textposition='inside', textinfo='percent+label')
        fig_card.update_layout(height=400)
        st.plotly_chart(fig_card, use_container_width=True)
    
    # Time series charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Transaction Volume Over Time**")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Scatter(
            x=volume_data['time'],
            y=volume_data['transaction_volume'],
            mode='lines+markers',
            name='Transaction Volume',
            line=dict(color='gold', width=2)
        ))
        fig_volume.add_trace(go.Scatter(
            x=volume_data['time'],
            y=volume_data['higher_threshold'],
            mode='lines',
            name='Higher Threshold',
            line=dict(color='red', dash='dash')
        ))
        fig_volume.add_trace(go.Scatter(
            x=volume_data['time'],
            y=volume_data['lower_threshold'],
            mode='lines',
            name='Lower Threshold',
            line=dict(color='blue', dash='dash')
        ))
        fig_volume.update_layout(height=400, yaxis_title="Transaction Volume")
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        st.markdown("**Declined Transactions by Type**")
        fig_decline = go.Figure()
        
        colors = ['gold', 'lightgray', 'lightgreen', 'lightblue']
        for i, (transaction_type, color) in enumerate(zip(['ATM', 'Debit', 'Credit', 'Check'], colors)):
            fig_decline.add_trace(go.Scatter(
                x=decline_data['time'],
                y=decline_data[transaction_type],
                mode='markers',
                name=transaction_type,
                marker=dict(color=color, size=8)
            ))
        
        fig_decline.update_layout(
            height=400,
            yaxis_title="Declined Transactions",
            yaxis=dict(range=[-2, 6])
        )
        st.plotly_chart(fig_decline, use_container_width=True)
    
    # Fraud Detection Section
    st.markdown("---")
    st.markdown('<div class="fraud-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 Fraud Detection System")
    st.markdown("**Masukkan detail transaksi untuk analisis fraud detection**")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input(
            "💰 Nominal Transaksi (Rp)",
            min_value=0,
            max_value=100000000,
            value=1000000,
            step=100000,
            help="Masukkan nominal transaksi dalam Rupiah"
        )
        
        transaction_type = st.selectbox(
            "💳 Jenis Transaksi",
            ["ATM", "Debit Card", "Credit Card", "Check", "Online Transfer"],
            help="Pilih jenis transaksi yang akan dianalisis"
        )
        
        time_hour = st.slider(
            "🕐 Jam Transaksi",
            0, 23, 14,
            help="Pilih jam transaksi (0-23)"
        )
    
    with col2:
        merchant_category = st.selectbox(
            "🏪 Kategori Merchant",
            ["Retail", "Restaurant", "Gas Station", "Online Shopping", 
             "ATM", "Online Gaming", "Cryptocurrency", "Money Transfer", "Other"],
            help="Pilih kategori merchant"
        )
        
        location = st.text_input(
            "📍 Lokasi Transaksi",
            value="Jakarta",
            help="Masukkan lokasi transaksi"
        )
        
        customer_id = st.text_input(
            "👤 Customer ID",
            value="CUST001",
            help="Masukkan ID customer"
        )
    
    # Analyze button
    if st.button("🔍 Analisis Fraud Detection", type="primary"):
        # Perform fraud detection
        fraud_status, risk_score, reasons = detect_fraud(
            amount, transaction_type, time_hour, merchant_category
        )
        
        # Create transaction record
        transaction_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'customer_id': customer_id,
            'amount': amount,
            'type': transaction_type,
            'location': location,
            'merchant_category': merchant_category,
            'time_hour': time_hour,
            'fraud_status': fraud_status,
            'risk_score': risk_score,
            'reasons': reasons
        }
        
        # Add to history
        st.session_state.transaction_history.append(transaction_record)
        
        # Display results
        st.markdown("### 📊 Hasil Analisis")
        
        if fraud_status == "FRAUD":
            st.markdown(f"""
            <div class="fraud-transaction">
                <h3>🚨 FRAUD DETECTED</h3>
                <p><strong>Risk Score:</strong> {risk_score}/100</p>
                <p><strong>Status:</strong> Transaksi berpotensi FRAUD</p>
                <p><strong>Alasan:</strong></p>
                <ul>
                    {''.join([f'<li>{reason}</li>' for reason in reasons])}
                </ul>
                <p><strong>Rekomendasi:</strong> Blokir transaksi dan lakukan verifikasi manual</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif fraud_status == "SUSPICIOUS":
            st.markdown(f"""
            <div class="warning-transaction">
                <h3>⚠️ SUSPICIOUS TRANSACTION</h3>
                <p><strong>Risk Score:</strong> {risk_score}/100</p>
                <p><strong>Status:</strong> Transaksi mencurigakan</p>
                <p><strong>Alasan:</strong></p>
                <ul>
                    {''.join([f'<li>{reason}</li>' for reason in reasons])}
                </ul>
                <p><strong>Rekomendasi:</strong> Lakukan verifikasi tambahan</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="safe-transaction">
                <h3>✅ SAFE TRANSACTION</h3>
                <p><strong>Risk Score:</strong> {risk_score}/100</p>
                <p><strong>Status:</strong> Transaksi aman</p>
                <p><strong>Rekomendasi:</strong> Transaksi dapat diproses normal</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display transaction details
        st.markdown("### 📋 Detail Transaksi")
        details_df = pd.DataFrame([{
            'Field': 'Nominal',
            'Value': f"Rp {amount:,}"
        }, {
            'Field': 'Jenis Transaksi',
            'Value': transaction_type
        }, {
            'Field': 'Jam Transaksi',
            'Value': f"{time_hour}:00"
        }, {
            'Field': 'Kategori Merchant',
            'Value': merchant_category
        }, {
            'Field': 'Lokasi',
            'Value': location
        }, {
            'Field': 'Customer ID',
            'Value': customer_id
        }])
        
        st.dataframe(details_df, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Transaction History
    if st.session_state.transaction_history:
        st.markdown("---")
        st.markdown("### 📚 History Transaksi")
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.transaction_history)
        
        # Display recent transactions
        st.markdown("**10 Transaksi Terakhir:**")
        recent_df = history_df.tail(10)[['timestamp', 'customer_id', 'amount', 'type', 'fraud_status', 'risk_score']]
        recent_df['amount'] = recent_df['amount'].apply(lambda x: f"Rp {x:,}")
        
        # Color code based on fraud status
        def highlight_fraud(row):
            if row['fraud_status'] == 'FRAUD':
                return ['background-color: #fef2f2'] * len(row)
            elif row['fraud_status'] == 'SUSPICIOUS':
                return ['background-color: #fefce8'] * len(row)
            else:
                return ['background-color: #dcfce7'] * len(row)
        
        styled_df = recent_df.style.apply(highlight_fraud, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fraud_count = len(history_df[history_df['fraud_status'] == 'FRAUD'])
            st.metric("🚨 Fraud Detected", fraud_count)
        
        with col2:
            suspicious_count = len(history_df[history_df['fraud_status'] == 'SUSPICIOUS'])
            st.metric("⚠️ Suspicious", suspicious_count)
        
        with col3:
            safe_count = len(history_df[history_df['fraud_status'] == 'SAFE'])
            st.metric("✅ Safe", safe_count)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.transaction_history = []
            st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("**Fraud Detection Parameters:**")
    fraud_threshold = st.slider("Risk Score Threshold", 0, 100, 50)
    
    st.markdown("**Time Settings:**")
    auto_refresh = st.checkbox("Auto Refresh Dashboard", value=False)
    
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 30)
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.success("✅ Fraud Detection: Online")
    st.success("✅ Database: Connected")
    st.success("✅ API: Running")
    
    st.markdown("---")
    st.markdown("### 📞 Support")
    st.info("Untuk bantuan teknis, hubungi IT Support")

if __name__ == "__main__":
    main()
