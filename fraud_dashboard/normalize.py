# normalize.py
import joblib
import pandas as pd

#Load FEATURES
FEATURES = joblib.load("fraud_dashboard/selected_features.joblib")

def load_scaler(path='scaler.joblib'):
    return joblib.load(path)

def normalize_data(df, scaler=None):
    if scaler is None:
        scaler = load_scaler()
    
    # Ambil hanya fitur numerik yang sudah dipilih
    data_to_scale = df[features].copy()
    scaled = scaler.transform(data_to_scale)
    
    return pd.DataFrame(scaled, columns=features, index=df.index)
