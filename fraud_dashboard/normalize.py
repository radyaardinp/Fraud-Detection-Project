# normalize.py
import joblib
import pandas as pd

def load_scaler(path='scaler.joblib'):
    return joblib.load(path)

def normalize_data(df, scaler=None):
    if scaler is None:
        scaler = load_scaler()
    if features is None:
        raise ValueError("Fitur tidak diberikan ke normalize_data")
    
    # Ambil hanya fitur numerik yang sudah dipilih
    data_to_scale = df[features].copy()
    scaled = scaler.transform(data_to_scale)
    
    return pd.DataFrame(scaled, columns=features, index=df.index)
