# normalize.py
import joblib
import pandas as pd
from selected_features import FEATURES

def load_scaler(path='scaler.joblib'):
    return joblib.load(path)

def normalize_data(df, scaler=None):
    if scaler is None:
        scaler = load_scaler()
    
    # Ambil hanya fitur numerik yang sudah dipilih
    data_to_scale = df[FEATURES].copy()
    scaled = scaler.transform(data_to_scale)
    
    return pd.DataFrame(scaled, columns=FEATURES, index=df.index)
