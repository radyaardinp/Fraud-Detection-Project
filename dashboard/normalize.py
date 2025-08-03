import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple

def load_scaler(path='fraud_dashboard/scaler.joblib'):
    try:
        return joblib.load(path)
    except Exception as e:
        raise Exception(f"Failed to load scaler: {str(e)}")

def load_selected_features(path='fraud_dashboard/selected_features.joblib'):
    try:
        return joblib.load(path)
    except Exception as e:
        raise Exception(f"Failed to load features: {str(e)}")

def create_and_save_scaler(df: pd.DataFrame, target_col: str = 'fraud', 
                          scaler_path: str = 'fraud_dashboard/scaler.joblib',
                          features_path: str = 'fraud_dashboard/selected_features.joblib') -> Tuple[MinMaxScaler, List[str]]:
    # Get feature columns (exclude target)
    features = [col for col in df.columns if col != target_col]
    
    # Create and fit scaler
    scaler = MinMaxScaler()
    scaler.fit(df[features])
    
    # Save scaler and features
    import os
    os.makedirs('fraud_dashboard', exist_ok=True)
    joblib.dump(scaler, scaler_path)
    joblib.dump(features, features_path)
    
    return scaler, features

def normalize_data(df: pd.DataFrame, scaler=None, features: Optional[List[str]] = None,
                  auto_create_scaler: bool = False, target_col: str = 'fraud') -> pd.DataFrame:
    try:
        # Try to load existing scaler and features
        if scaler is None:
            scaler = load_scaler()
        if features is None:
            features = load_selected_features()
            
    except Exception as e:
        if auto_create_scaler:
            print(f"Creating new scaler and feature list: {str(e)}")
            scaler, features = create_and_save_scaler(df, target_col)
        else:
            raise Exception(f"Scaler/features not found and auto_create_scaler=False: {str(e)}")
    
    # Ambil hanya fitur yang sudah dipilih, fill missing dengan 0
    data_to_scale = df.reindex(columns=features, fill_value=0)
    scaled = scaler.transform(data_to_scale)
    
    return pd.DataFrame(scaled, columns=features, index=df.index)

def normalize_data_with_existing_preprocessing(df: pd.DataFrame, 
                                             preprocessing_results: dict = None) -> Tuple[pd.DataFrame, List[str]]:
    # Get feature names (exclude 'fraud' if exists)
    target_col = 'fraud'
    feature_names = [col for col in df.columns if col != target_col]
    
    # Create scaler for current data
    scaler = MinMaxScaler()
    data_to_scale = df[feature_names]
    
    # Fit and transform
    scaled_data = scaler.fit_transform(data_to_scale)
    
    # Create normalized dataframe
    normalized_df = pd.DataFrame(scaled_data, columns=feature_names, index=df.index)
    
    # Add target column back if it exists
    if target_col in df.columns:
        normalized_df[target_col] = df[target_col].values
    
    return normalized_df, feature_names
