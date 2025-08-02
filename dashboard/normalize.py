import joblib
import pandas as pd


def load_scaler(path='fraud_dashboard/scaler.joblib'):
    """Load saved MinMaxScaler"""
    try:
        return joblib.load(path)
    except Exception as e:
        raise Exception(f"Failed to load scaler: {str(e)}")


def load_selected_features(path='fraud_dashboard/selected_features.joblib'):
    """Load selected features"""
    try:
        return joblib.load(path)
    except Exception as e:
        raise Exception(f"Failed to load features: {str(e)}")


def normalize_data(df: pd.DataFrame, scaler=None) -> pd.DataFrame:
    """
    Normalize data using saved MinMaxScaler
    
    Args:
        df: Input dataframe
        scaler: Optional pre-loaded scaler
    
    Returns:
        normalized_df: Normalized dataframe with selected features only
    """
    if scaler is None:
        scaler = load_scaler()
    
    # Load selected features
    features = load_selected_features()
    
    # Ambil hanya fitur yang sudah dipilih, fill missing dengan 0
    data_to_scale = df.reindex(columns=features, fill_value=0)
    scaled = scaler.transform(data_to_scale)
    
    return pd.DataFrame(scaled, columns=features, index=df.index)
