import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import joblib

def select_features_by_mi(df, label_col='fraud', threshold=0.01):
    df = df.copy()

    # Pisahkan fitur dan label
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Hitung MI
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})

    # Seleksi fitur berdasarkan threshold
    selected_features = mi_df[mi_df['MI Score'] > threshold]['Feature'].tolist()

    # Simpan hasil fitur terpilih
    joblib.dump(selected_features, "selected_features.pkl")

    # Return dataframe siap modeling
    return df[selected_features + [label_col]]
