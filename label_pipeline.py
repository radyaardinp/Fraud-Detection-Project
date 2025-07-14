import pandas as pd
from labeling_rules import (
    detect_anomaly1, detect_anomaly2, detect_anomaly3, detect_combined_anomaly
)
from threshold_utils import calculate_thresholds

def label_dataset(df):
    # Hitung threshold dari data
    thresholds = calculate_thresholds(df)

    # Labeling
    df['label1'] = df.apply(lambda row: detect_anomaly1(row, thresholds), axis=1)
    df['label2'] = df.apply(detect_anomaly2, axis=1)
    df['label3'] = df.apply(detect_anomaly3, axis=1)
    df['fraud'] = df.apply(lambda row: detect_combined_anomaly(row, thresholds), axis=1)

    return df
