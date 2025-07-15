def calculate_thresholds(df):
    return {
        'daily_freq': df['daily_freq'].quantile(0.95),
        'amount': df['amount'].quantile(0.95),
        'failed_count': df['failed_count'].quantile(0.95),
        'mismatch': df['mismatch'].quantile(0.95)
    }

def detect_anomaly1(row, thresholds):
    if row['daily_freq'] > thresholds['daily_freq']:
        if row['amount'] > thresholds['amount']:
            if row['failed_count'] > thresholds['failed_count']:
                if row['mismatch'] > thresholds['mismatch']:
                    return 'Fraud'
                else:
                    return 'Fraud'
            else:
                if row['mismatch'] > thresholds['mismatch']:
                    if row['mismatch_ratio'] < 0.01 and row['failed_count'] == 0:
                        return 'Not Fraud'
                    else:
                        return 'Fraud'
                else:
                    return 'Not Fraud'
        else:
            if row['failed_count'] > thresholds['failed_count'] and row['mismatch'] > thresholds['mismatch']:
                return 'Fraud'
            else:
                return 'Not Fraud'
    else:
        if row['amount'] > thresholds['amount']:
            if row['failed_count'] > thresholds['failed_count'] or row['mismatch'] > thresholds['mismatch']:
                return 'Fraud'
            else:
                return 'Not Fraud'
        else:
            return 'Not Fraud'

def detect_anomaly2(row):
    if row['failed_count'] > 2 * row['avg_failed']:
        if row['fail_ratio'] > 0.7:
            if row['avg_fail_interval'] < 60:
                return 'Fraud'
            else:
                return 'Fraud'
        else:
            if row['avg_fail_interval'] < 60:
                return 'Fraud'
            else:
                return 'Not Fraud'
    else:
        if row['fail_ratio'] > 0.4:
            return 'Fraud'
        else:
            return 'Not Fraud'

def detect_anomaly3(row):
    if row['mismatch_ratio'] > 1.0:
        return 'Fraud'
    else:
        return 'Not Fraud'

def detect_combined_anomaly(row, thresholds):
    result1 = detect_anomaly1(row, thresholds)
    result2 = detect_anomaly2(row)
    result3 = detect_anomaly3(row)
    if 'Fraud' in [result1, result2, result3]:
        return 'Fraud'
    else:
        return 'Not Fraud'

def process_dataframe(df):
    # Hitung threshold
    thresholds = calculate_thresholds(df)

    # Labeling berdasarkan masing-masing rule
    df['label1'] = df.apply(lambda row: detect_anomaly1(row, thresholds), axis=1)
    df['label2'] = df.apply(detect_anomaly2, axis=1)
    df['label3'] = df.apply(detect_anomaly3, axis=1)

    # Label akhir berdasarkan kombinasi
    df['fraud'] = df.apply(lambda row: detect_combined_anomaly(row, thresholds), axis=1)

    # Print distribusi label
    print(df[['label1', 'label2', 'label3', 'fraud']].apply(pd.Series.value_counts))

    # Kolom yang ingin dihapus
    columns_to_drop = [
        'createdDate', 'daily_freq', 'is_declined', 'failed_count',
        'is_nominal_tinggi', 'mismatch', 'avg_failed', 'fail_ratio',
        'avg_fail_interval', 'mismatch_ratio', 'label1', 'label2', 'label3'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df
