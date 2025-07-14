def calculate_thresholds(df):
    return {
        'daily_freq': df['daily_freq'].quantile(0.95),
        'amount': df['amount'].quantile(0.95),
        'failed_count': df['failed_count'].quantile(0.95),
        'mismatch': df['mismatch'].quantile(0.95)
    }
