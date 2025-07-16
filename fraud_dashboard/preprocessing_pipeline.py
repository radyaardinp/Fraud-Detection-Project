import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_data(df):
    """Clean and prepare basic data types"""
    columns_to_drop = ['id', 'inquiryId', 'networkReferenceId']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Handle missing values more safely
    df = df.copy()  # Prevent SettingWithCopyWarning
    df['settlementAmount'] = df['settlementAmount'].fillna(0)
    df['feeAmount'] = df['feeAmount'].fillna(0)
    df['discountAmount'] = df['discountAmount'].fillna(0) if 'discountAmount' in df.columns else 0

    # Convert datetime with better error handling
    df['updatedTime'] = pd.to_datetime(df['updatedTime'], errors='coerce')
    df['createdTime'] = pd.to_datetime(df['createdTime'], errors='coerce')

    # Convert numeric columns safely
    float_cols = ['amount', 'settlementAmount', 'feeAmount', 'discountAmount', 'inquiryAmount']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert categorical columns
    categorical_cols = ['merchantId', 'paymentSource', 'status', 'statusCode']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df


def generate_label_features(df):
    """Generate fraud detection features and labels"""
    df = df.copy()
    df['createdDate'] = pd.to_datetime(df['createdTime']).dt.date
    df['is_declined'] = df['status'].str.lower() == 'declined'

    # Daily frequency calculation
    frekuensi_harian = df.groupby(['merchantId', 'createdDate']).size().reset_index(name='daily_freq')
    df = df.merge(frekuensi_harian, on=['merchantId', 'createdDate'])

    # Failed transactions per day
    failed_per_day = df.groupby(['merchantId', 'createdDate'])['is_declined'].sum().reset_index(name='failed_count')
    df = df.merge(failed_per_day, on=['merchantId', 'createdDate'])

    # Additional features
    df['is_nominal_tinggi'] = df['amount'] > 8_000_000
    df['mismatch'] = abs(df['inquiryAmount'] - df['settlementAmount'])

    # Average failed per merchant
    avg_failed_per_merchant = failed_per_day.groupby('merchantId')['failed_count'].mean().reset_index(name='avg_failed')
    df = df.merge(avg_failed_per_merchant, on='merchantId', how='left')
    df['avg_failed'] = df['avg_failed'].fillna(0)

    # Failure ratio
    df['fail_ratio'] = df['failed_count'] / np.maximum(df['daily_freq'], 1)

    # Calculate failure intervals
    failed_trx = df[df['is_declined']].copy()
    if len(failed_trx) > 0:
        failed_trx = failed_trx.sort_values(by=['merchantId', 'createdTime'])
        failed_trx['prev_failed_time'] = failed_trx.groupby('merchantId')['createdTime'].shift(1)
        failed_trx['failed_time_diff'] = (failed_trx['createdTime'] - failed_trx['prev_failed_time']).dt.total_seconds()
        failed_trx['createdDate'] = failed_trx['createdTime'].dt.date

        failed_diff_daily = failed_trx.groupby(['merchantId', 'createdDate'])['failed_time_diff'].mean().reset_index(name='avg_fail_interval')
        failed_count_per_day = failed_trx.groupby(['merchantId', 'createdDate']).size().reset_index(name='count_failed')
        failed_diff_daily = failed_diff_daily.merge(failed_count_per_day, on=['merchantId', 'createdDate'])
        failed_diff_daily['avg_fail_interval'] = np.where(
            failed_diff_daily['count_failed'] < 2,
            0,
            failed_diff_daily['avg_fail_interval']
        )
        df = df.merge(failed_diff_daily[['merchantId', 'createdDate', 'avg_fail_interval']], on=['merchantId', 'createdDate'], how='left')
    
    df['avg_fail_interval'] = df['avg_fail_interval'].fillna(0)

    # Mismatch ratio
    df['mismatch_ratio'] = np.where(
        df['inquiryAmount'] == 0,
        0,
        abs(df['settlementAmount'] - df['inquiryAmount']) / df['inquiryAmount']
    )

    # Calculate thresholds
    thresholds = {
        'daily_freq': df['daily_freq'].quantile(0.95),
        'amount': df['amount'].quantile(0.95),
        'failed_count': df['failed_count'].quantile(0.95),
        'mismatch': df['mismatch'].quantile(0.95)
    }

    def detect_anomaly1(row):
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
        return 'Fraud' if row['mismatch_ratio'] > 1.0 else 'Not Fraud'

    def detect_combined_anomaly(row):
        results = [
            detect_anomaly1(row),
            detect_anomaly2(row),
            detect_anomaly3(row)
        ]
        return 'Fraud' if 'Fraud' in results else 'Not Fraud'

    # Apply detection functions
    df['label1'] = df.apply(detect_anomaly1, axis=1)
    df['label2'] = df.apply(detect_anomaly2, axis=1)
    df['label3'] = df.apply(detect_anomaly3, axis=1)
    df['fraud'] = df.apply(detect_combined_anomaly, axis=1)

    # Drop temporary columns
    columns_to_drop = ['createdDate', 'daily_freq', 'is_declined', 'failed_count', 'is_nominal_tinggi', 'mismatch',
                       'avg_failed', 'fail_ratio', 'avg_fail_interval', 'mismatch_ratio', 'label1', 'label2', 'label3']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df


def feature_engineering(df):
    """Create additional engineered features"""
    df = df.copy()
    epsilon = 1e-6
    
    # Ensure required columns exist
    if 'discountAmount' not in df.columns:
        df['discountAmount'] = 0
    if 'feeAmount' not in df.columns:
        df['feeAmount'] = 0
    
    df['discount_ratio'] = df['discountAmount'] / (df['amount'] + epsilon)
    df['fee_ratio'] = df['feeAmount'] / (df['amount'] + epsilon)
    
    # Check if datetime columns exist
    if 'updatedTime' in df.columns and 'createdTime' in df.columns:
        df['selisih_waktu (sec)'] = (df['updatedTime'] - df['createdTime']).dt.total_seconds()
    else:
        df['selisih_waktu (sec)'] = 0
    
    if 'createdTime' in df.columns:
        df['hour_of_day'] = df['createdTime'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df = df.drop(columns=['hour_of_day'], errors='ignore')
    else:
        df['hour_sin'] = 0
        df['hour_cos'] = 0
    
    # Drop datetime columns
    df = df.drop(columns=['createdTime', 'updatedTime'], errors='ignore')
    return df


def encode_categoricals(df):
    """Encode categorical variables"""
    df = df.copy()
    cat_cols = ['merchantId', 'paymentSource', 'status', 'statusCode']
    
    # Add other categorical columns if they exist
    potential_cat_cols = ['currency', 'type', 'Type Token']
    for col in potential_cat_cols:
        if col in df.columns:
            cat_cols.append(col)
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Handle missing values
            df[col] = df[col].fillna('unknown')
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def preprocess_for_prediction(df):
    """Main preprocessing pipeline function"""
    try:
        print("Starting preprocessing...")
        
        # Validate input
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        
        print(f"Input shape: {df.shape}")
        
        # Apply preprocessing steps
        df = clean_data(df)
        print("✓ Data cleaning completed")
        
        df = generate_label_features(df)
        print("✓ Label features generated")
        
        df = feature_engineering(df)
        print("✓ Feature engineering completed")
        
        df = encode_categoricals(df)
        print("✓ Categorical encoding completed")
        
        print(f"Final shape: {df.shape}")
        print("Preprocessing completed successfully!")
        
        return df
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise
