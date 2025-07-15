import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_data(df):
    columns_to_drop = ['id', 'inquiryId', 'networkReferenceId']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df['settlementAmount'].fillna(0, inplace=True)
    df['feeAmount'].fillna(0, inplace=True)

    df['updatedTime'] = pd.to_datetime(df['updatedTime'], dayfirst=True, format='mixed')
    df['createdTime'] = pd.to_datetime(df['createdTime'])

    float_cols = ['amount', 'settlementAmount', 'feeAmount', 'discountAmount', 'inquiryAmount']
    for col in float_cols:
        df[col] = df[col].astype(float)

    categorical_cols = ['merchantId', 'paymentSource', 'status', 'statusCode']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df


def generate_label_features(df):
    df['createdDate'] = pd.to_datetime(df['createdTime']).dt.date
    df['is_declined'] = df['status'].str.lower() == 'declined'

    frekuensi_harian = df.groupby(['merchantId', 'createdDate']).size().reset_index(name='daily_freq')
    df = df.merge(frekuensi_harian, on=['merchantId', 'createdDate'])

    failed_per_day = df.groupby(['merchantId', 'createdDate'])['is_declined'].sum().reset_index(name='failed_count')
    df = df.merge(failed_per_day, on=['merchantId', 'createdDate'])

    df['is_nominal_tinggi'] = df['amount'] > 8_000_000
    df['mismatch'] = abs(df['inquiryAmount'] - df['settlementAmount'])

    avg_failed_per_merchant = failed_per_day.groupby('merchantId')['failed_count'].mean().reset_index(name='avg_failed')
    df = df.merge(avg_failed_per_merchant, on='merchantId', how='left')

    df['fail_ratio'] = df['failed_count'] / df['daily_freq']

    failed_trx = df[df['is_declined']].copy()
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

    df['mismatch_ratio'] = np.where(
        df['inquiryAmount'] == 0,
        0,
        abs(df['settlementAmount'] - df['inquiryAmount']) / df['inquiryAmount']
    )

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

    df['label1'] = df.apply(detect_anomaly1, axis=1)
    df['label2'] = df.apply(detect_anomaly2, axis=1)
    df['label3'] = df.apply(detect_anomaly3, axis=1)
    df['fraud'] = df.apply(detect_combined_anomaly, axis=1)

    columns_to_drop = ['createdDate', 'daily_freq', 'is_declined', 'failed_count', 'is_nominal_tinggi', 'mismatch',
                       'avg_failed', 'fail_ratio', 'avg_fail_interval', 'mismatch_ratio', 'label1', 'label2', 'label3']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df


def feature_engineering(df):
    epsilon = 1e-6
    df['discount_ratio'] = df['discountAmount'] / (df['amount'] + epsilon)
    df['fee_ratio'] = df['feeAmount'] / (df['amount'] + epsilon)
    df['selisih_waktu (sec)'] = (df['updatedTime'] - df['createdTime']).dt.total_seconds()
    df['hour_of_day'] = df['createdTime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df.drop(columns=['hour_of_day', 'createdTime', 'updatedTime'], inplace=True)
    return df


def encode_categoricals(df):
    cat_cols = ['merchantId', 'paymentSource', 'status', 'statusCode', 'currency', 'type', 'Type Token']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def preprocess_for_prediction(df):
    df = clean_data(df)
    df = generate_label_features(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)
    return df
