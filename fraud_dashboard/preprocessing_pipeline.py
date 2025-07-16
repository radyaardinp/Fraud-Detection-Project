import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionPreprocessor:
    def __init__(self, config=None):
        """
        Initialize preprocessor with configurable parameters
        """
        self.config = config or {
            'quantile_threshold': 0.95,
            'mismatch_ratio_threshold': 0.01,
            'high_fail_ratio_threshold': 0.7,
            'low_fail_ratio_threshold': 0.4,
            'quick_fail_interval': 60,
            'extreme_mismatch_ratio': 1.0,
            'high_failed_multiplier': 2
        }
        self.label_encoders = {}
        
    def clean_data(self, df):
        """Clean and prepare basic data types"""
        df = df.copy()  # Prevent SettingWithCopyWarning
        
        # Drop unnecessary columns
        columns_to_drop = ['id', 'inquiryId', 'networkReferenceId']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Handle missing values
        numeric_fill_cols = ['settlementAmount', 'feeAmount', 'discountAmount']
        for col in numeric_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Convert datetime columns
        datetime_cols = ['updatedTime', 'createdTime']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        float_cols = ['amount', 'settlementAmount', 'feeAmount', 'discountAmount', 'inquiryAmount']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert categorical columns
        categorical_cols = ['merchantId', 'paymentSource', 'status', 'statusCode', 'currency', 'type', 'Type Token']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    def generate_label_features(self, df):
        """Generate fraud detection features and labels"""
        df = df.copy()
        
        # Create base features
        df['createdDate'] = df['createdTime'].dt.date
        df['is_declined'] = df['status'].str.lower() == 'declined'
        
        # Calculate daily frequency and failures
        daily_stats = df.groupby(['merchantId', 'createdDate']).agg({
            'is_declined': ['sum', 'count']
        }).reset_index()
        
        daily_stats.columns = ['merchantId', 'createdDate', 'failed_count', 'daily_freq']
        df = df.merge(daily_stats, on=['merchantId', 'createdDate'])
        
        # Calculate merchant averages
        merchant_avg = daily_stats.groupby('merchantId')['failed_count'].mean().reset_index(name='avg_failed')
        df = df.merge(merchant_avg, on='merchantId', how='left')
        
        # Calculate ratios and additional features
        df['fail_ratio'] = df['failed_count'] / np.maximum(df['daily_freq'], 1)
        df['is_nominal_tinggi'] = df['amount'] > 8_000_000
        df['mismatch'] = abs(df['inquiryAmount'] - df['settlementAmount'])
        df['mismatch_ratio'] = np.where(
            df['inquiryAmount'] == 0, 0,
            abs(df['settlementAmount'] - df['inquiryAmount']) / df['inquiryAmount']
        )
        
        # Calculate failure intervals
        df = self._calculate_failure_intervals(df)
        
        # Generate fraud labels
        df = self._generate_fraud_labels(df)
        
        # Clean up temporary columns
        temp_cols = ['createdDate', 'daily_freq', 'is_declined', 'failed_count', 
                    'is_nominal_tinggi', 'mismatch', 'avg_failed', 'fail_ratio', 
                    'avg_fail_interval', 'mismatch_ratio', 'label1', 'label2', 'label3']
        df = df.drop(columns=temp_cols, errors='ignore')
        
        return df
    
    def _calculate_failure_intervals(self, df):
        """Calculate average failure intervals for merchants"""
        failed_trx = df[df['is_declined']].copy()
        if len(failed_trx) == 0:
            df['avg_fail_interval'] = 0
            return df
            
        failed_trx = failed_trx.sort_values(by=['merchantId', 'createdTime'])
        failed_trx['prev_failed_time'] = failed_trx.groupby('merchantId')['createdTime'].shift(1)
        failed_trx['failed_time_diff'] = (
            failed_trx['createdTime'] - failed_trx['prev_failed_time']
        ).dt.total_seconds()
        
        failed_diff_daily = failed_trx.groupby(['merchantId', 'createdDate']).agg({
            'failed_time_diff': 'mean'
        }).reset_index()
        failed_diff_daily.columns = ['merchantId', 'createdDate', 'avg_fail_interval']
        
        # Count failed transactions per day
        failed_count_per_day = failed_trx.groupby(['merchantId', 'createdDate']).size().reset_index(name='count_failed')
        failed_diff_daily = failed_diff_daily.merge(failed_count_per_day, on=['merchantId', 'createdDate'])
        
        # Set interval to 0 if less than 2 failures
        failed_diff_daily['avg_fail_interval'] = np.where(
            failed_diff_daily['count_failed'] < 2, 0, failed_diff_daily['avg_fail_interval']
        )
        
        df = df.merge(
            failed_diff_daily[['merchantId', 'createdDate', 'avg_fail_interval']], 
            on=['merchantId', 'createdDate'], how='left'
        )
        df['avg_fail_interval'] = df['avg_fail_interval'].fillna(0)
        
        return df
    
    def _generate_fraud_labels(self, df):
        """Generate fraud detection labels using multiple strategies"""
        # Calculate thresholds
        thresholds = {
            'daily_freq': df['daily_freq'].quantile(self.config['quantile_threshold']),
            'amount': df['amount'].quantile(self.config['quantile_threshold']),
            'failed_count': df['failed_count'].quantile(self.config['quantile_threshold']),
            'mismatch': df['mismatch'].quantile(self.config['quantile_threshold'])
        }
        
        # Strategy 1: Volume and amount based detection
        df['label1'] = df.apply(lambda row: self._detect_anomaly1(row, thresholds), axis=1)
        
        # Strategy 2: Failure pattern based detection
        df['label2'] = df.apply(lambda row: self._detect_anomaly2(row), axis=1)
        
        # Strategy 3: Mismatch ratio based detection
        df['label3'] = df.apply(lambda row: self._detect_anomaly3(row), axis=1)
        
        # Combined strategy
        df['fraud'] = df.apply(lambda row: self._detect_combined_anomaly(row), axis=1)
        
        return df
    
    def _detect_anomaly1(self, row, thresholds):
        """Volume and amount based fraud detection"""
        if row['daily_freq'] > thresholds['daily_freq']:
            if row['amount'] > thresholds['amount']:
                if row['failed_count'] > thresholds['failed_count']:
                    return 'Fraud'
                else:
                    if row['mismatch'] > thresholds['mismatch']:
                        if (row['mismatch_ratio'] < self.config['mismatch_ratio_threshold'] and 
                            row['failed_count'] == 0):
                            return 'Not Fraud'
                        else:
                            return 'Fraud'
                    else:
                        return 'Not Fraud'
            else:
                if (row['failed_count'] > thresholds['failed_count'] and 
                    row['mismatch'] > thresholds['mismatch']):
                    return 'Fraud'
                else:
                    return 'Not Fraud'
        else:
            if row['amount'] > thresholds['amount']:
                if (row['failed_count'] > thresholds['failed_count'] or 
                    row['mismatch'] > thresholds['mismatch']):
                    return 'Fraud'
                else:
                    return 'Not Fraud'
            else:
                return 'Not Fraud'
    
    def _detect_anomaly2(self, row):
        """Failure pattern based fraud detection"""
        if row['failed_count'] > self.config['high_failed_multiplier'] * row['avg_failed']:
            if row['fail_ratio'] > self.config['high_fail_ratio_threshold']:
                return 'Fraud'
            else:
                if row['avg_fail_interval'] < self.config['quick_fail_interval']:
                    return 'Fraud'
                else:
                    return 'Not Fraud'
        else:
            if row['fail_ratio'] > self.config['low_fail_ratio_threshold']:
                return 'Fraud'
            else:
                return 'Not Fraud'
    
    def _detect_anomaly3(self, row):
        """Mismatch ratio based fraud detection"""
        return 'Fraud' if row['mismatch_ratio'] > self.config['extreme_mismatch_ratio'] else 'Not Fraud'
    
    def _detect_combined_anomaly(self, row):
        """Combined fraud detection strategy"""
        results = [row['label1'], row['label2'], row['label3']]
        return 'Fraud' if 'Fraud' in results else 'Not Fraud'
    
    def feature_engineering(self, df):
        """Create additional engineered features"""
        df = df.copy()
        
        epsilon = 1e-6
        df['discount_ratio'] = df['discountAmount'] / (df['amount'] + epsilon)
        df['fee_ratio'] = df['feeAmount'] / (df['amount'] + epsilon)
        df['selisih_waktu (sec)'] = (df['updatedTime'] - df['createdTime']).dt.total_seconds()
        
        # Time-based features
        df['hour_of_day'] = df['createdTime'].dt.hour
        df['day_of_week'] = df['createdTime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Clean up temporary columns
        df = df.drop(columns=['hour_of_day', 'day_of_week', 'createdTime', 'updatedTime'], 
                    errors='ignore')
        
        return df
    
    def encode_categoricals(self, df):
        """Encode categorical variables with proper handling"""
        df = df.copy()
        
        cat_cols = ['merchantId', 'paymentSource', 'status', 'statusCode', 'currency', 'type', 'Type Token']
        
        for col in cat_cols:
            if col in df.columns:
                # Handle missing values
                df[col] = df[col].fillna('unknown')
                
                # Use existing encoder or create new one
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = set(df[col].astype(str).unique())
                    known_values = set(self.label_encoders[col].classes_)
                    
                    if not unique_values.issubset(known_values):
                        # Add new categories to encoder
                        new_classes = list(known_values.union(unique_values))
                        self.label_encoders[col].classes_ = np.array(new_classes)
                    
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def preprocess_for_prediction(self, df):
        """Main preprocessing pipeline"""
        df = self.clean_data(df)
        df = self.generate_label_features(df)
        df = self.feature_engineering(df)
        df = self.encode_categoricals(df)
        
        return df
    
    def get_feature_names(self, df):
        """Get list of feature names after preprocessing"""
        return list(df.columns)
    
    def save_encoders(self, filepath):
        """Save label encoders for later use"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoders, f)
    
    def load_encoders(self, filepath):
        """Load previously saved label encoders"""
        import pickle
        with open(filepath, 'rb') as f:
            self.label_encoders = pickle.load(f)


# Usage example
def main():
    # Initialize preprocessor with custom config
    config = {
        'quantile_threshold': 0.95,
        'mismatch_ratio_threshold': 0.01,
        'high_fail_ratio_threshold': 0.7,
        'low_fail_ratio_threshold': 0.4,
        'quick_fail_interval': 60,
        'extreme_mismatch_ratio': 1.0,
        'high_failed_multiplier': 2
    }
    
    preprocessor = FraudDetectionPreprocessor(config)
    
    # Load and preprocess data
    # df = pd.read_csv('your_data.csv')
    # processed_df = preprocessor.preprocess_for_prediction(df)
    
    # Save encoders for later use
    # preprocessor.save_encoders('label_encoders.pkl')
    
    print("Preprocessing pipeline ready!")

if __name__ == "__main__":
    main()
