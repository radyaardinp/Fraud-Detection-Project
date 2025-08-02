import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple, List, Optional, Any


class PreprocessingPipeline:
    """Enhanced preprocessing pipeline with dashboard integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.stats = {}
        self.visualizations = {}
        
    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            'outlier_threshold': 0.95,
            'fraud_rules': {
                'daily_freq_threshold': 0.95,
                'amount_threshold': 0.95,
                'failed_count_threshold': 0.95,
                'mismatch_threshold': 0.95,
                'fail_ratio_high': 0.7,
                'fail_ratio_medium': 0.4,
                'mismatch_ratio_threshold': 1.0,
                'failed_multiplier': 2,
                'fail_interval_threshold': 60
            },
            'columns_to_drop': ['id', 'inquiryId', 'networkReferenceId'],
            'keep_intermediate_columns': False
        }
    
    def run_full_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run complete preprocessing pipeline with tracking"""
        try:
            results = {'steps': {}, 'visualizations': {}, 'summary': {}}
            original_shape = df.shape
            
            # Step 1: Data Cleaning
            df, cleaning_stats = self.clean_data(df)
            results['steps']['cleaning'] = cleaning_stats
            
            # Step 2: Missing Value Analysis
            df, missing_stats = self.analyze_missing_values(df)
            results['steps']['missing_values'] = missing_stats
            
            # Step 3: Rule-based Labeling
            df, labeling_stats = self.apply_rule_based_labeling(df)
            results['steps']['labeling'] = labeling_stats
            
            # Step 4: Outlier Detection
            df, outlier_stats = self.detect_outliers(df)
            results['steps']['outliers'] = outlier_stats
            
            # Step 5: Feature Engineering
            df, feature_stats = self.feature_engineering(df)
            results['steps']['feature_engineering'] = feature_stats
            
            # Step 6: Categorical Encoding
            df, encoding_stats = self.encode_categoricals(df)
            results['steps']['encoding'] = encoding_stats
            
            # Summary
            results['summary'] = {
                'original_shape': original_shape,
                'final_shape': df.shape,
                'total_features_created': df.shape[1] - original_shape[1],
                'processing_steps': 6,
                'fraud_percentage': (df['fraud'] == 'Fraud').mean() * 100 if 'fraud' in df.columns else 0
            }
            
            return df, results
            
        except Exception as e:
            raise Exception(f"Pipeline failed: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Clean and prepare basic data types with tracking"""
        original_shape = df.shape
        original_dtypes = df.dtypes.to_dict()
        
        # Drop columns
        columns_to_drop = self.config['columns_to_drop']
        existing_drops = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_drops, errors='ignore')
        
        # Handle missing values
        df = df.copy()
        missing_before = df.isnull().sum().to_dict()
        
        df['settlementAmount'] = df['settlementAmount'].fillna(0)
        df['feeAmount'] = df['feeAmount'].fillna(0)
        df['discountAmount'] = df['discountAmount'].fillna(0) if 'discountAmount' in df.columns else 0
        
        missing_after = df.isnull().sum().to_dict()
        
        # Convert datetime with tracking
        datetime_cols = ['updatedTime', 'createdTime']
        datetime_converted = []
        for col in datetime_cols:
            if col in df.columns:
                original_type = str(df[col].dtype)
                df[col] = pd.to_datetime(df[col], errors='coerce')
                datetime_converted.append(f"{col}: {original_type} → datetime64")
        
        # Convert numeric columns safely
        float_cols = ['amount', 'settlementAmount', 'feeAmount', 'discountAmount', 'inquiryAmount']
        numeric_converted = []
        for col in float_cols:
            if col in df.columns:
                original_type = str(df[col].dtype)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                numeric_converted.append(f"{col}: {original_type} → float64")
        
        # Convert categorical columns
        categorical_cols = ['merchantId', 'paymentSource', 'status', 'statusCode']
        categorical_converted = []
        for col in categorical_cols:
            if col in df.columns:
                original_type = str(df[col].dtype)
                df[col] = df[col].astype('category')
                categorical_converted.append(f"{col}: {original_type} → category")
        
        stats = {
            'original_shape': original_shape,
            'cleaned_shape': df.shape,
            'columns_dropped': existing_drops,
            'missing_values_before': missing_before,
            'missing_values_after': missing_after,
            'missing_values_filled': sum(missing_before.values()) - sum(missing_after.values()),
            'datetime_conversions': datetime_converted,
            'numeric_conversions': numeric_converted,
            'categorical_conversions': categorical_converted
        }
        
        return df, stats
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Analyze and handle missing values with visualization data"""
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df)) * 100
        
        # Create visualization data
        missing_data = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Filter only columns with missing values
        missing_data = missing_data[missing_data['Missing_Count'] > 0]
        
        stats = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': len(missing_data),
            'missing_data_table': missing_data.to_dict('records'),
            'highest_missing_column': missing_data.iloc[0]['Column'] if len(missing_data) > 0 else None,
            'highest_missing_percentage': missing_data.iloc[0]['Missing_Percentage'] if len(missing_data) > 0 else 0
        }
        
        return df, stats
    
    def calculate_daily_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily transaction metrics"""
        df = df.copy()
        df['createdDate'] = pd.to_datetime(df['createdTime']).dt.date
        df['is_declined'] = df['status'].str.lower() == 'declined'
        
        # Daily frequency calculation
        frekuensi_harian = df.groupby(['merchantId', 'createdDate']).size().reset_index(name='daily_freq')
        df = df.merge(frekuensi_harian, on=['merchantId', 'createdDate'])
        
        # Failed transactions per day
        failed_per_day = df.groupby(['merchantId', 'createdDate'])['is_declined'].sum().reset_index(name='failed_count')
        df = df.merge(failed_per_day, on=['merchantId', 'createdDate'])
        
        # Average failed per merchant
        avg_failed_per_merchant = failed_per_day.groupby('merchantId')['failed_count'].mean().reset_index(name='avg_failed')
        df = df.merge(avg_failed_per_merchant, on='merchantId', how='left')
        df['avg_failed'] = df['avg_failed'].fillna(0)
        
        # Failure ratio
        df['fail_ratio'] = df['failed_count'] / np.maximum(df['daily_freq'], 1)
        
        return df
    
    def calculate_failure_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time intervals between failed transactions"""
        df = df.copy()
        
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
        return df
    
    def calculate_thresholds(self, df: pd.DataFrame) -> Dict:
        """Calculate dynamic thresholds for fraud detection"""
        threshold_percentile = self.config['outlier_threshold']
        
        thresholds = {
            'daily_freq': df['daily_freq'].quantile(threshold_percentile),
            'amount': df['amount'].quantile(threshold_percentile),
            'failed_count': df['failed_count'].quantile(threshold_percentile),
            'mismatch': df['mismatch'].quantile(threshold_percentile)
        }
        
        return thresholds
    
    def apply_fraud_rule_1(self, df: pd.DataFrame, thresholds: Dict) -> pd.Series:
        """Fraud detection rule 1: High frequency + amount + failures"""
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
        
        return df.apply(detect_anomaly1, axis=1)
    
    def apply_fraud_rule_2(self, df: pd.DataFrame) -> pd.Series:
        """Fraud detection rule 2: Failure patterns"""
        config = self.config['fraud_rules']
        
        def detect_anomaly2(row):
            if row['failed_count'] > config['failed_multiplier'] * row['avg_failed']:
                if row['fail_ratio'] > config['fail_ratio_high']:
                    if row['avg_fail_interval'] < config['fail_interval_threshold']:
                        return 'Fraud'
                    else:
                        return 'Fraud'
                else:
                    if row['avg_fail_interval'] < config['fail_interval_threshold']:
                        return 'Fraud'
                    else:
                        return 'Not Fraud'
            else:
                if row['fail_ratio'] > config['fail_ratio_medium']:
                    return 'Fraud'
                else:
                    return 'Not Fraud'
        
        return df.apply(detect_anomaly2, axis=1)
    
    def apply_fraud_rule_3(self, df: pd.DataFrame) -> pd.Series:
        """Fraud detection rule 3: Mismatch detection"""
        threshold = self.config['fraud_rules']['mismatch_ratio_threshold']
        return df['mismatch_ratio'].apply(lambda x: 'Fraud' if x > threshold else 'Not Fraud')
    
    def apply_rule_based_labeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Apply rule-based fraud labeling with detailed tracking"""
        df = df.copy()
        
        # Calculate daily metrics
        df = self.calculate_daily_metrics(df)
        
        # Calculate additional features
        df['is_nominal_tinggi'] = df['amount'] > 8_000_000
        df['mismatch'] = abs(df['inquiryAmount'] - df['settlementAmount'])
        
        # Calculate failure intervals
        df = self.calculate_failure_intervals(df)
        
        # Mismatch ratio
        df['mismatch_ratio'] = np.where(
            df['inquiryAmount'] == 0,
            0,
            abs(df['settlementAmount'] - df['inquiryAmount']) / df['inquiryAmount']
        )
        
        # Calculate thresholds
        thresholds = self.calculate_thresholds(df)
        
        # Apply fraud rules
        df['label1'] = self.apply_fraud_rule_1(df, thresholds)
        df['label2'] = self.apply_fraud_rule_2(df)
        df['label3'] = self.apply_fraud_rule_3(df)
        
        # Combine labels
        def detect_combined_anomaly(row):
            results = [row['label1'], row['label2'], row['label3']]
            return 'Fraud' if 'Fraud' in results else 'Not Fraud'
        
        df['fraud'] = df.apply(detect_combined_anomaly, axis=1)
        
        # Generate statistics
        label_stats = {
            'rule1_fraud_count': (df['label1'] == 'Fraud').sum(),
            'rule2_fraud_count': (df['label2'] == 'Fraud').sum(),
            'rule3_fraud_count': (df['label3'] == 'Fraud').sum(),
            'combined_fraud_count': (df['fraud'] == 'Fraud').sum(),
            'total_transactions': len(df),
            'fraud_percentage': (df['fraud'] == 'Fraud').mean() * 100,
            'thresholds_used': thresholds,
            'rule_agreement': {
                'rule1_rule2': ((df['label1'] == df['label2'])).mean() * 100,
                'rule1_rule3': ((df['label1'] == df['label3'])).mean() * 100,
                'rule2_rule3': ((df['label2'] == df['label3'])).mean() * 100
            }
        }
        
        # Create visualization data for fraud distribution
        fraud_dist = df['fraud'].value_counts()
        label_stats['fraud_distribution'] = fraud_dist.to_dict()
        
        # Rule-wise comparison
        rule_comparison = pd.DataFrame({
            'Rule 1': df['label1'].value_counts(),
            'Rule 2': df['label2'].value_counts(), 
            'Rule 3': df['label3'].value_counts(),
            'Combined': df['fraud'].value_counts()
        }).fillna(0)
        label_stats['rule_comparison_table'] = rule_comparison.to_dict()
        
        # Clean up intermediate columns if not keeping them
        if not self.config['keep_intermediate_columns']:
            columns_to_drop = ['createdDate', 'daily_freq', 'is_declined', 'failed_count', 
                             'is_nominal_tinggi', 'mismatch', 'avg_failed', 'fail_ratio', 
                             'avg_fail_interval', 'mismatch_ratio', 'label1', 'label2', 'label3']
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        return df, label_stats
    
    def detect_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect outliers using multiple methods"""
        outlier_stats = {'methods': {}, 'summary': {}}
        
        # Numeric columns for outlier detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'fraud' in numeric_cols:
            numeric_cols.remove('fraud')
        
        # IQR Method
        iqr_outliers = {}
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                iqr_outliers[col] = {
                    'count': outliers_mask.sum(),
                    'percentage': (outliers_mask.sum() / len(df)) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
        
        # Z-Score Method
        zscore_outliers = {}
        for col in numeric_cols:
            if col in df.columns and df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_mask = z_scores > 3
                zscore_outliers[col] = {
                    'count': outliers_mask.sum(),
                    'percentage': (outliers_mask.sum() / len(df)) * 100
                }
        
        outlier_stats['methods'] = {
            'iqr': iqr_outliers,
            'zscore': zscore_outliers
        }
        
        outlier_stats['summary'] = {
            'total_numeric_columns': len(numeric_cols),
            'columns_analyzed': numeric_cols,
            'outlier_detection_methods': ['IQR', 'Z-Score']
        }
        
        return df, outlier_stats
    
    def feature_engineering(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Create additional engineered features with tracking"""
        original_columns = df.columns.tolist()
        df = df.copy()
        epsilon = 1e-6
        
        # Ensure required columns exist
        if 'discountAmount' not in df.columns:
            df['discountAmount'] = 0
        if 'feeAmount' not in df.columns:
            df['feeAmount'] = 0
        
        # Create ratio features
        df['discount_ratio'] = df['discountAmount'] / (df['amount'] + epsilon)
        df['fee_ratio'] = df['feeAmount'] / (df['amount'] + epsilon)
        
        # Time-based features
        if 'updatedTime' in df.columns and 'createdTime' in df.columns:
            df['selisih_waktu (sec)'] = (df['updatedTime'] - df['createdTime']).dt.total_seconds()
            time_features_created = ['selisih_waktu (sec)']
        else:
            df['selisih_waktu (sec)'] = 0
            time_features_created = ['selisih_waktu (sec) (default)']
        
        # Cyclical hour encoding
        if 'createdTime' in df.columns:
            df['hour_of_day'] = df['createdTime'].dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df = df.drop(columns=['hour_of_day'], errors='ignore')
            cyclical_features = ['hour_sin', 'hour_cos']
        else:
            df['hour_sin'] = 0
            df['hour_cos'] = 0
            cyclical_features = ['hour_sin (default)', 'hour_cos (default)']
        
        # Drop datetime columns
        datetime_cols_dropped = [col for col in ['createdTime', 'updatedTime'] if col in df.columns]
        df = df.drop(columns=datetime_cols_dropped, errors='ignore')
        
        # Calculate feature statistics
        new_columns = [col for col in df.columns if col not in original_columns]
        
        feature_stats = {
            'original_feature_count': len(original_columns),
            'new_feature_count': len(new_columns),
            'features_created': new_columns,
            'ratio_features': ['discount_ratio', 'fee_ratio'],
            'time_features': time_features_created,
            'cyclical_features': cyclical_features,
            'datetime_columns_dropped': datetime_cols_dropped,
            'final_feature_count': len(df.columns)
        }
        
        return df, feature_stats
    
    def encode_categoricals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables with tracking"""
        df = df.copy()
        cat_cols = ['merchantId', 'paymentSource', 'status', 'statusCode']
        
        # Add other categorical columns if they exist
        potential_cat_cols = ['currency', 'type', 'Type Token']
        for col in potential_cat_cols:
            if col in df.columns:
                cat_cols.append(col)
        
        encoding_info = {}
        
        for col in cat_cols:
            if col in df.columns:
                original_unique = df[col].nunique()
                original_type = str(df[col].dtype)
                
                le = LabelEncoder()
                # Handle missing values
                df[col] = df[col].astype(str)
                df[col] = df[col].replace('nan', 'unknown')
                df[col] = df[col].fillna('unknown')
                
                # Store unique values before encoding
                unique_values = df[col].unique().tolist()
                
                df[col] = le.fit_transform(df[col].astype(str))
                
                encoding_info[col] = {
                    'original_type': original_type,
                    'unique_count': original_unique,
                    'unique_values_sample': unique_values[:10],  # First 10 values
                    'encoded_range': f"0 to {df[col].max()}"
                }
        
        encoding_stats = {
            'columns_encoded': list(encoding_info.keys()),
            'encoding_details': encoding_info,
            'total_categorical_features': len(encoding_info)
        }
        
        return df, encoding_stats
    
    def get_data_overview(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive data overview for dashboard"""
        overview = {
            'basic_info': {
                'shape': df.shape,
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                'data_types': df.dtypes.value_counts().to_dict()
            },
            'missing_values': {
                'total_missing': df.isnull().sum().sum(),
                'columns_with_missing': (df.isnull().sum() > 0).sum(),
                'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            },
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {col: df[col].nunique() for col in df.select_dtypes(include=['object', 'category']).columns}
        }
        
        return overview


# Utility functions for Streamlit integration
def create_preprocessing_visualizations(stats: Dict) -> Dict:
    """Create visualization objects for Streamlit dashboard"""
    visualizations = {}
    
    # Missing values chart
    if 'missing_values' in stats['steps'] and stats['steps']['missing_values']['missing_data_table']:
        missing_data = pd.DataFrame(stats['steps']['missing_values']['missing_data_table'])
        fig_missing = px.bar(
            missing_data, 
            x='Column', 
            y='Missing_Percentage',
            title='Missing Values by Column',
            labels={'Missing_Percentage': 'Missing %'}
        )
        visualizations['missing_values'] = fig_missing
    
    # Fraud distribution chart
    if 'labeling' in stats['steps'] and 'fraud_distribution' in stats['steps']['labeling']:
        fraud_dist = stats['steps']['labeling']['fraud_distribution']
        fig_fraud = px.pie(
            values=list(fraud_dist.values()),
            names=list(fraud_dist.keys()),
            title='Fraud vs Not Fraud Distribution'
        )
        visualizations['fraud_distribution'] = fig_fraud
    
    # Rule comparison chart
    if 'labeling' in stats['steps'] and 'rule_comparison_table' in stats['steps']['labeling']:
        rule_comp = pd.DataFrame(stats['steps']['labeling']['rule_comparison_table'])
        fig_rules = px.bar(
            rule_comp.reset_index(),
            x='index',
            y=['Rule 1', 'Rule 2', 'Rule 3', 'Combined'],
            title='Fraud Detection by Rules',
            barmode='group'
        )
        visualizations['rule_comparison'] = fig_rules
    
    return visualizations


def preprocess_for_prediction(df: pd.DataFrame, selected_features: Optional[List[str]] = None, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """Main preprocessing function for Streamlit integration"""
    pipeline = PreprocessingPipeline(config)
    
    try:
        processed_df, results = pipeline.run_full_pipeline(df)
        
        # Apply feature selection if provided
        if selected_features is not None:
            missing = [f for f in selected_features if f not in processed_df.columns]
            if missing:
                results['feature_selection'] = {
                    'requested_features': len(selected_features),
                    'available_features': len([f for f in selected_features if f in processed_df.columns]),
                    'missing_features': missing
                }
            processed_df = processed_df.reindex(columns=selected_features, fill_value=0)
        
        # Create visualizations
        results['visualizations'] = create_preprocessing_visualizations(results)
        
        return processed_df, results
        
    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")


# Legacy function for backward compatibility
def preprocess_for_prediction_legacy(df, selected_features=None):
    """Legacy function - kept for backward compatibility"""
    processed_df, _ = preprocess_for_prediction(df, selected_features)
    return processed_df
