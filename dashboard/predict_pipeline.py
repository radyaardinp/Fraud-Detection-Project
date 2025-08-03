import streamlit as st
from preprocessing_pipeline import preprocess_for_prediction
from normalize import normalize_data_with_existing_preprocessing
from resampling import apply_resampling_method, prepare_data_for_resampling
from elm_model import train_elm_manual, optimize_elm_auto
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def run_complete_pipeline(df, resampling_method='SMOTE', training_mode='manual', 
                         hidden_neurons=100, activation='sigmoid', threshold=0.5, 
                         n_trials=50, random_state=42):
    try:
        # Step 1: Preprocessing
        st.info("🔄 Step 1: Preprocessing...")
        processed_df, preprocessing_results = preprocess_for_prediction(df)
        
        # Step 2: Normalize with feature names tracking
        st.info("🔢 Step 2: Normalizing...")
        normalized_df, feature_names = normalize_data_with_existing_preprocessing(
            processed_df, preprocessing_results
        )
        
        # Step 3: Resample
        st.info(f"⚖️ Step 3: Resampling with {resampling_method}...")
        X, y = prepare_data_for_resampling(normalized_df, 'fraud')
        resampled_df, *_ = apply_resampling_method(X, y, resampling_method, random_state)
        
        # Step 4: Train-test split
        st.info("🔀 Step 4: Splitting data...")
        X = resampled_df.drop(columns=['fraud'])
        y = (resampled_df['fraud'] == 'Fraud').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        
        # Step 5: Train model
        if training_mode == 'manual':
            st.info(f"🧠 Step 5: Training ELM (Manual)...")
            results = train_elm_manual(X_train.values, y_train.values, X_test.values, y_test.values,
                                     hidden_neurons, activation, threshold, random_state)
        else:
            st.info(f"🤖 Step 5: Training ELM (Auto-tuning)...")
            results = optimize_elm_auto(X_train.values, y_train.values, X_test.values, y_test.values,
                                      n_trials, random_state)
        
        # Convert results to tables
        metrics_table = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [results['metrics']['accuracy'], results['metrics']['precision'], 
                     results['metrics']['recall'], results['metrics']['f1_score']]
        })
        
        confusion_matrix = pd.DataFrame(results['confusion_matrix'], 
                                      columns=['Predicted: Not Fraud', 'Predicted: Fraud'],
                                      index=['Actual: Not Fraud', 'Actual: Fraud'])
        
        st.success("✅ Pipeline completed!")
        
        return {
            'metrics_table': metrics_table,
            'confusion_matrix': confusion_matrix,
            'model_results': results,
            # NEW: Data untuk LIME integration
            'lime_data': {
                'model_weights': results['model_weights'],  # From modified ELM
                'feature_names': feature_names,             # From normalize step
                'X_train': X_train,                         # For LIME explainer setup
                'X_test': X_test,                          # For explanation examples
                'y_test': y_test                           # For actual labels
            },
            # Additional info
            'preprocessing_results': preprocessing_results,
            'training_mode': training_mode,
            'resampling_method': resampling_method
        }
        
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        return None

# Untuk dashboard yang lebih simple
def quick_fraud_detection(df, method='SMOTE'):
    return run_complete_pipeline(df, resampling_method=method, training_mode='manual')

# Helper functions untuk LIME integration
def extract_lime_data(pipeline_results):
    if pipeline_results is None:
        return None
        
    try:
        return pipeline_results['lime_data']
    except KeyError:
        # Fallback untuk compatibility dengan old results
        return None

def validate_pipeline_results(pipeline_results):
    if pipeline_results is None:
        return False, "Pipeline results is None"
    
    if 'lime_data' not in pipeline_results:
        return False, "No LIME data in pipeline results"
    
    lime_data = pipeline_results['lime_data']
    required_keys = ['model_weights', 'feature_names', 'X_train', 'X_test', 'y_test']
    
    for key in required_keys:
        if key not in lime_data:
            return False, f"Missing {key} in LIME data"
    
    # Validate model weights structure
    model_weights = lime_data['model_weights']
    required_weights = ['input_weights', 'biases', 'output_weights', 'activation']
    
    for key in required_weights:
        if key not in model_weights:
            return False, f"Missing {key} in model weights"
        if model_weights[key] is None:
            return False, f"{key} is None in model weights"
    
    return True, "Valid"
