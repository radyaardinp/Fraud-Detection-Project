import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import dari modules yang sudah terintegrasi
from predict_pipeline import run_complete_pipeline, validate_pipeline_results
from lime_explainer import create_lime_explainer_from_pipeline, explain_test_instance, explain_custom_instance
from integration_test import test_complete_integration, print_pipeline_summary

# Advanced resampling methods only
RESAMPLING_METHODS = {
    'SMOTE': 'Synthetic Minority Oversampling Technique',
    'ADASYN': 'Adaptive Synthetic Sampling',
    'ENN': 'Edited Nearest Neighbours',
    'TomekLinks': 'Tomek Links',
    'SMOTEENN': 'SMOTE + Edited Nearest Neighbours',
    'SMOTETomek': 'SMOTE + Tomek Links'
}

# Initialize session state
def init_session_state():
    """Initialize semua session state variables"""
    defaults = {
        'current_page': 'upload',
        'uploaded_data': None,
        'pipeline_results': None,
        'lime_explainer': None,
        'processing_complete': False,
        'selected_resampling': 'SMOTE',  # Default ke SMOTE
        'selected_training_mode': 'manual',
        'selected_hidden_neurons': 100,
        'selected_activation': 'sigmoid',
        'selected_threshold': 0.5
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Page configuration
st.set_page_config(
    page_title="🛡️ Fraud Detection System Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.2rem;
        padding: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #666;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .description-text {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        line-height: 1.6;
        margin: 1rem auto;
        max-width: 800px;
        padding: 0.2rem;
    }
    
    .highlight-text {
        color: #2E86AB;
        font-weight: 600;
    }
    
    .success-box {
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(90deg, #f8d7da 0%, #f1aeb5 100%);
        border: 1px solid #dc3545;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .preprocessing-box {
        background: linear-gradient(90deg, #e2e3e5 0%, #d1ecf1 100%);
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .step-indicator {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .step {
        padding: 0.5rem 1rem;
        margin: 0 0.5rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .step-active {
        background-color: #2E86AB;
        color: white;
    }
    
    .step-completed {
        background-color: #28a745;
        color: white;
    }
    
    .step-pending {
        background-color: #e9ecef;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

def show_step_indicator(current_step):
    """Show progress indicator sesuai dengan flow diagram"""
    steps = ["📤 Upload Data", "🔧 Preprocessing", "🤖 Analisis Data", "🔍 Interpretasi LIME"]
    step_mapping = {"upload": 0, "process": 1, "analysis": 2, "explanation": 3}
    current_idx = step_mapping.get(current_step, 0)
    
    step_html = '<div class="step-indicator">'
    for i, step in enumerate(steps):
        if i < current_idx:
            step_class = "step step-completed"
        elif i == current_idx:
            step_class = "step step-active"
        else:
            step_class = "step step-pending"
        step_html += f'<div class="{step_class}">{step}</div>'
    step_html += '</div>'
    
    st.markdown(step_html, unsafe_allow_html=True)

def page_upload():
    """Page 1: Upload Data - Sesuai dengan flow diagram"""
    
    show_step_indicator("upload")
    
    # Main header
    st.markdown('<div class="main-header">🛡️ Fraud Detection System Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced AI-Powered Transaction Analysis</div>', unsafe_allow_html=True)

    # Description
    st.markdown("""
    <div class="description-text">
    Dashboard ini menggunakan Algoritma Machine Learning <span class="highlight-text">Extreme Learning Machine (ELM)</span> 
    yang telah terintegrasi dengan <span class="highlight-text">LIME (Local Interpretable Model-agnostic Explanations)</span> 
    untuk mendeteksi fraud dengan akurasi tinggi dan memberikan penjelasan yang dapat dipahami.
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown("### 📁 Upload Transaction Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your transaction data in CSV format. Supported formats: CSV with standard transaction columns."
    )

    if uploaded_file is not None:
        st.markdown("---")
        
        try:
            df = pd.read_csv(uploaded_file)
            
            # Success message with file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Rows", f"{len(df):,}")
            with col2:
                st.metric("📋 Total Columns", f"{df.shape[1]}")
            with col3:
                st.metric("💾 File Size", f"{uploaded_file.size / (1024*1024):.2f} MB")

            # Data preview - sesuai dengan "Preview Dataset" di diagram
            st.markdown("### 📋 Preview Dataset")
            st.dataframe(df.head(10), use_container_width=True)

            # Data quality assessment
            st.markdown("### 🔍 Data Quality Assessment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Missing values - sesuai dengan "Identifikasi Missing value" di diagram
                missing_data = df.isnull().sum()
                missing_pct = (missing_data / len(df)) * 100
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': missing_pct.values
                }).sort_values('Missing %', ascending=False)
                
                st.markdown("#### Identifikasi Missing Values")
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
                
                if missing_df['Missing Count'].sum() == 0:
                    st.success("✅ No missing values found!")
            
            with col2:
                # Data types
                st.markdown("#### Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(dtype_df, use_container_width=True)

            # Column information
            st.markdown("### 📊 Column Information")
            
            # Enhanced column descriptions
            enhanced_descriptions = {
                'id': 'Unique transaction identifier',
                'createdTime': 'Transaction creation timestamp',
                'updatedTime': 'Transaction update timestamp', 
                'currency': 'Transaction currency code',
                'amount': 'Transaction amount (primary field for fraud detection)',
                'inquiryAmount': 'Amount requested during inquiry phase',
                'inquiryId': 'Unique inquiry process identifier',
                'merchantId': 'Merchant identifier (key fraud indicator)',
                'type': 'Transaction type classification',
                'paymentSource': 'Payment method source (CARD/BANK/WALLET)',
                'status': 'Final transaction status',
                'statusCode': 'Numeric status code',
                'networkReferenceId': 'Payment network reference',
                'settlementAmount': 'Amount settled to merchant',
                'discountAmount': 'Discount applied to transaction',
                'feeAmount': 'Transaction processing fee',
                'typeToken': 'Tokenization type for security'
            }

            def get_enhanced_description(col_name):
                return enhanced_descriptions.get(col_name, f'Data field: {col_name}')

            col_info_enhanced = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count().values,
                'Unique Values': [df[col].nunique() for col in df.columns],
                'Description': [get_enhanced_description(col) for col in df.columns]
            })

            st.dataframe(col_info_enhanced, use_container_width=True)

            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown("### 📈 Numeric Columns Summary")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)

            # Data validation warnings
            warnings = []
            
            if len(df) < 100:
                warnings.append("⚠️ Dataset is quite small (< 100 rows). Consider using more data for better results.")
            
            if missing_df['Missing %'].max() > 50:
                warnings.append("⚠️ Some columns have > 50% missing values. Data quality may be affected.")
            
            if len(numeric_cols) < 3:
                warnings.append("⚠️ Few numeric columns detected. Ensure your data contains transaction amounts and related numeric features.")
            
            if warnings:
                st.markdown("### ⚠️ Data Quality Warnings")
                for warning in warnings:
                    st.warning(warning)

            # Analysis Configuration - hanya advanced resampling
            st.markdown("---")
            st.markdown("### ⚙️ Analysis Configuration")
            
            st.info("🚀 **Advanced Resampling Mode**: Using sophisticated resampling methods for optimal fraud detection performance.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Configuration")
                
                st.session_state.selected_resampling = st.selectbox(
                    "Resampling Method",
                    options=list(RESAMPLING_METHODS.keys()),
                    index=list(RESAMPLING_METHODS.keys()).index(st.session_state.selected_resampling),
                    format_func=lambda x: f"{x} - {RESAMPLING_METHODS[x]}",
                    help="Choose resampling method to handle imbalanced data optimally"
                )
                
                st.session_state.selected_training_mode = st.selectbox(
                    "Training Mode",
                    options=['manual', 'optuna'],
                    index=0,
                    help="Manual: Use specified parameters, Optuna: Auto-optimize parameters"
                )
            
            with col2:
                st.markdown("#### Manual Parameters")
                if st.session_state.selected_training_mode == 'manual':
                    st.session_state.selected_hidden_neurons = st.slider(
                        "Hidden Neurons",
                        min_value=50,
                        max_value=500,
                        value=100,
                        step=50,
                        help="Number of hidden neurons in ELM model"
                    )
                    
                    st.session_state.selected_activation = st.selectbox(
                        "Activation Function",
                        options=['sigmoid', 'tanh', 'relu'],
                        index=0,
                        help="Activation function for hidden layer"
                    )
                    
                    st.session_state.selected_threshold = st.slider(
                        "Classification Threshold",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.5,
                        step=0.1,
                        help="Threshold for fraud classification"
                    )
                else:
                    st.info("Optuna will automatically optimize: hidden_neurons, activation, threshold")

            # Start Preprocessing button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("🚀 Start Preprocessing", key="preprocessing_btn", use_container_width=True):
                    # Validate minimum requirements
                    if len(df) < 50:
                        st.error("❌ Dataset too small. Minimum 50 rows required.")
                        return
                    
                    # Store data and proceed
                    st.session_state.uploaded_data = df
                    st.session_state.current_page = 'process'
                    st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with standard transaction columns.")

    # Footer
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; text-align: center; color: #666; border-top: 1px solid #eee;">
        <p>🛡️ Advanced Fraud Detection System | Powered by ELM + LIME Integration</p>
        <p><small>Built with Streamlit • Machine Learning • Explainable AI</small></p>
    </div>
    """, unsafe_allow_html=True)

def page_process():
    """Page 2: Preprocessing - Sesuai dengan flow diagram preprocessing"""
    
    show_step_indicator("process")
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Upload", key="back_btn"):
            st.session_state.current_page = 'upload'
            st.rerun()
    
    with col2:
        st.markdown('<div class="main-header">🔧 Preprocessing Data</div>', unsafe_allow_html=True)
    
    # Check data availability
    if st.session_state.uploaded_data is None:
        st.error("❌ No data found. Please upload data first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return
    
    df = st.session_state.uploaded_data
    
    # Preprocessing configuration summary
    st.markdown("### ⚙️ Preprocessing Configuration")
    
    st.markdown("""
    <div class="preprocessing-box">
    <h4>🔄 Preprocessing Steps</h4>
    <p>Sistem akan menjalankan tahapan preprocessing sesuai dengan diagram:</p>
    <ol>
        <li><strong>Missing Value Handling</strong> - Identifikasi dan penanganan nilai yang hilang</li>
        <li><strong>Rule Based Labelling</strong> - Pemberian label berdasarkan aturan fraud detection</li>
        <li><strong>Outlier Detection</strong> - Identifikasi dan penanganan outlier</li>
        <li><strong>Data Visualization</strong> - Visualisasi distribusi data</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info(f"""
        **Model Configuration:**
        - Advanced Resampling: {st.session_state.selected_resampling}
        - Training Mode: {st.session_state.selected_training_mode}
        """)
        
        # Session information
        if st.session_state.uploaded_data is not None:
            st.markdown("---")
            st.markdown("### 📋 Session Info")
            st.write(f"**Rows:** {len(st.session_state.uploaded_data):,}")
            st.write(f"**Columns:** {st.session_state.uploaded_data.shape[1]}")
            
            if st.session_state.pipeline_results:
                metrics = st.session_state.pipeline_results['model_results']['metrics']
                st.write(f"**Accuracy:** {metrics['accuracy']:.3f}")
                st.write(f"**Advanced Resampling:** {st.session_state.selected_resampling}")
        
        st.markdown("---")
        
        # Control buttons
        st.markdown("### 🛠️ Controls")
        
        if st.button("🔄 Reset Session", key="reset_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        if st.button("📥 Export Session", key="export_session"):
            if st.session_state.pipeline_results:
                export_data = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'metrics': st.session_state.pipeline_results['model_results']['metrics'],
                    'configuration': {
                        'advanced_resampling': st.session_state.selected_resampling,
                        'training_mode': st.session_state.selected_training_mode,
                        'hidden_neurons': st.session_state.selected_hidden_neurons,
                        'activation': st.session_state.selected_activation,
                        'threshold': st.session_state.selected_threshold,
                        'model': 'ELM',
                        'explainer': 'LIME'
                    },
                    'data_info': {
                        'total_rows': len(st.session_state.uploaded_data),
                        'total_columns': st.session_state.uploaded_data.shape[1]
                    } if st.session_state.uploaded_data is not None else {}
                }
                
                import json
                export_json = json.dumps(export_data, indent=2)
                
                st.download_button(
                    label="💾 Download Session Data",
                    data=export_json,
                    file_name=f"fraud_detection_session_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No session data to export yet.")
        
        # Help section
        st.markdown("---")
        st.markdown("### ❓ Help")
        
        with st.expander("🚨 Common Issues"):
            st.markdown("""
            **Advanced Resampling Errors:**
            - Try SMOTE for most stable results
            - ADASYN works well for adaptive sampling
            - Check data quality for preprocessing issues
            
            **LIME Explanation Errors:**
            - Ensure preprocessing completed successfully
            - Try smaller number of features to explain
            - Check for data compatibility issues
            
            **ELM Model Issues:**
            - Adjust hidden neurons count
            - Try different activation functions
            - Use Optuna for automatic optimization
            """)
        
        with st.expander("📊 Advanced Resampling Guide"):
            st.markdown("""
            **Available Methods:**
            - **SMOTE**: Synthetic oversampling (recommended)
            - **ADASYN**: Adaptive synthetic sampling
            - **ENN**: Edited Nearest Neighbours (undersampling)
            - **TomekLinks**: Remove borderline samples
            - **SMOTEENN**: Combined over + undersampling
            - **SMOTETomek**: SMOTE + Tomek Links
            
            **Best Practices:**
            - Start with SMOTE for most datasets
            - Use ADASYN for highly imbalanced data
            - Combined methods for complex distributions
            """)
        
        with st.expander("🔍 LIME Interpretation Guide"):
            st.markdown("""
            **Understanding LIME Results:**
            - **Red bars**: Features pushing toward fraud
            - **Green bars**: Features pushing toward non-fraud
            - **Bar length**: Strength of contribution
            
            **Best Practices:**
            - Focus on top 5-10 most important features
            - Compare explanations across similar cases
            - Use custom scenarios to test edge cases
            """)
    
    # Main content area with error handling
    try:
        if st.session_state.current_page == 'upload':
            page_upload()
        elif st.session_state.current_page == 'process':
            page_process()
        elif st.session_state.current_page == 'analysis':
            page_analysis()
        elif st.session_state.current_page == 'explanation':
            page_explanation()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        
        # Error recovery options
        st.markdown("### 🔧 Error Recovery")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏠 Go to Upload", key="error_home"):
                st.session_state.current_page = 'upload'
                st.rerun()
        
        with col2:
            if st.button("🔄 Try SMOTE", key="error_smote"):
                st.session_state.selected_resampling = 'SMOTE'
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset All", key="error_reset"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Debug information
        with st.expander("🔍 Debug Information"):
            st.code(f"""
            Error Type: {type(e).__name__}
            Error Message: {str(e)}
            Current Page: {st.session_state.current_page}
            Advanced Resampling: {st.session_state.selected_resampling}
            Has Data: {st.session_state.uploaded_data is not None}
            Processing Complete: {st.session_state.processing_complete}
            """)

if __name__ == "__main__":
    main()
    
    with config_col2:
        if st.session_state.selected_training_mode == 'manual':
            st.info(f"""
            **Manual Parameters:**
            - Hidden Neurons: {st.session_state.selected_hidden_neurons}
            - Activation: {st.session_state.selected_activation}
            - Threshold: {st.session_state.selected_threshold}
            """)
        else:
            st.info("**Auto-Optimization:**\nOptuna will find best parameters")
    
    # Process button
    if not st.session_state.processing_complete:
        st.markdown("---")
        
        if st.button("🔄 Start Preprocessing & Analysis", key="process_btn", use_container_width=True):
            # Progress tracking sesuai dengan tahapan preprocessing
            progress_bar = st.progress(0)
            status_text = st.empty()
            error_container = st.empty()
            
            try:
                # Step 1: Missing Value Handling
                status_text.text("Step 1/6: Identifikasi Missing Value...")
                progress_bar.progress(0.15)
                
                # Step 2: Rule Based Labelling
                status_text.text("Step 2/6: Rule Based Labelling...")
                progress_bar.progress(0.3)
                
                # Step 3: Outlier Detection
                status_text.text("Step 3/6: Identifikasi Outlier...")
                progress_bar.progress(0.45)
                
                # Step 4: Data Visualization
                status_text.text("Step 4/6: Visualisasi Data...")
                progress_bar.progress(0.6)
                
                # Step 5: Advanced Resampling
                status_text.text("Step 5/6: Advanced Resampling...")
                progress_bar.progress(0.8)
                
                # Step 6: ELM Training
                status_text.text("Step 6/6: Training ELM Model...")
                progress_bar.progress(0.9)
                
                # Run complete pipeline dengan parameter yang dipilih
                pipeline_params = {
                    'df': df,
                    'resampling_method': st.session_state.selected_resampling,
                    'training_mode': st.session_state.selected_training_mode,
                    'random_state': 42
                }
                
                # Add manual parameters if needed
                if st.session_state.selected_training_mode == 'manual':
                    pipeline_params.update({
                        'hidden_neurons': st.session_state.selected_hidden_neurons,
                        'activation': st.session_state.selected_activation,
                        'threshold': st.session_state.selected_threshold
                    })
                
                # Run pipeline
                pipeline_results = run_complete_pipeline(**pipeline_params)
                
                # Finalize LIME Integration
                progress_bar.progress(1.0)
                status_text.text("Finalizing: Setting up LIME explainer...")
                
                if pipeline_results is None:
                    error_container.error("❌ Pipeline processing failed - no results returned!")
                    return
                
                # Validate results
                is_valid, error_msg = validate_pipeline_results(pipeline_results)
                if not is_valid:
                    error_container.error(f"❌ Pipeline validation failed: {error_msg}")
                    return
                
                # Create LIME explainer
                lime_explainer = create_lime_explainer_from_pipeline(pipeline_results)
                if lime_explainer is None:
                    error_container.error("❌ Failed to create LIME explainer!")
                    return
                
                # Store results
                st.session_state.pipeline_results = pipeline_results
                st.session_state.lime_explainer = lime_explainer
                st.session_state.processing_complete = True
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                error_container.empty()
                
                st.success("✅ Preprocessing and analysis completed successfully!")
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                error_container.error(f"❌ Processing error: {str(e)}")
                
                # Error recovery suggestions
                st.markdown("### 🔧 Troubleshooting Suggestions")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Try SMOTE", key="fallback_smote"):
                        st.session_state.selected_resampling = 'SMOTE'
                        st.rerun()
                
                with col2:
                    if st.button("📋 Show Error Details", key="show_error"):
                        st.code(str(e))
                
                return
    
    # Show results if processing is complete
    if st.session_state.processing_complete and st.session_state.pipeline_results is not None:
        st.markdown("---")
        st.markdown("### 🎉 Preprocessing Results")
        
        results = st.session_state.pipeline_results
        
        # Model Performance Metrics
        metrics = results['model_results']['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("🔍 Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("📈 Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("⚖️ F1-Score", f"{metrics['f1_score']:.4f}")
        
        # Data Processing Summary
        st.markdown("#### 📊 Data Processing Summary")
        
        lime_data = results['lime_data']
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Data Splits:**
            - Training samples: {len(lime_data['X_train']):,}
            - Test samples: {len(lime_data['X_test']):,}
            - Features selected: {len(lime_data['feature_names'])}
            """)
        
        with col2:
            st.info(f"""
            **Model Configuration:**
            - Advanced Resampling: {st.session_state.selected_resampling}
            - Training: {results['model_results']['mode']}
            - LIME Integration:  Ready
            """)
        
        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🧪 Run Integration Test", key="test_btn"):
                with st.spinner("Running integration test..."):
                    try:
                        test_success = test_complete_integration(df)
                        if test_success:
                            st.success("✅ Integration test passed!")
                        else:
                            st.error("❌ Integration test failed!")
                    except Exception as test_error:
                        st.error(f"❌ Integration test error: {str(test_error)}")
        
        with col2:
            if st.button("📊 View Analysis", key="analysis_btn", use_container_width=True):
                st.session_state.current_page = 'analysis'
                st.rerun()
        
        with col3:
            if st.button("🔍 LIME Interpretation", key="explanation_btn"):
                st.session_state.current_page = 'explanation'
                st.rerun()

def page_analysis():
    """Page 3: Analisis Data - Sesuai dengan flow diagram"""
    
    show_step_indicator("analysis")
    
    # Header with navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Preprocessing", key="back_btn"):
            st.session_state.current_page = 'process'
            st.rerun()
    
    with col2:
        st.markdown('<div class="main-header">📊 Analisis Data</div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("Next: LIME Interpretation →", key="next_btn"):
            st.session_state.current_page = 'explanation'
            st.rerun()
    
    # Check data availability
    if st.session_state.pipeline_results is None:
        st.error("❌ No analysis results found. Please run preprocessing first.")
        if st.button("Go to Preprocessing"):
            st.session_state.current_page = 'process'
            st.rerun()
        return
    
    results = st.session_state.pipeline_results
    lime_data = results['lime_data']
    
    # Analysis Overview - sesuai dengan tahap "Analisis Data" di diagram
    st.markdown("### 🎯 Model Performance Overview")
    
    metrics = results['model_results']['metrics']
    
    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_acc = (metrics['accuracy'] - 0.5) * 100
        st.metric("🎯 Accuracy", f"{metrics['accuracy']:.4f}", delta=f"{delta_acc:.1f}%")
    
    with col2:
        delta_prec = (metrics['precision'] - 0.5) * 100
        st.metric("🔍 Precision", f"{metrics['precision']:.4f}", delta=f"{delta_prec:.1f}%")
    
    with col3:
        delta_rec = (metrics['recall'] - 0.5) * 100
        st.metric("📈 Recall", f"{metrics['recall']:.4f}", delta=f"{delta_rec:.1f}%")
    
    with col4:
        delta_f1 = (metrics['f1_score'] - 0.5) * 100
        st.metric("⚖️ F1-Score", f"{metrics['f1_score']:.4f}", delta=f"{delta_f1:.1f}%")
    
    # Performance interpretation
    if metrics['accuracy'] > 0.9:
        st.success("🎉 Excellent model performance! Advanced resampling achieved high accuracy.")
    elif metrics['accuracy'] > 0.8:
        st.info("👍 Good model performance with advanced resampling methods.")
    else:
        st.warning("⚠️ Model performance could be improved. Consider trying different advanced resampling methods.")
    
    # Configuration info
    st.info(f"**Current Configuration:** Advanced Resampling: {st.session_state.selected_resampling} • {len(lime_data['feature_names'])} features • ELM Model")
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Model Analysis", "📈 Data Insights", "🔧 Technical Details"])
    
    with tab1:
        st.markdown("#### Confusion Matrix Analysis")
        
        if 'confusion_matrix' in results:
            cm_df = results['confusion_matrix']
            
            # Enhanced confusion matrix visualization
            fig = go.Figure(data=go.Heatmap(
                z=cm_df.values,
                x=['Predicted Non-Fraud', 'Predicted Fraud'],
                y=['Actual Non-Fraud', 'Actual Fraud'],
                colorscale='Blues',
                text=cm_df.values,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorbar=dict(title="Count")
            ))
            
            fig.update_layout(
                title="ELM Model Performance - Confusion Matrix",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            tn, fp, fn, tp = cm_df.values.ravel()
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **True Predictions:**
                - True Negatives (Non-Fraud -> Non-Fraud): {tn:,}
                - True Positives (Fraud -> Fraud): {tp:,}
                - **Total Correct**: {tn + tp:,}
                """)
            
            with col2:
                st.warning(f"""
                **False Predictions:**
                - False Positives (Non-Fraud -> Fraud): {fp:,}
                - False Negatives (Fraud -> Non-Fraud): {fn:,}
                - **Total Incorrect**: {fp + fn:,}
                """)
        
        # Performance metrics comparison
        st.markdown("#### 📈 Performance Metrics Breakdown")
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        
        fig = go.Figure(data=[
            go.Bar(name='ELM Performance', x=metric_names, y=metric_values,
                   marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ])
        
        fig.update_layout(
            title="ELM Model Performance Metrics with Advanced Resampling",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        # Add benchmark line
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                      annotation_text="Good Performance Threshold (0.8)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### 📊 Dataset Information")
        
        # Data distribution after advanced resampling
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Dataset Overview:**
            - Total Samples: {len(lime_data['X_train']) + len(lime_data['X_test']):,}
            - Training Samples: {len(lime_data['X_train']):,}
            - Test Samples: {len(lime_data['X_test']):,}
            - Features Used: {len(lime_data['feature_names'])}
            """)
        
        with col2:
            # Calculate class distribution in test set
            test_fraud_count = lime_data['y_test'].sum()
            test_total = len(lime_data['y_test'])
            fraud_rate = (test_fraud_count / test_total) * 100
            
            st.info(
                f"**Test Set Distribution:**\n"
                f"- Fraud Cases: {test_fraud_count:,}\n"
                f"- Non-Fraud Cases: {test_total - test_fraud_count:,}\n"
                f"- Fraud Rate: {fraud_rate:.2f}%\n"
                f"- Balance Ratio: {(test_total - test_fraud_count) / max(test_fraud_count, 1):.1f}:1"
                )
        
        # Advanced resampling impact
        st.markdown("#### 🔄 Advanced Resampling Impact")
        st.success(f"✅ **{st.session_state.selected_resampling}** successfully applied to handle class imbalance and improve model performance.")
        
        # Feature overview
        st.markdown("#### 🏷️ Selected Features")
        
        feature_names = lime_data['feature_names']
        
        # Display features in a structured way
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, feature in enumerate(feature_names):
            col_idx = i % num_cols
            with cols[col_idx]:
                st.markdown(f"**{i+1}.** `{feature}`")
        
        # Feature statistics from training data
        if hasattr(lime_data['X_train'], 'columns'):
            train_df = lime_data['X_train']
        else:
            train_df = pd.DataFrame(lime_data['X_train'], columns=feature_names)
        
        st.markdown("#### 📈 Feature Statistics (After Preprocessing)")
        st.dataframe(train_df.describe(), use_container_width=True)
        
        # Feature correlation heatmap
        if len(feature_names) <= 20:
            st.markdown("#### 🔗 Feature Correlation Matrix")
            
            corr_matrix = train_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Feature Correlation Heatmap (Processed Data)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.markdown("#### 🔧 Technical Configuration")
        
        model_results = results['model_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### ELM Model Parameters")
            if 'parameters' in model_results:
                params = model_results['parameters']
                for key, value in params.items():
                    st.write(f"**{key}:** {value}")
            else:
                st.write(f"**Mode:** {model_results['mode']}")
                st.write(f"**Advanced Resampling:** {st.session_state.selected_resampling}")
        
        with col2:
            st.markdown("##### Processing Summary")
            
            processing_info = results.get('preprocessing_results', {})
            if processing_info:
                for key, value in processing_info.items():
                    if isinstance(value, (int, float, str)):
                        st.write(f"**{key}:** {value}")
        
        # ELM Model architecture
        st.markdown("#### 🧠 ELM Model Architecture")
        
        model_weights = lime_data['model_weights']
        
        input_features = len(lime_data['feature_names'])
        hidden_neurons = model_weights['input_weights'].shape[1] if 'input_weights' in model_weights else 'Unknown'
        output_neurons = 1
        
        arch_info = f"""
        **ELM Architecture:**
        - Input Layer: {input_features} features
        - Hidden Layer: {hidden_neurons} neurons
        - Activation: {model_weights.get('activation', 'sigmoid')}
        - Output Layer: {output_neurons} neuron (fraud probability)
        - Threshold: {model_weights.get('threshold', 0.5)}
        - Advanced Resampling: {st.session_state.selected_resampling}
        """
        
        st.info(arch_info)
        
        # Pipeline summary
        st.markdown("#### 📋 Complete Pipeline Summary")
        
        if st.button("📊 Show Detailed Pipeline Summary", key="summary_btn"):
            with st.expander("Complete Pipeline Summary", expanded=True):
                summary_output = st.empty()
                
                import io
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                
                try:
                    print_pipeline_summary(results)
                    summary_text = buffer.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                summary_output.text(summary_text)

def page_explanation():
    """Page 4: Interpretasi LIME - Sesuai dengan flow diagram"""
    
    show_step_indicator("explanation")
    
    # Header with navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Analysis", key="back_btn"):
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    with col2:
        st.markdown('<div class="main-header">🔍 Interpretasi LIME</div>', unsafe_allow_html=True)
    
    # Check data availability
    if st.session_state.pipeline_results is None or st.session_state.lime_explainer is None:
        st.error("❌ No LIME explanation data found. Please run preprocessing first.")
        if st.button("Go to Preprocessing"):
            st.session_state.current_page = 'process'
            st.rerun()
        return
    
    results = st.session_state.pipeline_results
    lime_data = results['lime_data']
    
    st.markdown("""
    ### 🔍 Understanding AI Decisions with LIME
    
    LIME (Local Interpretable Model-agnostic Explanations) membantu menjelaskan keputusan individual 
    dari model ELM dengan menunjukkan fitur mana yang paling berkontribusi terhadap prediksi fraud.
    """)
    
    # Explanation tabs
    tab1, tab2 = st.tabs(["🔍 Test Instance Explanation", "📝 Custom Instance Explanation"])
    
    with tab1:
        st.markdown("#### Pilih Transaksi Test untuk Dijelaskan")
        
        X_test = lime_data['X_test']
        y_test = lime_data['y_test']
        
        # Verify LIME system
        with st.spinner("Initializing LIME explanation system..."):
            try:
                sample_explanation = explain_test_instance(results, instance_idx=0, num_features=5)
                if sample_explanation:
                    st.success("✅ LIME explanation system ready!")
                else:
                    st.error("❌ LIME system not ready. Please check the integration.")
                    return
            except Exception as e:
                st.error(f"❌ LIME initialization failed: {str(e)}")
                return
        
        # Transaction filters
        col1, col2 = st.columns(2)
        
        with col1:
            explanation_filter = st.selectbox(
                "Filter transactions by:",
                ["All Transactions", "Fraud Cases Only", "Non-Fraud Cases Only"]
            )
        
        with col2:
            max_display = st.slider("Max transactions to show:", 10, 100, 50)
        
        # Get filtered indices
        available_indices = []
        
        if explanation_filter == "Fraud Cases Only":
            available_indices = [i for i in range(len(y_test)) if y_test.iloc[i] == 1][:max_display]
        elif explanation_filter == "Non-Fraud Cases Only":
            available_indices = [i for i in range(len(y_test)) if y_test.iloc[i] == 0][:max_display]
        else:
            available_indices = list(range(min(max_display, len(y_test))))
        
        if not available_indices:
            st.warning("⚠️ No transactions match the selected filter.")
            return
        
        # Transaction selection
        selected_idx = st.selectbox(
            "Choose transaction to explain:",
            available_indices,
            format_func=lambda x: f"Transaction {x} - Actual: {'🚨 FRAUD' if y_test.iloc[x] == 1 else '✅ NON-FRAUD'}"
        )
        
        # Transaction details
        st.markdown("##### 📋 Transaction Details")
        
        selected_instance = X_test.iloc[selected_idx] if hasattr(X_test, 'iloc') else X_test[selected_idx]
        feature_names = lime_data['feature_names']
        
        # Display key features
        num_cols = 4
        cols = st.columns(num_cols)
        
        for i, feature in enumerate(feature_names[:12]):
            col_idx = i % num_cols
            with cols[col_idx]:
                if hasattr(selected_instance, 'iloc'):
                    value = selected_instance[feature]
                else:
                    value = selected_instance[i] if isinstance(selected_instance, np.ndarray) else selected_instance[feature]
                
                if isinstance(value, float):
                    st.metric(feature, f"{value:.4f}")
                else:
                    st.metric(feature, str(value))
        
        # Generate explanation
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            num_features_explain = st.slider("Number of features to explain:", 5, 20, 10)
        
        with col2:
            if st.button("🔍 Generate LIME Explanation", key="explain_btn", use_container_width=True):
                with st.spinner("🤖 LIME is analyzing the transaction..."):
                    try:
                        explanation_result = explain_test_instance(
                            results, 
                            instance_idx=selected_idx, 
                            num_features=num_features_explain
                        )
                        
                        if explanation_result is None:
                            st.error("❌ Failed to generate LIME explanation.")
                            return
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎯 LIME Explanation Results")
                        
                        # Prediction summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            prediction_class = explanation_result['predicted_class']
                            class_color = "🚨" if prediction_class == "Fraud" else "✅"
                            st.metric("🤖 ELM Prediction", f"{class_color} {prediction_class}")
                        
                        with col2:
                            actual_class = explanation_result['actual_class']
                            actual_color = "🚨" if actual_class == "Fraud" else "✅"
                            st.metric("🎯 Actual Label", f"{actual_color} {actual_class}")
                        
                        with col3:
                            confidence = explanation_result['confidence']
                            st.metric("📊 Confidence", f"{confidence:.1%}")
                        
                        # Accuracy indicator
                        is_correct = explanation_result['predicted_class'] == explanation_result['actual_class']
                        if is_correct:
                            st.success("✅ Correct ELM Prediction!")
                        else:
                            st.error("❌ Incorrect ELM Prediction")
                        
                        # LIME feature importance
                        st.markdown("#### 📊 LIME Feature Importance Analysis")
                        
                        explanation_df = explanation_result['explanation_data']['explanation_df']
                        
                        # Interactive visualization
                        fig = go.Figure()
                        
                        colors = ['red' if imp > 0 else 'green' for imp in explanation_df['Importance']]
                        
                        fig.add_trace(go.Bar(
                            y=explanation_df['Feature'],
                            x=explanation_df['Importance'],
                            orientation='h',
                            marker_color=colors,
                            text=[f"{imp:.4f}" for imp in explanation_df['Importance']],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title="LIME Feature Contributions to ELM Decision",
                            xaxis_title="LIME Contribution Score",
                            yaxis_title="Features",
                            height=400 + len(explanation_df) * 20,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed explanation table
                        st.markdown("#### 📋 Detailed LIME Feature Analysis")
                        
                        display_df = explanation_df.copy()
                        display_df['Contribution'] = display_df['Importance'].apply(
                            lambda x: f"{'⬆️' if x > 0 else '⬇️'} {abs(x):.4f}"
                        )
                        display_df['Impact Direction'] = display_df['Impact']
                        
                        st.dataframe(
                            display_df[['Feature', 'Contribution', 'Impact Direction']],
                            use_container_width=True
                        )
                        
                        # LIME interpretation summary
                        st.markdown("#### 💡 LIME Interpretation Summary")
                        
                        top_fraud_features = explanation_df[explanation_df['Importance'] > 0]['Feature'].head(3).tolist()
                        top_safe_features = explanation_df[explanation_df['Importance'] < 0]['Feature'].head(3).tolist()
                        
                        summary_text = f"""
                        **LIME Analysis of ELM Decision:**
                        
                        The ELM model predicted this transaction as **{prediction_class}** with **{confidence:.1%} confidence**.
                        LIME analysis reveals the following feature contributions:
                        
                        """
                        
                        if top_fraud_features:
                            summary_text += f"""
                            **\U0001F6A8 Features increasing fraud risk (according to LIME):**
                            {', '.join(top_fraud_features)}
                            
                            """
                        
                        if top_safe_features:
                            summary_text += f"""
                            **✅ Features decreasing fraud risk (according to LIME):**
                            {', '.join(top_safe_features)}
                            """
                        
                        st.markdown(summary_text)
                        
                        # Download LIME explanation
                        if st.button("📥 Download LIME Explanation Report"):
                            report_data = {
                                'Transaction_ID': selected_idx,
                                'ELM_Prediction': prediction_class,
                                'Actual_Label': actual_class,
                                'Confidence': confidence,
                                'Correct_Prediction': is_correct,
                                'LIME_Feature_Explanations': explanation_df.to_dict('records'),
                                'Configuration': {
                                    'Advanced_Resampling': st.session_state.selected_resampling,
                                    'Training_Mode': st.session_state.selected_training_mode,
                                    'Model': 'ELM',
                                    'Explainer': 'LIME'
                                }
                            }
                            
                            import json
                            report_json = json.dumps(report_data, indent=2)
                            
                            st.download_button(
                                label="📄 Download LIME Report (JSON)",
                                data=report_json,
                                file_name=f"lime_explanation_transaction_{selected_idx}.json",
                                mime="application/json"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ LIME explanation generation failed: {str(e)}")
                        
                        with st.expander("🔍 Debug Information"):
                            st.code(f"""
                            Error type: {type(e).__name__}
                            Error message: {str(e)}
                            Selected transaction: {selected_idx}
                            Advanced resampling: {st.session_state.selected_resampling}
                            """)
    
    with tab2:
        st.markdown("#### Create Custom Transaction for LIME Explanation")
        
        st.info("""
        💡 **Custom LIME Explanation Feature**
        
        Create a custom transaction with your own values to see how LIME would explain 
        the ELM model's decision. This helps understand model behavior on specific scenarios.
        """)
        
        feature_names = lime_data['feature_names']
        
        # Get sample statistics
        X_train = lime_data['X_train']
        if hasattr(X_train, 'describe'):
            stats = X_train.describe()
        else:
            train_df = pd.DataFrame(X_train, columns=feature_names)
            stats = train_df.describe()
        
        st.markdown("##### 📝 Enter Custom Values")
        
        # Create input fields
        custom_values = {}
        
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, feature in enumerate(feature_names):
            col_idx = i % num_cols
            
            with cols[col_idx]:
                if feature in stats.columns:
                    mean_val = float(stats[feature]['mean'])
                    min_val = float(stats[feature]['min'])
                    max_val = float(stats[feature]['max'])
                    std_val = float(stats[feature]['std'])
                    
                    lower_bound = max(min_val, mean_val - 3 * std_val)
                    upper_bound = min(max_val, mean_val + 3 * std_val)
                    
                    custom_values[feature] = st.number_input(
                        f"{feature}",
                        value=mean_val,
                        min_value=lower_bound,
                        max_value=upper_bound,
                        step=std_val / 10,
                        help=f"Range: {min_val:.2f} to {max_val:.2f}, Mean: {mean_val:.2f}"
                    )
                else:
                    custom_values[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        help="Enter a numeric value for this feature"
                    )
        
        # Quick presets
        st.markdown("##### 🎛️ Quick Presets")
        
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("💰 High Amount Scenario"):
                for feature in feature_names:
                    if 'amount' in feature.lower() and feature in stats.columns:
                        custom_values[feature] = float(stats[feature]['75%']) * 2
                st.rerun()
        
        with preset_col2:
            if st.button("🌙 Suspicious Pattern"):
                st.info("Preset values set for suspicious transaction pattern")
        
        with preset_col3:
            if st.button("🔄 Reset to Averages"):
                for feature in feature_names:
                    if feature in stats.columns:
                        custom_values[feature] = float(stats[feature]['mean'])
                st.rerun()
        
        # Generate custom explanation
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            custom_num_features = st.slider("Features to explain:", 5, 20, 10, key="custom_features")
        
        with col2:
            if st.button("🔍 Generate Custom LIME Explanation", key="custom_explain_btn", use_container_width=True):
                with st.spinner("🤖 LIME analyzing custom transaction..."):
                    try:
                        custom_instance = np.array([custom_values[feature] for feature in feature_names])
                        
                        custom_explanation = explain_custom_instance(
                            results,
                            custom_instance,
                            num_features=custom_num_features
                        )
                        
                        if custom_explanation is None:
                            st.error("❌ Failed to explain custom transaction with LIME.")
                            return
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎯 Custom Transaction LIME Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            prediction_class = custom_explanation['predicted_class']
                            class_color = "🚨" if prediction_class == "Fraud" else "✅"
                            st.metric("🤖 ELM Prediction", f"{class_color} {prediction_class}")
                        
                        with col2:
                            confidence = custom_explanation['confidence']
                            st.metric("📊 Confidence", f"{confidence:.1%}")
                        
                        # Risk assessment
                        if prediction_class == "Fraud":
                            if confidence > 0.8:
                                st.error("🚨 HIGH RISK: Strong fraud indicators detected by ELM!")
                            elif confidence > 0.6:
                                st.warning("⚠️ MEDIUM RISK: Some fraud patterns detected by ELM.")
                            else:
                                st.info("📊 LOW-MEDIUM RISK: Weak fraud indicators.")
                        else:
                            if confidence > 0.8:
                                st.success("✅ LOW RISK: Strong legitimate transaction indicators.")
                            else:
                                st.info("📊 UNCERTAIN: Mixed signals in transaction pattern.")
                        
                        # LIME feature importance
                        explanation_df = custom_explanation['explanation_data']['explanation_df']
                        
                        fig = go.Figure()
                        
                        colors = ['red' if imp > 0 else 'green' for imp in explanation_df['Importance']]
                        
                        fig.add_trace(go.Bar(
                            y=explanation_df['Feature'],
                            x=explanation_df['Importance'],
                            orientation='h',
                            marker_color=colors,
                            text=[f"{imp:.4f}" for imp in explanation_df['Importance']],
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title="Custom Transaction - LIME Feature Contributions",
                            xaxis_title="LIME Contribution Score",
                            yaxis_title="Features",
                            height=400 + len(explanation_df) * 20,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed analysis
                        st.markdown("#### 📋 LIME Feature Impact Details")
                        
                        display_df = explanation_df.copy()
                        display_df['Your Value'] = [custom_values[feature] for feature in display_df['Feature']]
                        display_df['LIME Contribution'] = display_df['Importance'].apply(
                            lambda x: f"{'📈' if x > 0 else '📉'} {abs(x):.4f}"
                        )
                        
                        st.dataframe(
                            display_df[['Feature', 'Your Value', 'LIME Contribution', 'Impact']],
                            use_container_width=True
                        )
                        
                        # LIME insights
                        st.markdown("#### 💡 LIME Analysis Insights")
                        
                        top_risk_features = explanation_df[explanation_df['Importance'] > 0].head(3)
                        top_safe_features = explanation_df[explanation_df['Importance'] < 0].head(3)
                        
                        if len(top_risk_features) > 0:
                            st.markdown("**🚨 Features increasing fraud risk (LIME analysis):**")
                            for _, row in top_risk_features.iterrows():
                                feature = row['Feature']
                                value = custom_values[feature]
                                impact = row['Importance']
                                st.write(f"- **{feature}**: {value:.4f} (LIME impact: +{impact:.4f})")
                        
                        if len(top_safe_features) > 0:
                            st.markdown("**✅ Features indicating legitimate transaction (LIME analysis):**")
                            for _, row in top_safe_features.iterrows():
                                feature = row['Feature']
                                value = custom_values[feature]
                                impact = abs(row['Importance'])
                                st.write(f"- **{feature}**: {value:.4f} (LIME impact: -{impact:.4f})")
                        
                        # Download custom LIME analysis
                        if st.button("📥 Download Custom LIME Analysis", key="download_custom"):
                            custom_report = {
                                'Analysis_Type': 'Custom_Transaction_LIME',
                                'ELM_Prediction': prediction_class,
                                'Confidence': confidence,
                                'Input_Values': custom_values,
                                'LIME_Feature_Explanations': explanation_df.to_dict('records'),
                                'Configuration': {
                                    'Advanced_Resampling': st.session_state.selected_resampling,
                                    'Training_Mode': st.session_state.selected_training_mode,
                                    'Model': 'ELM',
                                    'Explainer': 'LIME'
                                },
                                'Timestamp': pd.Timestamp.now().isoformat()
                            }
                            
                            import json
                            custom_json = json.dumps(custom_report, indent=2)
                            
                            st.download_button(
                                label="📄 Download Custom LIME Analysis",
                                data=custom_json,
                                file_name=f"custom_lime_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Custom LIME explanation failed: {str(e)}")
                        
                        with st.expander("🔍 Debug Information"):
                            st.code(f"""
                            Error type: {type(e).__name__}
                            Error message: {str(e)}
                            Advanced resampling: {st.session_state.selected_resampling}
                            Number of features: {len(feature_names)}
                            """)

# Main navigation function
def main():
    """Enhanced main navigation sesuai dengan flow diagram"""
    
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Page navigation sesuai flow diagram
        pages = {
            'upload': '📤 Upload Data',
            'process': '🔧 Preprocessing Data', 
            'analysis': '📊 Analisis Data',
            'explanation': '🔍 Interpretasi LIME'
        }
        
        for page_key, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                # Check prerequisites
                if page_key == 'process' and st.session_state.uploaded_data is None:
                    st.error("Please upload data first!")
                    continue
                elif page_key in ['analysis', 'explanation'] and st.session_state.pipeline_results is None:
                    st.error("Please complete preprocessing first!")
                    continue
                
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # Progress indicator
        st.markdown("### 📊 Progress")
        
        progress_items = [
            ("Data Upload", st.session_state.uploaded_data is not None),
            ("Preprocessing", st.session_state.processing_complete),  
            ("LIME Ready", st.session_state.lime_explainer is not None)
        ]
        
        for item, completed in progress_items:
            status = "✅" if completed else "⏳"
            st.write(f"{status} {item}")
        
        st.markdown("---")
        
        # Current configuration
        st.markdown("### ⚙️ Current Configuration")
        st.info(f"""
        **Advanced Resampling:** {st.session_state.selected_resampling}
        **Training Mode:** {st.session_state.selected_training_mode}
        **Model:** ELM (Extreme Learning Machine)
        **Explainer:** LIME
        """)
