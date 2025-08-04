import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import modules (pastikan semua file ada di folder yang sama)
try:
    from preprocessing_pipeline import PreprocessingPipeline, preprocess_for_prediction
    from normalize import normalize_data_with_existing_preprocessing
    from resampling import apply_resampling_method, get_available_methods, compare_resampling_methods
    from elm_model import train_elm_manual, optimize_elm_auto, get_activation_options
    from predict_pipeline import run_complete_pipeline, validate_pipeline_results
    from lime_explainer import explain_test_instance, explain_custom_instance
    from sklearn.model_selection import train_test_split
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.error("Pastikan semua file module ada di folder yang sama dengan dashboard ini")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 1rem 0;
    }
    .metrics-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'preprocessing_results' not in st.session_state:
        st.session_state.preprocessing_results = None
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1

# Header
def show_header():
    st.markdown('<h1 class="main-header">🔍 Fraud Detection System Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")

# Sidebar navigation
def show_sidebar():
    st.sidebar.title("📊 Navigation")
    
    steps = {
        1: "📤 Upload Data",
        2: "🔧 Preprocessing", 
        3: "⚖️ Resampling Analysis",
        4: "🧠 Model Training",
        5: "📈 Evaluation Results",
        6: "🔬 LIME Interpretation"
    }
    
    current_step = st.sidebar.radio(
        "Select Step:",
        options=list(steps.keys()),
        format_func=lambda x: steps[x],
        index=st.session_state.current_step - 1
    )
    
    st.session_state.current_step = current_step
    
    # Show progress
    st.sidebar.markdown("### 📋 Progress")
    progress = current_step / len(steps)
    st.sidebar.progress(progress)
    st.sidebar.caption(f"Step {current_step} of {len(steps)}")
    
    return current_step

# Step 1: Upload Data
def step_upload_data():
    st.markdown('<h2 class="step-header">📤 Step 1: Upload Data</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload your transaction data in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Preview data
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Basic info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                with col4:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with col2:
        st.info("""
        ### 📝 Data Requirements
        Your CSV file should contain:
        - Transaction amounts
        - Timestamps (createdTime, updatedTime)
        - Merchant information
        - Payment details
        - Settlement information
        
        ### 🔍 Sample Columns
        - amount, inquiryAmount
        - settlementAmount, feeAmount
        - merchantId, paymentSource
        - status, statusCode
        - createdTime, updatedTime
        """)
    
    # Next step button
    if st.session_state.uploaded_data is not None:
        if st.button("➡️ Proceed to Preprocessing", type="primary"):
            st.session_state.current_step = 2
            st.rerun()

# Step 2: Preprocessing
def step_preprocessing():
    st.markdown('<h2 class="step-header">🔧 Step 2: Data Preprocessing</h2>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.uploaded_data
    
    # Preprocessing configuration
    with st.expander("⚙️ Preprocessing Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            outlier_threshold = st.slider("Outlier Detection Threshold", 0.90, 0.99, 0.95, 0.01)
            keep_intermediate = st.checkbox("Keep Intermediate Columns", value=False)
        with col2:
            fraud_rules_config = st.checkbox("Custom Fraud Rules", value=False)
            if fraud_rules_config:
                st.info("Using default fraud detection rules")
    
    # Run preprocessing
    if st.button("🚀 Run Preprocessing", type="primary"):
        
        config = {
            'outlier_threshold': outlier_threshold,
            'keep_intermediate_columns': keep_intermediate
        }
        
        with st.spinner("Processing data..."):
            try:
                processed_df, preprocessing_results = preprocess_for_prediction(df, config=config)
                
                st.session_state.processed_data = processed_df
                st.session_state.preprocessing_results = preprocessing_results
                
                st.success("✅ Preprocessing completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {e}")
                return
    
    # Show results if available
    if st.session_state.preprocessing_results is not None:
        results = st.session_state.preprocessing_results
        
        # Summary metrics
        st.subheader("📊 Preprocessing Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Features", results['summary']['original_shape'][1])
        with col2:
            st.metric("Final Features", results['summary']['final_shape'][1])
        with col3:
            st.metric("Features Added", results['summary']['total_features_created'])
        with col4:
            if 'fraud_percentage' in results['summary']:
                st.metric("Fraud Rate", f"{results['summary']['fraud_percentage']:.1f}%")
        
        # Detailed results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Missing Values", "📊 Rule-based Labels", "⚠️ Outliers", "🔧 Features"])
        
        with tab1:
            if 'missing_values' in results['steps']:
                missing_stats = results['steps']['missing_values']
                if missing_stats['missing_data_table']:
                    missing_df = pd.DataFrame(missing_stats['missing_data_table'])
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Missing values chart
                    if len(missing_df) > 0:
                        fig = px.bar(missing_df, x='Column', y='Missing_Percentage',
                                   title='Missing Values by Column')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing values found!")
        
        with tab2:
            if 'labeling' in results['steps']:
                label_stats = results['steps']['labeling']
                
                # Fraud distribution
                if 'fraud_distribution' in label_stats:
                    dist = label_stats['fraud_distribution']
                    fig = px.pie(values=list(dist.values()), names=list(dist.keys()),
                               title='Fraud vs Non-Fraud Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Rule comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rule 1 Frauds", label_stats.get('rule1_fraud_count', 0))
                    st.metric("Rule 2 Frauds", label_stats.get('rule2_fraud_count', 0))
                with col2:
                    st.metric("Rule 3 Frauds", label_stats.get('rule3_fraud_count', 0))
                    st.metric("Combined Frauds", label_stats.get('combined_fraud_count', 0))
        
        with tab3:
            if 'outliers' in results['steps']:
                outlier_stats = results['steps']['outliers']
                
                st.write("### IQR Method Results")
                if outlier_stats['methods']['iqr']:
                    iqr_data = []
                    for col, stats in outlier_stats['methods']['iqr'].items():
                        iqr_data.append({
                            'Column': col,
                            'Outliers': stats['count'],
                            'Percentage': f"{stats['percentage']:.2f}%"
                        })
                    st.dataframe(pd.DataFrame(iqr_data), use_container_width=True)
                
                st.write("### Z-Score Method Results")
                if outlier_stats['methods']['zscore']:
                    zscore_data = []
                    for col, stats in outlier_stats['methods']['zscore'].items():
                        zscore_data.append({
                            'Column': col,
                            'Outliers': stats['count'],
                            'Percentage': f"{stats['percentage']:.2f}%"
                        })
                    st.dataframe(pd.DataFrame(zscore_data), use_container_width=True)
        
        with tab4:
            if 'feature_engineering' in results['steps']:
                feat_stats = results['steps']['feature_engineering']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Features Created:**")
                    for feature in feat_stats['features_created']:
                        st.write(f"• {feature}")
                
                with col2:
                    st.write("**Feature Categories:**")
                    st.write(f"• Ratio features: {len(feat_stats['ratio_features'])}")
                    st.write(f"• Time features: {len(feat_stats['time_features'])}")
                    st.write(f"• Cyclical features: {len(feat_stats['cyclical_features'])}")
        
        # Data preview
        st.subheader("📋 Processed Data Preview")
        st.dataframe(st.session_state.processed_data.head(), use_container_width=True)
        
        # Next step button
        if st.button("➡️ Proceed to Resampling Analysis", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

# Step 3: Resampling Analysis
def step_resampling():
    st.markdown('<h2 class="step-header">⚖️ Step 3: Resampling Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please complete preprocessing first!")
        return
    
    df = st.session_state.processed_data
    
    # Check if fraud column exists
    if 'fraud' not in df.columns:
        st.error("❌ 'fraud' column not found in processed data!")
        return
    
    # Show current class distribution
    fraud_dist = df['fraud'].value_counts()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Fraud Cases", fraud_dist.get('Fraud', 0))
    with col3:
        fraud_pct = (fraud_dist.get('Fraud', 0) / len(df)) * 100
        st.metric("Fraud Rate", f"{fraud_pct:.1f}%")
    
    # Current distribution chart
    fig = px.pie(values=fraud_dist.values, names=fraud_dist.index,
                title='Current Class Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Resampling method selection
    st.subheader("🔧 Select Resampling Method")
    
    available_methods = get_available_methods()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_method = st.selectbox(
            "Choose resampling method:",
            options=list(available_methods.keys()),
            index=0
        )
    
    with col2:
        method_info = available_methods[selected_method]
        st.info(f"""
        **{selected_method}** ({method_info['type']})
        
        {method_info['description']}
        
        **Best for:** {method_info['best_for']}
        """)
    
    # Compare all methods option
    if st.checkbox("🔍 Compare All Methods", value=False):
        if st.button("🚀 Compare All Resampling Methods"):
            with st.spinner("Comparing resampling methods..."):
                try:
                    # Prepare data
                    from resampling import prepare_data_for_resampling
                    X, y = prepare_data_for_resampling(df, 'fraud')
                    
                    # Normalize data first
                    normalized_df, feature_names = normalize_data_with_existing_preprocessing(
                        pd.concat([X, y], axis=1), st.session_state.preprocessing_results
                    )
                    X_norm = normalized_df.drop(columns=['fraud'])
                    y_norm = normalized_df['fraud']
                    
                    results, comparison_stats = compare_resampling_methods(X_norm, y_norm)
                    
                    # Show comparison results
                    st.subheader("📊 Method Comparison Results")
                    
                    if comparison_stats['comparison_table']:
                        comparison_df = pd.DataFrame([item for item in comparison_stats['comparison_table'] 
                                                    if 'Error' not in item])
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Visualizations
                        if len(comparison_df) > 0:
                            # Size comparison
                            fig1 = px.bar(comparison_df, x='Method', 
                                        y=['Original_Size', 'Resampled_Size'],
                                        title='Dataset Size Comparison',
                                        barmode='group')
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            # Fraud percentage comparison
                            fig2 = px.bar(comparison_df, x='Method', 
                                        y='Final_Fraud_Percentage',
                                        title='Final Fraud Percentage by Method',
                                        color='Type')
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    # Recommendations
                    if 'recommendations' in comparison_stats:
                        rec = comparison_stats['recommendations']
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'most_balanced' in rec:
                                st.success(f"""
                                **Most Balanced:** {rec['most_balanced']['method']}
                                
                                Fraud Rate: {rec['most_balanced']['fraud_percentage']:.1f}%
                                
                                {rec['most_balanced']['reason']}
                                """)
                        
                        with col2:
                            if 'least_data_change' in rec:
                                st.info(f"""
                                **Least Change:** {rec['least_data_change']['method']}
                                
                                Change: {rec['least_data_change']['change_percentage']:.1f}%
                                
                                {rec['least_data_change']['reason']}
                                """)
                
                except Exception as e:
                    st.error(f"❌ Comparison failed: {e}")
    
    # Store selected method and proceed
    if 'selected_resampling_method' not in st.session_state:
        st.session_state.selected_resampling_method = selected_method
    
    st.session_state.selected_resampling_method = selected_method
    
    if st.button("➡️ Proceed to Model Training", type="primary"):
        st.session_state.current_step = 4
        st.rerun()

# Step 4: Model Training
def step_model_training():
    st.markdown('<h2 class="step-header">🧠 Step 4: ELM Model Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please complete preprocessing first!")
        return
    
    # Training mode selection
    col1, col2 = st.columns(2)
    
    with col1:
        training_mode = st.radio(
            "🎯 Training Mode:",
            options=['manual', 'auto'],
            format_func=lambda x: 'Manual Configuration' if x == 'manual' else 'Auto-Optimization (Optuna)',
            index=0
        )
    
    with col2:
        if training_mode == 'manual':
            st.info("""
            **Manual Mode:**
            - Set parameters manually
            - Faster training
            - Full control over configuration
            """)
        else:
            st.info("""
            **Auto-Optimization Mode:**
            - Uses Optuna for hyperparameter tuning
            - Takes longer but finds optimal parameters
            - Better performance
            """)
    
    # Configuration based on mode
    if training_mode == 'manual':
        st.subheader("⚙️ Manual Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hidden_neurons = st.slider("Hidden Neurons", 10, 200, 100, 10)
        
        with col2:
            activation_options = get_activation_options()
            activation = st.selectbox("Activation Function", 
                                    options=list(activation_options.keys()),
                                    format_func=lambda x: activation_options[x])
        
        with col3:
            threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.1)
        
        # Additional parameters
        n_trials = None
        
    else:
        st.subheader("🤖 Auto-Optimization Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_trials = st.slider("Optuna Trials", 20, 100, 50, 10)
        
        with col2:
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        # Set default values for manual parameters (not used in auto mode)
        hidden_neurons = 100
        activation = 'sigmoid'
        threshold = 0.5
    
    # Training button
    if st.button("🚀 Train Model", type="primary"):
        
        # Get selected resampling method
        resampling_method = st.session_state.get('selected_resampling_method', 'SMOTE')
        
        with st.spinner(f"Training ELM model with {resampling_method} resampling..."):
            try:
                # Run complete pipeline
                pipeline_results = run_complete_pipeline(
                    df=st.session_state.processed_data,
                    resampling_method=resampling_method,
                    training_mode=training_mode,
                    hidden_neurons=hidden_neurons,
                    activation=activation,
                    threshold=threshold,
                    n_trials=n_trials if training_mode == 'auto' else 50,
                    random_state=42
                )
                
                if pipeline_results is not None:
                    st.session_state.pipeline_results = pipeline_results
                    st.success("✅ Model training completed successfully!")
                else:
                    st.error("❌ Model training failed!")
                    return
                
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                return
    
    # Show results if available
    if st.session_state.pipeline_results is not None:
        results = st.session_state.pipeline_results
        
        st.subheader("📊 Training Results")
        
        # Model performance metrics
        metrics = results['metrics_table']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.iloc[0]['Value']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics.iloc[1]['Value']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics.iloc[2]['Value']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics.iloc[3]['Value']:.4f}")
        
        # Configuration used
        model_results = results['model_results']
        st.subheader("⚙️ Model Configuration")
        
        config_data = []
        config_data.append(["Training Mode", model_results['mode']])
        config_data.append(["Resampling Method", results['resampling_method']])
        
        if 'parameters' in model_results:
            params = model_results['parameters']
            for key, value in params.items():
                config_data.append([key.replace('_', ' ').title(), str(value)])
        
        config_df = pd.DataFrame(config_data, columns=['Parameter', 'Value'])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        # Next step button
        if st.button("➡️ View Evaluation Results", type="primary"):
            st.session_state.current_step = 5
            st.rerun()

# Step 5: Evaluation Results
def step_evaluation():
    st.markdown('<h2 class="step-header">📈 Step 5: Evaluation Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.pipeline_results is None:
        st.warning("⚠️ Please complete model training first!")
        return
    
    results = st.session_state.pipeline_results
    
    # Metrics overview
    st.subheader("🎯 Performance Metrics")
    
    metrics_df = results['metrics_table']
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = metrics_df.iloc[0]['Value']
        st.metric("Accuracy", f"{accuracy:.4f}", 
                 delta=f"{(accuracy-0.5)*100:.1f}%" if accuracy > 0.5 else None)
    
    with col2:
        precision = metrics_df.iloc[1]['Value'] 
        st.metric("Precision", f"{precision:.4f}")
    
    with col3:
        recall = metrics_df.iloc[2]['Value']
        st.metric("Recall", f"{recall:.4f}")
    
    with col4:
        f1 = metrics_df.iloc[3]['Value']
        st.metric("F1-Score", f"{f1:.4f}")
    
    # Metrics visualization
    fig_metrics = px.bar(
        x=metrics_df['Metric'], 
        y=metrics_df['Value'],
        title='Model Performance Metrics',
        color=metrics_df['Value'],
        color_continuous_scale='viridis'
    )
    fig_metrics.update_layout(showlegend=False)
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
    
    confusion_matrix = results['confusion_matrix']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(confusion_matrix, use_container_width=True)
    
    with col2:
        # Confusion matrix heatmap
        fig_cm = px.imshow(
            confusion_matrix.values,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Not Fraud', 'Fraud'],
            y=['Not Fraud', 'Fraud'],
            title='Confusion Matrix Heatmap',
            color_continuous_scale='Blues',
            text_auto=True
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Classification Report
    if 'classification_report' in results['model_results']:
        st.subheader("📋 Detailed Classification Report")
        
        class_report = results['model_results']['classification_report']
        
        # Convert to dataframe for better display
        report_data = []
        for key, values in class_report.items():
            if isinstance(values, dict) and key not in ['accuracy', 'macro avg', 'weighted avg']:
                report_data.append({
                    'Class': key,
                    'Precision': values.get('precision', 0),
                    'Recall': values.get('recall', 0),
                    'F1-Score': values.get('f1-score', 0),
                    'Support': values.get('support', 0)
                })
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)
    
    # Model insights
    st.subheader("💡 Model Insights")
    
    model_results = results['model_results']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Strengths:**")
        if accuracy > 0.8:
            st.write("✅ High accuracy performance")
        if precision > 0.7:
            st.write("✅ Good precision - low false positives")
        if recall > 0.7:
            st.write("✅ Good recall - detects most frauds")
        if f1 > 0.7:
            st.write("✅ Well-balanced precision and recall")
    
    with col2:
        st.write("**Areas for Improvement:**")
        if accuracy < 0.7:
            st.write("⚠️ Consider hyperparameter tuning")
        if precision < 0.6:
            st.write("⚠️ High false positive rate")
        if recall < 0.6:
            st.write("⚠️ Missing fraud cases")
        if abs(precision - recall) > 0.2:
            st.write("⚠️ Imbalanced precision-recall trade-off")
    
    # Training configuration summary
    st.subheader("⚙️ Training Summary")
    
    summary_data = [
        ["Training Mode", model_results.get('mode', 'N/A')],
        ["Resampling Method", results.get('resampling_method', 'N/A')],
        ["Total Features", len(results['lime_data']['feature_names']) if 'lime_data' in results else 'N/A'],
    ]
    
    if 'parameters' in model_results:
        params = model_results['parameters']
        for key, value in params.items():
            summary_data.append([key.replace('_', ' ').title(), str(value)])
    
    summary_df = pd.DataFrame(summary_data, columns=['Configuration', 'Value'])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Performance comparison (if available)
    if model_results.get('mode') == 'Optuna Optimization':
        st.subheader("🎯 Optimization Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Score", f"{model_results.get('best_score', 0):.4f}")
        with col2:
            st.metric("Total Trials", model_results.get('total_trials', 0))
        
        st.info("💡 The model was automatically optimized to find the best hyperparameters.")
    
    # Next step button
    if st.button("➡️ Proceed to LIME Interpretation", type="primary"):
        st.session_state.current_step = 6
        st.rerun()

# Step 6: LIME Interpretation
def step_lime_interpretation():
    st.markdown('<h2 class="step-header">🔬 Step 6: LIME Interpretation</h2>', unsafe_allow_html=True)
    
    if st.session_state.pipeline_results is None:
        st.warning("⚠️ Please complete model training first!")
        return
    
    # Validate LIME data availability
    valid, error_msg = validate_pipeline_results(st.session_state.pipeline_results)
    if not valid:
        st.error(f"❌ LIME data not available: {error_msg}")
        return
    
    results = st.session_state.pipeline_results
    lime_data = results['lime_data']
    
    st.subheader("🎯 Model Interpretation Options")
    
    # Interpretation mode
    interpretation_mode = st.radio(
        "Choose interpretation mode:",
        options=['test_instance', 'custom_instance'],
        format_func=lambda x: 'Explain Test Instance' if x == 'test_instance' else 'Explain Custom Instance',
        index=0
    )
    
    if interpretation_mode == 'test_instance':
        # Test instance explanation
        st.subheader("🔍 Test Instance Explanation")
        
        # Instance selection
        total_test_samples = len(lime_data['X_test'])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            instance_idx = st.slider(
                "Select test instance:", 
                0, total_test_samples - 1, 0
            )
        
        with col2:
            num_features = st.slider(
                "Number of features to explain:", 
                3, min(15, len(lime_data['feature_names'])), 10
            )
        
        # Explain button
        if st.button("🔍 Generate Explanation", type="primary"):
            with st.spinner("Generating LIME explanation..."):
                try:
                    explanation_result = explain_test_instance(
                        results, instance_idx, num_features
                    )
                    
                    if explanation_result is not None:
                        show_lime_explanation(explanation_result, lime_data, instance_idx)
                    else:
                        st.error("❌ Failed to generate explanation!")
                
                except Exception as e:
                    st.error(f"❌ Explanation failed: {e}")
    
    else:
        # Custom instance explanation
        st.subheader("🛠️ Custom Instance Explanation")
        
        st.info("Create a custom transaction instance for explanation")
        
        # Feature input form
        feature_names = lime_data['feature_names']
        
        with st.form("custom_instance_form"):
            st.write("**Configure Transaction Features:**")
            
            # Create input fields for features
            custom_values = {}
            
            # Group features for better organization
            col1, col2, col3 = st.columns(3)
            
            for i, feature in enumerate(feature_names):
                col = [col1, col2, col3][i % 3]
                
                with col:
                    # Get feature statistics for default values
                    X_train = lime_data['X_train']
                    if hasattr(X_train, feature):
                        mean_val = float(X_train[feature].mean())
                        std_val = float(X_train[feature].std())
                        min_val = float(X_train[feature].min())
                        max_val = float(X_train[feature].max())
                    else:
                        mean_val = 0.5
                        min_val = 0.0
                        max_val = 1.0
                    
                    custom_values[feature] = st.number_input(
                        f"{feature}:",
                        value=mean_val,
                        min_value=min_val,
                        max_value=max_val,
                        step=(max_val - min_val) / 100,
                        key=f"custom_{feature}"
                    )
            
            num_features_custom = st.slider(
                "Number of features to explain:", 
                3, min(15, len(feature_names)), 10,
                key="custom_num_features"
            )
            
            submitted = st.form_submit_button("🔍 Explain Custom Instance", type="primary")
            
            if submitted:
                with st.spinner("Generating explanation for custom instance..."):
                    try:
                        # Create custom instance array
                        custom_instance = np.array([custom_values[f] for f in feature_names])
                        
                        explanation_result = explain_custom_instance(
                            results, custom_instance, num_features_custom
                        )
                        
                        if explanation_result is not None:
                            show_custom_lime_explanation(explanation_result, custom_values, feature_names)
                        else:
                            st.error("❌ Failed to generate explanation!")
                    
                    except Exception as e:
                        st.error(f"❌ Custom explanation failed: {e}")

def show_lime_explanation(explanation_result, lime_data, instance_idx):
    """Display LIME explanation results for test instance"""
    
    # Prediction overview
    st.subheader("🎯 Prediction Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Instance Index", instance_idx)
    
    with col2:
        pred_class = explanation_result['predicted_class']
        st.metric("Predicted Class", pred_class)
    
    with col3:
        actual_class = explanation_result['actual_class']
        color = "normal" if pred_class == actual_class else "inverse"
        st.metric("Actual Class", actual_class)
    
    with col4:
        confidence = explanation_result['confidence']
        st.metric("Confidence", f"{confidence:.3f}")
    
    # Prediction correctness indicator
    if pred_class == actual_class:
        st.success("✅ Correct Prediction!")
    else:
        st.error("❌ Incorrect Prediction!")
    
    # Probability breakdown
    prob_fraud = explanation_result['prediction_proba'][1]
    prob_not_fraud = explanation_result['prediction_proba'][0]
    
    st.subheader("📊 Prediction Probabilities")
    
    prob_df = pd.DataFrame({
        'Class': ['Not Fraud', 'Fraud'],
        'Probability': [prob_not_fraud, prob_fraud]
    })
    
    fig_prob = px.bar(
        prob_df, x='Class', y='Probability',
        title='Class Probabilities',
        color='Probability',
        color_continuous_scale='RdYlBu_r'
    )
    fig_prob.update_layout(showlegend=False)
    st.plotly_chart(fig_prob, use_container_width=True)
    
    # Feature explanations
    st.subheader("🔍 Feature Explanations")
    
    explanation_df = explanation_result['explanation_data']['explanation_df']
    
    # Feature importance chart
    fig_importance = px.bar(
        explanation_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for This Prediction',
        color='Importance',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    fig_importance.update_layout(height=max(400, len(explanation_df) * 30))
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Detailed explanation table
    st.subheader("📋 Detailed Feature Analysis")
    
    # Add feature values to explanation
    if 'instance_data' in explanation_result and explanation_result['instance_data'] is not None:
        instance_data = explanation_result['instance_data']
        explanation_df_detailed = explanation_df.copy()
        explanation_df_detailed['Feature_Value'] = explanation_df_detailed['Feature'].apply(
            lambda x: f"{instance_data.get(x, 'N/A'):.4f}" if x in instance_data else 'N/A'
        )
        
        # Reorder columns
        explanation_df_detailed = explanation_df_detailed[['Feature', 'Feature_Value', 'Importance', 'Impact']]
        st.dataframe(explanation_df_detailed, use_container_width=True, hide_index=True)
    else:
        st.dataframe(explanation_df, use_container_width=True, hide_index=True)
    
    # Explanation insights
    st.subheader("💡 Interpretation Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Features Increasing Fraud Risk:**")
        increasing_features = explanation_df[explanation_df['Importance'] > 0]
        if len(increasing_features) > 0:
            for _, row in increasing_features.head(3).iterrows():
                st.write(f"• **{row['Feature']}**: +{row['Importance']:.3f}")
        else:
            st.write("No features strongly indicate fraud for this instance.")
    
    with col2:
        st.write("**Features Decreasing Fraud Risk:**")
        decreasing_features = explanation_df[explanation_df['Importance'] < 0]
        if len(decreasing_features) > 0:
            for _, row in decreasing_features.head(3).iterrows():
                st.write(f"• **{row['Feature']}**: {row['Importance']:.3f}")
        else:
            st.write("No features strongly indicate legitimate transaction.")

def show_custom_lime_explanation(explanation_result, custom_values, feature_names):
    """Display LIME explanation results for custom instance"""
    
    # Prediction overview
    st.subheader("🎯 Custom Instance Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pred_class = explanation_result['predicted_class']
        st.metric("Predicted Class", pred_class)
    
    with col2:
        confidence = explanation_result['confidence']
        st.metric("Confidence", f"{confidence:.3f}")
    
    # Probability breakdown
    prob_fraud = explanation_result['prediction_proba'][1]
    prob_not_fraud = explanation_result['prediction_proba'][0]
    
    prob_df = pd.DataFrame({
        'Class': ['Not Fraud', 'Fraud'],
        'Probability': [prob_not_fraud, prob_fraud]
    })
    
    fig_prob = px.bar(
        prob_df, x='Class', y='Probability',
        title='Prediction Probabilities',
        color='Probability',
        color_continuous_scale='RdYlBu_r'
    )
    st.plotly_chart(fig_prob, use_container_width=True)
    
    # Feature explanations
    st.subheader("🔍 Feature Explanations")
    
    explanation_df = explanation_result['explanation_data']['explanation_df']
    
    # Add custom values to explanation
    explanation_df_detailed = explanation_df.copy()
    explanation_df_detailed['Custom_Value'] = explanation_df_detailed['Feature'].apply(
        lambda x: f"{custom_values.get(x, 0):.4f}"
    )
    
    # Reorder columns
    explanation_df_detailed = explanation_df_detailed[['Feature', 'Custom_Value', 'Importance', 'Impact']]
    
    # Feature importance chart
    fig_importance = px.bar(
        explanation_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for Custom Instance',
        color='Importance',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    fig_importance.update_layout(height=max(400, len(explanation_df) * 30))
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Detailed explanation table
    st.dataframe(explanation_df_detailed, use_container_width=True, hide_index=True)
    
    # Custom instance summary
    st.subheader("📝 Custom Instance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Risk Factors:**")
        increasing_features = explanation_df[explanation_df['Importance'] > 0]
        if len(increasing_features) > 0:
            for _, row in increasing_features.head(3).iterrows():
                feature_val = custom_values.get(row['Feature'], 0)
                st.write(f"• **{row['Feature']}**: {feature_val:.3f} (+{row['Importance']:.3f})")
        else:
            st.write("No significant fraud indicators found.")
    
    with col2:
        st.write("**Top Protective Factors:**")
        decreasing_features = explanation_df[explanation_df['Importance'] < 0]
        if len(decreasing_features) > 0:
            for _, row in decreasing_features.head(3).iterrows():
                feature_val = custom_values.get(row['Feature'], 0)
                st.write(f"• **{row['Feature']}**: {feature_val:.3f} ({row['Importance']:.3f})")
        else:
            st.write("No significant legitimacy indicators found.")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🔍 <strong>Fraud Detection System Dashboard</strong></p>
        <p>Built with Streamlit • ELM Algorithm • LIME Interpretability</p>
        <p><em>End-to-end fraud detection with explainable AI</em></p>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    init_session_state()
    show_header()
    
    current_step = show_sidebar()
    
    # Route to appropriate step
    if current_step == 1:
        step_upload_data()
    elif current_step == 2:
        step_preprocessing()
    elif current_step == 3:
        step_resampling()
    elif current_step == 4:
        step_model_training()
    elif current_step == 5:
        step_evaluation()
    elif current_step == 6:
        step_lime_interpretation()
    
    show_footer()

if __name__ == "__main__":
    main()
