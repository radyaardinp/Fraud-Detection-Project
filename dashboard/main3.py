# streamlit_app.py - Refactored version yang memanfaatkan existing modules
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import existing modules - INI YANG SEHARUSNYA DIPAKAI!
from predict_pipeline import run_complete_pipeline, validate_pipeline_results
from lime_explainer import create_lime_explainer_from_pipeline, explain_test_instance, explain_custom_instance
from integration_test import test_complete_integration, print_pipeline_summary

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

def configure_app():
    """Configure Streamlit page and load CSS"""
    st.set_page_config(
        page_title="🛡️ Fraud Detection System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 1rem;
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
        .step-active { background-color: #2E86AB; color: white; }
        .step-completed { background-color: #28a745; color: white; }
        .step-pending { background-color: #e9ecef; color: #6c757d; }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'current_page': 'upload',
        'uploaded_data': None,
        'pipeline_results': None,
        'lime_explainer': None,
        'processing_complete': False,
        'selected_resampling': 'SMOTE',
        'selected_training_mode': 'manual',
        'selected_hidden_neurons': 100,
        'selected_activation': 'sigmoid',
        'selected_threshold': 0.5
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def show_step_indicator(current_step):
    """Show progress indicator"""
    steps = ["📤 Upload", "🔧 Process", "📊 Analysis", "🔍 LIME"]
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

def validate_uploaded_data(df):
    """Simple data validation"""
    warnings = []
    
    if len(df) < 100:
        warnings.append("⚠️ Dataset kecil (< 100 baris). Pertimbangkan data lebih banyak.")
    
    missing_pct = (df.isnull().sum() / len(df) * 100).max()
    if missing_pct > 50:
        warnings.append("⚠️ Ada kolom dengan > 50% missing values.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 3:
        warnings.append("⚠️ Sedikit kolom numerik. Pastikan data berisi fitur transaksi.")
    
    return True, "Data valid", warnings

def render_sidebar():
    """Render navigation sidebar"""
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        pages = {
            'upload': '📤 Upload Data',
            'process': '🔧 Process Data', 
            'analysis': '📊 Analysis',
            'explanation': '🔍 LIME Explanation'
        }
        
        for page_key, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                if can_navigate_to(page_key):
                    st.session_state.current_page = page_key
                    st.rerun()
                else:
                    st.error(f"Complete previous steps first!")
        
        st.markdown("---")
        
        # Progress
        st.markdown("### 📊 Progress")
        progress_items = [
            ("Upload", st.session_state.uploaded_data is not None),
            ("Processing", st.session_state.processing_complete),
            ("LIME Ready", st.session_state.lime_explainer is not None)
        ]
        
        for item, completed in progress_items:
            icon = "✅" if completed else "⏳"
            st.write(f"{icon} {item}")
        
        st.markdown("---")
        
        # Config summary
        st.markdown("### ⚙️ Configuration")
        st.info(f"""
        **Resampling:** {st.session_state.selected_resampling}
        **Mode:** {st.session_state.selected_training_mode}
        **Model:** ELM + LIME
        """)

def can_navigate_to(page_key):
    """Check navigation permissions"""
    if page_key == 'upload':
        return True
    elif page_key == 'process':
        return st.session_state.uploaded_data is not None
    elif page_key in ['analysis', 'explanation']:
        return st.session_state.pipeline_results is not None
    return False

# =============================================================================
# PAGE FUNCTIONS - SIMPLIFIED
# =============================================================================

def page_upload():
    """Upload page - simplified version"""
    show_step_indicator("upload")
    
    st.markdown('<div class="main-header">🛡️ Fraud Detection System</div>', 
                unsafe_allow_html=True)
    st.markdown("### 📁 Upload Transaction Data")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Rows", f"{len(df):,}")
            with col2:
                st.metric("📋 Columns", f"{df.shape[1]}")
            with col3:
                st.metric("💾 Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
            
            # Preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validation - delegate to utility
            is_valid, message, warnings = validate_uploaded_data(df)
            
            if is_valid:
                st.session_state.uploaded_data = df
                
                if warnings:
                    for warning in warnings:
                        st.warning(warning)
                
                # Configuration
                st.markdown("### ⚙️ Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.session_state.selected_resampling = st.selectbox(
                        "Resampling Method",
                        ['SMOTE', 'ADASYN', 'ENN', 'TomekLinks', 'SMOTEENN', 'SMOTETomek'],
                        help="Advanced resampling for imbalanced data"
                    )
                
                with col2:
                    st.session_state.selected_training_mode = st.selectbox(
                        "Training Mode", 
                        ['manual', 'optuna'],
                        help="Manual parameters or auto-optimization"
                    )
                
                # Manual parameters (if needed)
                if st.session_state.selected_training_mode == 'manual':
                    with st.expander("Manual Parameters"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.session_state.selected_hidden_neurons = st.slider("Hidden Neurons", 50, 500, 100)
                        with col2:
                            st.session_state.selected_activation = st.selectbox("Activation", ['sigmoid', 'tanh', 'relu'])
                        with col3:
                            st.session_state.selected_threshold = st.slider("Threshold", 0.1, 0.9, 0.5)
                
                # Start button
                if st.button("🚀 Start Processing", use_container_width=True):
                    st.session_state.current_page = 'process'
                    st.rerun()
            else:
                st.error(message)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def page_process():
    """Processing page - delegates to existing pipeline"""
    show_step_indicator("process")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Upload"):
            st.session_state.current_page = 'upload'
            st.rerun()
    with col2:
        st.markdown('<div class="main-header">🔧 Processing Pipeline</div>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None:
        st.error("No data found. Please upload data first.")
        return
    
    if not st.session_state.processing_complete:
        st.markdown("### 🔄 Ready to Process")
        
        st.info(f"""
        **Configuration:**
        - Dataset: {len(st.session_state.uploaded_data):,} rows, {st.session_state.uploaded_data.shape[1]} columns
        - Resampling: {st.session_state.selected_resampling}
        - Training: {st.session_state.selected_training_mode}
        """)
        
        if st.button("🚀 Run Complete Pipeline", use_container_width=True):
            run_processing_pipeline()
    else:
        show_processing_results()

def run_processing_pipeline():
    """Execute complete pipeline using existing modules"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Prepare parameters
        pipeline_params = {
            'df': st.session_state.uploaded_data,
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
        
        # Processing steps
        status_text.text("Step 1/4: Running complete pipeline...")
        progress_bar.progress(0.25)
        
        # Use existing pipeline function - NO DUPLICATION!
        results = run_complete_pipeline(**pipeline_params)
        
        status_text.text("Step 2/4: Validating results...")
        progress_bar.progress(0.5)
        
        # Use existing validation function
        is_valid, error_msg = validate_pipeline_results(results)
        if not is_valid:
            st.error(f"Pipeline validation failed: {error_msg}")
            return
        
        status_text.text("Step 3/4: Creating LIME explainer...")
        progress_bar.progress(0.75)
        
        # Use existing LIME function
        lime_explainer = create_lime_explainer_from_pipeline(results)
        if lime_explainer is None:
            st.error("Failed to create LIME explainer!")
            return
        
        status_text.text("Step 4/4: Finalizing...")
        progress_bar.progress(1.0)
        
        # Store results
        st.session_state.pipeline_results = results
        st.session_state.lime_explainer = lime_explainer
        st.session_state.processing_complete = True
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Processing completed successfully!")
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Processing error: {str(e)}")
        
        # Recovery options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Try SMOTE"):
                st.session_state.selected_resampling = 'SMOTE'
                st.rerun()
        with col2:
            if st.button("🔧 Manual Mode"):
                st.session_state.selected_training_mode = 'manual'
                st.rerun()

def show_processing_results():
    """Show pipeline results using existing data"""
    st.markdown("### 🎉 Processing Results")
    
    results = st.session_state.pipeline_results
    metrics = results['model_results']['metrics']
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("🔍 Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("📈 Recall", f"{metrics['recall']:.4f}")
    with col4:
        st.metric("⚖️ F1-Score", f"{metrics['f1_score']:.4f}")
    
    # Dataset info
    lime_data = results['lime_data']
    st.info(f"""
    **Processing Summary:**
    - Training samples: {len(lime_data['X_train']):,}
    - Test samples: {len(lime_data['X_test']):,}
    - Features used: {len(lime_data['feature_names'])}
    - Resampling: {st.session_state.selected_resampling}
    """)
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧪 Integration Test"):
            with st.spinner("Running integration test..."):
                # Use existing test function
                test_success = test_complete_integration(st.session_state.uploaded_data)
                if test_success:
                    st.success("✅ Integration test passed!")
                else:
                    st.error("❌ Integration test failed!")
    
    with col2:
        if st.button("📊 View Analysis", use_container_width=True):
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    with col3:
        if st.button("🔍 LIME Explanation", use_container_width=True):
            st.session_state.current_page = 'explanation'
            st.rerun()

def page_analysis():
    """Analysis page - focused on results visualization"""
    show_step_indicator("analysis")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Process"):
            st.session_state.current_page = 'process'
            st.rerun()
    with col2:
        st.markdown('<div class="main-header">📊 Model Analysis</div>', unsafe_allow_html=True)
    with col3:
        if st.button("LIME →"):
            st.session_state.current_page = 'explanation'
            st.rerun()
    
    if st.session_state.pipeline_results is None:
        st.error("No results found. Please run processing first.")
        return
    
    results = st.session_state.pipeline_results
    metrics = results['model_results']['metrics']
    
    # Performance overview
    st.markdown("### 🎯 Model Performance")
    
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
        st.success("🎉 Excellent performance with advanced resampling!")
    elif metrics['accuracy'] > 0.8:
        st.info("👍 Good performance achieved.")
    else:
        st.warning("⚠️ Consider trying different resampling methods.")
    
    # Tabs for detailed analysis
    tab1, tab2 = st.tabs(["📈 Performance Charts", "📊 Technical Details"])
    
    with tab1:
        # Metrics bar chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        
        fig = go.Figure(data=[
            go.Bar(name='ELM Performance', x=metric_names, y=metric_values,
                   marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ])
        
        fig.update_layout(
            title="ELM Model Performance with Advanced Resampling",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                      annotation_text="Good Performance (0.8)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix (if available)
        if 'confusion_matrix' in results:
            cm_df = results['confusion_matrix']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_df.values,
                x=['Predicted Non-Fraud', 'Predicted Fraud'],
                y=['Actual Non-Fraud', 'Actual Fraud'],
                colorscale='Blues',
                text=cm_df.values,
                texttemplate="%{text}",
                textfont={"size": 16}
            ))
            
            fig_cm.update_layout(title="Confusion Matrix", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab2:
        # Technical details
        lime_data = results['lime_data']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset Information:**")
            st.info(f"""
            - Total Training: {len(lime_data['X_train']):,}
            - Total Testing: {len(lime_data['X_test']):,}
            - Features Used: {len(lime_data['feature_names'])}
            - Advanced Resampling: {st.session_state.selected_resampling}
            """)
        
        with col2:
            st.markdown("**Model Configuration:**")
            model_info = results['model_results']
            st.info(f"""
            - Training Mode: {model_info['mode']}
            - Model Type: ELM (Extreme Learning Machine)
            - Explainer: LIME Integration Ready
            """)
        
        # Feature list
        st.markdown("**Selected Features:**")
        feature_names = lime_data['feature_names']
        
        # Display in columns
        num_cols = 3
        cols = st.columns(num_cols)
        for i, feature in enumerate(feature_names):
            col_idx = i % num_cols
            with cols[col_idx]:
                st.write(f"**{i+1}.** `{feature}`")

def page_explanation():
    """LIME explanation page - uses existing LIME functions"""
    show_step_indicator("explanation")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Analysis"):
            st.session_state.current_page = 'analysis'
            st.rerun()
    with col2:
        st.markdown('<div class="main-header">🔍 LIME Explanations</div>', unsafe_allow_html=True)
    
    if st.session_state.lime_explainer is None:
        st.error("LIME not ready. Please complete processing first.")
        return
    
    results = st.session_state.pipeline_results
    
    st.markdown("### 🔍 AI Decision Explanations with LIME")
    st.info("LIME explains individual predictions by showing which features contributed most to the decision.")
    
    # Tabs for different explanation types
    tab1, tab2 = st.tabs(["🔍 Test Instance Explanation", "📝 Custom Instance"])
    
    with tab1:
        render_test_explanation(results)
    
    with tab2:
        render_custom_explanation(results)

def render_test_explanation(results):
    """Render test instance explanation using existing LIME function"""
    lime_data = results['lime_data']
    y_test = lime_data['y_test']
    
    # Instance selection
    col1, col2 = st.columns(2)
    with col1:
        explanation_filter = st.selectbox(
            "Filter by:",
            ["All Transactions", "Fraud Cases", "Non-Fraud Cases"]
        )
    
    with col2:
        max_display = st.slider("Max to show:", 10, 100, 50)
    
    # Get filtered indices
    if explanation_filter == "Fraud Cases":
        available_indices = [i for i in range(len(y_test)) if y_test.iloc[i] == 1][:max_display]
    elif explanation_filter == "Non-Fraud Cases":
        available_indices = [i for i in range(len(y_test)) if y_test.iloc[i] == 0][:max_display]
    else:
        available_indices = list(range(min(max_display, len(y_test))))
    
    if not available_indices:
        st.warning("No transactions match filter.")
        return
    
    selected_idx = st.selectbox(
        "Select transaction:",
        available_indices,
        format_func=lambda x: f"Transaction {x} - {'🚨 FRAUD' if y_test.iloc[x] == 1 else '✅ NON-FRAUD'}"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        num_features = st.slider("Features to explain:", 5, 20, 10)
    
    with col2:
        if st.button("🔍 Generate Explanation", use_container_width=True):
            with st.spinner("LIME analyzing transaction..."):
                # Use existing LIME function - NO DUPLICATION!
                explanation = explain_test_instance(results, selected_idx, num_features)
                
                if explanation:
                    display_lime_explanation(explanation)

def render_custom_explanation(results):
    """Render custom instance explanation"""
    lime_data = results['lime_data']
    feature_names = lime_data['feature_names']
    
    st.markdown("#### Create Custom Transaction")
    
    # Get training data statistics for reasonable defaults
    X_train = lime_data['X_train']
    if hasattr(X_train, 'describe'):
        stats = X_train.describe()
    else:
        train_df = pd.DataFrame(X_train, columns=feature_names)
        stats = train_df.describe()
    
    # Input fields
    custom_values = {}
    num_cols = 3
    cols = st.columns(num_cols)
    
    for i, feature in enumerate(feature_names):
        col_idx = i % num_cols
        with cols[col_idx]:
            if feature in stats.columns:
                mean_val = float(stats[feature]['mean'])
                std_val = float(stats[feature]['std'])
                custom_values[feature] = st.number_input(
                    feature,
                    value=mean_val,
                    step=std_val / 10,
                    format="%.4f"
                )
            else:
                custom_values[feature] = st.number_input(feature, value=0.0)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        custom_num_features = st.slider("Features to explain:", 5, 20, 10, key="custom_features")
    
    with col2:
        if st.button("🔍 Explain Custom Transaction", use_container_width=True):
            with st.spinner("LIME analyzing custom transaction..."):
                custom_instance = np.array([custom_values[f] for f in feature_names])
                
                # Use existing LIME function
                explanation = explain_custom_instance(results, custom_instance, custom_num_features)
                
                if explanation:
                    display_lime_explanation(explanation, is_custom=True)

def display_lime_explanation(explanation, is_custom=False):
    """Display LIME explanation results"""
    st.markdown("---")
    st.markdown("### 🎯 LIME Explanation Results")
    
    # Prediction summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction = explanation['predicted_class']
        icon = "🚨" if prediction == "Fraud" else "✅"
        st.metric("🤖 ELM Prediction", f"{icon} {prediction}")
    
    with col2:
        if not is_custom:
            actual = explanation['actual_class']
            icon = "🚨" if actual == "Fraud" else "✅"
            st.metric("🎯 Actual", f"{icon} {actual}")
        else:
            st.metric("📝 Type", "Custom Transaction")
    
    with col3:
        confidence = explanation['confidence']
        st.metric("📊 Confidence", f"{confidence:.1%}")
    
    # Accuracy check (for test instances)
    if not is_custom:
        is_correct = explanation['predicted_class'] == explanation['actual_class']
        if is_correct:
            st.success("✅ Correct prediction!")
        else:
            st.error("❌ Incorrect prediction")
    
    # LIME feature importance chart
    explanation_df = explanation['explanation_data']['explanation_df']
    
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
        title="LIME Feature Contributions",
        xaxis_title="Contribution Score",
        yaxis_title="Features",
        height=400 + len(explanation_df) * 20,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature details table
    st.markdown("#### 📋 Feature Contributions")
    display_df = explanation_df.copy()
    display_df['Impact'] = display_df['Importance'].apply(
        lambda x: f"{'⬆️ +' if x > 0 else '⬇️ '}{abs(x):.4f}"
    )
    
    st.dataframe(
        display_df[['Feature', 'Impact']],
        use_container_width=True
    )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    configure_app()
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Route to appropriate page
    try:
        if st.session_state.current_page == 'upload':
            page_upload()
        elif st.session_state.current_page == 'process':
            page_process()
        elif st.session_state.current_page == 'analysis':
            page_analysis()
        elif st.session_state.current_page == 'explanation':
            page_explanation()
        else:
            st.error("Page not found!")
            
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        
        # Error recovery
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏠 Go to Upload"):
                st.session_state.current_page = 'upload'
                st.rerun()
        with col2:
            if st.button("🔄 Try SMOTE"):
                st.session_state.selected_resampling = 'SMOTE'
                st.rerun()
        with col3:
            if st.button("🔄 Reset Session"):
                for key in list(st.session_state.keys()):
                    if key.startswith(('uploaded_', 'pipeline_', 'lime_', 'processing_', 'selected_')):
                        del st.session_state[key]
                init_session_state()
                st.rerun()
        
        # Debug info
        with st.expander("🔍 Debug Information"):
            st.code(f"""
            Error: {str(e)}
            Page: {st.session_state.current_page}
            Resampling: {st.session_state.selected_resampling}
            Data Available: {st.session_state.uploaded_data is not None}
            Processing Complete: {st.session_state.processing_complete}
            """)

if __name__ == "__main__":
    main()
