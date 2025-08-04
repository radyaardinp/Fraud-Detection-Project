# main_app.py - Entry point yang simpel
import streamlit as st
from config.app_config import configure_page
from utils.session_manager import init_session_state
from pages import upload_page, process_page, analysis_page, explanation_page
from components.sidebar import render_sidebar
from components.common import show_error_boundary

def main():
    """Main application entry point"""
    configure_page()
    init_session_state()
    
    # Render sidebar navigation
    render_sidebar()
    
    # Route to appropriate page with error boundary
    try:
        page_router = {
            'upload': upload_page.render,
            'process': process_page.render,
            'analysis': analysis_page.render,
            'explanation': explanation_page.render
        }
        
        current_page = st.session_state.get('current_page', 'upload')
        if current_page in page_router:
            page_router[current_page]()
        else:
            st.error("Page not found!")
            
    except Exception as e:
        show_error_boundary(e)

if __name__ == "__main__":
    main()

# ---

# pages/upload_page.py - Hanya fokus upload & preview
import streamlit as st
import pandas as pd
from components.common import show_step_indicator, show_data_preview
from utils.data_validator import validate_uploaded_data

def render():
    """Render upload page"""
    show_step_indicator("upload")
    
    st.markdown('<div class="main-header">🛡️ Fraud Detection System</div>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Delegate validation to utility
            is_valid, message, warnings = validate_uploaded_data(df)
            
            if is_valid:
                st.session_state.uploaded_data = df
                show_data_preview(df)  # Component handles preview
                
                if warnings:
                    for warning in warnings:
                        st.warning(warning)
                
                # Simple configuration
                _render_config_section()
                
                if st.button("🚀 Start Processing", use_container_width=True):
                    st.session_state.current_page = 'process'
                    st.rerun()
            else:
                st.error(message)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def _render_config_section():
    """Render configuration section"""
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.selected_resampling = st.selectbox(
            "Resampling Method",
            ['SMOTE', 'ADASYN', 'ENN', 'TomekLinks'],
            help="Advanced resampling for imbalanced data"
        )
    
    with col2:
        st.session_state.selected_training_mode = st.selectbox(
            "Training Mode", 
            ['manual', 'optuna'],
            help="Manual params or auto-optimization"
        )

# ---

# pages/process_page.py - Hanya orchestrate pipeline
import streamlit as st
from predict_pipeline import run_complete_pipeline, validate_pipeline_results
from lime_explainer import create_lime_explainer_from_pipeline
from components.common import show_step_indicator, show_processing_progress

def render():
    """Render processing page"""
    show_step_indicator("process")
    
    if st.session_state.uploaded_data is None:
        st.error("No data found. Please upload data first.")
        return
    
    st.markdown("### 🔧 Processing Pipeline")
    
    if not st.session_state.processing_complete:
        if st.button("🚀 Run Pipeline", use_container_width=True):
            _run_pipeline()
    else:
        _show_pipeline_results()

def _run_pipeline():
    """Execute the complete pipeline"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Use existing pipeline - no duplication!
        pipeline_params = {
            'df': st.session_state.uploaded_data,
            'resampling_method': st.session_state.selected_resampling,
            'training_mode': st.session_state.selected_training_mode,
            'random_state': 42
        }
        
        # Show progress
        show_processing_progress(progress_bar, status_text)
        
        # Run pipeline (existing function)
        results = run_complete_pipeline(**pipeline_params)
        
        # Validate results (existing function)
        is_valid, error_msg = validate_pipeline_results(results)
        if not is_valid:
            st.error(f"Pipeline validation failed: {error_msg}")
            return
        
        # Create LIME explainer (existing function)
        lime_explainer = create_lime_explainer_from_pipeline(results)
        
        # Store results
        st.session_state.pipeline_results = results
        st.session_state.lime_explainer = lime_explainer
        st.session_state.processing_complete = True
        
        progress_bar.progress(1.0)
        status_text.text("✅ Pipeline completed successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def _show_pipeline_results():
    """Show pipeline results summary"""
    results = st.session_state.pipeline_results
    metrics = results['model_results']['metrics']
    
    # Simple metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 View Analysis", use_container_width=True):
            st.session_state.current_page = 'analysis'
            st.rerun()
    with col2:
        if st.button("🔍 LIME Explanation", use_container_width=True):
            st.session_state.current_page = 'explanation'
            st.rerun()

# ---

# pages/analysis_page.py - Fokus visualisasi hasil
import streamlit as st
import plotly.graph_objects as go
from components.common import show_step_indicator
from components.charts import create_confusion_matrix, create_metrics_chart

def render():
    """Render analysis page"""
    show_step_indicator("analysis")
    
    if st.session_state.pipeline_results is None:
        st.error("No results found. Please run pipeline first.")
        return
    
    results = st.session_state.pipeline_results
    
    st.markdown("### 📊 Model Analysis")
    
    # Use components for charts
    tab1, tab2 = st.tabs(["📈 Performance", "📊 Details"])
    
    with tab1:
        metrics = results['model_results']['metrics']
        
        # Delegate chart creation to components
        fig_metrics = create_metrics_chart(metrics)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        if 'confusion_matrix' in results:
            fig_cm = create_confusion_matrix(results['confusion_matrix'])
            st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab2:
        # Technical details
        lime_data = results['lime_data']
        st.info(f"""
        **Dataset Info:**
        - Training: {len(lime_data['X_train']):,} samples
        - Testing: {len(lime_data['X_test']):,} samples  
        - Features: {len(lime_data['feature_names'])}
        - Resampling: {st.session_state.selected_resampling}
        """)

# ---

# pages/explanation_page.py - Fokus LIME explanation
import streamlit as st
from lime_explainer import explain_test_instance, explain_custom_instance
from components.common import show_step_indicator
from components.lime_components import render_explanation_result, render_custom_form

def render():
    """Render LIME explanation page"""
    show_step_indicator("explanation")
    
    if st.session_state.lime_explainer is None:
        st.error("LIME not ready. Please complete processing first.")
        return
    
    st.markdown("### 🔍 LIME Explanations")
    
    tab1, tab2 = st.tabs(["🔍 Test Instances", "📝 Custom Instance"])
    
    with tab1:
        _render_test_explanation()
    
    with tab2:
        _render_custom_explanation()

def _render_test_explanation():
    """Render test instance explanation"""
    results = st.session_state.pipeline_results
    y_test = results['lime_data']['y_test']
    
    # Simple instance selection
    available_indices = list(range(min(50, len(y_test))))
    selected_idx = st.selectbox("Select transaction:", available_indices)
    
    if st.button("🔍 Explain", use_container_width=True):
        with st.spinner("Generating LIME explanation..."):
            # Use existing LIME function
            explanation = explain_test_instance(results, selected_idx, num_features=10)
            
            if explanation:
                # Delegate rendering to component
                render_explanation_result(explanation)

def _render_custom_explanation():
    """Render custom instance explanation"""
    results = st.session_state.pipeline_results
    feature_names = results['lime_data']['feature_names']
    
    # Use component for form
    custom_values = render_custom_form(feature_names, results['lime_data']['X_train'])
    
    if st.button("🔍 Explain Custom", use_container_width=True):
        with st.spinner("Generating LIME explanation..."):
            # Use existing LIME function
            custom_instance = [custom_values[f] for f in feature_names]
            explanation = explain_custom_instance(results, custom_instance, num_features=10)
            
            if explanation:
                render_explanation_result(explanation)

# ---

# components/sidebar.py - Sidebar terpisah
import streamlit as st

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
                # Validation logic
                if _can_navigate_to(page_key):
                    st.session_state.current_page = page_key
                    st.rerun()
                else:
                    st.error(f"Complete previous steps to access {page_name}")
        
        st.markdown("---")
        _render_progress()
        _render_config_summary()

def _can_navigate_to(page_key):
    """Check if navigation to page is allowed"""
    if page_key == 'upload':
        return True
    elif page_key == 'process':
        return st.session_state.uploaded_data is not None
    elif page_key in ['analysis', 'explanation']:
        return st.session_state.pipeline_results is not None
    return False

def _render_progress():
    """Show progress indicators"""
    st.markdown("### 📊 Progress")
    progress_items = [
        ("Upload", st.session_state.uploaded_data is not None),
        ("Processing", st.session_state.processing_complete),
        ("LIME Ready", st.session_state.lime_explainer is not None)
    ]
    
    for item, completed in progress_items:
        icon = "✅" if completed else "⏳"
        st.write(f"{icon} {item}")

def _render_config_summary():
    """Show current configuration"""
    st.markdown("### ⚙️ Configuration")
    st.info(f"""
    **Resampling:** {st.session_state.get('selected_resampling', 'SMOTE')}
    **Mode:** {st.session_state.get('selected_training_mode', 'manual')}
    **Model:** ELM + LIME
    """)

# ---

# config/app_config.py - Configuration terpisah
import streamlit as st

def configure_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="🛡️ Fraud Detection Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

def get_custom_css():
    """Return custom CSS styles"""
    return """
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
        }
    </style>
    """

# ---

# utils/session_manager.py - Session state management
import streamlit as st

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'current_page': 'upload',
        'uploaded_data': None,
        'pipeline_results': None,
        'lime_explainer': None,
        'processing_complete': False,
        'selected_resampling': 'SMOTE',
        'selected_training_mode': 'manual'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_session():
    """Reset session state"""
    for key in list(st.session_state.keys()):
        if key.startswith(('uploaded_', 'pipeline_', 'lime_', 'processing_', 'selected_')):
            del st.session_state[key]
    
    init_session_state()
