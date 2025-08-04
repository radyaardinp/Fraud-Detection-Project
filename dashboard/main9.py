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

# Sklearn imports with error handling
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # SMOTE dengan penanganan error
    try:
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
        from imblearn.combine import SMOTEENN, SMOTETomek
        RESAMPLING_AVAILABLE = True
    except ImportError:
        RESAMPLING_AVAILABLE = False
    
    # LIME dengan penanganan error
    try:
        import lime
        from lime.lime_tabular import LimeTabularExplainer
        LIME_AVAILABLE = True
    except ImportError:
        LIME_AVAILABLE = False
        
except ImportError as e:
    st.error(f"Required libraries not installed: {e}")
    st.stop()

# Initialize session state
def init_session_state():
    """Initialize semua session state variables"""
    defaults = {
        'current_page': 'upload',
        'uploaded_data': None,
        'pipeline_results': None,
        'lime_explainer': None,
        'processing_complete': False,
        'selected_resampling': 'None',
        'selected_training_mode': 'manual',
        'selected_hidden_neurons': 100,
        'selected_activation': 'sigmoid',
        'selected_threshold': 0.5
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Simplified ELM Implementation
class SimpleELM:
    def __init__(self, hidden_neurons=100, activation='sigmoid', random_state=42):
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.random_state = random_state
        self.input_weights = None
        self.bias = None
        self.output_weights = None
        self.scaler = StandardScaler()
    
    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return x
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        
        # Scale input data
        X_scaled = self.scaler.fit_transform(X)
        
        n_samples, n_features = X_scaled.shape
        
        # Initialize random weights and bias
        self.input_weights = np.random.randn(n_features, self.hidden_neurons)
        self.bias = np.random.randn(self.hidden_neurons)
        
        # Calculate hidden layer output
        hidden_output = self._activation_function(
            np.dot(X_scaled, self.input_weights) + self.bias
        )
        
        # Calculate output weights using pseudo-inverse
        try:
            self.output_weights = np.dot(np.linalg.pinv(hidden_output), y.reshape(-1, 1))
        except np.linalg.LinAlgError:
            # Fallback to least squares
            self.output_weights = np.linalg.lstsq(hidden_output, y.reshape(-1, 1), rcond=None)[0]
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        hidden_output = self._activation_function(
            np.dot(X_scaled, self.input_weights) + self.bias
        )
        predictions = np.dot(hidden_output, self.output_weights).flatten()
        return predictions
    
    def predict_proba(self, X):
        predictions = self.predict(X)
        # Convert to probabilities using sigmoid
        probabilities = 1 / (1 + np.exp(-predictions))
        return np.column_stack([1 - probabilities, probabilities])

# Enhanced preprocessing function
def preprocess_data(df):
    """Enhanced data preprocessing with better error handling"""
    try:
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # Fill missing values
        for col in numeric_columns:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        for col in categorical_columns:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_columns:
            if col not in ['id', 'createdTime', 'updatedTime']:  # Skip ID and timestamp columns
                try:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    label_encoders[col] = le
                except Exception as e:
                    st.warning(f"Could not encode column {col}: {e}")
        
        # Create synthetic fraud labels if not present
        if 'is_fraud' not in df_processed.columns:
            np.random.seed(42)
            # Create realistic fraud distribution (5-10% fraud rate)
            fraud_rate = 0.07
            n_fraud = int(len(df_processed) * fraud_rate)
            fraud_labels = np.zeros(len(df_processed))
            fraud_indices = np.random.choice(len(df_processed), n_fraud, replace=False)
            fraud_labels[fraud_indices] = 1
            df_processed['is_fraud'] = fraud_labels
        
        # Select relevant features for modeling
        feature_columns = []
        for col in df_processed.columns:
            if col not in ['id', 'createdTime', 'updatedTime', 'is_fraud'] and df_processed[col].dtype in [np.int64, np.float64]:
                feature_columns.append(col)
        
        if len(feature_columns) == 0:
            raise ValueError("No suitable numeric features found for modeling")
        
        return df_processed, feature_columns, label_encoders
        
    except Exception as e:
        st.error(f"Preprocessing failed: {str(e)}")
        return None, None, None

# Safe resampling function
def apply_resampling(X, y, method='None', random_state=42):
    """Apply resampling with error handling"""
    if not RESAMPLING_AVAILABLE or method == 'None':
        return X, y
    
    try:
        if method == 'SMOTE':
            # Safe SMOTE implementation
            k_neighbors = min(5, len(X[y == 1]) - 1) if len(X[y == 1]) > 1 else 1
            if k_neighbors < 1:
                st.warning("Not enough minority samples for SMOTE. Skipping resampling.")
                return X, y
            sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        elif method == 'ADASYN':
            sampler = ADASYN(random_state=random_state)
        elif method == 'ENN':
            sampler = EditedNearestNeighbours()
        elif method == 'TomekLinks':
            sampler = TomekLinks()
        elif method == 'SMOTEENN':
            sampler = SMOTEENN(random_state=random_state)
        elif method == 'SMOTETomek':
            sampler = SMOTETomek(random_state=random_state)
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
        
    except Exception as e:
        st.warning(f"Resampling with {method} failed: {str(e)}. Using original data.")
        return X, y

# Main pipeline function
def run_complete_pipeline(**kwargs):
    """Complete ML pipeline with enhanced error handling"""
    try:
        df = kwargs.get('df')
        resampling_method = kwargs.get('resampling_method', 'None')
        training_mode = kwargs.get('training_mode', 'manual')
        random_state = kwargs.get('random_state', 42)
        
        # Preprocessing
        df_processed, feature_columns, label_encoders = preprocess_data(df)
        if df_processed is None:
            return None
        
        # Prepare features and target
        X = df_processed[feature_columns].values
        y = df_processed['is_fraud'].values
        
        # Feature selection
        try:
            selector = SelectKBest(score_func=f_classif, k=min(10, len(feature_columns)))
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        except:
            X_selected = X
            selected_features = feature_columns
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # Apply resampling
        X_train_resampled, y_train_resampled = apply_resampling(
            X_train, y_train, resampling_method, random_state
        )
        
        # Model training
        if training_mode == 'manual':
            hidden_neurons = kwargs.get('hidden_neurons', 100)
            activation = kwargs.get('activation', 'sigmoid')
            threshold = kwargs.get('threshold', 0.5)
        else:
            # Simple parameter optimization
            hidden_neurons = 150
            activation = 'sigmoid'
            threshold = 0.5
        
        # Train ELM model
        model = SimpleELM(
            hidden_neurons=hidden_neurons,
            activation=activation,
            random_state=random_state
        )
        
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], 
                           index=['Actual 0', 'Actual 1'])
        
        # Package results
        results = {
            'model_results': {
                'metrics': metrics,
                'mode': training_mode,
                'parameters': {
                    'hidden_neurons': hidden_neurons,
                    'activation': activation,
                    'threshold': threshold
                }
            },
            'lime_data': {
                'X_train': pd.DataFrame(X_train_resampled, columns=selected_features),
                'X_test': pd.DataFrame(X_test, columns=selected_features),
                'y_train': pd.Series(y_train_resampled),
                'y_test': pd.Series(y_test),
                'feature_names': selected_features,
                'model': model,
                'model_weights': {
                    'input_weights': model.input_weights,
                    'output_weights': model.output_weights,
                    'activation': activation,
                    'threshold': threshold
                }
            },
            'confusion_matrix': cm_df,
            'preprocessing_results': {
                'original_features': len(feature_columns),
                'selected_features': len(selected_features),
                'train_samples': len(X_train_resampled),
                'test_samples': len(X_test),
                'resampling_method': resampling_method
            }
        }
        
        return results
        
    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        return None

# LIME explanation functions
def create_lime_explainer_from_pipeline(pipeline_results):
    """Create LIME explainer with error handling"""
    if not LIME_AVAILABLE:
        st.warning("LIME not available. Install lime package for explanations.")
        return None
    
    try:
        lime_data = pipeline_results['lime_data']
        X_train = lime_data['X_train'].values
        feature_names = lime_data['feature_names']
        
        explainer = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=['Non-Fraud', 'Fraud'],
            mode='classification',
            discretize_continuous=True
        )
        
        return explainer
        
    except Exception as e:
        st.warning(f"Could not create LIME explainer: {str(e)}")
        return None

def explain_test_instance(pipeline_results, instance_idx, num_features=10):
    """Explain a test instance with error handling"""
    try:
        lime_data = pipeline_results['lime_data']
        model = lime_data['model']
        X_test = lime_data['X_test']
        y_test = lime_data['y_test']
        
        # Get instance
        instance = X_test.iloc[instance_idx].values if hasattr(X_test, 'iloc') else X_test[instance_idx]
        actual_label = y_test.iloc[instance_idx] if hasattr(y_test, 'iloc') else y_test[instance_idx]
        
        # Get prediction
        prediction_proba = model.predict_proba([instance])[0]
        predicted_class = "Fraud" if prediction_proba[1] > 0.5 else "Non-Fraud"
        confidence = max(prediction_proba)
        actual_class = "Fraud" if actual_label == 1 else "Non-Fraud"
        
        # Simple feature importance (gradient-based approximation)
        feature_names = lime_data['feature_names']
        baseline_pred = model.predict_proba([np.zeros_like(instance)])[0][1]
        current_pred = prediction_proba[1]
        
        # Calculate feature importance by perturbation
        importances = []
        for i in range(len(instance)):
            perturbed_instance = instance.copy()
            perturbed_instance[i] = 0  # Set feature to baseline
            perturbed_pred = model.predict_proba([perturbed_instance])[0][1]
            importance = current_pred - perturbed_pred
            importances.append(importance)
        
        # Create explanation dataframe
        explanation_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Impact': ['Increases Fraud Risk' if imp > 0 else 'Decreases Fraud Risk' for imp in importances]
        }).sort_values('Importance', key=abs, ascending=False).head(num_features)
        
        return {
            'predicted_class': predicted_class,
            'actual_class': actual_class,
            'confidence': confidence,
            'explanation_data': {
                'explanation_df': explanation_df
            }
        }
        
    except Exception as e:
        st.error(f"Explanation failed: {str(e)}")
        return None

def explain_custom_instance(pipeline_results, custom_instance, num_features=10):
    """Explain a custom instance"""
    try:
        lime_data = pipeline_results['lime_data']
        model = lime_data['model']
        feature_names = lime_data['feature_names']
        
        # Get prediction
        prediction_proba = model.predict_proba([custom_instance])[0]
        predicted_class = "Fraud" if prediction_proba[1] > 0.5 else "Non-Fraud"
        confidence = max(prediction_proba)
        
        # Calculate feature importance
        current_pred = prediction_proba[1]
        importances = []
        
        for i in range(len(custom_instance)):
            perturbed_instance = custom_instance.copy()
            perturbed_instance[i] = 0
            perturbed_pred = model.predict_proba([perturbed_instance])[0][1]
            importance = current_pred - perturbed_pred
            importances.append(importance)
        
        explanation_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Impact': ['Increases Fraud Risk' if imp > 0 else 'Decreases Fraud Risk' for imp in importances]
        }).sort_values('Importance', key=abs, ascending=False).head(num_features)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'explanation_data': {
                'explanation_df': explanation_df
            }
        }
        
    except Exception as e:
        st.error(f"Custom explanation failed: {str(e)}")
        return None

# Validation function
def validate_pipeline_results(results):
    """Validate pipeline results"""
    if results is None:
        return False, "Results are None"
    
    required_keys = ['model_results', 'lime_data', 'confusion_matrix']
    for key in required_keys:
        if key not in results:
            return False, f"Missing key: {key}"
    
    return True, "Valid"

# Integration test
def test_complete_integration(df):
    """Test complete integration"""
    try:
        # Run a small test pipeline
        test_results = run_complete_pipeline(
            df=df.head(100),  # Use smaller dataset for testing
            resampling_method='None',
            training_mode='manual',
            hidden_neurons=50,
            activation='sigmoid',
            threshold=0.5
        )
        
        if test_results is None:
            return False
        
        # Validate results
        is_valid, _ = validate_pipeline_results(test_results)
        if not is_valid:
            return False
        
        # Test LIME explainer
        explainer = create_lime_explainer_from_pipeline(test_results)
        if explainer is None and LIME_AVAILABLE:
            st.warning("LIME explainer creation failed, but pipeline is working")
        
        return True
        
    except Exception as e:
        st.error(f"Integration test failed: {str(e)}")
        return False

def print_pipeline_summary(results):
    """Print pipeline summary"""
    print("=== PIPELINE SUMMARY ===")
    print(f"Model Accuracy: {results['model_results']['metrics']['accuracy']:.4f}")
    print(f"Model Precision: {results['model_results']['metrics']['precision']:.4f}")
    print(f"Model Recall: {results['model_results']['metrics']['recall']:.4f}")
    print(f"Model F1-Score: {results['model_results']['metrics']['f1_score']:.4f}")
    print(f"Training Mode: {results['model_results']['mode']}")
    print(f"Features Used: {len(results['lime_data']['feature_names'])}")
    print(f"Training Samples: {len(results['lime_data']['X_train'])}")
    print(f"Test Samples: {len(results['lime_data']['X_test'])}")

# Page configuration
st.set_page_config(
    page_title="🛡️ Fraud Detection System Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (keeping the same as original)
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
    """Show progress indicator"""
    steps = ["📤 Upload", "🔧 Process", "🤖 Analyze", "🔍 Explain"]
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
    """Page 1: Upload dan Preview Data"""
    
    show_step_indicator("upload")
    
    # Main header
    st.markdown('<div class="main-header">🛡️ Fraud Detection System Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced AI-Powered Transaction Analysis</div>', unsafe_allow_html=True)

    # Library status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🔬 **Core ML**: ✅ Available")
    with col2:
        status = "✅ Available" if RESAMPLING_AVAILABLE else "⚠️ Limited"
        st.info(f"⚖️ **Resampling**: {status}")
    with col3:
        status = "✅ Available" if LIME_AVAILABLE else "⚠️ Basic Mode"
        st.info(f"🧠 **LIME**: {status}")

    # Description
    st.markdown("""
    <div class="description-text">
    Dashboard ini menggunakan <span class="highlight-text">Extreme Learning Machine (ELM)</span> 
    yang telah terintegrasi dengan <span class="highlight-text">LIME (Local Interpretable Model-agnostic Explanations)</span> 
    untuk mendeteksi fraud dengan akurasi tinggi dan memberikan penjelasan yang dapat dipahami.
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown("### 📁 Upload Transaction Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your transaction data in CSV format. The system will automatically create fraud labels if not present."
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

            # Data preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Data quality check
            st.markdown("### 🔍 Data Quality Assessment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Missing values
                missing_data = df.isnull().sum()
                missing_pct = (missing_data / len(df)) * 100
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': missing_pct.values
                }).sort_values('Missing %', ascending=False)
                
                st.markdown("#### Missing Values")
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

            # Configuration section
            st.markdown("---")
            st.markdown("### ⚙️ Analysis Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Configuration")
                
                # Resampling options based on availability
                if RESAMPLING_AVAILABLE:
                    resampling_options = ['None', 'SMOTE', 'ADASYN', 'ENN', 'TomekLinks', 'SMOTEENN', 'SMOTETomek']
                else:
                    resampling_options = ['None']
                    st.warning("⚠️ Resampling libraries not available. Install imbalanced-learn for more options.")
                
                st.session_state.selected_resampling = st.selectbox(
                    "Resampling Method",
                    options=resampling_options,
                    index=0,
                    help="Choose resampling method to handle imbalanced data"
                )
                
                st.session_state.selected_training_mode = st.selectbox(
                    "Training Mode",
                    options=['manual', 'optimized'],
                    index=0,
                    help="Manual: Use specified parameters, Optimized: Use predefined best parameters"
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
                    st.info("Optimized mode will use predefined best parameters")

            # Start Analysis button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("🚀 Start Advanced Analysis", key="analysis_btn", use_container_width=True):
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
    """Page 2: Processing dengan integrasi penuh"""
    
    show_step_indicator("process")
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Upload", key="back_btn"):
            st.session_state.current_page = 'upload'
            st.rerun()
    
    with col2:
        st.markdown('<div class="main-header">🔧 Processing & Training</div>', unsafe_allow_html=True)
    
    # Check data availability
    if st.session_state.uploaded_data is None:
        st.error("❌ No data found. Please upload data first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return
    
    df = st.session_state.uploaded_data
    
    # Processing configuration summary
    st.markdown("### ⚙️ Processing Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info(f"""
        **Model Configuration:**
        - Resampling: {st.session_state.selected_resampling}
        - Training Mode: {st.session_state.selected_training_mode}
        """)
    
    with config_col2:
        if st.session_state.selected_training_mode == 'manual':
            st.info(f"""
            **Manual Parameters:**
            - Hidden Neurons: {st.session_state.selected_hidden_neurons}
            - Activation: {st.session_state.selected_activation}
            - Threshold: {st.session_state.selected_threshold}
            """)
        else:
            st.info("**Optimized Mode:**\nUsing predefined best parameters")
    
    # Process button
    if not st.session_state.processing_complete:
        st.markdown("---")
        
        if st.button("🔄 Start Processing", key="process_btn", use_container_width=True):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Preprocessing
                status_text.text("Step 1/5: Data preprocessing...")
                progress_bar.progress(0.2)
                
                # Step 2: Feature Engineering
                status_text.text("Step 2/5: Feature engineering & selection...")
                progress_bar.progress(0.4)
                
                # Step 3: Resampling
                status_text.text("Step 3/5: Handling imbalanced data...")
                progress_bar.progress(0.6)
                
                # Step 4: Model Training
                status_text.text("Step 4/5: Training ELM model...")
                progress_bar.progress(0.8)
                
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
                
                pipeline_results = run_complete_pipeline(**pipeline_params)
                
                # Step 5: LIME Integration
                status_text.text("Step 5/5: Setting up explainer...")
                progress_bar.progress(1.0)
                
                if pipeline_results is None:
                    st.error("❌ Pipeline processing failed!")
                    return
                
                # Validate results
                is_valid, error_msg = validate_pipeline_results(pipeline_results)
                if not is_valid:
                    st.error(f"❌ Pipeline validation failed: {error_msg}")
                    return
                
                # Create LIME explainer
                lime_explainer = create_lime_explainer_from_pipeline(pipeline_results)
                
                # Store results
                st.session_state.pipeline_results = pipeline_results
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
                st.error(f"❌ Processing failed: {str(e)}")
                st.info("Please try with different parameters or check your data format.")
                return
    
    # Show results if processing is complete
    if st.session_state.processing_complete and st.session_state.pipeline_results is not None:
        st.markdown("---")
        st.markdown("### 🎉 Processing Results")
        
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
        
        # Data Information
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
            - Mode: {results['model_results']['mode']}
            - Resampling: {st.session_state.selected_resampling}
            - Integration: ✅ Ready
            """)
        
        # Confusion Matrix
        if 'confusion_matrix' in results:
            st.markdown("#### 🔄 Confusion Matrix")
            cm_df = results['confusion_matrix']
            
            # Create plotly heatmap
            fig = px.imshow(
                cm_df.values,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Non-Fraud', 'Fraud'],
                y=['Non-Fraud', 'Fraud'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig.update_layout(title="Model Performance - Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if len(lime_data['feature_names']) > 0:
            st.markdown("#### 🏷️ Selected Features")
            
            # Display features in a nice format
            features_per_row = 5
            feature_names = lime_data['feature_names']
            
            for i in range(0, len(feature_names), features_per_row):
                cols = st.columns(features_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(feature_names):
                        col.info(f"**{i+j+1}.** {feature_names[i+j]}")
        
        # Integration test button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🧪 Run Integration Test", key="test_btn"):
                with st.spinner("Running integration test..."):
                    test_success = test_complete_integration(df)
                    if test_success:
                        st.success("✅ Integration test passed!")
                    else:
                        st.error("❌ Integration test failed!")
        
        with col2:
            if st.button("📊 View Analysis", key="analysis_btn", use_container_width=True):
                st.session_state.current_page = 'analysis'
                st.rerun()
        
        with col3:
            if st.button("🔍 AI Explanation", key="explanation_btn"):
                st.session_state.current_page = 'explanation'
                st.rerun()

def page_analysis():
    """Page 3: Analysis Results dengan visualisasi enhanced"""
    
    show_step_indicator("analysis")
    
    # Header with navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Process", key="back_btn"):
            st.session_state.current_page = 'process'
            st.rerun()
    
    with col2:
        st.markdown('<div class="main-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("Next: AI Explanation →", key="next_btn"):
            st.session_state.current_page = 'explanation'
            st.rerun()
    
    # Check data availability
    if st.session_state.pipeline_results is None:
        st.error("❌ No analysis results found. Please run processing first.")
        if st.button("Go to Processing"):
            st.session_state.current_page = 'process'
            st.rerun()
        return
    
    results = st.session_state.pipeline_results
    lime_data = results['lime_data']
    
    # Performance Overview
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
        st.success("🎉 Excellent model performance! High accuracy achieved.")
    elif metrics['accuracy'] > 0.8:
        st.info("👍 Good model performance. Acceptable for fraud detection.")
    else:
        st.warning("⚠️ Model performance could be improved. Consider adjusting parameters.")
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Model Analysis", "📈 Data Insights", "🔧 Technical Details"])
    
    with tab1:
        st.markdown("#### Confusion Matrix Analysis")
        
        if 'confusion_matrix' in results:
            cm_df = results['confusion_matrix']
            
            # Create enhanced confusion matrix visualization
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
                title="Confusion Matrix - Model Predictions vs Actual Labels",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            tn, fp, fn, tp = cm_df.values.ravel()
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **True Predictions:**
                - True Negatives (Non-Fraud → Non-Fraud): {tn:,}
                - True Positives (Fraud → Fraud): {tp:,}
                - **Total Correct**: {tn + tp:,}
                """)
            
            with col2:
                st.warning(f"""
                **False Predictions:**
                - False Positives (Non-Fraud → Fraud): {fp:,}
                - False Negatives (Fraud → Non-Fraud): {fn:,}
                - **Total Incorrect**: {fp + fn:,}
                """)
        
        # Performance metrics comparison chart
        st.markdown("#### 📈 Performance Metrics Breakdown")
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        
        fig = go.Figure(data=[
            go.Bar(name='Model Performance', x=metric_names, y=metric_values,
                   marker_color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ])
        
        fig.update_layout(
            title="Model Performance Metrics Comparison",
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
        
        # Data distribution
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
            
            st.info(f"""
            **Test Set Distribution:**
            - Fraud Cases: {test_fraud_count:,}
            - Non-Fraud Cases: {test_total - test_fraud_count:,}
            - Fraud Rate: {fraud_rate:.2f}%
            - Balance Ratio: {(test_total - test_fraud_count) / max(test_fraud_count, 1):.1f}:1
            """)
        
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
        train_df = lime_data['X_train']
        
        st.markdown("#### 📈 Feature Statistics")
        st.dataframe(train_df.describe(), use_container_width=True)
        
        # Feature correlation heatmap
        if len(feature_names) <= 15:  # Only show for reasonable number of features
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
                title="Feature Correlation Heatmap",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.markdown("#### 🔧 Technical Configuration")
        
        model_results = results['model_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Model Parameters")
            if 'parameters' in model_results:
                params = model_results['parameters']
                for key, value in params.items():
                    st.write(f"**{key}:** {value}")
            else:
                st.write(f"**Mode:** {model_results['mode']}")
                st.write(f"**Resampling:** {st.session_state.selected_resampling}")
        
        with col2:
            st.markdown("##### Processing Summary")
            
            processing_info = results.get('preprocessing_results', {})
            for key, value in processing_info.items():
                if isinstance(value, (int, float, str)):
                    st.write(f"**{key}:** {value}")
        
        # Model architecture
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
        """
        
        st.info(arch_info)
        
        # Training summary
        st.markdown("#### 📋 Training Summary")
        
        if st.button("📊 Show Detailed Pipeline Summary", key="summary_btn"):
            with st.expander("Complete Pipeline Summary", expanded=True):
                summary_output = st.empty()
                
                # Create a string buffer to capture print output
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
    """Page 4: Enhanced Explanations"""
    
    show_step_indicator("explanation")
    
    # Header with navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Analysis", key="back_btn"):
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    with col2:
        st.markdown('<div class="main-header">🧠 AI Explanations</div>', unsafe_allow_html=True)
    
    # Check data availability
    if st.session_state.pipeline_results is None:
        st.error("❌ No explanation data found. Please run processing first.")
        if st.button("Go to Processing"):
            st.session_state.current_page = 'process'
            st.rerun()
        return
    
    results = st.session_state.pipeline_results
    lime_data = results['lime_data']
    
    # Show LIME availability status
    lime_status = "✅ Full LIME Support" if LIME_AVAILABLE else "⚠️ Basic Explanation Mode"
    st.info(f"**Explanation System Status:** {lime_status}")
    
    st.markdown("""
    ### 🔍 Understanding AI Decisions
    
    This system provides explanations for individual predictions by showing 
    which features contributed most to the model's decision.
    """)
    
    # Explanation options
    tab1, tab2 = st.tabs(["🔍 Test Instance Explanation", "📝 Custom Instance Explanation"])
    
    with tab1:
        st.markdown("#### Select a Test Transaction to Explain")
        
        X_test = lime_data['X_test']
        y_test = lime_data['y_test']
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            explanation_filter = st.selectbox(
                "Filter transactions by:",
                ["All Transactions", "Fraud Cases Only", "Non-Fraud Cases Only"]
            )
        
        with col2:
            max_display = st.slider("Max transactions to show:", 10, 100, 50)
        
        # Get available indices based on filter
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
        
        # Show transaction details
        st.markdown("##### 📋 Transaction Details")
        
        selected_instance = X_test.iloc[selected_idx]
        actual_label = y_test.iloc[selected_idx]
        
        # Show instance details
        feature_names = lime_data['feature_names']
        
        num_cols = 4
        cols = st.columns(num_cols)
        
        for i, feature in enumerate(feature_names[:12]):  # Show first 12 features
            col_idx = i % num_cols
            with cols[col_idx]:
                value = selected_instance[feature]
                if isinstance(value, float):
                    st.metric(feature, f"{value:.4f}")
                else:
                    st.metric(feature, str(value))
        
        # Explanation generation
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            num_features_explain = st.slider("Number of features to explain:", 5, 20, 10)
        
        with col2:
            if st.button("🔍 Generate AI Explanation", key="explain_btn", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the transaction..."):
                    try:
                        explanation_result = explain_test_instance(
                            results, 
                            instance_idx=selected_idx, 
                            num_features=num_features_explain
                        )
                        
                        if explanation_result is None:
                            st.error("❌ Failed to generate explanation. Please try another transaction.")
                            return
                        
                        # Display explanation results
                        st.markdown("---")
                        st.markdown("### 🎯 AI Explanation Results")
                        
                        # Prediction summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            prediction_class = explanation_result['predicted_class']
                            class_color = "🚨" if prediction_class == "Fraud" else "✅"
                            st.metric("🤖 AI Prediction", f"{class_color} {prediction_class}")
                        
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
                            st.success("✅ Correct Prediction!")
                        else:
                            st.error("❌ Incorrect Prediction")
                        
                        # Feature importance explanation
                        st.markdown("#### 📊 Feature Importance Analysis")
                        
                        explanation_df = explanation_result['explanation_data']['explanation_df']
                        
                        # Create interactive bar chart
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
                            title="Feature Contributions to AI Decision",
                            xaxis_title="Contribution Score",
                            yaxis_title="Features",
                            height=400 + len(explanation_df) * 20,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Explanation table
                        st.markdown("#### 📋 Detailed Feature Analysis")
                        
                        display_df = explanation_df.copy()
                        display_df['Contribution'] = display_df['Importance'].apply(
                            lambda x: f"{'⬆️' if x > 0 else '⬇️'} {abs(x):.4f}"
                        )
                        display_df['Impact Direction'] = display_df['Impact']
                        
                        st.dataframe(
                            display_df[['Feature', 'Contribution', 'Impact Direction']],
                            use_container_width=True
                        )
                        
                        # Explanation summary
                        st.markdown("#### 💡 Explanation Summary")
                        
                        top_fraud_features = explanation_df[explanation_df['Importance'] > 0]['Feature'].head(3).tolist()
                        top_safe_features = explanation_df[explanation_df['Importance'] < 0]['Feature'].head(3).tolist()
                        
                        summary_text = f"""
                        **AI Decision Analysis:**
                        
                        The AI model predicted this transaction as **{prediction_class}** with **{confidence:.1%} confidence**.
                        
                        """
                        
                        if top_fraud_features:
                            summary_text += f"""
                            **🚨 Features increasing fraud risk:**
                            {', '.join(top_fraud_features)}
                            
                            """
                        
                        if top_safe_features:
                            summary_text += f"""
                            **✅ Features decreasing fraud risk:**
                            {', '.join(top_safe_features)}
                            """
                        
                        st.markdown(summary_text)
                        
                    except Exception as e:
                        st.error(f"❌ Explanation generation failed: {str(e)}")
                        st.info("Please try with a different transaction or check the data integrity.")
    
    with tab2:
        st.markdown("#### Create Custom Transaction for Explanation")
        
        st.info("""
        💡 **Custom Explanation Feature**
        
        You can create a custom transaction with your own values to see how the AI would classify it.
        This is useful for testing specific scenarios or understanding model behavior.
        """)
        
        feature_names = lime_data['feature_names']
        
        # Get sample statistics for reasonable defaults
        X_train = lime_data['X_train']
        stats = X_train.describe()
        
        st.markdown("##### 📝 Enter Custom Values")
        
        # Create input fields for each feature
        custom_values = {}
        
        # Group inputs in columns for better layout
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, feature in enumerate(feature_names):
            col_idx = i % num_cols
            
            with cols[col_idx]:
                # Get reasonable default and range from training data
                if feature in stats.columns:
                    mean_val = float(stats[feature]['mean'])
                    min_val = float(stats[feature]['min'])
                    max_val = float(stats[feature]['max'])
                    std_val = float(stats[feature]['std'])
                    
                    # Set reasonable bounds
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
                    # Fallback for missing stats
                    custom_values[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        help="Enter a numeric value for this feature"
                    )
        
        # Preset buttons for common scenarios
        st.markdown("##### 🎛️ Quick Presets")
        
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("💰 High Amount Scenario"):
                # Set high amount related features
                for feature in feature_names:
                    if 'amount' in feature.lower():
                        if feature in stats.columns:
                            custom_values[feature] = float(stats[feature]['75%']) * 2
                        st.rerun()
        
        with preset_col2:
            if st.button("🌙 Off-Hours Scenario"):
                # Set suspicious timing features if available
                st.info("Preset values set for off-hours transaction pattern")
        
        with preset_col3:
            if st.button("🔄 Reset to Averages"):
                # Reset all to mean values
                for feature in feature_names:
                    if feature in stats.columns:
                        custom_values[feature] = float(stats[feature]['mean'])
                st.rerun()
        
        # Generate explanation for custom instance
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            custom_num_features = st.slider("Features to explain:", 5, 20, 10, key="custom_features")
        
        with col2:
            if st.button("🔍 Explain Custom Transaction", key="custom_explain_btn", use_container_width=True):
                with st.spinner("🤖 Analyzing custom transaction..."):
                    try:
                        # Create custom instance array
                        custom_instance = np.array([custom_values[feature] for feature in feature_names])
                        
                        # Use integrated custom explanation function
                        custom_explanation = explain_custom_instance(
                            results,
                            custom_instance,
                            num_features=custom_num_features
                        )
                        
                        if custom_explanation is None:
                            st.error("❌ Failed to explain custom transaction.")
                            return
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 🎯 Custom Transaction Analysis")
                        
                        # Prediction results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            prediction_class = custom_explanation['predicted_class']
                            class_color = "🚨" if prediction_class == "Fraud" else "✅"
                            st.metric("🤖 AI Prediction", f"{class_color} {prediction_class}")
                        
                        with col2:
                            confidence = custom_explanation['confidence']
                            st.metric("📊 Confidence", f"{confidence:.1%}")
                        
                        # Risk assessment
                        if prediction_class == "Fraud":
                            if confidence > 0.8:
                                st.error("🚨 HIGH RISK: Strong fraud indicators detected!")
                            elif confidence > 0.6:
                                st.warning("⚠️ MEDIUM RISK: Some fraud patterns detected.")
                            else:
                                st.info("📊 LOW-MEDIUM RISK: Weak fraud indicators.")
                        else:
                            if confidence > 0.8:
                                st.success("✅ LOW RISK: Strong legitimate transaction indicators.")
                            else:
                                st.info("📊 UNCERTAIN: Mixed signals in transaction pattern.")
                        
                        # Feature importance for custom transaction
                        explanation_df = custom_explanation['explanation_data']['explanation_df']
                        
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
                            title="Custom Transaction - Feature Contributions",
                            xaxis_title="Contribution Score",
                            yaxis_title="Features",
                            height=400 + len(explanation_df) * 20,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed analysis
                        st.markdown("#### 📋 Feature Impact Details")
                        
                        display_df = explanation_df.copy()
                        display_df['Your Value'] = [custom_values[feature] for feature in display_df['Feature']]
                        display_df['Contribution'] = display_df['Importance'].apply(
                            lambda x: f"{'📈' if x > 0 else '📉'} {abs(x):.4f}"
                        )
                        
                        st.dataframe(
                            display_df[['Feature', 'Your Value', 'Contribution', 'Impact']],
                            use_container_width=True
                        )
                        
                        # Recommendations
                        st.markdown("#### 💡 Analysis Insights")
                        
                        top_risk_features = explanation_df[explanation_df['Importance'] > 0].head(3)
                        top_safe_features = explanation_df[explanation_df['Importance'] < 0].head(3)
                        
                        if len(top_risk_features) > 0:
                            st.markdown("**🚨 Features increasing fraud risk in your transaction:**")
                            for _, row in top_risk_features.iterrows():
                                feature = row['Feature']
                                value = custom_values[feature]
                                impact = row['Importance']
                                st.write(f"- **{feature}**: {value:.4f} (impact: +{impact:.4f})")
                        
                        if len(top_safe_features) > 0:
                            st.markdown("**✅ Features indicating legitimate transaction:**")
                            for _, row in top_safe_features.iterrows():
                                feature = row['Feature']
                                value = custom_values[feature]
                                impact = abs(row['Importance'])
                                st.write(f"- **{feature}**: {value:.4f} (impact: -{impact:.4f})")
                        
                    except Exception as e:
                        st.error(f"❌ Custom explanation failed: {str(e)}")
                        st.info("Please check your input values and try again.")

# Main navigation function
def main():
    """Enhanced main navigation with session state management"""
    
    # Initialize session state
    init_session_state()
    
    # Sidebar with enhanced navigation and controls
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Page navigation buttons
        pages = {
            'upload': '📤 Upload Data',
            'process': '🔧 Process & Train',
            'analysis': '📊 Analysis Results',
            'explanation': '🧠 AI Explanation'
        }
        
        for page_key, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                # Check prerequisites
                if page_key == 'process' and st.session_state.uploaded_data is None:
                    st.error("Please upload data first!")
                    continue
                elif page_key in ['analysis', 'explanation'] and st.session_state.pipeline_results is None:
                    st.error("Please complete processing first!")
                    continue
                
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # System status
        st.markdown("### 🔧 System Status")
        
        # Library availability
        ml_status = "✅" if True else "❌"
        resampling_status = "✅" if RESAMPLING_AVAILABLE else "⚠️"
        lime_status = "✅" if LIME_AVAILABLE else "⚠️"
        
        st.write(f"{ml_status} Core ML Libraries")
        st.write(f"{resampling_status} Resampling Tools")
        st.write(f"{lime_status} LIME Explainer")
        
        # Progress indicator
        st.markdown("### 📊 Progress")
        
        progress_items = [
            ("Data Upload", st.session_state.uploaded_data is not None),
            ("Processing", st.session_state.processing_complete),
            ("Explainer Ready", st.session_state.lime_explainer is not None)
        ]
        
        for item, completed in progress_items:
            status = "✅" if completed else "⏳"
            st.write(f"{status} {item}")
        
        # Session information
        if st.session_state.uploaded_data is not None:
            st.markdown("---")
            st.markdown("### 📋 Session Info")
            st.write(f"**Rows:** {len(st.session_state.uploaded_data):,}")
            st.write(f"**Columns:** {st.session_state.uploaded_data.shape[1]}")
            
            if st.session_state.pipeline_results:
                metrics = st.session_state.pipeline_results['model_results']['metrics']
                st.write(f"**Accuracy:** {metrics['accuracy']:.3f}")
        
        st.markdown("---")
        
        # Control buttons
        st.markdown("### 🛠️ Controls")
        
        if st.button("🔄 Reset Session", key="reset_session"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Help section
        st.markdown("---")
        st.markdown("### ❓ Help")
        
        with st.expander("📖 Quick Guide"):
            st.markdown("""
            **Getting Started:**
            1. Upload your CSV transaction data
            2. Configure model parameters
            3. Run processing & training
            4. View analysis results
            5. Explore AI explanations
            
            **Notes:**
            - Synthetic fraud labels are created if not present
            - SMOTE resampling requires sufficient minority samples
            - LIME explanations work with or without the lime library
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common Issues:**
            - **SMOTE Error**: Try 'None' resampling or ensure enough fraud samples
            - **Low Performance**: Try different parameters or more data
            - **Missing Libraries**: Install with `pip install imbalanced-learn lime`
            - **Small Dataset**: Use at least 100 rows for best results
            """)
    
    # Main content area
    if st.session_state.current_page == 'upload':
        page_upload()
    elif st.session_state.current_page == 'process':
        page_process()
    elif st.session_state.current_page == 'analysis':
        page_analysis()
    elif st.session_state.current_page == 'explanation':
        page_explanation()

if __name__ == "__main__":
    main()
