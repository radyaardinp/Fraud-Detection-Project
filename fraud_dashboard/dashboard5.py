import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# ELM Implementation
class ELMClassifier:
    """Extreme Learning Machine Classifier"""
    
    def __init__(self, n_hidden=100, activation='relu', random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        
    def _activation_function(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            return x
    
    def fit(self, X, y):
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Random input weights and biases
        self.input_weights = np.random.randn(n_features, self.n_hidden)
        self.biases = np.random.randn(self.n_hidden)
        
        # Calculate hidden layer output
        H = self._activation_function(np.dot(X, self.input_weights) + self.biases)
        
        # Handle binary and multiclass classification
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification
            y_encoded = np.where(y == self.classes_[0], -1, 1)
            # Calculate output weights using Moore-Penrose pseudoinverse
            self.output_weights = np.dot(np.linalg.pinv(H), y_encoded)
        else:
            # Multiclass classification - one-hot encoding
            y_encoded = np.zeros((len(y), n_classes))
            for i, class_label in enumerate(self.classes_):
                y_encoded[y == class_label, i] = 1
            # Calculate output weights
            self.output_weights = np.dot(np.linalg.pinv(H), y_encoded)
        
        return self
    
    def predict(self, X):
        H = self._activation_function(np.dot(X, self.input_weights) + self.biases)
        output = np.dot(H, self.output_weights)
        
        if len(self.classes_) == 2:
            # Binary classification
            predictions = np.where(output > 0, self.classes_[1], self.classes_[0])
        else:
            # Multiclass classification
            predictions = self.classes_[np.argmax(output, axis=1)]
        
        return predictions
    
    def predict_proba(self, X):
        H = self._activation_function(np.dot(X, self.input_weights) + self.biases)
        output = np.dot(H, self.output_weights)
        
        if len(self.classes_) == 2:
            # Binary classification - convert to probabilities
            proba_positive = 1 / (1 + np.exp(-output))
            probabilities = np.column_stack([1 - proba_positive, proba_positive])
        else:
            # Multiclass classification - softmax
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        
        return probabilities

# Page configuration
st.set_page_config(
    page_title="🚀 ML Pipeline Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    default_states = {
        'current_page': 'upload',
        'uploaded_data': None,
        'eda_complete': False,
        'preprocessed_data': None,
        'labeled_data': None,
        'feature_selected_data': None,
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
        'trained_models': {},
        'best_model': None,
        'hyperparameter_results': None,
        'evaluation_results': None,
        'target_column': None,
        'feature_columns': None,
        'label_encoders': {},
        'scaler': None
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 1rem;
            padding: 1rem;
            background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
        }
        
        .step-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #495057;
            margin: 1rem 0;
            padding: 0.5rem;
            border-left: 4px solid #28a745;
            background-color: #f8f9fa;
        }
        
        .metric-card {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .sidebar-nav {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .stButton > button {
            width: 100%;
            background-color: #007bff;
            color: white;
            border: none;
            padding: 0.75rem;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
        }
        
        .success-btn > button {
            background-color: #28a745 !important;
        }
        
        .danger-btn > button {
            background-color: #dc3545 !important;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Sidebar Navigation
def sidebar_navigation():
    """Create sidebar navigation"""
    st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🧭 Navigation")
    
    pages = {
        'upload': '📁 1. Upload Data',
        'eda': '📊 2. Exploratory Data Analysis', 
        'preprocessing': '🔧 3. Preprocessing',
        'labeling': '🏷️ 4. Data Labeling',
        'feature_engineering': '⚙️ 5. Feature Engineering',
        'modeling': '🤖 6. ELM Modeling & Resampling',
        'evaluation': '📈 7. Model Evaluation',
        'xai': '🧠 8. Explainable AI'
    }
    
    for page_key, page_name in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}"):
            st.session_state.current_page = page_key
            st.rerun()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Progress tracker
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Progress")
    
    progress_status = {
        'upload': st.session_state.uploaded_data is not None,
        'eda': st.session_state.eda_complete,
        'preprocessing': st.session_state.preprocessed_data is not None,
        'labeling': st.session_state.labeled_data is not None,
        'feature_engineering': st.session_state.feature_selected_data is not None,
        'modeling': len(st.session_state.trained_models) > 0,
        'evaluation': st.session_state.evaluation_results is not None,
        'xai': st.session_state.evaluation_results is not None
    }
    
    completed = sum(progress_status.values())
    total = len(progress_status)
    progress = completed / total
    
    st.sidebar.progress(progress)
    st.sidebar.write(f"Progress: {completed}/{total} steps completed")
    
    for step, completed in progress_status.items():
        icon = "✅" if completed else "⏳"
        st.sidebar.write(f"{icon} {pages[step].split('. ')[1]}")

# Page 1: Upload Data (unchanged)
def page_upload():
    st.markdown('<div class="main-header">🚀 ML Pipeline Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-header">📁 Step 1: Upload Data</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Complete ML Pipeline Dashboard!
    
    This dashboard will guide you through the entire machine learning workflow:
    1. **Data Upload & EDA** - Upload and explore your data
    2. **Preprocessing** - Clean and prepare your data
    3. **Labeling** - Define target variables and features
    4. **Feature Engineering** - Select and create features
    5. **ELM Modeling** - Train ELM models with various resampling techniques
    6. **Evaluation** - Assess model performance
    7. **Explainable AI** - Understand model predictions
    """)
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            
            st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")
            
            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", f"{df.shape[1]}")
            with col3:
                st.metric("Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
            with col4:
                st.metric("Missing Values", f"{df.isnull().sum().sum()}")
            
            # Preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column info
            st.markdown("### 📊 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Next step button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Start EDA", key="start_eda", use_container_width=True):
                    st.session_state.current_page = 'eda'
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# Page 2: EDA (unchanged)
def page_eda():
    st.markdown('<div class="step-header">📊 Step 2: Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.uploaded_data
    
    # EDA Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistical Summary", "📊 Distributions", "🔗 Correlations", "❌ Missing Values"])
    
    with tab1:
        st.markdown("### 📈 Statistical Summary")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.markdown("#### Numeric Columns")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Categorical columns summary
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            st.markdown("#### Categorical Columns")
            cat_summary = pd.DataFrame({
                'Column': cat_cols,
                'Unique Values': [df[col].nunique() for col in cat_cols],
                'Most Frequent': [df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A' for col in cat_cols],
                'Frequency': [df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0 for col in cat_cols]
            })
            st.dataframe(cat_summary, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Data Distributions")
        
        # Select columns for visualization
        viz_cols = st.multiselect(
            "Select columns to visualize:",
            df.columns.tolist(),
            default=df.columns.tolist()[:5]
        )
        
        if viz_cols:
            cols_per_row = 2
            
            for i in range(0, len(viz_cols), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col_name in enumerate(viz_cols[i:i+cols_per_row]):
                    with cols[j]:
                        if df[col_name].dtype in ['int64', 'float64']:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.hist(df[col_name].dropna(), bins=30, alpha=0.7, color='skyblue')
                            ax.set_title(f'{col_name} Distribution')
                            ax.set_xlabel(col_name)
                            ax.set_ylabel('Frequency')
                            st.pyplot(fig)
                            plt.close()
                        else:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            value_counts = df[col_name].value_counts().head(10)
                            ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
                            ax.set_title(f'{col_name} Top 10 Values')
                            ax.set_xticks(range(len(value_counts)))
                            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                            ax.set_ylabel('Count')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
    
    with tab3:
        st.markdown("### 🔗 Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Plotly heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # High correlations
            st.markdown("#### High Correlations (|r| > 0.7)")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if high_corr:
                st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
            else:
                st.info("No high correlations found (|r| > 0.7)")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    with tab4:
        st.markdown("### ❌ Missing Values Analysis")
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            # Missing values chart
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data.plot(kind='bar', ax=ax, color='salmon')
            ax.set_title('Missing Values by Column')
            ax.set_ylabel('Number of Missing Values')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Missing values table
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("🎉 No missing values found in the dataset!")
    
    # Mark EDA as complete
    st.session_state.eda_complete = True
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Upload", key="back_to_upload"):
            st.session_state.current_page = 'upload'
            st.rerun()
    with col3:
        if st.button("Next: Preprocessing →", key="to_preprocessing"):
            st.session_state.current_page = 'preprocessing'
            st.rerun()

# Page 3: Preprocessing (unchanged)
def page_preprocessing():
    st.markdown('<div class="step-header">🔧 Step 3: Data Preprocessing</div>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.uploaded_data.copy()
    
    st.markdown("### Configure Preprocessing Steps")
    
    # Preprocessing options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧹 Data Cleaning")
        
        # Missing value handling
        missing_strategy = st.selectbox(
            "Missing Value Strategy:",
            ["Drop rows with missing values", "Fill with mean/mode", "Forward fill", "Backward fill", "Custom value"]
        )
        
        # Duplicate handling
        handle_duplicates = st.checkbox("Remove duplicate rows", value=True)
        
        # Outlier detection
        outlier_method = st.selectbox(
            "Outlier Detection Method:",
            ["None", "IQR Method", "Z-Score", "Isolation Forest"]
        )
        
    with col2:
        st.markdown("#### 🔄 Data Transformation")
        
        # Encoding
        encoding_method = st.selectbox(
            "Categorical Encoding:",
            ["Label Encoding", "One-Hot Encoding", "Target Encoding"]
        )
        
        # Scaling
        scaling_method = st.selectbox(
            "Numerical Scaling:",
            ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
        )
        
        # Date handling
        handle_dates = st.checkbox("Convert date columns to datetime")
    
    # Apply preprocessing
    if st.button("🔄 Apply Preprocessing", key="apply_preprocessing"):
        with st.spinner("Processing data..."):
            processed_df = df.copy()
            processing_log = []
            
            try:
                # Handle missing values
                if missing_strategy == "Drop rows with missing values":
                    original_len = len(processed_df)
                    processed_df = processed_df.dropna()
                    processing_log.append(f"Dropped {original_len - len(processed_df)} rows with missing values")
                
                elif missing_strategy == "Fill with mean/mode":
                    for col in processed_df.columns:
                        if processed_df[col].dtype in ['int64', 'float64']:
                            processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                        else:
                            processed_df[col].fillna(processed_df[col].mode().iloc[0] if len(processed_df[col].mode()) > 0 else 'Unknown', inplace=True)
                    processing_log.append("Filled missing values with mean/mode")
                
                # Handle duplicates
                if handle_duplicates:
                    original_len = len(processed_df)
                    processed_df = processed_df.drop_duplicates()
                    processing_log.append(f"Removed {original_len - len(processed_df)} duplicate rows")
                
                # Handle outliers
                if outlier_method == "IQR Method":
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    outliers_removed = 0
                    for col in numeric_cols:
                        Q1 = processed_df[col].quantile(0.25)
                        Q3 = processed_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        original_len = len(processed_df)
                        processed_df = processed_df[(processed_df[col] >= lower) & (processed_df[col] <= upper)]
                        outliers_removed += original_len - len(processed_df)
                    processing_log.append(f"Removed {outliers_removed} outliers using IQR method")
                
                # Encode categorical variables
                if encoding_method == "Label Encoding":
                    label_encoders = {}
                    cat_cols = processed_df.select_dtypes(include=['object']).columns
                    for col in cat_cols:
                        le = LabelEncoder()
                        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                        label_encoders[col] = le
                    st.session_state.label_encoders = label_encoders
                    processing_log.append(f"Applied label encoding to {len(cat_cols)} categorical columns")
                
                elif encoding_method == "One-Hot Encoding":
                    cat_cols = processed_df.select_dtypes(include=['object']).columns
                    processed_df = pd.get_dummies(processed_df, columns=cat_cols, prefix=cat_cols)
                    processing_log.append(f"Applied one-hot encoding to {len(cat_cols)} categorical columns")
                
                # Scale numerical features
                if scaling_method != "None":
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    if scaling_method == "StandardScaler":
                        scaler = StandardScaler()
                    elif scaling_method == "MinMaxScaler":
                        scaler = MinMaxScaler()
                    
                    processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
                    st.session_state.scaler = scaler
                    processing_log.append(f"Applied {scaling_method} to {len(numeric_cols)} numerical columns")
                
                # Store processed data
                st.session_state.preprocessed_data = processed_df
                
                # Show results
                st.success("✅ Preprocessing completed successfully!")
                
                # Show processing log
                st.markdown("### 📋 Processing Log")
                for log_entry in processing_log:
                    st.write(f"• {log_entry}")
                
                # Show before/after comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Before Preprocessing")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Missing values: {df.isnull().sum().sum()}")
                    st.write(f"Duplicates: {df.duplicated().sum()}")
                
                with col2:
                    st.markdown("#### After Preprocessing")
                    st.write(f"Shape: {processed_df.shape}")
                    st.write(f"Missing values: {processed_df.isnull().sum().sum()}")
                    st.write(f"Duplicates: {processed_df.duplicated().sum()}")
                
                # Show processed data preview
                st.markdown("### 📊 Processed Data Preview")
                st.dataframe(processed_df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {str(e)}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to EDA", key="back_to_eda"):
            st.session_state.current_page = 'eda'
            st.rerun()
    with col3:
        if st.button("Next: Labeling →", key="to_labeling", disabled=st.session_state.preprocessed_data is None):
            st.session_state.current_page = 'labeling'
            st.rerun()

# Page 4: Labeling (unchanged)
def page_labeling():
    st.markdown('<div class="step-header">🏷️ Step 4: Data Labeling</div>', unsafe_allow_html=True)
    
    if st.session_state.preprocessed_data is None:
        st.warning("⚠️ Please complete preprocessing first!")
        return
    
    df = st.session_state.preprocessed_data
    
    st.markdown("### Define Target Variable and Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Target Variable")
        target_column = st.selectbox(
            "Select target column:",
            df.columns.tolist(),
            help="Choose the column you want to predict"
        )
        
        if target_column:
            st.write(f"**Target column:** {target_column}")
            st.write(f"**Data type:** {df[target_column].dtype}")
            st.write(f"**Unique values:** {df[target_column].nunique()}")
            
            # Show target distribution
            if df[target_column].nunique() < 20:
                fig, ax = plt.subplots(figsize=(8, 4))
                df[target_column].value_counts().plot(kind='bar', ax=ax, color='lightblue')
                ax.set_title(f'Distribution of {target_column}')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with col2:
        st.markdown("#### 📊 Feature Selection")
        
        available_features = [col for col in df.columns if col != target_column]
        
        # Feature selection method
        selection_method = st.radio(
            "Feature selection method:",
            ["Select manually", "Select all", "Auto-select numeric only"]
        )
        
        if selection_method == "Select manually":
            feature_columns = st.multiselect(
                "Select feature columns:",
                available_features,
                default=available_features[:10] if len(available_features) > 10 else available_features
            )
        elif selection_method == "Select all":
            feature_columns = available_features
        else:  # Auto-select numeric only
            feature_columns = df[available_features].select_dtypes(include=[np.number]).columns.tolist()
        
        st.write(f"**Selected features:** {len(feature_columns)}")
        if feature_columns:
            st.write(f"**Features:** {', '.join(feature_columns[:5])}{'...' if len(feature_columns) > 5 else ''}")
    
    # Problem type detection
    if target_column:
        unique_values = df[target_column].nunique()
        if unique_values == 2:
            problem_type = "Binary Classification"
        elif unique_values < 20 and df[target_column].dtype in ['int64', 'object']:
            problem_type = "Multi-class Classification"
        else:
            problem_type = "Regression"
        
        st.info(f"**Detected problem type:** {problem_type}")
    
    # Apply labeling
    if st.button("🏷️ Apply Labeling", key="apply_labeling"):
        if not target_column:
            st.error("Please select a target column!")
        elif not feature_columns:
            st.error("Please select at least one feature column!")
        else:
            try:
                # Create labeled dataset
                labeled_df = df[[target_column] + feature_columns].copy()
                
                # Store in session state
                st.session_state.labeled_data = labeled_df
                st.session_state.target_column = target_column
                st.session_state.feature_columns = feature_columns
                
                st.success("✅ Labeling completed successfully!")
                
                # Show summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(labeled_df))
                with col2:
                    st.metric("Features", len(feature_columns))
                with col3:
                    st.metric("Target Classes", labeled_df[target_column].nunique())
                
                # Show labeled data preview
                st.markdown("### 📊 Labeled Data Preview")
                st.dataframe(labeled_df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Labeling failed: {str(e)}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Preprocessing", key="back_to_preprocessing"):
            st.session_state.current_page = 'preprocessing'
            st.rerun()
    with col3:
        if st.button("Next: Feature Engineering →", key="to_feature_engineering", disabled=st.session_state.labeled_data is None):
            st.session_state.current_page = 'feature_engineering'
            st.rerun()

# Page 5: Feature Engineering (Modified to use MI only)
def page_feature_engineering():
    st.markdown('<div class="step-header">⚙️ Step 5: Feature Engineering</div>', unsafe_allow_html=True)
    
    if st.session_state.labeled_data is None:
        st.warning("⚠️ Please complete data labeling first!")
        return
    
    df = st.session_state.labeled_data
    X = df[st.session_state.feature_columns]
    y = df[st.session_state.target_column]
    
    st.markdown("### Feature Selection Using Mutual Information")
    
    st.markdown("""
    **Mutual Information (MI)** measures the dependency between variables. 
    Higher MI values indicate stronger relationships between features and the target variable.
    """)
    
    # MI parameters
    col1, col2 = st.columns(2)
    with col1:
        k_features = st.slider("Number of features to select:", 1, len(X.columns), min(10, len(X.columns)))
    with col2:
        discrete_features = st.checkbox("Treat features as discrete", value=False)
    
    if st.button("🔍 Apply Mutual Information Selection", key="mi_selection"):
        try:
            # Apply MI feature selection
            if discrete_features:
                mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)
            else:
                mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
            
            # Create feature scores dataframe
            feature_scores = pd.DataFrame({
                'Feature': X.columns,
                'MI_Score': mi_scores
            }).sort_values('MI_Score', ascending=False)
            
            # Select top k features
            selected_features = feature_scores.head(k_features)['Feature'].tolist()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Mutual Information Scores")
                st.dataframe(feature_scores, use_container_width=True)
            
            with col2:
                st.markdown("##### MI Score Visualization")
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = ['green' if feat in selected_features else 'lightgray' for feat in feature_scores['Feature']]
                ax.barh(range(len(feature_scores)), feature_scores['MI_Score'], color=colors, alpha=0.7)
                ax.set_yticks(range(len(feature_scores)))
                ax.set_yticklabels(feature_scores['Feature'])
                ax.set_xlabel('Mutual Information Score')
                ax.set_title(f'MI Scores (Top {k_features} Selected)')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Store selected features
            st.session_state.selected_features_method = "Mutual Information"
            st.session_state.selected_features_list = selected_features
            
            # Show selection summary
            st.markdown("---")
            st.markdown("### ✅ Feature Selection Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Features", len(X.columns))
            with col2:
                st.metric("Selected Features", len(selected_features))
            with col3:
                st.metric("Reduction", f"{((len(X.columns) - len(selected_features)) / len(X.columns) * 100):.1f}%")
            
            st.write("**Selected Features:**", ', '.join(selected_features))
            
        except Exception as e:
            st.error(f"❌ MI selection failed: {str(e)}")
    
    # Apply feature selection
    if hasattr(st.session_state, 'selected_features_list'):
        if st.button("✅ Apply Feature Selection", key="apply_feature_selection"):
            try:
                selected_df = df[st.session_state.selected_features_list + [st.session_state.target_column]]
                st.session_state.feature_selected_data = selected_df
                st.session_state.feature_columns = st.session_state.selected_features_list
                
                st.success("✅ Feature selection applied successfully!")
                st.dataframe(selected_df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Failed to apply feature selection: {str(e)}")
    
    # Skip feature selection option
    st.markdown("---")
    if st.button("⏭️ Skip Feature Selection (Use All Features)", key="skip_feature_selection"):
        st.session_state.feature_selected_data = df
        st.success("✅ Using all features without selection.")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Labeling", key="back_to_labeling"):
            st.session_state.current_page = 'labeling'
            st.rerun()
    with col3:
        if st.button("Next: ELM Modeling →", key="to_modeling", disabled=st.session_state.feature_selected_data is None):
            st.session_state.current_page = 'modeling'
            st.rerun()

# Page 6: ELM Modeling with Resampling (New)
def page_modeling():
    st.markdown('<div class="step-header">🤖 Step 6: ELM Modeling with Resampling</div>', unsafe_allow_html=True)
    
    if st.session_state.feature_selected_data is None:
        st.warning("⚠️ Please complete feature engineering first!")
        return
    
    df = st.session_state.feature_selected_data
    X = df[st.session_state.feature_columns]
    y = df[st.session_state.target_column]
    
    # Train-test split
    st.markdown("### 🔄 Train-Test Split Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test size:", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Random state:", value=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    
    st.success(f"✅ Data split: Train({len(X_train)}) | Test({len(X_test)})")
    
    # Show class distribution
    st.markdown("### 📊 Class Distribution Analysis")
    
    class_distribution = y_train.value_counts()
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(class_distribution.to_frame('Count'), use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        class_distribution.plot(kind='bar', ax=ax, color='lightblue')
        ax.set_title('Training Set Class Distribution')
        ax.set_ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Calculate imbalance ratio
    if len(class_distribution) == 2:
        majority_class = class_distribution.max()
        minority_class = class_distribution.min()
        imbalance_ratio = majority_class / minority_class
        
        st.info(f"**Imbalance Ratio:** {imbalance_ratio:.2f}:1 (Majority:Minority)")
        
        if imbalance_ratio > 2:
            st.warning("⚠️ Dataset appears to be imbalanced. Consider using resampling techniques.")
        else:
            st.success("✅ Dataset appears to be relatively balanced.")
    
    # Modeling with different resampling techniques
    st.markdown("### 🎯 ELM Model Training with Resampling Techniques")
    
    # ELM parameters
    st.markdown("#### ⚙️ ELM Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_hidden = st.slider("Number of hidden neurons:", 10, 1000, 100, 10)
    with col2:
        activation = st.selectbox("Activation function:", ["relu", "sigmoid", "tanh"])
    with col3:
        enable_tuning = st.checkbox("Enable hyperparameter tuning", value=False)
    
    # Resampling techniques selection
    st.markdown("#### ⚖️ Resampling Techniques")
    
    resampling_methods = st.multiselect(
        "Select resampling methods to test:",
        ["No Resampling", "SMOTE", "ADASYN", "Tomek Links", "ENN", "SMOTE + Tomek", "SMOTE + ENN"],
        default=["No Resampling", "SMOTE", "ADASYN"]
    )
    
    if not resampling_methods:
        st.warning("Please select at least one resampling method!")
        return
    
    # Hyperparameter tuning configuration
    tuning_params = {}
    if enable_tuning:
        st.markdown("#### 🎛️ Hyperparameter Tuning Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hidden_range = st.multiselect("Hidden neurons to test:", [50, 100, 200, 300, 500, 1000], default=[100, 200, 300])
        with col2:
            activation_options = st.multiselect("Activations to test:", ["relu", "sigmoid", "tanh"], default=[activation])
        with col3:
            cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
        
        tuning_params = {
            'n_hidden': hidden_range,
            'activation': activation_options,
            'cv_folds': cv_folds
        }
    
    # Train models
    if st.button("🚀 Train ELM Models", key="train_elm_models"):
        if not resampling_methods:
            st.warning("Please select at least one resampling method!")
            return
        
        results = {}
        trained_models = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, method in enumerate(resampling_methods):
            status_text.text(f"Training with {method}...")
            
            try:
                # Apply resampling
                X_resampled, y_resampled = apply_resampling(X_train, y_train, method)
                
                if enable_tuning and len(tuning_params['n_hidden']) > 1:
                    # Hyperparameter tuning
                    best_params, best_score = tune_elm_hyperparameters(
                        X_resampled, y_resampled, tuning_params
                    )
                    
                    # Train best model
                    elm = ELMClassifier(
                        n_hidden=best_params['n_hidden'],
                        activation=best_params['activation'],
                        random_state=random_state
                    )
                    
                else:
                    # Use default parameters
                    elm = ELMClassifier(
                        n_hidden=n_hidden,
                        activation=activation,
                        random_state=random_state
                    )
                    best_params = {'n_hidden': n_hidden, 'activation': activation}
                    best_score = None
                
                # Train model
                elm.fit(X_resampled, y_resampled)
                
                # Make predictions
                y_pred = elm.predict(X_test)
                y_pred_proba = elm.predict_proba(X_test)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                if len(np.unique(y_test)) == 2:
                    metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                results[method] = {
                    'model': elm,
                    'metrics': metrics,
                    'best_params': best_params,
                    'cv_score': best_score,
                    'resampled_size': len(X_resampled)
                }
                
                trained_models[method] = elm
                
            except Exception as e:
                st.error(f"❌ Failed to train with {method}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(resampling_methods))
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            # Store results
            st.session_state.trained_models = trained_models
            st.session_state.elm_results = results
            
            # Find best model
            best_method = max(results.keys(), key=lambda x: results[x]['metrics']['f1'])
            st.session_state.best_model = results[best_method]['model']
            st.session_state.hyperparameter_results = {
                'model_name': f'ELM with {best_method}',
                'best_params': results[best_method]['best_params'],
                'best_score': results[best_method]['cv_score'] or results[best_method]['metrics']['f1'],
                'resampling_method': best_method
            }
            
            # Display results comparison
            st.markdown("### 📊 Results Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            for method, result in results.items():
                row = {
                    'Resampling Method': method,
                    'Training Size': result['resampled_size'],
                    **result['metrics']
                }
                if result['cv_score']:
                    row['CV Score'] = result['cv_score']
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Highlight best performers
            styled_df = comparison_df.style.highlight_max(
                subset=[col for col in comparison_df.columns if col not in ['Resampling Method', 'Training Size']],
                axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Performance visualization
            st.markdown("### 📈 Performance Visualization")
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            if 'auc' in results[list(results.keys())[0]]['metrics']:
                metrics_to_plot.append('auc')
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics_to_plot):
                if i < len(axes):
                    methods = list(results.keys())
                    values = [results[method]['metrics'][metric] for method in methods]
                    
                    bars = axes[i].bar(methods, values, color='skyblue', alpha=0.7)
                    axes[i].set_title(f'{metric.capitalize()} Comparison')
                    axes[i].set_ylabel(metric.capitalize())
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                                   f'{value:.3f}', ha='center', va='bottom')
            
            # Training size comparison
            if len(axes) > len(metrics_to_plot):
                methods = list(results.keys())
                sizes = [results[method]['resampled_size'] for method in methods]
                
                bars = axes[len(metrics_to_plot)].bar(methods, sizes, color='lightcoral', alpha=0.7)
                axes[len(metrics_to_plot)].set_title('Training Set Size After Resampling')
                axes[len(metrics_to_plot)].set_ylabel('Number of Samples')
                axes[len(metrics_to_plot)].tick_params(axis='x', rotation=45)
                axes[len(metrics_to_plot)].grid(True, alpha=0.3)
                
                for bar, size in zip(bars, sizes):
                    axes[len(metrics_to_plot)].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                                                  f'{size}', ha='center', va='bottom')
            
            # Hide unused subplots
            for j in range(len(metrics_to_plot) + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Best model summary
            st.markdown("### 🏆 Best Model Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Method", best_method)
            with col2:
                st.metric("Best F1-Score", f"{results[best_method]['metrics']['f1']:.4f}")
            with col3:
                st.metric("Hidden Neurons", results[best_method]['best_params']['n_hidden'])
            with col4:
                st.metric("Activation", results[best_method]['best_params']['activation'])
            
            st.success(f"✅ Successfully trained {len(results)} ELM models with different resampling techniques!")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Feature Engineering", key="back_to_feature_engineering"):
            st.session_state.current_page = 'feature_engineering'
            st.rerun()
    with col3:
        if st.button("Next: Evaluation →", key="to_evaluation", disabled=st.session_state.best_model is None):
            st.session_state.current_page = 'evaluation'
            st.rerun()

# Helper functions for resampling and hyperparameter tuning
def apply_resampling(X_train, y_train, method):
    """Apply the specified resampling method"""
    
    if method == "No Resampling":
        return X_train, y_train
    
    elif method == "SMOTE":
        sampler = SMOTE(random_state=42)
        
    elif method == "ADASYN":
        sampler = ADASYN(random_state=42)
        
    elif method == "Tomek Links":
        sampler = TomekLinks()
        
    elif method == "ENN":
        sampler = EditedNearestNeighbours()
        
    elif method == "SMOTE + Tomek":
        sampler = SMOTETomek(random_state=42)
        
    elif method == "SMOTE + ENN":
        sampler = SMOTEENN(random_state=42)
    
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    return sampler.fit_resample(X_train, y_train)

def tune_elm_hyperparameters(X_train, y_train, tuning_params):
    """Perform hyperparameter tuning for ELM using cross-validation"""
    from sklearn.model_selection import cross_val_score
    
    best_score = -1
    best_params = {}
    
    # Grid search over parameters
    for n_hidden in tuning_params['n_hidden']:
        for activation in tuning_params['activation']:
            
            # Create ELM model
            elm = ELMClassifier(
                n_hidden=n_hidden,
                activation=activation,
                random_state=42
            )
            
            # Perform cross-validation
            try:
                scores = cross_val_score(
                    elm, X_train, y_train, 
                    cv=tuning_params['cv_folds'], 
                    scoring='f1_weighted',
                    n_jobs=-1
                )
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'n_hidden': n_hidden,
                        'activation': activation
                    }
                    
            except Exception as e:
                continue
    
    return best_params, best_score

# Page 7: Model Evaluation (Modified for ELM)
def page_evaluation():
    st.markdown('<div class="step-header">📈 Step 7: Model Evaluation</div>', unsafe_allow_html=True)
    
    if st.session_state.best_model is None:
        st.warning("⚠️ Please complete ELM modeling first!")
        return
    
    best_model = st.session_state.best_model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Calculate comprehensive metrics
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                classification_report, confusion_matrix, roc_auc_score, roc_curve)
    
    # Basic metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    }
    
    if len(np.unique(y_test)) == 2:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    # Store evaluation results
    st.session_state.evaluation_results = {
        'metrics': metrics,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Display metrics
    st.markdown("### 📊 ELM Model Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics_list = list(metrics.items())
    
    for i, (metric, value) in enumerate(metrics_list):
        col = [col1, col2, col3, col4, col5][i % 5]
        with col:
            st.metric(metric, f"{value:.4f}")
    
    # Model information
    st.markdown("### ℹ️ Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Resampling Method:** {st.session_state.hyperparameter_results['resampling_method']}")
    with col2:
        st.info(f"**Hidden Neurons:** {st.session_state.hyperparameter_results['best_params']['n_hidden']}")
    with col3:
        st.info(f"**Activation:** {st.session_state.hyperparameter_results['best_params']['activation']}")
    
    # Detailed evaluation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Confusion Matrix", "📈 ROC Curve", "📋 Classification Report", "📊 Resampling Comparison"])
    
    with tab1:
        st.markdown("#### 🎯 Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Plotly confusion matrix
        fig = px.imshow(cm, 
                       text_auto=True, 
                       aspect="auto",
                       title="Confusion Matrix",
                       labels=dict(x="Predicted", y="Actual"),
                       color_continuous_scale='Blues')
        
        # Add labels
        unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
        fig.update_xaxes(tickvals=list(range(len(unique_labels))), ticktext=[f"Class {label}" for label in unique_labels])
        fig.update_yaxes(tickvals=list(range(len(unique_labels))), ticktext=[f"Class {label}" for label in unique_labels])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix interpretation
        if len(unique_labels) == 2:
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            **Confusion Matrix Breakdown:**
            - True Negatives (TN): {tn}
            - False Positives (FP): {fp}
            - False Negatives (FN): {fn}
            - True Positives (TP): {tp}
            
            **Interpretation:**
            - Sensitivity (Recall): {tp/(tp+fn):.4f}
            - Specificity: {tn/(tn+fp):.4f}
            - Precision: {tp/(tp+fp):.4f}
            """)
    
    with tab2:
        st.markdown("#### 📈 ROC Curve")
        
        if len(np.unique(y_test)) == 2:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            # Plot ROC curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ELM ROC Curve (AUC = {auc_score:.4f})', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Classifier', line=dict(dash='dash', color='red')))
            
            fig.update_layout(
                title='ROC Curve - ELM Model',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=600,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find optimal threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            st.info(f"**Optimal Threshold:** {optimal_threshold:.4f} (Youden's Index)")
            
        else:
            st.info("ROC curve is only available for binary classification.")
    
    with tab3:
        st.markdown("#### 📋 Detailed Classification Report")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Style the dataframe
        st.dataframe(
            report_df.style.highlight_max(axis=1, subset=['precision', 'recall', 'f1-score'])
                           .format(precision=4),
            use_container_width=True
        )
        
        # Class-wise performance
        st.markdown("#### 📊 Class-wise Performance")
        
        if len(np.unique(y_test)) <= 10:  # Only for reasonable number of classes
            classes = [str(cls) for cls in sorted(np.unique(y_test)) if str(cls) in report_df.index]
            
            if classes:
                metrics_to_plot = ['precision', 'recall', 'f1-score']
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                for i, metric in enumerate(metrics_to_plot):
                    values = [report_df.loc[cls, metric] for cls in classes]
                    axes[i].bar(classes, values, color='skyblue', alpha=0.7)
                    axes[i].set_title(f'{metric.capitalize()} by Class')
                    axes[i].set_ylabel(metric.capitalize())
                    axes[i].set_xlabel('Class')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for j, v in enumerate(values):
                        axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab4:
        st.markdown("#### 📊 Resampling Methods Comparison")
        
        if hasattr(st.session_state, 'elm_results'):
            results = st.session_state.elm_results
            
            # Create comparison dataframe
            comparison_data = []
            for method, result in results.items():
                comparison_data.append({
                    'Method': method,
                    'Accuracy': result['metrics']['accuracy'],
                    'Precision': result['metrics']['precision'],
                    'Recall': result['metrics']['recall'],
                    'F1-Score': result['metrics']['f1'],
                    'Training Size': result['resampled_size']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
                use_container_width=True
            )
            
            # Comparison visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                bars = ax.bar(comparison_df['Method'], comparison_df[metric], 
                            color='lightgreen', alpha=0.7)
                ax.set_title(f'{metric} Comparison Across Resampling Methods')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Highlight best performer
                best_idx = comparison_df[metric].idxmax()
                bars[best_idx].set_color('gold')
                
                # Add value labels
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{comparison_df[metric].iloc[j]:.3f}',
                           ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Best method summary
            best_method = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Method']
            st.success(f"🏆 **Best performing method:** {best_method}")
            
        else:
            st.info("No resampling comparison data available.")
    
    # Model summary
    st.markdown("---")
    st.markdown("### 🎯 ELM Model Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model Configuration:**")
        st.write(f"- **Algorithm:** Extreme Learning Machine (ELM)")
        st.write(f"- **Resampling Method:** {st.session_state.hyperparameter_results['resampling_method']}")
        st.write(f"- **Hidden Neurons:** {st.session_state.hyperparameter_results['best_params']['n_hidden']}")
        st.write(f"- **Activation Function:** {st.session_state.hyperparameter_results['best_params']['activation']}")
        st.write(f"- **Test Accuracy:** {metrics['Accuracy']:.4f}")
    
    with col2:
        st.markdown("**Dataset Information:**")
        st.write(f"- **Total Samples:** {len(st.session_state.feature_selected_data)}")
        st.write(f"- **Features:** {len(st.session_state.feature_columns)}")
        st.write(f"- **Train/Test Split:** {len(st.session_state.X_train)}/{len(st.session_state.X_test)}")
        st.write(f"- **Feature Selection:** {getattr(st.session_state, 'selected_features_method', 'Manual')}")
        st.write(f"- **Classes:** {len(np.unique(y_test))}")
    
    # Save model option
    st.markdown("### 💾 Save ELM Model")
    if st.button("💾 Save Trained ELM Model", key="save_elm_model"):
        try:
            # Save model and preprocessing components
            model_package = {
                'model': best_model,
                'model_type': 'ELM',
                'feature_columns': st.session_state.feature_columns,
                'target_column': st.session_state.target_column,
                'scaler': st.session_state.scaler,
                'label_encoders': st.session_state.label_encoders,
                'metrics': metrics,
                'hyperparameters': st.session_state.hyperparameter_results,
                'resampling_method': st.session_state.hyperparameter_results['resampling_method']
            }
            
            # Use joblib to save
            joblib.dump(model_package, 'trained_elm_model.joblib')
            st.success("✅ ELM model saved successfully as 'trained_elm_model.joblib'")
            
        except Exception as e:
            st.error(f"❌ Failed to save model: {str(e)}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Modeling", key="back_to_modeling"):
            st.session_state.current_page = 'modeling'
            st.rerun()
    with col3:
        if st.button("Next: Explainable AI →", key="to_xai"):
            st.session_state.current_page = 'xai'
            st.rerun()

# Page 8: Explainable AI (Modified for ELM)
def page_xai():
    st.markdown('<div class="step-header">🧠 Step 8: Explainable AI</div>', unsafe_allow_html=True)
    
    if st.session_state.evaluation_results is None:
        st.warning("⚠️ Please complete model evaluation first!")
        return
    
    best_model = st.session_state.best_model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    feature_columns = st.session_state.feature_columns
    
    st.markdown("""
    ### 🎯 Understanding ELM Model Predictions
    
    Since ELM (Extreme Learning Machine) is a neural network with randomly assigned input weights,
    traditional feature importance methods may not directly apply. However, we can still use
    model-agnostic explainability techniques:
    
    - **LIME (Local Interpretable Model-agnostic Explanations)**: Local explanations for individual predictions
    - **Permutation Feature Importance**: Global feature importance based on prediction changes
    - **Prediction Analysis**: Understanding model behavior and confidence
    """)
    
    # XAI Method Selection
    xai_method = st.selectbox(
        "Choose Explainability Method:",
        ["Permutation Feature Importance", "LIME Analysis", "Prediction Analysis"]
    )
    
    if xai_method == "Permutation Feature Importance":
        st.markdown("#### 🔄 Permutation Feature Importance")
        
        st.markdown("""
        **Permutation Feature Importance** measures how much model performance decreases 
        when a feature's values are randomly shuffled, breaking its relationship with the target.
        """)
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            n_repeats = st.slider("Number of permutations:", 5, 50, 10)
        with col2:
            scoring_metric = st.selectbox("Scoring metric:", ["accuracy", "f1_weighted"])
        
        if st.button("🔄 Calculate Permutation Importance", key="calc_perm_importance"):
            try:
                from sklearn.inspection import permutation_importance
                
                with st.spinner("Calculating permutation importance..."):
                    # Calculate permutation importance
                    perm_importance = permutation_importance(
                        best_model, X_test, y_test,
                        n_repeats=n_repeats,
                        scoring=scoring_metric,
                        random_state=42,
                        n_jobs=-1
                    )
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance_Mean': perm_importance.importances_mean,
                    'Importance_Std': perm_importance.importances_std
                }).sort_values('Importance_Mean', ascending=False)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Feature Importance Rankings")
                    st.dataframe(importance_df, use_container_width=True)
                
                with col2:
                    st.markdown("##### Importance Visualization")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Select top 15 features for visualization
                    top_features = importance_df.head(15)
                    
                    y_pos = np.arange(len(top_features))
                    bars = ax.barh(y_pos, top_features['Importance_Mean'], 
                                  xerr=top_features['Importance_Std'],
                                  color='lightblue', alpha=0.7, capsize=5)
                    
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_features['Feature'])
                    ax.set_xlabel('Permutation Importance')
                    ax.set_title('Top 15 Feature Importance (with std error bars)')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Interpretation
                st.markdown("##### 📖 Interpretation Guide")
                top_feature = importance_df.iloc[0]
                st.write(f"🏆 **Most important feature:** {top_feature['Feature']} (importance: {top_feature['Importance_Mean']:.4f})")
                
                significant_features = importance_df[importance_df['Importance_Mean'] > 0.01]
                st.write(f"📊 **Significant features (importance > 0.01):** {len(significant_features)} out of {len(importance_df)}")
                
                if len(significant_features) < len(importance_df):
                    st.info(f"💡 Consider removing features with very low importance for model simplification.")
                
            except Exception as e:
                st.error(f"❌ Permutation importance calculation failed: {str(e)}")
    
    elif xai_method == "LIME Analysis":
        st.markdown("#### 🍋 LIME Analysis for ELM")
        
        try:
            # Prepare data for LIME
            X_train_values = st.session_state.X_train.values
            
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                X_train_values,
                feature_names=feature_columns,
                class_names=[f'Class {i}' for i in sorted(np.unique(y_test))],
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            
            # Instance selection
            st.markdown("##### 🎯 Select Instance to Explain")
            
            # Show sample of test data with predictions
            sample_size = min(20, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            y_sample = y_test.loc[X_sample.index]
            pred_sample = best_model.predict(X_sample)
            pred_proba_sample = best_model.predict_proba(X_sample)
            
            # Create selection dataframe
            selection_df = pd.DataFrame({
                'Index': range(len(X_sample)),
                'Actual': y_sample.values,
                'Predicted': pred_sample,
                'Confidence': np.max(pred_proba_sample, axis=1),
                'Match': y_sample.values == pred_sample
            })
            
            st.dataframe(selection_df, use_container_width=True)
            
            # Select instance
            selected_idx = st.selectbox(
                "Choose instance to explain:",
                range(len(X_sample)),
                format_func=lambda x: f"Instance {x}: Actual={y_sample.iloc[x]}, Predicted={pred_sample[x]}, Conf={selection_df.iloc[x]['Confidence']:.3f}, {'✅' if y_sample.iloc[x] == pred_sample[x] else '❌'}"
            )
            
            if st.button("🍋 Generate LIME Explanation", key="generate_lime_elm"):
                with st.spinner("Generating LIME explanation for ELM..."):
                    # Get the instance to explain
                    instance = X_sample.iloc[selected_idx].values
                    
                    # Generate explanation
                    explanation = explainer.explain_instance(
                        instance,
                        best_model.predict_proba,
                        num_features=min(10, len(feature_columns)),
                        num_samples=500  # Reduced for ELM performance
                    )
                    
                    # Display explanation
                    st.markdown("##### 📊 LIME Explanation for ELM")
                    
                    # Get explanation as list
                    exp_list = explanation.as_list()
                    
                    # Create explanation dataframe
                    lime_df = pd.DataFrame(exp_list, columns=['Feature', 'Importance'])
                    lime_df = lime_df.sort_values('Importance', key=abs, ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(lime_df, use_container_width=True)
                    
                    with col2:
                        # Plot explanation
                        fig, ax = plt.subplots(figsize=(8, 6))
                        colors = ['green' if imp > 0 else 'red' for imp in lime_df['Importance']]
                        ax.barh(range(len(lime_df)), lime_df['Importance'], color=colors, alpha=0.7)
                        ax.set_yticks(range(len(lime_df)))
                        ax.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in lime_df['Feature']])
                        ax.set_xlabel('Feature Importance')
                        ax.set_title('LIME Feature Importance for ELM')
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    # Show prediction details
                    actual_class = y_sample.iloc[selected_idx]
                    predicted_class = pred_sample[selected_idx]
                    probabilities = pred_proba_sample[selected_idx]
                    
                    st.markdown("##### 🎯 ELM Prediction Details")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Actual Class", actual_class)
                    with col2:
                        st.metric("Predicted Class", predicted_class)
                    with col3:
                        st.metric("Prediction Confidence", f"{max(probabilities):.4f}")
                    
                    # Show feature values for this instance
                    st.markdown("##### 📋 Instance Feature Values")
                    instance_df = pd.DataFrame({
                        'Feature': feature_columns,
                        'Value': instance
                    })
                    st.dataframe(instance_df, use_container_width=True)
                    
                    # ELM-specific interpretation guide
                    st.markdown("""
                    ##### 📖 How to Interpret LIME Results for ELM:
                    - **Green bars**: Features that support the predicted class
                    - **Red bars**: Features that oppose the predicted class
                    - **Longer bars**: Features with stronger local influence
                    - **ELM Note**: ELM uses random input weights, so feature importance represents 
                      the local linear approximation of the ELM's decision boundary
                    """)
                    
        except Exception as e:
            st.error(f"❌ LIME analysis failed: {str(e)}")
            st.info("Please ensure all required libraries are installed and try again.")
    
    elif xai_method == "Prediction Analysis":
        st.markdown("#### 🔍 ELM Prediction Analysis")
        
        # Model behavior analysis
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Prediction confidence analysis
        st.markdown("##### 📊 Prediction Confidence Distribution")
        
        confidence_scores = np.max(y_pred_proba, axis=1)
        correct_predictions = (y_test == y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(confidence_scores[correct_predictions], bins=20, alpha=0.7, 
                   label='Correct Predictions', color='green', density=True)
            ax.hist(confidence_scores[~correct_predictions], bins=20, alpha=0.7, 
                   label='Incorrect Predictions', color='red', density=True)
            ax.set_xlabel('Prediction Confidence')
            ax.set_ylabel('Density')
            ax.set_title('ELM Prediction Confidence Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Confidence vs Accuracy
            confidence_bins = np.linspace(0, 1, 11)
            bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
            bin_accuracies = []
            bin_counts = []
            
            for i in range(len(confidence_bins) - 1):
                mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i+1])
                if mask.sum() > 0:
                    bin_accuracies.append(correct_predictions[mask].mean())
                    bin_counts.append(mask.sum())
                else:
                    bin_accuracies.append(0)
                    bin_counts.append(0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='skyblue')
            ax.set_xlabel('Confidence Bin')
            ax.set_ylabel('Accuracy')
            ax.set_title('ELM Accuracy vs Confidence')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for bar, count in zip(bars, bin_counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'n={count}', ha='center', va='bottom', fontsize=8)
            
            st.pyplot(fig)
            plt.close()
        
        # Class-wise prediction analysis
        st.markdown("##### 🎯 Class-wise ELM Performance")
        
        unique_classes = sorted(np.unique(y_test))
        class_analysis = []
        
        for cls in unique_classes:
            class_mask = (y_test == cls)
            class_predictions = y_pred[class_mask]
            class_confidences = confidence_scores[class_mask]
            
            analysis = {
                'Class': cls,
                'Total_Samples': class_mask.sum(),
                'Correct_Predictions': (class_predictions == cls).sum(),
                'Accuracy': (class_predictions == cls).mean(),
                'Avg_Confidence': class_confidences.mean(),
                'Min_Confidence': class_confidences.min(),
                'Max_Confidence': class_confidences.max()
            }
            class_analysis.append(analysis)
        
        class_df = pd.DataFrame(class_analysis)
        st.dataframe(class_df, use_container_width=True)
        
        # ELM Model insights
        st.markdown("##### 🧠 ELM Model Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Confidence", f"{confidence_scores.mean():.4f}")
            st.metric("High Confidence Predictions (>0.8)", f"{(confidence_scores > 0.8).sum()}")
        
        with col2:
            st.metric("Low Confidence Predictions (<0.6)", f"{(confidence_scores < 0.6).sum()}")
            st.metric("Hidden Neurons", st.session_state.hyperparameter_results['best_params']['n_hidden'])
        
        with col3:
            st.metric("Activation Function", st.session_state.hyperparameter_results['best_params']['activation'])
            st.metric("Resampling Method", st.session_state.hyperparameter_results['resampling_method'])
        
        # Model reliability assessment
        high_conf_correct = ((confidence_scores > 0.8) & correct_predictions).sum()
        high_conf_total = (confidence_scores > 0.8).sum()
        
        if high_conf_total > 0:
            high_conf_accuracy = high_conf_correct / high_conf_total
            st.success(f"✅ **High Confidence Accuracy:** {high_conf_accuracy:.4f} ({high_conf_correct}/{high_conf_total})")
        
        low_conf_incorrect = ((confidence_scores < 0.6) & ~correct_predictions).sum()
        low_conf_total = (confidence_scores < 0.6).sum()
        
        if low_conf_total > 0:
            st.info(f"⚠️ **Low Confidence Predictions:** {low_conf_total} samples need attention ({low_conf_incorrect} incorrect)")
    
    # Global Model Insights
    st.markdown("---")
    st.markdown("### 🌍 Global ELM Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Model Performance Summary")
        metrics = st.session_state.evaluation_results['metrics']
        for metric, value in metrics.items():
            st.write(f"**{metric}:** {value:.4f}")
    
    with col2:
        st.markdown("#### 🎯 Key Takeaways")
        
        # Generate insights based on model performance
        best_metric = max(metrics.items(), key=lambda x: x[1])
        worst_metric = min(metrics.items(), key=lambda x: x[1])
        
        st.write(f"🏆 **Strongest metric:** {best_metric[0]} ({best_metric[1]:.4f})")
        st.write(f"⚠️ **Weakest metric:** {worst_metric[0]} ({worst_metric[1]:.4f})")
        
        # ELM-specific insights
        resampling_method = st.session_state.hyperparameter_results['resampling_method']
        if resampling_method != "No Resampling":
            st.write(f"🔄 **Resampling helped:** Used {resampling_method}")
        
        if metrics.get('Accuracy', 0) > 0.9:
            st.write("✅ **Excellent** ELM performance")
        elif metrics.get('Accuracy', 0) > 0.8:
            st.write("✅ **Good** ELM performance")
        elif metrics.get('Accuracy', 0) > 0.7:
            st.write("⚠️ **Fair** ELM performance - consider more hidden neurons")
        else:
            st.write("❌ **Poor** ELM performance - try different activation or more data")
    
    # Export insights
    st.markdown("### 📥 Export ELM Model Insights")
    
    if st.button("📊 Generate ELM Model Report", key="generate_elm_report"):
        try:
            # Create comprehensive report
            report_data = {
                'Model Information': {
                    'Algorithm': 'Extreme Learning Machine (ELM)',
                    'Hidden Neurons': st.session_state.hyperparameter_results['best_params']['n_hidden'],
                    'Activation Function': st.session_state.hyperparameter_results['best_params']['activation'],
                    'Resampling Method': st.session_state.hyperparameter_results['resampling_method']
                },
                'Dataset Information': {
                    'Total Samples': len(st.session_state.feature_selected_data),
                    'Features': len(st.session_state.feature_columns),
                    'Train/Test Split': f"{len(st.session_state.X_train)}/{len(st.session_state.X_test)}",
                    'Feature Selection': getattr(st.session_state, 'selected_features_method', 'Manual'),
                    'Classes': len(np.unique(y_test))
                },
                'Performance Metrics': st.session_state.evaluation_results['metrics'],
                'Feature Columns': st.session_state.feature_columns
            }
            
            # Convert to JSON format for download
            import json
            report_json = json.dumps(report_data, indent=2, default=str)
            
            st.download_button(
                label="📄 Download ELM Model Report (JSON)",
                data=report_json,
                file_name="elm_model_report.json",
                mime="application/json"
            )
            
            # Also create a text summary
            report_text = f"""
# ELM (Extreme Learning Machine) Model Report

## Model Information
- Algorithm: Extreme Learning Machine (ELM)
- Hidden Neurons: {report_data['Model Information']['Hidden Neurons']}
- Activation Function: {report_data['Model Information']['Activation Function']}
- Resampling Method: {report_data['Model Information']['Resampling Method']}

## Dataset Information
- Total Samples: {report_data['Dataset Information']['Total Samples']}
- Features: {report_data['Dataset Information']['Features']}
- Train/Test Split: {report_data['Dataset Information']['Train/Test Split']}
- Feature Selection: {report_data['Dataset Information']['Feature Selection']}
- Classes: {report_data['Dataset Information']['Classes']}

## Performance Metrics
"""
            for metric, value in report_data['Performance Metrics'].items():
                report_text += f"- {metric}: {value:.4f}\n"
            
            report_text += f"""
## Features Used
{', '.join(report_data['Feature Columns'])}

## ELM Advantages
- Fast training due to random input weights
- No need for iterative tuning of input weights
- Good generalization performance
- Universal approximation capability

## Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            st.download_button(
                label="📝 Download ELM Model Report (Text)",
                data=report_text,
                file_name="elm_model_report.txt",
                mime="text/plain"
            )
            
            st.success("✅ ELM model report generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to generate report: {str(e)}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Evaluation", key="back_to_evaluation"):
            st.session_state.current_page = 'evaluation'
            st.rerun()
    with col2:
        if st.button("🔄 Start New Project", key="new_project"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                if key != 'current_page':
                    del st.session_state[key]
            st.session_state.current_page = 'upload'
            st.rerun()
    with col3:
        st.success("🎉 ELM Pipeline Complete!")

# Main application
def main():
    """Main application function"""
    
    # Load CSS
    load_css()
    
    # Sidebar navigation
    sidebar_navigation()
    
    # Page routing
    page_functions = {
        'upload': page_upload,
        'eda': page_eda,
        'preprocessing': page_preprocessing,
        'labeling': page_labeling,
        'feature_engineering': page_feature_engineering,
        'modeling': page_modeling,
        'evaluation': page_evaluation,
        'xai': page_xai
    }
    
    # Execute current page
    current_page = st.session_state.current_page
    if current_page in page_functions:
        page_functions[current_page]()
    else:
        st.error(f"Page '{current_page}' not found!")
        st.session_state.current_page = 'upload'
        st.rerun()

# Run the application
if __name__ == "__main__":
    main()
