import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from typing import Dict, List, Tuple, Any, Optional

class LimeExplainer:
    def __init__(self):
        self.explainer = None
        self.feature_names = None
        
    def activation_function(self, x, func_type='sigmoid'):
        """Activation function yang sama dengan ELM model"""
        if func_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif func_type == 'tanh':
            return np.tanh(x)
        elif func_type == 'relu':
            return np.maximum(0, x)
        else:
            return x
    
    def create_elm_predict_function(self, model_weights: Dict):
        def predict_proba(X):
            if hasattr(X, 'values'):  # if DataFrame
                X = X.values
            
            # Forward pass
            H = self.activation_function(
                np.dot(X, model_weights['input_weights']) + model_weights['biases'],
                func_type=model_weights['activation']
            )
            logits = np.dot(H, model_weights['output_weights'])
            
            # Convert to probabilities
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))  # sigmoid
            
            # Return [prob_class_0, prob_class_1]
            if len(probs.shape) == 1:
                probs = probs.reshape(-1, 1)
            
            return np.column_stack([1 - probs, probs])
        
        return predict_proba
    
    def setup_explainer(self, X_train: np.ndarray, feature_names: List[str]):
        self.feature_names = feature_names
        
        self.explainer = LimeTabularExplainer(
            training_data=X_train,
            mode="classification",
            feature_names=feature_names,
            class_names=['Non Fraud', 'Fraud'],
            discretize_continuous=True,
            random_state=42
        )
        
        return self.explainer
    
    def explain_instance(self, instance: np.ndarray, model_weights: Dict, 
                        num_features: int = 10) -> Tuple[Any, np.ndarray]:
        if self.explainer is None:
            raise ValueError("Explainer belum di-setup! Panggil setup_explainer() dulu.")
        
        # Create prediction function
        predict_fn = self.create_elm_predict_function(model_weights)
        
        # Get prediction
        prediction_proba = predict_fn(np.array([instance]))[0]
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features
        )
        
        return explanation, prediction_proba
    
    def get_explanation_data(self, explanation) -> Dict:
        exp_list = explanation.as_list()
        
        explanation_data = {
            'features': [item[0] for item in exp_list],
            'importance': [item[1] for item in exp_list],
            'explanation_df': pd.DataFrame(exp_list, columns=['Feature', 'Importance'])
        }
        
        # Sort by absolute importance
        explanation_data['explanation_df']['Abs_Importance'] = explanation_data['explanation_df']['Importance'].abs()
        explanation_data['explanation_df'] = explanation_data['explanation_df'].sort_values('Abs_Importance', ascending=False)
        explanation_data['explanation_df']['Impact'] = explanation_data['explanation_df']['Importance'].apply(
            lambda x: 'Increases Fraud Risk' if x > 0 else 'Decreases Fraud Risk'
        )
        
        return explanation_data


# Integration functions dengan predict_pipeline
def create_lime_explainer_from_pipeline(pipeline_results: Dict) -> Optional[LimeExplainer]:
    if pipeline_results is None:
        return None
    
    try:
        lime_data = pipeline_results['lime_data']
        
        # Create explainer
        explainer = LimeExplainer()
        
        # Setup dengan training data
        X_train = lime_data['X_train'].values if hasattr(lime_data['X_train'], 'values') else lime_data['X_train']
        explainer.setup_explainer(X_train, lime_data['feature_names'])
        
        return explainer
        
    except Exception as e:
        print(f"Failed to create LIME explainer: {str(e)}")
        return None

def explain_test_instance(pipeline_results: Dict, instance_idx: int = 0, 
                         num_features: int = 10) -> Optional[Dict]:
    try:
        lime_data = pipeline_results['lime_data']
        
        # Create explainer
        explainer = create_lime_explainer_from_pipeline(pipeline_results)
        if explainer is None:
            return None
        
        # Get instance
        X_test = lime_data['X_test']
        y_test = lime_data['y_test']
        
        if instance_idx >= len(X_test):
            raise ValueError(f"Instance index {instance_idx} out of range (max: {len(X_test)-1})")
        
        instance = X_test.iloc[instance_idx].values if hasattr(X_test, 'iloc') else X_test[instance_idx]
        actual_label = y_test.iloc[instance_idx] if hasattr(y_test, 'iloc') else y_test[instance_idx]
        
        # Generate explanation
        explanation, prediction_proba = explainer.explain_instance(
            instance, lime_data['model_weights'], num_features
        )
        
        # Extract explanation data
        explanation_data = explainer.get_explanation_data(explanation)
        
        return {
            'explanation': explanation,
            'explanation_data': explanation_data,
            'prediction_proba': prediction_proba,
            'predicted_class': 'Fraud' if prediction_proba[1] > 0.5 else 'Non Fraud',
            'actual_label': int(actual_label),
            'actual_class': 'Fraud' if actual_label == 1 else 'Non Fraud',
            'instance_data': X_test.iloc[instance_idx] if hasattr(X_test, 'iloc') else None,
            'confidence': max(prediction_proba)
        }
        
    except Exception as e:
        print(f"Failed to explain instance: {str(e)}")
        return None

def explain_custom_instance(pipeline_results: Dict, custom_instance: np.ndarray, 
                           num_features: int = 10) -> Optional[Dict]:
    try:
        # Create explainer
        explainer = create_lime_explainer_from_pipeline(pipeline_results)
        if explainer is None:
            return None
        
        lime_data = pipeline_results['lime_data']
        
        # Generate explanation
        explanation, prediction_proba = explainer.explain_instance(
            custom_instance, lime_data['model_weights'], num_features
        )
        
        # Extract explanation data
        explanation_data = explainer.get_explanation_data(explanation)
        
        return {
            'explanation': explanation,
            'explanation_data': explanation_data,
            'prediction_proba': prediction_proba,
            'predicted_class': 'Fraud' if prediction_proba[1] > 0.5 else 'Non Fraud',
            'confidence': max(prediction_proba)
        }
        
    except Exception as e:
        print(f"Failed to explain custom instance: {str(e)}")
        return None
