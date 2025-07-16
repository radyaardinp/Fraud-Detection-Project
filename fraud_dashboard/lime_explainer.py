# lime_explainer.py
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
from predict_pipeline import activation_function
from selected_features import FEATURES

def get_elm_predict_proba(model_dict):
    def predict(X):
        if hasattr(X, 'values'):  # if DataFrame
            X = X.values
        H = activation_function(np.dot(X, model_dict['input_weights']) + model_dict['biases'],
                                func_type=model_dict['activation_type'])
        logits = np.dot(H, model_dict['output_weights'])
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        return np.column_stack([1 - probs, probs])
    return predict

def explain_instance(instance_scaled_row, training_data_scaled, model_path='hyperparameter_ELM.pkl', num_features=5):
    """
    Buat penjelasan LIME untuk satu instance scaled.
    
    Params:
    - instance_scaled_row: 1D array, hasil normalisasi satu baris transaksi
    - training_data_scaled: 2D array dari data training hasil scaling (digunakan untuk explainer)
    - model_path: path ke model ELM pickle
    - num_features: berapa banyak fitur yang dijelaskan
    
    Returns:
    - exp: objek LIME explanation
    """
    model = joblib.load(model_path)
    elm_predict_proba = get_elm_predict_proba(model)

    explainer = LimeTabularExplainer(
        training_data=training_data_scaled,
        mode="classification",
        feature_names=FEATURES,
        class_names=['Non Fraud', 'Fraud'],
        discretize_continuous=True,
        random_state=42
    )

    exp = explainer.explain_instance(
        instance_scaled_row,
        elm_predict_proba,
        num_features=num_features
    )
    return exp
