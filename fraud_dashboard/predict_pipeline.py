# predict_pipeline.py
import joblib
import numpy as np

# Fungsi aktivasi
def activation_function(x, func_type='sigmoid'):
    if func_type == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif func_type == 'tanh':
        return np.tanh(x)
    elif func_type == 'relu':
        return np.maximum(0, x)
    else:
        return x

# Fungsi prediksi manual ELM
def elm_predict(X, input_weights, biases, output_weights, activation):
    H = activation_function(np.dot(X, input_weights) + biases, func_type=activation)
    y_pred_raw = np.dot(H, output_weights)
    return y_pred_raw

# Load dan prediksi
def predict_fraud(X_scaled, model_path="hyperparameter_ELM.pkl"):
    """
    Prediksi fraud menggunakan model ELM custom hasil Optuna.

    Parameters:
        X_scaled: data input yang sudah discale
        model_path: lokasi file pickle model
    
    Returns:
        np.ndarray: array prediksi biner (0 = normal, 1 = fraud)
    """
    model = joblib.load(model_path)

    y_raw = elm_predict(
        X_scaled,
        model['input_weights'],
        model['biases'],
        model['output_weights'],
        model['activation_type']
    )
    
    # Ambil threshold dan ubah ke prediksi biner
    threshold = model['threshold']
    y_pred = (y_raw >= threshold).astype(int)

    return y_pred
