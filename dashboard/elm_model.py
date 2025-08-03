import numpy as np
import optuna
from typing import Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


class ELMModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.W = None
        self.b = None 
        self.beta = None
        self.activation_type = None
        self.is_trained = False
        np.random.seed(random_state)
    
    def activation_function(self, x, func_type='sigmoid'):
        if func_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif func_type == 'tanh':
            return np.tanh(x)
        elif func_type == 'relu':
            return np.maximum(0, x)
        else:
            return x
    
    def elm_train(self, X_train, y_train, hidden_neurons, activation):
        np.random.seed(self.random_state)
        
        input_weights = np.random.randn(X_train.shape[1], hidden_neurons)
        biases = np.random.randn(hidden_neurons)
        H = self.activation_function(np.dot(X_train, input_weights) + biases, func_type=activation)
        output_weights = np.linalg.pinv(H).dot(y_train)
        
        return input_weights, biases, output_weights
    
    def elm_predict(self, X, input_weights, biases, output_weights, activation, threshold=0.5):
        H = self.activation_function(np.dot(X, input_weights) + biases, func_type=activation)
        y_pred_raw = np.dot(H, output_weights)
        y_pred = (y_pred_raw >= threshold).astype(int)
        
        return y_pred
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray,
                   hidden_neurons: int = 100, activation: str = 'sigmoid', 
                   threshold: float = 0.5):   
        # Train model
        self.W, self.b, self.beta = self.elm_train(X_train, y_train, hidden_neurons, activation)
        self.activation_type = activation  # Store activation type
        self.is_trained = True
        
        # Predict
        y_pred = self.elm_predict(X_test, self.W, self.b, self.beta, activation, threshold)
        
        # Calculate metrics
        results = {
            'mode': 'Manual Selection',
            'parameters': {
                'hidden_neurons': hidden_neurons,
                'activation': activation,
                'threshold': threshold
            },
            'metrics': {
                'accuracy': round(accuracy_score(y_test, y_pred), 4),
                'precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                'recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                'f1_score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            # NEW: Add model weights untuk LIME
            'model_weights': {
                'input_weights': self.W,
                'biases': self.b,
                'output_weights': self.beta,
                'activation': activation,
                'threshold': threshold
            }
        }
        
        return results
    
    def optimize_parameters(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          n_trials: int = 50):
        # Define objective function sesuai code Colab
        def objective(trial):
            activation = trial.suggest_categorical('activation', ['sigmoid', 'tanh', 'relu'])
            hidden_neurons = trial.suggest_int('hidden_neurons', 10, 200)
            threshold = trial.suggest_float('threshold', 0.1, 0.8, step=0.1)

            # Train
            W, b, beta = self.elm_train(X_train, y_train, hidden_neurons=hidden_neurons, activation=activation)

            # Predict
            y_pred = self.elm_predict(X_test, W, b, beta, activation=activation, threshold=threshold)

            # Evaluasi
            return accuracy_score(y_test, y_pred)
        
        # Jalankan Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best parameters
        best_params = study.best_trial.params
        
        # Train final model dengan parameter terbaik
        self.W, self.b, self.beta = self.elm_train(
            X_train, y_train,
            hidden_neurons=best_params['hidden_neurons'],
            activation=best_params['activation']
        )
        self.activation_type = best_params['activation']  # Store activation type
        self.is_trained = True
        
        # Final prediction dengan parameter terbaik
        y_pred_best = self.elm_predict(
            X_test, self.W, self.b, self.beta, 
            best_params['activation'], best_params['threshold']
        )
        
        # Calculate metrics
        results = {
            'mode': 'Optuna Optimization',
            'parameters': best_params,
            'best_score': round(study.best_value, 4),
            'total_trials': n_trials,
            'metrics': {
                'accuracy': round(accuracy_score(y_test, y_pred_best), 4),
                'precision': round(precision_score(y_test, y_pred_best, average='weighted', zero_division=0), 4),
                'recall': round(recall_score(y_test, y_pred_best, average='weighted', zero_division=0), 4),
                'f1_score': round(f1_score(y_test, y_pred_best, average='weighted', zero_division=0), 4)
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred_best).tolist(),
            'classification_report': classification_report(y_test, y_pred_best, output_dict=True, zero_division=0),
            # NEW: Add model weights untuk LIME
            'model_weights': {
                'input_weights': self.W,
                'biases': self.b,
                'output_weights': self.beta,
                'activation': best_params['activation'],
                'threshold': best_params['threshold']
            }
        }
        
        return results


# Main functions untuk dashboard (unchanged)
def train_elm_manual(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    hidden_neurons: int, activation: str, threshold: float,
                    random_state: int = 42):
    elm = ELMModel(random_state=random_state)
    results = elm.train_model(X_train, y_train, X_test, y_test, 
                             hidden_neurons, activation, threshold)
    return results


def optimize_elm_auto(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     n_trials: int = 50, random_state: int = 42):
    elm = ELMModel(random_state=random_state)
    results = elm.optimize_parameters(X_train, y_train, X_test, y_test, n_trials)
    return results


def get_activation_options():
    """Get activation function options untuk dropdown"""
    return {
        'sigmoid': 'Sigmoid (0,1)',
        'tanh': 'Tanh (-1,1)', 
        'relu': 'ReLU [0,∞)'
    }
