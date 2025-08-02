import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go

# Resampling imports
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek


class ResamplingMethods:
    """Resampling methods for imbalanced dataset with dashboard integration"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.available_methods = {
            'SMOTE': self._apply_smote,
            'ADASYN': self._apply_adasyn, 
            'TomekLinks': self._apply_tomek,
            'ENN': self._apply_enn,
            'SMOTEENN': self._apply_smote_enn,
            'SMOTETomek': self._apply_smote_tomek
        }
    
    def _apply_smote(self, X, y) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply SMOTE resampling"""
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        method_info = {
            'method': 'SMOTE',
            'type': 'Over-sampling',
            'description': 'Synthetic Minority Oversampling Technique'
        }
        
        return X_resampled, y_resampled, method_info
    
    def _apply_adasyn(self, X, y) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply ADASYN resampling"""
        adasyn = ADASYN(random_state=self.random_state)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        method_info = {
            'method': 'ADASYN',
            'type': 'Over-sampling',
            'description': 'Adaptive Synthetic Sampling'
        }
        
        return X_resampled, y_resampled, method_info
    
    def _apply_tomek(self, X, y) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply TomekLinks resampling"""
        tl = TomekLinks()
        X_resampled, y_resampled = tl.fit_resample(X, y)
        
        method_info = {
            'method': 'TomekLinks',
            'type': 'Under-sampling',
            'description': 'Remove Tomek links (borderline examples)'
        }
        
        return X_resampled, y_resampled, method_info
    
    def _apply_enn(self, X, y) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply Edited Nearest Neighbours resampling"""
        enn = EditedNearestNeighbours()
        X_resampled, y_resampled = enn.fit_resample(X, y)
        
        method_info = {
            'method': 'ENN',
            'type': 'Under-sampling', 
            'description': 'Edited Nearest Neighbours'
        }
        
        return X_resampled, y_resampled, method_info
    
    def _apply_smote_enn(self, X, y) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply SMOTEENN resampling"""
        smote_enn = SMOTEENN(random_state=self.random_state)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        
        method_info = {
            'method': 'SMOTEENN',
            'type': 'Combined',
            'description': 'SMOTE + Edited Nearest Neighbours'
        }
        
        return X_resampled, y_resampled, method_info
    
    def _apply_smote_tomek(self, X, y) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply SMOTETomek resampling"""
        smote_tomek = SMOTETomek(random_state=self.random_state)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        method_info = {
            'method': 'SMOTETomek',
            'type': 'Combined',
            'description': 'SMOTE + Tomek Links'
        }
        
        return X_resampled, y_resampled, method_info
    
    def apply_resampling(self, X: pd.DataFrame, y: pd.Series, 
                        method: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply selected resampling method
        
        Args:
            X: Feature matrix (normalized)
            y: Target labels  
            method: Resampling method name
            
        Returns:
            resampled_df: Complete resampled dataframe
            stats: Comprehensive statistics for dashboard
        """
        
        if method not in self.available_methods:
            raise ValueError(f"Method '{method}' not available. Choose from: {list(self.available_methods.keys())}")
        
        # Store original distribution
        original_distribution = Counter(y)
        original_shape = X.shape
        
        # Apply resampling
        try:
            X_resampled, y_resampled, method_info = self.available_methods[method](X, y)
            
            # New distribution
            new_distribution = Counter(y_resampled)
            new_shape = X_resampled.shape
            
            # Create resampled dataframe
            resampled_df = pd.concat([
                pd.DataFrame(X_resampled, columns=X.columns),
                pd.DataFrame(y_resampled, columns=['fraud'])
            ], axis=1)
            
            # Calculate comprehensive stats
            stats = {
                'method_info': method_info,
                'original_data': {
                    'shape': original_shape,
                    'total_samples': len(y),
                    'distribution': dict(original_distribution),
                    'fraud_percentage': (original_distribution.get('Fraud', 0) / len(y)) * 100,
                    'imbalance_ratio': max(original_distribution.values()) / min(original_distribution.values()) if min(original_distribution.values()) > 0 else float('inf')
                },
                'resampled_data': {
                    'shape': new_shape,
                    'total_samples': len(y_resampled),
                    'distribution': dict(new_distribution),
                    'fraud_percentage': (new_distribution.get('Fraud', 0) / len(y_resampled)) * 100,
                    'imbalance_ratio': max(new_distribution.values()) / min(new_distribution.values()) if min(new_distribution.values()) > 0 else 1.0
                },
                'resampling_impact': {
                    'samples_added': len(y_resampled) - len(y),
                    'samples_removed': max(0, len(y) - len(y_resampled)),
                    'net_change': len(y_resampled) - len(y),
                    'size_change_percentage': ((len(y_resampled) - len(y)) / len(y)) * 100
                }
            }
            
            return resampled_df, stats
            
        except Exception as e:
            raise Exception(f"Resampling with {method} failed: {str(e)}")
    
    def compare_all_methods(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, Dict]:
        """
        Compare all resampling methods
        
        Returns:
            results: Dict of resampled dataframes for each method
            comparison_stats: Comparison statistics and visualizations
        """
        
        results = {}
        comparison_data = []
        original_distribution = Counter(y)
        
        # Apply each method
        for method_name in self.available_methods.keys():
            try:
                resampled_df, stats = self.apply_resampling(X, y, method_name)
                results[method_name] = resampled_df
                
                # Store comparison data
                comparison_data.append({
                    'Method': method_name,
                    'Type': stats['method_info']['type'],
                    'Original_Size': stats['original_data']['total_samples'],
                    'Resampled_Size': stats['resampled_data']['total_samples'],
                    'Size_Change': stats['resampling_impact']['net_change'],
                    'Change_Percentage': stats['resampling_impact']['size_change_percentage'],
                    'Final_Fraud_Percentage': stats['resampled_data']['fraud_percentage'],
                    'Final_Imbalance_Ratio': stats['resampled_data']['imbalance_ratio']
                })
                
            except Exception as e:
                print(f"Warning: {method_name} failed: {e}")
                comparison_data.append({
                    'Method': method_name,
                    'Type': 'Failed',
                    'Error': str(e)
                })
        
        # Create comparison stats
        comparison_stats = {
            'original_distribution': dict(original_distribution),
            'methods_tested': len(self.available_methods),
            'methods_successful': len([r for r in comparison_data if 'Error' not in r]),
            'comparison_table': comparison_data,
            'recommendations': self._generate_recommendations(comparison_data)
        }
        
        return results, comparison_stats
    
    def _generate_recommendations(self, comparison_data: list) -> Dict:
        """Generate recommendations based on resampling results"""
        successful_methods = [item for item in comparison_data if 'Error' not in item]
        
        if not successful_methods:
            return {'recommendation': 'No methods succeeded', 'reason': 'All methods failed'}
        
        # Find most balanced result
        best_balance = min(successful_methods, key=lambda x: abs(x['Final_Fraud_Percentage'] - 50))
        
        # Find method with least data change
        least_change = min(successful_methods, key=lambda x: abs(x['Change_Percentage']))
        
        recommendations = {
            'most_balanced': {
                'method': best_balance['Method'],
                'fraud_percentage': best_balance['Final_Fraud_Percentage'],
                'reason': 'Closest to 50-50 class balance'
            },
            'least_data_change': {
                'method': least_change['Method'],
                'change_percentage': least_change['Change_Percentage'],
                'reason': 'Minimal change to original dataset size'
            }
        }
        
        return recommendations


def create_resampling_visualizations(original_distribution: Dict, 
                                   resampled_distribution: Dict, 
                                   method_name: str) -> Dict:
    """Create before/after resampling visualizations"""
    
    visualizations = {}
    
    # 1. Distribution comparison bar chart
    categories = list(set(list(original_distribution.keys()) + list(resampled_distribution.keys())))
    
    original_counts = [original_distribution.get(cat, 0) for cat in categories]
    resampled_counts = [resampled_distribution.get(cat, 0) for cat in categories]
    
    fig_bar = go.Figure(data=[
        go.Bar(name='Before Resampling', x=categories, y=original_counts, marker_color='lightblue'),
        go.Bar(name=f'After {method_name}', x=categories, y=resampled_counts, marker_color='lightcoral')
    ])
    
    fig_bar.update_layout(
        title=f'📊 Class Distribution: Before vs After {method_name}',
        xaxis_title='Class Label',
        yaxis_title='Number of Samples',
        barmode='group'
    )
    
    visualizations['distribution_comparison'] = fig_bar
    
    # 2. Pie charts comparison
    fig_pie = go.Figure()
    
    # Before pie chart
    fig_pie.add_trace(go.Pie(
        labels=list(original_distribution.keys()),
        values=list(original_distribution.values()),
        name="Before",
        domain=dict(x=[0, 0.48]),
        title="Before Resampling"
    ))
    
    # After pie chart  
    fig_pie.add_trace(go.Pie(
        labels=list(resampled_distribution.keys()),
        values=list(resampled_distribution.values()),
        name="After", 
        domain=dict(x=[0.52, 1]),
        title=f"After {method_name}"
    ))
    
    fig_pie.update_layout(
        title=f'🥧 Class Balance Comparison: {method_name}',
        showlegend=True
    )
    
    visualizations['pie_comparison'] = fig_pie
    
    return visualizations


# Main functions for dashboard integration
def apply_resampling_method(X: pd.DataFrame, y: pd.Series, 
                          method: str, random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply single resampling method with dashboard stats
    
    Args:
        X: Feature matrix (should be normalized)
        y: Target labels
        method: Resampling method name
        random_state: Random seed
        
    Returns:
        resampled_df: Resampled dataframe
        stats: Statistics and visualizations for dashboard
    """
    
    resampler = ResamplingMethods(random_state=random_state)
    resampled_df, stats = resampler.apply_resampling(X, y, method)
    
    # Create visualizations
    visualizations = create_resampling_visualizations(
        stats['original_data']['distribution'],
        stats['resampled_data']['distribution'],
        method
    )
    
    stats['visualizations'] = visualizations
    
    return resampled_df, stats


def compare_resampling_methods(X: pd.DataFrame, y: pd.Series, 
                             random_state: int = 42) -> Tuple[Dict, Dict]:
    """
    Compare all available resampling methods
    
    Returns:
        results: Dict of resampled dataframes for each method
        comparison_stats: Comprehensive comparison statistics
    """
    
    resampler = ResamplingMethods(random_state=random_state)
    results, comparison_stats = resampler.compare_all_methods(X, y)
    
    # Create overall comparison visualization
    if comparison_stats['comparison_table']:
        comparison_df = pd.DataFrame([item for item in comparison_stats['comparison_table'] if 'Error' not in item])
        
        if not comparison_df.empty:
            # Methods comparison chart
            fig_comparison = px.bar(
                comparison_df,
                x='Method',
                y=['Original_Size', 'Resampled_Size'],
                title='📈 Dataset Size Comparison Across Methods',
                barmode='group'
            )
            
            # Fraud percentage comparison
            fig_fraud_pct = px.bar(
                comparison_df,
                x='Method',
                y='Final_Fraud_Percentage',
                title='⚖️ Final Fraud Percentage by Method',
                color='Type'
            )
            
            comparison_stats['visualizations'] = {
                'methods_comparison': fig_comparison,
                'fraud_percentage_comparison': fig_fraud_pct
            }
    
    return results, comparison_stats


def get_available_methods() -> Dict:
    """Get information about available resampling methods"""
    return {
        'SMOTE': {
            'type': 'Over-sampling',
            'description': 'Synthetic Minority Oversampling Technique - Creates synthetic examples',
            'best_for': 'Small minority class'
        },
        'ADASYN': {
            'type': 'Over-sampling', 
            'description': 'Adaptive Synthetic Sampling - Focuses on harder examples',
            'best_for': 'Complex decision boundaries'
        },
        'TomekLinks': {
            'type': 'Under-sampling',
            'description': 'Removes borderline examples between classes',
            'best_for': 'Cleaning noisy data'
        },
        'ENN': {
            'type': 'Under-sampling',
            'description': 'Edited Nearest Neighbours - Removes misclassified examples',
            'best_for': 'Removing noisy samples'
        },
        'SMOTEENN': {
            'type': 'Combined',
            'description': 'SMOTE followed by Edited Nearest Neighbours',
            'best_for': 'Balance dataset and clean noise'
        },
        'SMOTETomek': {
            'type': 'Combined',
            'description': 'SMOTE followed by Tomek Links removal',
            'best_for': 'Balance dataset and remove borderline cases'
        }
    }


# Utility function for easy integration
def prepare_data_for_resampling(df: pd.DataFrame, target_column: str = 'fraud') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare dataframe for resampling by separating features and target
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        
    Returns:
        X: Feature matrix
        y: Target series
    """
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y
