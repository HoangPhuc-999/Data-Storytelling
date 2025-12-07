"""
Model Training and Evaluation Module
====================================
Train, optimize, and evaluate machine learning models.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import DataConfig

warnings.filterwarnings('ignore')


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

class OptimizedRounder:
    """
    Optimize decision thresholds to maximize Quadratic Weighted Kappa.
    
    Converts regression predictions to ordinal classifications by finding
    optimal thresholds that maximize agreement with true labels.
    """
    
    def __init__(self):
        """Initialize with default thresholds."""
        self.coef_ = [0.5, 1.5, 2.5]
    
    def _kappa_loss(self, coef: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate negative kappa score (for minimization).
        
        Args:
            coef: Threshold coefficients
            X: Predictions
            y: True labels
            
        Returns:
            Negative QWK score
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 1
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 2
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        
        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Find optimal thresholds.
        
        Args:
            X: Regression predictions
            y: True labels
        """
        loss_partial = lambda coef: self._kappa_loss(coef, X, y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = minimize(loss_partial, initial_coef, method='nelder-mead')['x']
    
    def predict(self, X: np.ndarray, coef: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply thresholds to convert predictions to classes.
        
        Args:
            X: Regression predictions
            coef: Custom thresholds (uses fitted if None)
            
        Returns:
            Class predictions
        """
        if coef is None:
            coef = self.coef_
        
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 1
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 2
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        
        return X_p
    
    def coefficients(self) -> np.ndarray:
        """Get fitted coefficients."""
        return self.coef_


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> RandomForestClassifier:
    """
    Train Random Forest classifier with class balancing.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model hyperparameters
        
    Returns:
        Trained model
    """
    default_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    if params:
        default_params.update(params)
    
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    
    return model


def train_random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> RandomForestRegressor:
    """
    Train Random Forest regressor (for ordinal regression).
    
    Args:
        X_train: Training features
        y_train: Training labels
        sample_weight: Sample weights for imbalanced data
        params: Model hyperparameters
        
    Returns:
        Trained model
    """
    default_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    if params:
        default_params.update(params)
    
    model = RandomForestRegressor(**default_params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> XGBRegressor:
    """
    Train XGBoost regressor.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sample_weight: Sample weights
        params: Model hyperparameters
        
    Returns:
        Trained model
    """
    default_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if params:
        default_params.update(params)
    
    model = XGBRegressor(**default_params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> LGBMRegressor:
    """
    Train LightGBM regressor.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sample_weight: Sample weights
        params: Model hyperparameters
        
    Returns:
        Trained model
    """
    default_params = {
        'objective': 'regression',
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    if params:
        default_params.update(params)
    
    model = LGBMRegressor(**default_params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None
) -> CatBoostRegressor:
    """
    Train CatBoost regressor.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sample_weight: Sample weights
        params: Model hyperparameters
        
    Returns:
        Trained model
    """
    default_params = {
        'loss_function': 'RMSE',
        'iterations': 300,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
        'random_state': 42,
        'verbose': 0
    }
    
    if params:
        default_params.update(params)
    
    model = CatBoostRegressor(**default_params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    return model


# ============================================================================
# SAMPLE WEIGHTING
# ============================================================================

def compute_sample_weights(y_train: pd.Series, method: str = 'balanced') -> np.ndarray:
    """
    Compute sample weights to handle class imbalance.
    
    Args:
        y_train: Training labels
        method: 'balanced' or 'custom'
        
    Returns:
        Sample weights array
    """
    from sklearn.utils.class_weight import compute_sample_weight
    
    if method == 'balanced':
        return compute_sample_weight('balanced', y_train)
    elif method == 'custom':
        # Custom weighting for severe imbalance
        class_counts = y_train.value_counts().sort_index()
        class_weights = {cls: len(y_train) / (len(class_counts) * count) 
                        for cls, count in class_counts.items()}
        return y_train.map(class_weights).values
    else:
        return None


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name for display
        
    Returns:
        Dictionary of metric scores
    """
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'qwk': qwk,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    
    print(f"\n{'='*60}")
    print(f"{model_name} Performance")
    print(f"{'='*60}")
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")
    print(f"F1 Score (Macro):        {f1_macro:.4f}")
    print(f"F1 Score (Weighted):     {f1_weighted:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return metrics


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_best_model(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path = None,
    metric: str = 'qwk'
) -> Tuple[str, Path]:
    """
    Save the best performing model to disk.
    
    Args:
        results: Dictionary of model results
        output_dir: Directory to save model (default: DataConfig.MODELS_DIR)
        metric: Metric to select best model
        
    Returns:
        Tuple of (best_model_name, saved_path)
    """
    if output_dir is None:
        output_dir = DataConfig.MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find best model
    best_score = -np.inf
    best_model_name = None
    best_model_data = None
    
    for model_name, model_results in results.items():
        test_metrics = model_results.get('test_metrics', {})
        score = test_metrics.get(metric, -np.inf)
        
        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model_data = model_results
    
    if best_model_name is None:
        raise ValueError("No valid models found in results")
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"best_model_{best_model_name}_{timestamp}.pkl"
    model_path = output_dir / model_filename
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_data['model'], f)
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'timestamp': timestamp,
        'best_metric': metric,
        'best_score': float(best_score),
        'train_metrics': {k: float(v) for k, v in best_model_data['train_metrics'].items()},
        'test_metrics': {k: float(v) for k, v in best_model_data['test_metrics'].items()}
    }
    
    metadata_path = output_dir / f"best_model_{best_model_name}_{timestamp}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL SAVED")
    print(f"{'='*70}")
    print(f"Model: {best_model_name}")
    print(f"QWK Score: {best_score:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*70}\n")
    
    return best_model_name, model_path


def load_saved_model(model_path: Path):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded from {model_path}")
    return model


def save_all_models(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path = None
) -> Dict[str, Path]:
    """
    Save all trained models to disk.
    
    Args:
        results: Dictionary of model results
        output_dir: Directory to save models (default: DataConfig.MODELS_DIR)
        
    Returns:
        Dictionary mapping model names to saved paths
    """
    if output_dir is None:
        output_dir = DataConfig.MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = {}
    
    for model_name, model_results in results.items():
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = output_dir / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_results['model'], f)
        
        saved_paths[model_name] = model_path
        print(f"✓ Saved {model_name} to {model_path}")
    
    return saved_paths


# ============================================================================
# FULL PIPELINE
# ============================================================================

def train_and_evaluate_models(
    X_train_path: Path,
    y_train_path: Path,
    X_test_path: Path,
    y_test_path: Path,
    models_to_train: list = None,
    save_best: bool = True,
    output_dir: Path = None
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate multiple models with threshold optimization.
    
    Args:
        X_train_path: Path to training features
        y_train_path: Path to training labels
        X_test_path: Path to test features
        y_test_path: Path to test labels
        models_to_train: List of model names to train
        save_best: Save best model to disk
        output_dir: Directory to save models
        
    Returns:
        Dictionary of results for each model
    """
    # Load data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    
    # Compute sample weights
    sample_weights = compute_sample_weights(y_train, method='balanced')
    
    # Default models
    if models_to_train is None:
        models_to_train = ['rf_regressor', 'xgboost', 'lightgbm', 'catboost']
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'#'*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'#'*60}")
        
        # Train model
        if model_name == 'rf_classifier':
            model = train_random_forest_classifier(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
        elif model_name == 'rf_regressor':
            model = train_random_forest_regressor(X_train, y_train, sample_weights)
            y_pred_train_raw = model.predict(X_train)
            y_pred_test_raw = model.predict(X_test)
            
            # Optimize thresholds
            rounder = OptimizedRounder()
            rounder.fit(y_pred_train_raw, y_train)
            
            y_pred_train = rounder.predict(y_pred_train_raw)
            y_pred_test = rounder.predict(y_pred_test_raw)
            
            print(f"Optimized thresholds: {rounder.coefficients()}")
            
        elif model_name == 'xgboost':
            model = train_xgboost(X_train, y_train, sample_weights)
            y_pred_train_raw = model.predict(X_train)
            y_pred_test_raw = model.predict(X_test)
            
            rounder = OptimizedRounder()
            rounder.fit(y_pred_train_raw, y_train)
            
            y_pred_train = rounder.predict(y_pred_train_raw)
            y_pred_test = rounder.predict(y_pred_test_raw)
            
            print(f"Optimized thresholds: {rounder.coefficients()}")
            
        elif model_name == 'lightgbm':
            model = train_lightgbm(X_train, y_train, sample_weights)
            y_pred_train_raw = model.predict(X_train)
            y_pred_test_raw = model.predict(X_test)
            
            rounder = OptimizedRounder()
            rounder.fit(y_pred_train_raw, y_train)
            
            y_pred_train = rounder.predict(y_pred_train_raw)
            y_pred_test = rounder.predict(y_pred_test_raw)
            
            print(f"Optimized thresholds: {rounder.coefficients()}")
            
        elif model_name == 'catboost':
            model = train_catboost(X_train, y_train, sample_weights)
            y_pred_train_raw = model.predict(X_train)
            y_pred_test_raw = model.predict(X_test)
            
            rounder = OptimizedRounder()
            rounder.fit(y_pred_train_raw, y_train)
            
            y_pred_train = rounder.predict(y_pred_train_raw)
            y_pred_test = rounder.predict(y_pred_test_raw)
            
            print(f"Optimized thresholds: {rounder.coefficients()}")
        
        # Evaluate
        print("\n--- TRAIN SET ---")
        train_metrics = evaluate_model(y_train, y_pred_train, f"{model_name} (Train)")
        
        print("\n--- TEST SET ---")
        test_metrics = evaluate_model(y_test, y_pred_test, f"{model_name} (Test)")
        
        results[model_name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    # Save best model
    if save_best and results:
        save_best_model(results, output_dir=output_dir, metric='qwk')
    
    return results


if __name__ == "__main__":
    results = train_and_evaluate_models(
        X_train_path=DataConfig.PROCESSED_DIR / 'X_train_featured.csv',
        y_train_path=DataConfig.PROCESSED_DIR / 'y_train.csv',
        X_test_path=DataConfig.PROCESSED_DIR / 'X_test_featured.csv',
        y_test_path=DataConfig.PROCESSED_DIR / 'y_test.csv',
        models_to_train=['rf_regressor', 'xgboost', 'lightgbm', 'catboost']
        # save_best=True by default, sử dụng DataConfig.MODELS_DIR
    )
