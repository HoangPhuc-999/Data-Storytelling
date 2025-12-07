"""
Evaluation Metrics Module
=========================
Metrics and evaluation utilities for model assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Quadratic Weighted Kappa score.
    
    QWK measures inter-rater agreement for ordinal data, where disagreements
    are weighted quadratically by their distance from perfect agreement.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        QWK score (0 to 1, higher is better)
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of metric scores
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'qwk': quadratic_weighted_kappa(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    detailed: bool = True
) -> Dict[str, float]:
    """
    Print comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model name for display
        detailed: Print detailed classification report
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = compute_all_metrics(y_true, y_pred)
    
    print(f"\n{'='*70}")
    print(f"{model_name} - Evaluation Report")
    print(f"{'='*70}")
    print(f"Accuracy:                {metrics['accuracy']:.4f}")
    print(f"Quadratic Weighted Kappa: {metrics['qwk']:.4f}")
    print(f"F1 Score (Macro):        {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted):     {metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro):       {metrics['precision_macro']:.4f}")
    print(f"Precision (Weighted):    {metrics['precision_weighted']:.4f}")
    print(f"Recall (Macro):          {metrics['recall_macro']:.4f}")
    print(f"Recall (Weighted):       {metrics['recall_weighted']:.4f}")
    
    if detailed:
        print(f"\n{'-'*70}")
        print("Classification Report:")
        print(f"{'-'*70}")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print(f"\n{'-'*70}")
        print("Confusion Matrix:")
        print(f"{'-'*70}")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"\nNormalized Confusion Matrix (%):")
        print(np.round(cm_normalized * 100, 2))
    
    print(f"{'='*70}\n")
    
    return metrics


def compare_models(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'qwk',
    dataset: str = 'test'
) -> pd.DataFrame:
    """
    Compare multiple models based on metrics.
    
    Args:
        results: Dictionary of model results from train_and_evaluate_models
        metric: Metric to sort by ('qwk', 'f1_macro', 'f1_weighted', etc.)
        dataset: 'train' or 'test'
        
    Returns:
        Comparison DataFrame sorted by metric
    """
    comparison = []
    
    for model_name, model_results in results.items():
        metrics = model_results.get(f'{dataset}_metrics', {})
        comparison.append({
            'Model': model_name,
            'QWK': metrics.get('qwk', 0),
            'F1 (Macro)': metrics.get('f1_macro', 0),
            'F1 (Weighted)': metrics.get('f1_weighted', 0)
        })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values(by='QWK', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"Model Comparison ({dataset.upper()} set)")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}\n")
    
    return df


def analyze_class_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Analyze per-class performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        DataFrame with per-class metrics
    """
    classes = np.unique(y_true)
    
    results = []
    for cls in classes:
        # Binary mask for this class
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        results.append({
            'Class': cls,
            'Support': np.sum(y_true_binary),
            'Precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'Recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'F1': f1_score(y_true_binary, y_pred_binary, zero_division=0)
        })
    
    df = pd.DataFrame(results)
    
    print("\nPer-Class Performance:")
    print(df.to_string(index=False))
    
    return df


def calculate_cost_sensitive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate cost-sensitive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        cost_matrix: Cost matrix for misclassifications
        
    Returns:
        Dictionary with cost-based metrics
    """
    if cost_matrix is None:
        # Default: quadratic distance cost
        n_classes = len(np.unique(y_true))
        cost_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                cost_matrix[i, j] = (i - j) ** 2
    
    cm = confusion_matrix(y_true, y_pred)
    total_cost = np.sum(cm * cost_matrix)
    avg_cost = total_cost / len(y_true)
    
    metrics = {
        'total_cost': total_cost,
        'avg_cost_per_sample': avg_cost,
        'normalized_cost': avg_cost / np.max(cost_matrix)
    }
    
    return metrics
