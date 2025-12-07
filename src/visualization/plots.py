"""
Data Visualization Module
=========================
Visualization utilities for EDA and model results.
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# DISTRIBUTION PLOTS
# ============================================================================

def plot_target_distribution(
    y: pd.Series,
    title: str = "Severity Distribution",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot target variable distribution with counts and percentages.
    
    Args:
        y: Target variable
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    value_counts = y.value_counts().sort_index()
    axes[0].bar(value_counts.index, value_counts.values, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Severity Level', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Severity Count Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(value_counts.values):
        axes[0].text(value_counts.index[i], v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Percentage plot
    percentages = (value_counts / len(y) * 100).values
    axes[1].bar(value_counts.index, percentages, color='coral', alpha=0.8)
    axes[1].set_xlabel('Severity Level', fontsize=12)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_title('Severity Percentage Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(percentages):
        axes[1].text(value_counts.index[i], v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    ncols: int = 3,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot distributions for multiple features.
    
    Args:
        df: DataFrame with features
        features: List of feature names
        ncols: Number of columns in subplot grid
        save_path: Path to save figure
        figsize: Figure size
    """
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if len(features) > 1 else [axes]
    
    for idx, feature in enumerate(features):
        if feature in df.columns:
            if df[feature].dtype in ['int64', 'float64']:
                df[feature].hist(bins=50, ax=axes[idx], color='skyblue', edgecolor='black', alpha=0.7)
            else:
                df[feature].value_counts().plot(kind='bar', ax=axes[idx], color='lightcoral', alpha=0.7)
            
            axes[idx].set_title(feature, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def plot_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = 'pearson',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame with features
        features: List of features (None = all numeric)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        save_path: Path to save figure
        figsize: Figure size
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr = df[features].corr(method=method)
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        mask=mask,
        annot=True, 
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_feature_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 20,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot top N features correlated with target.
    
    Args:
        X: Features DataFrame
        y: Target variable
        top_n: Number of top features to show
        save_path: Path to save figure
        figsize: Figure size
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns
    correlations = X[numeric_features].corrwith(y).abs().sort_values(ascending=False)
    
    top_features = correlations.head(top_n)
    
    plt.figure(figsize=figsize)
    colors = ['green' if x > 0.5 else 'orange' if x > 0.3 else 'red' for x in top_features.values]
    top_features.plot(kind='barh', color=colors, alpha=0.7)
    
    plt.xlabel('Absolute Correlation', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Features Correlated with Target', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


# ============================================================================
# TIME SERIES ANALYSIS
# ============================================================================

def plot_temporal_patterns(
    df: pd.DataFrame,
    time_col: str = 'Start_Time',
    target_col: str = 'Severity',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot temporal patterns in accident data.
    
    Args:
        df: DataFrame with time and target columns
        time_col: Time column name
        target_col: Target column name
        save_path: Path to save figure
        figsize: Figure size
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # By year
    yearly = df.groupby(df[time_col].dt.year).size()
    axes[0, 0].plot(yearly.index, yearly.values, marker='o', linewidth=2, markersize=8)
    axes[0, 0].set_title('Accidents by Year', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(alpha=0.3)
    
    # By month
    monthly = df.groupby(df[time_col].dt.month).size()
    axes[0, 1].bar(monthly.index, monthly.values, color='steelblue', alpha=0.7)
    axes[0, 1].set_title('Accidents by Month', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # By day of week
    dow = df.groupby(df[time_col].dt.dayofweek).size()
    dow.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(range(7), dow.values, color='coral', alpha=0.7)
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(dow.index)
    axes[1, 0].set_title('Accidents by Day of Week', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # By hour
    hourly = df.groupby(df[time_col].dt.hour).size()
    axes[1, 1].plot(hourly.index, hourly.values, marker='o', linewidth=2, markersize=6, color='green')
    axes[1, 1].set_title('Accidents by Hour of Day', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Temporal Patterns in Accident Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


# ============================================================================
# MODEL EVALUATION PLOTS
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        normalize: Normalize values
        save_path: Path to save figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['qwk', 'f1_macro', 'f1_weighted'],
    dataset: str = 'test',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot model comparison across multiple metrics.
    
    Args:
        results: Dictionary of model results
        metrics: List of metrics to compare
        dataset: 'train' or 'test'
        save_path: Path to save figure
        figsize: Figure size
    """
    comparison_data = []
    
    for model_name, model_results in results.items():
        metric_values = model_results.get(f'{dataset}_metrics', {})
        comparison_data.append({
            'Model': model_name,
            **{m: metric_values.get(m, 0) for m in metrics}
        })
    
    df = pd.DataFrame(comparison_data).set_index('Model')
    
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind='bar', ax=ax, alpha=0.8, width=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison ({dataset.upper()} Set)', fontsize=14, fontweight='bold')
    ax.legend(title='Metrics', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save figure
        figsize: Figure size
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importances[indices], alpha=0.7, color='teal')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


# ============================================================================
# GEOGRAPHIC VISUALIZATION
# ============================================================================

def plot_geographic_distribution(
    df: pd.DataFrame,
    state_col: str = 'State',
    top_n: int = 20,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot geographic distribution of accidents.
    
    Args:
        df: DataFrame with location data
        state_col: State column name
        top_n: Number of top states to show
        save_path: Path to save figure
        figsize: Figure size
    """
    state_counts = df[state_col].value_counts().head(top_n)
    
    plt.figure(figsize=figsize)
    state_counts.plot(kind='bar', color='mediumseagreen', alpha=0.7)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.title(f'Top {top_n} States by Accident Count', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Available functions:")
    print("  - plot_target_distribution")
    print("  - plot_feature_distributions")
    print("  - plot_correlation_matrix")
    print("  - plot_temporal_patterns")
    print("  - plot_confusion_matrix")
    print("  - plot_model_comparison")
    print("  - plot_feature_importance")
    print("  - plot_geographic_distribution")
