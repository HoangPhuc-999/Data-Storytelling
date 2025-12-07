"""
Model Evaluation Visualizations
================================
Generate model performance plots and save to figures/model/
"""

import sys
from pathlib import Path
import warnings
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    f1_score, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import DataConfig

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Output directory
MODEL_DIR = DataConfig.FIGURES_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data_and_models():
    """Load featured data and trained models."""
    print("Loading data...")
    X_train = pd.read_csv(DataConfig.X_TRAIN_FEATURED_PATH)
    X_test = pd.read_csv(DataConfig.X_TEST_FEATURED_PATH)
    y_train = pd.read_csv(DataConfig.Y_TRAIN_PATH).values.ravel()
    y_test = pd.read_csv(DataConfig.Y_TEST_PATH).values.ravel()
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Load models if they exist
    models = {}
    model_files = list(DataConfig.MODELS_DIR.glob("*.pkl"))
    
    if model_files:
        print(f"\nLoading {len(model_files)} saved models...")
        for model_file in model_files:
            model_name = model_file.stem
            with open(model_file, 'rb') as f:
                models[model_name] = pickle.load(f)
            print(f"  - {model_name}")
    else:
        print("\nNo saved models found in models/")
        print("Please run: python src\\model\\training.py first")
        return None, None, None, None, None
    
    return X_train, X_test, y_train, y_test, models


def plot_confusion_matrices(models, X_test, y_test):
    """Plot 1: Confusion matrices for all models."""
    print("\n[1/7] Generating confusion matrices...")
    
    n_models = len(models)
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    ax=axes[idx], cbar=True, square=True,
                    xticklabels=[1,2,3,4], yticklabels=[1,2,3,4])
        
        axes[idx].set_title(f'{name}', fontsize=12, pad=10)
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: confusion_matrices.png")


def plot_classification_reports(models, X_test, y_test):
    """Plot 2: Classification metrics heatmap."""
    print("\n[2/7] Generating classification reports...")
    
    metrics_data = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        for severity in ['1', '2', '3', '4']:
            if severity in report:
                metrics_data.append({
                    'Model': name,
                    'Severity': severity,
                    'Precision': report[severity]['precision'],
                    'Recall': report[severity]['recall'],
                    'F1-Score': report[severity]['f1-score']
                })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create heatmaps for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, len(models)*1.5))
    
    for idx, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
        pivot = df_metrics.pivot(index='Model', columns='Severity', values=metric)
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                    ax=axes[idx], vmin=0, vmax=1, cbar_kws={'label': metric},
                    linewidths=0.5, linecolor='gray')
        
        axes[idx].set_title(f'{metric} by Severity', fontsize=12, pad=10)
        axes[idx].set_xlabel('Severity Class', fontsize=10)
        axes[idx].set_ylabel('Model', fontsize=10)
    
    plt.suptitle('Classification Metrics Heatmap', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'classification_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: classification_metrics_heatmap.png")


def plot_roc_curves(models, X_test, y_test):
    """Plot 3: ROC curves (One-vs-Rest for multiclass)."""
    print("\n[3/7] Generating ROC curves...")
    
    n_classes = 4
    classes = [1, 2, 3, 4]
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=classes)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for class_idx, severity in enumerate(classes):
        ax = axes[class_idx]
        
        for model_idx, (name, model) in enumerate(models.items()):
            # Get probability predictions
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
                
                # Compute ROC curve for this class
                fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score[:, class_idx])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=colors[model_idx % len(colors)], 
                       lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'ROC Curve - Severity {severity}', fontsize=11, pad=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('ROC Curves - All Severity Classes (One-vs-Rest)', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'roc_curves_multiclass.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: roc_curves_multiclass.png")


def plot_precision_recall_curves(models, X_test, y_test):
    """Plot 4: Precision-Recall curves."""
    print("\n[4/7] Generating precision-recall curves...")
    
    n_classes = 4
    classes = [1, 2, 3, 4]
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=classes)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for class_idx, severity in enumerate(classes):
        ax = axes[class_idx]
        
        for model_idx, (name, model) in enumerate(models.items()):
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
                
                precision, recall, _ = precision_recall_curve(
                    y_test_bin[:, class_idx], y_score[:, class_idx]
                )
                
                ax.plot(recall, precision, color=colors[model_idx % len(colors)], 
                       lw=2, label=f'{name}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(f'Precision-Recall Curve - Severity {severity}', fontsize=11, pad=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Precision-Recall Curves - All Severity Classes', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: precision_recall_curves.png")


def plot_feature_importance(models, X_test):
    """Plot 5: Feature importance comparison."""
    print("\n[5/7] Generating feature importance plots...")
    
    feature_names = X_test.columns.tolist()
    models_with_importance = {}
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            models_with_importance[name] = model.feature_importances_
    
    if not models_with_importance:
        print("   Skipped: No models with feature_importances_")
        return
    
    n_models = len(models_with_importance)
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 10))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, importances) in enumerate(models_with_importance.items()):
        # Get top 20 features
        indices = np.argsort(importances)[::-1][:20]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        axes[idx].barh(range(len(top_features)), top_importances, color='#5e9ce8')
        axes[idx].set_yticks(range(len(top_features)))
        axes[idx].set_yticklabels(top_features, fontsize=9)
        axes[idx].set_xlabel('Importance', fontsize=10)
        axes[idx].set_title(f'{name}\nTop 20 Features', fontsize=12, pad=10)
        axes[idx].invert_yaxis()
        
        # Add values on bars
        for i, v in enumerate(top_importances):
            axes[idx].text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
    
    plt.suptitle('Feature Importance Comparison', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: feature_importance.png")


def plot_performance_comparison(models, X_test, y_test):
    """Plot 6: Overall performance metrics comparison."""
    print("\n[6/7] Generating performance comparison...")
    
    metrics_data = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'Kappa': cohen_kappa_score(y_test, y_pred)
        })
    
    df_performance = pd.DataFrame(metrics_data)
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_performance))
    width = 0.15
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, metric in enumerate(metrics):
        offset = width * (idx - 2)
        bars = ax.bar(x + offset, df_performance[metric], width, 
                     label=metric, color=colors[idx], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Performance Comparison - Weighted Metrics', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df_performance['Model'], rotation=0, ha='center')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: performance_comparison.png")


def plot_class_distribution_predictions(models, X_test, y_test):
    """Plot 7: Predicted vs Actual class distribution."""
    print("\n[7/7] Generating prediction distribution plots...")
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5*(n_models+1), 5))
    
    # Plot actual distribution
    actual_counts = pd.Series(y_test).value_counts().sort_index()
    axes[0].bar(actual_counts.index, actual_counts.values, color='#2ca02c', alpha=0.7)
    axes[0].set_title('Actual Distribution', fontsize=12, pad=10)
    axes[0].set_xlabel('Severity', fontsize=10)
    axes[0].set_ylabel('Count', fontsize=10)
    axes[0].set_xticks([1, 2, 3, 4])
    
    for bar in axes[0].patches:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    # Plot predictions for each model
    for idx, (name, model) in enumerate(models.items(), start=1):
        y_pred = model.predict(X_test)
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        
        axes[idx].bar(pred_counts.index, pred_counts.values, color='#ff7f0e', alpha=0.7)
        axes[idx].set_title(f'{name}\nPredictions', fontsize=12, pad=10)
        axes[idx].set_xlabel('Severity', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].set_xticks([1, 2, 3, 4])
        axes[idx].set_ylim(axes[0].get_ylim())
        
        for bar in axes[idx].patches:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 50,
                          f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Severity Class Distribution - Actual vs Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'class_distribution_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: class_distribution_predictions.png")


def generate_all_model_plots():
    """Generate all model evaluation plots."""
    print("\n" + "="*70)
    print("GENERATING MODEL EVALUATION PLOTS")
    print("="*70)
    
    # Load data and models
    X_train, X_test, y_train, y_test, models = load_data_and_models()
    
    if models is None or len(models) == 0:
        print("\nCannot generate plots without trained models.")
        print("Please run: python src\\model\\training.py")
        return
    
    # Generate all plots
    plot_confusion_matrices(models, X_test, y_test)
    plot_classification_reports(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_precision_recall_curves(models, X_test, y_test)
    plot_feature_importance(models, X_test)
    plot_performance_comparison(models, X_test, y_test)
    plot_class_distribution_predictions(models, X_test, y_test)
    
    print("\n" + "="*70)
    print("ALL MODEL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Saved to: {MODEL_DIR}")
    print("="*70)


if __name__ == "__main__":
    generate_all_model_plots()
