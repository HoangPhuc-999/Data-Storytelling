"""
EDA Visualizations - Exact Reproduction from Notebooks
======================================================
Reproduce all plots from 03_eda_processed.ipynb
"""

import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import DataConfig

warnings.filterwarnings('ignore')
sns.set_theme(style="white")

# Output directories
FIGURES_DIR = DataConfig.FIGURES_DIR
EDA_DIR = FIGURES_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load raw and processed data."""
    df_raw = pd.read_csv(DataConfig.TRAIN_RAW_PATH)
    df_processed = pd.read_csv(DataConfig.TRAIN_PROCESSED_PATH)
    return df_raw, df_processed


def plot_outlier_detection(df_processed):
    """Plot 1: Outlier Detection - Boxplots of Numerical Features."""
    print("[1/14] Outlier Detection Boxplots...")
    
    numerical_vars = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    n = len(numerical_vars)
    
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, n * 1.5))
    uniform_color = '#5e9ce8'
    
    for i, col in enumerate(numerical_vars):
        ax = axes[i] if n > 1 else axes
        
        sns.boxplot(
            x=df_processed[col],
            orient='h',
            color=uniform_color,
            width=0.5,
            ax=ax,
            fliersize=3,
            linewidth=1.2
        )
        
        ax.set_title(col, fontsize=12, loc='left', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        sns.despine(ax=ax, left=True, top=True, right=True)
    
    plt.suptitle("Outlier Detection - Boxplots of Numerical Features", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'outlier_detection_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_vif(df_processed):
    """Plot 2: Variance Inflation Factor (VIF)."""
    print("[2/14] VIF Analysis...")
    
    numerical_vars = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    target_var = 'Severity'
    X_numeric = df_processed[numerical_vars].drop(columns=[target_var])
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_numeric.columns
    vif_data['VIF'] = [variance_inflation_factor(X_numeric.dropna().values, i) 
                       for i in range(X_numeric.shape[1])]
    
    vif_data = vif_data.sort_values(by='VIF', ascending=False)
    
    vif_threshold = 5
    colors = ['#5e9ce8' if x >= vif_threshold else '#d3d3d3' for x in vif_data['VIF']]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='VIF', y='Feature', data=vif_data, palette=colors)
    sns.despine(left=True, bottom=True)
    
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.5, 
                p.get_y() + p.get_height() / 2, 
                f'{width:.2f}', 
                ha='left', va='center', fontsize=10, color='black')
    
    plt.axvline(x=vif_threshold, color='gray', linestyle='--', alpha=0.5)
    plt.text(vif_threshold + 0.5, len(vif_data)-0.5, f'Threshold ({vif_threshold})', 
             color='gray', fontsize=9, style='italic')
    
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.title('Variance Inflation Factor (VIF) per Numeric Feature', fontsize=14, loc='center')
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'vif_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_hist(column, df1, df2, save_name, color1="#5e9ce8", color2='#ff9f1c', figsize=(14,6)):
    """Plot comparison histograms for numerical features."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    
    x_min = min(df1[column].min(), df2[column].min())
    x_max = max(df1[column].max(), df2[column].max())
    
    data_pairs = [
        (axes[0], df1, 'Raw Data', color1),
        (axes[1], df2, 'Processed Data', color2)
    ]
    
    for ax, df, suffix, color in data_pairs:
        sns.histplot(df[column], kde=True, bins=50, ax=ax, 
                     color=color, alpha=0.6, edgecolor='white', linewidth=0.5)
        
        mean_val = df[column].mean()
        median_val = df[column].median()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'{suffix}', fontsize=14, color='#333333', loc='center', pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        sns.despine(ax=ax, left=True)
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.legend(loc='upper right', frameon=False, fontsize=10)
        ax.set_xlim(x_min, x_max)
    
    fig.suptitle(f"Distribution Comparison: {column}", fontsize=16, y=1.05)
    plt.savefig(EDA_DIR / save_name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_numerical_distributions(df_raw, df_processed):
    """Plots 3-5: Temperature, Humidity, Visibility distributions."""
    print("[3/14] Temperature Distribution...")
    plot_comparison_hist('Temperature(F)', df_raw, df_processed, 'distribution_temperature.png')
    
    print("[4/14] Humidity Distribution...")
    plot_comparison_hist('Humidity(%)', df_raw, df_processed, 'distribution_humidity.png')
    
    print("[5/14] Visibility Distribution...")
    plot_comparison_hist('Visibility(mi)', df_raw, df_processed, 'distribution_visibility.png')


def plot_categorical_comparison_sunrise(column, df1, df2, save_name, color1="#70a7e9", color2='#ff9f1c', figsize=(14, 6)):
    """Plot categorical comparison for Sunrise_Sunset."""
    u1 = df1[column].dropna().unique()
    u2 = df2[column].dropna().unique()
    unique_vals = set(u1) | set(u2)
    categories = sorted(list(unique_vals), key=str)
    
    count1 = df1[column].value_counts().max()
    count2 = df2[column].value_counts().max()
    
    if np.isnan(count1): count1 = 0
    if np.isnan(count2): count2 = 0
        
    max_count = max(count1, count2)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    
    data_pairs = [
        (axes[0], df1, 'Raw Data', color1),
        (axes[1], df2, 'Processed Data', color2)
    ]
    
    for ax, df, suffix, color in data_pairs:
        counts = df[column].value_counts().reindex(categories, fill_value=0)
        x_labels = [str(x) for x in categories]
        bars = ax.bar(x_labels, counts, color=color, alpha=0.85, width=0.6)
        
        ax.set_title(f'{suffix}', fontsize=14, color='#333333', loc='center', pad=15)
        ax.set_xlabel('')
        ax.set_axisbelow(True)
        sns.despine(ax=ax, left=True, right=True, top=True)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=11, length=0)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + (max_count*0.01), 
                        f'{int(height):,}', 
                        ha='center', va='bottom', fontsize=10, color='black')
    
    fig.suptitle(f"Categorical Distribution Comparison: {column}", fontsize=16, y=1.05)
    plt.savefig(EDA_DIR / save_name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_location_vs_severity(df1, df2, categorical_cols, colors=None, top_n=3):
    """Plots 7-9: City, County, State vs Severity."""
    colors = colors or ["#F32308", "#FF5D47", "#FFB5AD", "#FFE3E0"]
    
    for idx, feature in enumerate(categorical_cols, start=7):
        print(f"[{idx}/14] {feature} vs Severity...")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        for ax, df, title in zip(axes, [df1, df2], ['Raw', 'Processed']):
            if df[feature].nunique() > top_n:
                top_categories = df[feature].value_counts().head(top_n).index[::-1]
                plot_df = df[df[feature].isin(top_categories)]
                title_suffix = f' (Top {top_n})'
            else:
                plot_df = df
                top_categories = None
                title_suffix = ''
            
            severity_counts = plot_df.groupby([feature, 'Severity']).size().unstack(fill_value=0)
            if top_categories is not None:
                severity_counts = severity_counts.reindex(top_categories)
            severity_counts = severity_counts.reindex(columns=[1,2,3,4], fill_value=0)[[4,3,2,1]]
            
            severity_counts.plot(kind='barh', ax=ax, stacked=False, logx=True, width=0.8, color=colors)
            ax.set_title(f'{feature} vs Severity - {title}{title_suffix}', fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], title='Severity', fontsize=9, loc='lower right', frameon=False)
        
        plt.tight_layout()
        plt.savefig(EDA_DIR / f'{feature.lower()}_vs_severity.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_numerical_vs_severity(df1, df2, numerical_cols, colors=None):
    """Plots 10-12: Temperature, Humidity, Visibility vs Severity boxplots."""
    colors = colors or ['#70a7e9', '#ff9f1c']
    
    plot_names = {
        'Temperature(F)': 'temperature_vs_severity',
        'Humidity(%)': 'humidity_vs_severity',
        'Visibility(mi)': 'visibility_vs_severity'
    }
    
    for idx, col in enumerate(numerical_cols, start=10):
        print(f"[{idx}/14] {col} vs Severity...")
        
        y_min = min(df1[col].min(), df2[col].min())
        y_max = max(df1[col].max(), df2[col].max())
        y_range = y_max - y_min
        ylim = (y_min - y_range*0.05, y_max + y_range*0.05)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        
        flier_props = dict(marker='o', markerfacecolor='gray', markersize=3,
                           linestyle='none', markeredgecolor='none', alpha=0.4)
        
        data_pairs = [
            (axes[0], df1, 'Raw Data', colors[0]),
            (axes[1], df2, 'Processed Data', colors[1])
        ]
        
        for ax, df, suffix, color in data_pairs:
            sns.boxplot(
                data=df,
                x='Severity',
                y=col,
                ax=ax,
                color=color,
                width=0.6,
                linewidth=1.2,
                flierprops=flier_props
            )
            
            ax.set_title(f'{suffix}', fontsize=14, color='#333333', loc='center', pad=15)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_ylim(ylim)
            sns.despine(ax=ax)
            ax.tick_params(axis='y', length=0)
            ax.tick_params(axis='x', length=0)
        
        fig.suptitle(f'Distribution Comparison: {col} by Severity', fontsize=16, y=1.05)
        plt.savefig(EDA_DIR / f'{plot_names[col]}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_weather_condition_vs_severity(df_raw, df_processed):
    """Plot 13: Weather_Condition vs Severity."""
    print("[13/14] Weather_Condition vs Severity...")
    
    colors = ["#F32308", "#FF5D47", "#FFB5AD", "#FFE3E0"]
    feature = 'Weather_Condition'
    top_n = 3
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for ax, df, title in zip(axes, [df_raw, df_processed], ['Raw', 'Processed']):
        if df[feature].nunique() > top_n:
            top_categories = df[feature].value_counts().head(top_n).index[::-1]
            plot_df = df[df[feature].isin(top_categories)]
            title_suffix = f' (Top {top_n})'
        else:
            plot_df = df
            title_suffix = ''
        
        severity_counts = plot_df.groupby([feature, 'Severity']).size().unstack(fill_value=0)
        if df[feature].nunique() > top_n:
            severity_counts = severity_counts.reindex(top_categories)
        severity_counts = severity_counts.reindex(columns=[1,2,3,4], fill_value=0)[[4,3,2,1]]
        
        severity_counts.plot(kind='barh', ax=ax, stacked=False, logx=True, width=0.8, color=colors)
        ax.set_title(f'{feature} vs Severity - {title}{title_suffix}', fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Severity', fontsize=9, loc='lower right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'weather_condition_vs_severity.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_sunrise_sunset_vs_severity(df_raw, df_processed):
    """Plot 14: Sunrise_Sunset vs Severity."""
    print("[14/14] Sunrise_Sunset vs Severity...")
    
    colors = ["#F32308", "#FF5D47", "#FFB5AD", "#FFE3E0"]
    feature = 'Sunrise_Sunset'
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for ax, df, title in zip(axes, [df_raw, df_processed], ['Raw', 'Processed']):
        plot_df = df
        
        severity_counts = plot_df.groupby([feature, 'Severity']).size().unstack(fill_value=0)
        severity_counts = severity_counts.reindex(columns=[1,2,3,4], fill_value=0)[[4,3,2,1]]
        
        severity_counts.plot(kind='barh', ax=ax, stacked=False, logx=True, width=0.8, color=colors)
        ax.set_title(f'{feature} vs Severity - {title}', fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Severity', fontsize=9, loc='lower right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'sunrise_sunset_vs_severity.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(df_processed):
    """Plot 15: Correlation Matrix for Numerical Features."""
    print("[15/16] Correlation Matrix (Numerical)...")
    
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df_processed[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.75},
                vmin=-1, vmax=1,
                annot_kws={"size": 11})
    
    plt.title('Correlation Matrix - Numerical Features After Processing', fontsize=14, pad=15)
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'correlation_matrix_numerical.png', dpi=300, bbox_inches='tight')
    plt.close()


def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    
    if min_dim == 0:
        return 0
    
    cramers_v_stat = np.sqrt(chi2 / (n * min_dim))
    return cramers_v_stat


def plot_cramers_v_matrix(df_processed):
    """Plot 16: Cramér's V Matrix for Categorical Features."""
    print("[16/16] Cramér's V Matrix (Categorical)...")
    
    categorical_cols_for_corr = df_processed.select_dtypes(include=['object']).columns.drop(
        ['ID', 'Start_Time', 'End_Time', 'Description', 'Street', 'Zipcode', 
         'Weather_Timestamp', 'City', 'County', 'Airport_Code'], errors='ignore'
    ).tolist()
    
    if 'Severity' not in categorical_cols_for_corr:
        categorical_cols_for_corr.append('Severity')
    
    n_features = len(categorical_cols_for_corr)
    cramers_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                cramers_matrix[i, j] = 1.0
            elif i < j:
                v = cramers_v(df_processed[categorical_cols_for_corr[i]], 
                             df_processed[categorical_cols_for_corr[j]])
                cramers_matrix[i, j] = v
                cramers_matrix[j, i] = v
    
    cramers_df = pd.DataFrame(cramers_matrix, 
                              index=categorical_cols_for_corr, 
                              columns=categorical_cols_for_corr)
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cramers_df, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0.5,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.75},
                vmin=0, vmax=1,
                annot_kws={"size": 11})
    
    plt.title("Correlation Matrix - Categorical Features After Processing (Cramér's V)", fontsize=14, pad=15)
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(EDA_DIR / 'correlation_matrix_categorical_cramers_v.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_eda_plots():
    """Generate all EDA plots from notebook."""
    print("\n" + "="*70)
    print("GENERATING ALL EDA PLOTS FROM NOTEBOOK")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df_raw, df_processed = load_data()
    print(f"Raw: {df_raw.shape}, Processed: {df_processed.shape}")
    
    # Generate all plots
    plot_outlier_detection(df_processed)
    plot_vif(df_processed)
    plot_numerical_distributions(df_raw, df_processed)
    
    print("[6/14] Sunrise_Sunset Distribution...")
    plot_categorical_comparison_sunrise('Sunrise_Sunset', df_raw, df_processed, 
                                       'distribution_sunrise_sunset.png')
    
    plot_location_vs_severity(df_raw, df_processed, ['City', 'County', 'State'])
    plot_numerical_vs_severity(df_raw, df_processed, 
                              ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)'])
    plot_weather_condition_vs_severity(df_raw, df_processed)
    plot_sunrise_sunset_vs_severity(df_raw, df_processed)
    plot_correlation_matrix(df_processed)
    plot_cramers_v_matrix(df_processed)
    
    print("\n" + "="*70)
    print("ALL EDA PLOTS GENERATED SUCCESSFULLY!")
    print(f"Saved to: {EDA_DIR}")
    print("="*70)


if __name__ == "__main__":
    generate_all_eda_plots()
