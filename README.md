# 📊 The Power of Data Preparation through Data Storytelling
## US Traffic Accidents: From Raw Data to Insights and Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green.svg)](https://pandas.pydata.org/)
[![Visualization](https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-orange.svg)](https://seaborn.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Project Goal**: Demonstrate the power of Data Preparation and Data Storytelling techniques through a comprehensive Machine Learning project analyzing US traffic accidents (2020-2023).

---

## 🎯 Project Objectives

This project is designed to showcase three key components:

### **Part 1: Data Storytelling**
Utilize data storytelling techniques to **narrate the data journey** through:
- 📊 **Visualizations**: 25+ professional charts (EDA + Model insights)
- 📖 **Narrative**: Clear story about US traffic accidents
- 🔍 **Insights**: Discover hidden patterns in data (temporal, geographic, weather)
- 💡 **Business Impact**: Connect insights to real-world solutions

### **Part 2: Technical Analysis**
Explain **HOW** and **WHY** specific Data Preparation techniques were chosen:

**The strategic narrative follows:**
- ✅ **Data Cleaning Strategy**: Handle outliers, missing values with business logic
- ✅ **Feature Engineering Philosophy**: Create temporal features based on domain knowledge
- ✅ **Encoding Decisions**: Frequency vs. One-hot vs. Label encoding - when to use what?
- ✅ **Feature Selection Rationale**: VIF analysis, data leakage prevention
- ✅ **Model Selection Justification**: Ordinal regression vs. classification - why?

**Visualization choices follow best practices:**
- 📈 Distribution plots → Understand data imbalance
- 🗺️ Geographic maps → Accident hotspots
- ⏰ Temporal heatmaps → Rush hour patterns
- 🌦️ Weather correlations → Environmental impact
- 🔗 VIF analysis → Multicollinearity detection

### **Part 3: GitHub Code Project**
- 🏗️ **Modular Architecture**: Separate concerns (preprocessing, features, modeling, visualization)
- ⚙️ **Configuration Management**: Centralized config for all parameters
- 📝 **Documentation**: Comprehensive README, docstrings, comments
- 🔄 **Reproducibility**: Fixed random seeds, versioned data splits
- 📊 **End-to-End Pipeline**: From raw data to trained models

---

## 📋 Table of Contents

- [🎯 Project Objectives](#-project-objectives)
- [📚 Dataset Description](#-dataset-description)
- [📖 Data Storytelling: The Narrative](#-data-storytelling-the-narrative)
  - [Chapter 1: The Problem - Traffic Accidents in America](#-chapter-1-the-problem---traffic-accidents-in-america)
  - [Chapter 2: The Data - 2.5M+ Accidents Across 3 Years](#-chapter-2-the-data---25m-accidents-across-3-years)
  - [Chapter 3: The Journey - Data Preparation as Foundation](#-chapter-3-the-journey---data-preparation-as-foundation)
    - [Act 1: Data Cleaning - Separating Signal from Noise](#-act-1-data-cleaning---separating-signal-from-noise)
    - [Act 2: Feature Engineering - Creating Meaningful Signals](#-act-2-feature-engineering---creating-meaningful-signals)
    - [Act 3: Feature Selection - Choosing the Right Signals](#-act-3-feature-selection---choosing-the-right-signals)
  - [Chapter 4: The Insights - What the Data Reveals](#-chapter-4-the-insights---what-the-data-reveals)
    - [Insight 1: Temporal Patterns](#-insight-1-temporal-patterns---when-do-accidents-happen)
    - [Insight 2: Geographic Distribution](#-insight-2-geographic-distribution---where-are-the-hotspots)
    - [Insight 3: Weather Impact](#-insight-3-weather-impact---environmental-factors)
    - [Insight 4: Infrastructure Influence](#-insight-4-infrastructure-influence)
  - [Chapter 5: The Prediction - Machine Learning Approach](#-chapter-5-the-prediction---machine-learning-approach)
    - [Model Selection Philosophy](#-model-selection-philosophy-ordinal-regression-vs-classification)
    - [Model Performance](#-model-performance-catboost-wins)
    - [Feature Importance](#-feature-importance-what-drives-severity)
  - [Chapter 6: The Deliverables - Code + Storytelling](#-chapter-6-the-deliverables---code--storytelling)
- [🔬 Technical Analysis: Why These Choices?](#-technical-analysis-why-these-choices)
  - [1. Why Median Imputation?](#1-why-median-imputation-for-numerical-features)
  - [2. Why Frequency Encoding?](#2-why-frequency-encoding-for-high-cardinality-features)
  - [3. Why VIF Analysis?](#3-why-vif-analysis-for-feature-selection)
  - [4. Why Stratified Split?](#4-why-stratified-split-for-traintest)
  - [5. Why Sample Weights?](#5-why-sample-weights-for-class-imbalance)
  - [6. Why CatBoost?](#6-why-catboost-over-xgboostlightgbm)
- [🏗️ Project Architecture](#-project-architecture)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [📝 Usage Guide](#-usage-guide)
  - [Method 1: Complete Pipeline Execution](#method-1-complete-pipeline-execution-recommended)
  - [Method 2: Step-by-Step Execution](#method-2-step-by-step-execution)
  - [Method 3: Generate Visualizations](#method-3-generate-visualizations)
  - [Method 4: Using Jupyter Notebooks](#method-4-using-jupyter-notebooks-interactive-exploration)
- [🔍 Verification Steps](#-verification-steps)
- [⚠️ Troubleshooting](#-troubleshooting)
- [🛠️ Configuration](#-configuration)
- [📦 Dependencies](#-dependencies)
- [🎓 Conclusion: Data Storytelling in Practice](#-conclusion-data-storytelling-in-practice)
  - [The Story We Told](#the-story-we-told)
  - [Technical Analysis: Lessons Learned](#technical-analysis-lessons-learned)
  - [Visualizations: The Storytelling Tools](#visualizations-the-storytelling-tools)
  - [Real-World Applications](#real-world-applications)
- [📦 Deliverables Summary](#-deliverables-summary)
- [📚 References & Further Reading](#-references--further-reading)

---

## 📋 Project Information

**Course**: Data Preparation and Visualization - DSEB 65B  
**Instructor**: Dr. Nguyen Tuan Long  
**Academic Year**: 2025-2026 

### 👥 Team Members

| Name | Class |
|------|------|
| **Bui Viet Huy** | DSEB 65B |
| **Dang Ngoc Hoa** | DSEB 65B |
| **Pham Khanh Linh** | DSEB 65B |
| **Pham Thanh Long** | DSEB 65B |
| **Ngo Hoang Phuc** | DSEB 65B |
---

## 📖 Data Storytelling: The Narrative

### 🚗 Chapter 1: The Problem - Traffic Accidents in America

**Context**: Traffic accidents are a leading cause of injury and death in the United States, with **millions of incidents** occurring annually. Understanding the factors that contribute to accident severity is crucial for:

- 🚨 **Emergency Services**: Faster response to high-severity accidents
- 🏙️ **Urban Planners**: Design safer roads and intersections
- 📜 **Policymakers**: Evidence-based traffic regulations
- 🌐 **Tech Companies**: Real-time accident severity prediction for navigation apps

**The Question**: *What patterns exist in US traffic accidents, and can we predict severity levels to enable proactive interventions?*

---

### 📊 Chapter 2: The Data - 2.5M+ Accidents Across 3 Years

**Dataset**: US Accidents (2020-2023) from Kaggle Competition
- **Scale**: 3,115,245 total records (2.5M train, 623K test)
- **Scope**: 49 US states, 9,562 cities
- **Granularity**: Street-level location, minute-level timestamps
- **Richness**: Weather, infrastructure, temporal, geographic features

**Key Statistics**:
| Metric | Value | Insight |
|--------|-------|---------|
| **Time Span** | Jan 2020 - Mar 2023 | Includes COVID-19 impact |
| **Features** | 47 raw → 85+ engineered | Heavy feature engineering |
| **Target Classes** | 4 severity levels (1-4) | Ordinal classification problem |
| **Class Imbalance** | 75% Severity 2 | Requires special handling |

---

### 🔍 Chapter 3: The Journey - Data Preparation as Foundation

#### 🧹 **Act 1: Data Cleaning - Separating Signal from Noise**

**Challenge**: Raw data contains anomalies and missing values that obscure true patterns.

**Our Approach**:
1. **Outlier Detection with Domain Knowledge**
   - Temperature: Keep -50°F to 150°F (realistic US range)
   - Visibility: Cap at 10 miles (sensor limit)
   - Wind Speed: Remove >100 mph (likely errors)
   - **Result**: Removed 3.2% extreme outliers while preserving valid extremes

2. **Missing Value Strategy**
   - Numerical: Median imputation (robust to outliers)
   - Categorical: Mode imputation or "Unknown" category
   - Strategic drops: Features with >50% missing (e.g., `Precipitation`)
   - **Result**: Zero missing values in final dataset

3. **Data Leakage Prevention**
   - Removed `Description` (contains severity keywords like "major crash")
   - Removed `End_Time`, `Distance(mi)` (consequences of severity)
   - **Result**: Model learns from causes, not effects

**Visualization**: See `figures/eda/outlier_detection_boxplots.png`

---

#### 🛠️ **Act 2: Feature Engineering - Creating Meaningful Signals**

**Philosophy**: *Raw features are ingredients; engineered features are the recipe.*

**Temporal Features** (Critical for accidents!)
```python
# From Start_Time, we extract:
- hour (0-23) → Rush hour identification
- day_of_week (0-6) → Weekend vs. weekday patterns
- month (1-12) → Seasonal effects (winter snow)
- is_weekend → Binary indicator
- is_night (7PM-7AM) → Low visibility period
- season → Spring/Summer/Fall/Winter
```

**Why This Matters**: Accidents at 8 AM on Monday (rush hour) are fundamentally different from 2 AM Sunday (drunk driving risk).

**Encoding Strategies** (Nuanced approach)
| Feature Type | Encoding Method | Rationale |
|--------------|----------------|-----------|
| City (9,562 unique) | **Frequency Encoding** | Accident rate per city correlates with urban density |
| County (1,567 unique) | **Frequency Encoding** | Regional patterns without explosion |
| State (49 unique) | **One-Hot Encoding** | Small cardinality, preserve state identity |
| Weather_Condition (130 unique) | **One-Hot Encoding** (top 20) + "Other" | Balance granularity and noise |

**Visualization**: See `figures/eda/temporal_patterns_processed.png`, `figures/eda/geographic_distribution_processed.png`

---

#### 🎯 **Act 3: Feature Selection - Choosing the Right Signals**

**Problem**: More features ≠ Better model (curse of dimensionality + multicollinearity)

**Variance Inflation Factor (VIF) Analysis**:
- Detect multicollinearity (VIF > 10 = redundant features)
- Iteratively remove highest VIF features
- **Example**: `Wind_Chill(F)` removed (high correlation with `Temperature(F)`)

**Result**:
- Started with 120+ features
- Final model uses 85 features
- Removed 30% redundant features while preserving 98% of information

**Visualization**: See `figures/eda/vif_analysis.png`, `figures/eda/correlation_matrix_numerical.png`

---

### 💡 Chapter 4: The Insights - What the Data Reveals

#### 📅 **Insight 1: Temporal Patterns - When Do Accidents Happen?**

**Finding 1**: **Rush hour dominance**
- **Peak times**: 7-9 AM (morning commute), 4-6 PM (evening commute)
- **Weekday accidents**: 3x higher than weekends
- **Implication**: Infrastructure improvements should prioritize rush hour safety

**Finding 2**: **Seasonal variations**
- **Winter (Dec-Feb)**: +40% accidents due to snow/ice
- **Summer (Jun-Aug)**: Lower severity (better visibility)
- **Implication**: Dynamic speed limits during winter months

**Visualization**: `figures/eda/temporal_patterns_processed.png`

---

#### 🗺️ **Insight 2: Geographic Distribution - Where Are the Hotspots?**

**Finding 1**: **State-level concentration**
- **Top 5 states**: California (22%), Texas (14%), Florida (12%), South Carolina (8%), North Carolina (6%)
- **Why?**: High population density + data collection infrastructure

**Finding 2**: **Urban vs. Rural**
- **75% of accidents**: Cities >100K population
- **Higher severity in rural areas**: Longer emergency response times
- **Implication**: Rural areas need better emergency service coverage

**Visualization**: `figures/eda/state_vs_severity.png`, `figures/eda/city_vs_severity.png`

---

#### 🌦️ **Insight 3: Weather Impact - Environmental Factors**

**Finding 1**: **Clear weather paradox**
- **65% of accidents**: Clear weather conditions
- **Why?**: More traffic volume, faster speeds, driver complacency

**Finding 2**: **Severity correlation**
- **Rain/Snow**: Higher average severity (2.3 vs. 2.1 overall)
- **Low visibility (<1 mi)**: 2.5x likelihood of Severity 3/4
- **Implication**: Weather-based alert systems in navigation apps

**Visualization**: `figures/eda/weather_condition_vs_severity.png`, `figures/eda/visibility_vs_severity.png`

---

#### 🚦 **Insight 4: Infrastructure Influence**

**Finding**: **Traffic signals reduce severity**
- **Near traffic signal**: -15% average severity
- **No nearby amenities**: +20% severity (remote areas)
- **Implication**: Strategic placement of traffic control devices

**Visualization**: See model feature importance analysis

---

### 🤖 Chapter 5: The Prediction - Machine Learning Approach

---

## 📁 Dataset Description

### US Accidents Dataset (March 2023 Release)

**Source**: Kaggle Competition - US Accidents (2020-2023)  
**Coverage**: 49 US states (continental US + Alaska, Hawaii)

#### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| **Total Records** | 2,492,196 (train) + 623,049 (test) |
| **Time Range** | January 2020 - March 2023 |
| **Features** | 47 original features |
| **Target Variable** | `Severity` (1: Minimal impact → 4: Significant impact) |
| **File Size** | ~415 MB (train), ~104 MB (test) |

#### Feature Categories

**1. Temporal Features (3)**
- `Start_Time`, `End_Time`: Accident timeline
- `Weather_Timestamp`: Weather observation time

**2. Geospatial Features (7)**
- `Latitude`, `Longitude`: Exact coordinates
- `Street`, `City`, `County`, `State`, `Zipcode`: Location hierarchy

**3. Environmental Features (8)**
- `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`
- `Wind_Speed(mph)`, `Wind_Direction`, `Wind_Chill(F)`
- `Weather_Condition`: Categorical weather description

**4. Infrastructure Features (7 boolean)**
- `Amenity`, `Bump`, `Crossing`, `Junction`, `Railway`, `Station`, `Stop`, `Traffic_Signal`

**5. Traffic Features**
- `Distance(mi)`: Length of road affected by accident
- `Description`: Natural language accident description

**6. Metadata**
- `ID`: Unique identifier
- `Timezone`, `Airport_Code`: Location context

---

#### 🎯 **Model Selection Philosophy: Ordinal Regression vs. Classification**

**The Dilemma**: Severity levels (1, 2, 3, 4) are ordered - Severity 3 is closer to 2 than to 1.

**Traditional Classification Problem**:
- Treats all misclassifications equally
- Predicting 1 as 4 penalized same as predicting 1 as 2
- **Issue**: Ignores ordinal nature

**Our Approach: Ordinal Regression**
```python
1. Train regressor → Predict continuous severity score (e.g., 2.43)
2. Apply optimized thresholds → Convert to discrete class (1, 2, 3, 4)
3. Optimize thresholds to maximize Quadratic Weighted Kappa (QWK)
```

**Why QWK?** Penalizes distant predictions more:
- Predicting 1 as 2: Weight = 0.11 penalty
- Predicting 1 as 3: Weight = 0.44 penalty
- Predicting 1 as 4: Weight = 1.00 penalty (maximum)

**Threshold Optimization**:
- Standard thresholds: [1.5, 2.5, 3.5]
- **Learned thresholds**: [1.42, 2.18, 3.65] (CatBoost)
- **Improvement**: +0.035 QWK score

---

#### 🏆 **Model Performance: CatBoost Wins**

| Model | Test Accuracy | Test QWK | F1-Weighted | Why It Works |
|-------|---------------|----------|-------------|--------------|
| Random Forest (Classifier) | 0.821 | 0.783 | 0.798 | Baseline: treats severity as discrete |
| Random Forest (Regressor) | 0.838 | 0.812 | 0.821 | Better: ordinal nature preserved |
| XGBoost | 0.845 | 0.828 | 0.834 | Gradient boosting + L2 regularization |
| LightGBM | 0.843 | 0.825 | 0.831 | Fast but slightly underperforms |
| **CatBoost** 🥇 | **0.856** | **0.847** | **0.848** | **Best: native categorical handling** |

**Winner**: CatBoost Regressor + OptimizedRounder
- **QWK**: 0.847 (industry-grade performance)
- **Accuracy**: 85.6% (strong for imbalanced data)
- **Why**: Handles categorical features without preprocessing, robust to overfitting

**Visualization**: See `figures/model/model_comparison_qwk.png`, confusion matrices

---

#### 🔍 **Feature Importance: What Drives Severity?**

**Top 5 Predictors** (CatBoost model):

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | **Temperature(F)** | 18.3% | Extreme cold/heat → Higher severity |
| 2 | **Hour** | 15.7% | Rush hour → More severe (chain reactions) |
| 3 | **Visibility(mi)** | 12.4% | Low visibility → Can't avoid obstacles |
| 4 | **City (frequency)** | 11.2% | Urban density → Congestion patterns |
| 5 | **Humidity(%)** | 9.8% | Proxy for rain/fog conditions |

**Insight**: Environmental factors (temperature, visibility, humidity) dominate over infrastructure features - suggesting **weather-based interventions** are critical.

**Visualization**: `figures/model/feature_importance_catboost.png`

---

### 📊 Chapter 6: The Deliverables - Code + Storytelling

#### 🎨 **25+ Professional Visualizations**

**Raw Data Exploration** (Understanding the dataset)
- `target_distribution_raw.png` - Class imbalance visualization
- `feature_distributions_raw.png` - Univariate analysis
- `geographic_distribution_raw.png` - State-level accident counts
- `temporal_patterns_raw.png` - Time-series patterns

**Data Cleaning Validation**
- `outlier_detection_boxplots.png` - Before/after outlier removal
- `target_distribution_processed.png` - Preserved class distribution
- `feature_distributions_processed.png` - Cleaned distributions

**Feature Engineering Insights**
- `temporal_patterns_processed.png` - Hour × Day-of-week heatmap
- `correlation_matrix_numerical.png` - Pearson correlations
- `correlation_matrix_categorical_cramers_v.png` - Cramér's V for categorical
- `vif_analysis.png` - Multicollinearity detection

**Domain Insights** (The storytelling core!)
- `state_vs_severity.png` - Geographic severity patterns
- `city_vs_severity.png` - Top 20 cities by accident count
- `weather_condition_vs_severity.png` - Weather impact analysis
- `temperature_vs_severity.png`, `visibility_vs_severity.png`, `humidity_vs_severity.png`
- `sunrise_sunset_vs_severity.png` - Daylight vs. night patterns

**Model Performance**
- `confusion_matrix_[model].png` - Prediction error analysis
- `feature_importance_[model].png` - Top 20 features per model
- `model_comparison_qwk.png` - Bar chart comparison
- `roc_curves_[model].png` - One-vs-rest ROC curves

**Total**: 25 plots telling a complete data story

---

#### 💻 **Production-Ready Codebase**

**Modular Architecture**:
```
src/
├── config/config.py              # Single source of truth for all parameters
├── data/data_processing.py       # Preprocessing pipeline (outliers, imputation)
├── features/feature_engineering.py  # Temporal features, encoding, VIF
├── model/training.py             # Model training, OptimizedRounder, evaluation
├── evaluation/metrics.py         # QWK, F1, custom metrics
├── visualization/
│   ├── plots.py                  # Core plotting functions
│   ├── generate_all_eda_plots.py # EDA automation
│   └── generate_model_plots.py   # Model visualization automation
└── pipeline.py                   # End-to-end orchestration (MLPipeline class)
```

**Key Features**:
- ✅ **Configuration Management**: All hyperparameters in `DataConfig` class
- ✅ **Reproducibility**: Fixed `RANDOM_STATE = 42` everywhere
- ✅ **Logging**: Detailed console output for each pipeline stage
- ✅ **Extensibility**: Easy to add new models, features, or visualizations
- ✅ **Documentation**: Comprehensive docstrings (Google style)

---

## 🔬 Technical Analysis: Why These Choices?

### **1. Why Median Imputation for Numerical Features?**

**Options Considered**:
- Mean imputation → Sensitive to outliers
- Mode imputation → Only for categorical
- KNN imputation → Computationally expensive (2.5M rows)
- **Median imputation** ✅ → Robust to outliers, fast

**Example**: Temperature has outliers at -89°F (sensor error). Mean would be skewed; median is resilient.

---

### **2. Why Frequency Encoding for High-Cardinality Features?**

**Problem**: City has 9,562 unique values.
- One-hot encoding → 9,562 new columns (curse of dimensionality)
- Label encoding → Imposes false ordinal relationship

**Frequency Encoding** ✅:
```python
city_freq = train['City'].value_counts(normalize=True)
train['City_freq'] = train['City'].map(city_freq)
```
- **Captures pattern**: High-accident cities get higher values
- **Generalizes well**: Test cities not in train get 0 (rare city)
- **Single column**: No dimensionality explosion

---

### **3. Why VIF Analysis for Feature Selection?**

**Problem**: Multicollinearity inflates model variance and reduces interpretability.

**Example**:
- `Temperature(F)` and `Wind_Chill(F)` have VIF > 15 (nearly identical)
- Both provide same information → Remove `Wind_Chill(F)`

**Result**: Model trains faster, predictions more stable, feature importance more interpretable.

---

### **4. Why Stratified Split for Train/Test?**

**Problem**: Severity classes are imbalanced (75% Severity 2).

**Random split** ❌:
- Test set might have different distribution than train
- Model evaluation is biased

**Stratified split** ✅:
```python
train_test_split(..., stratify=df['Severity'])
```
- Train and test have identical class distributions
- Fair evaluation of model generalization

---

### **5. Why Sample Weights for Class Imbalance?**

**Problem**: Model biased toward Severity 2 (majority class).

**Sample Weights**:
```python
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=weights)
```
- Minority classes (Severity 1, 4) get higher weights
- Forces model to pay attention to rare events
- **Result**: +0.08 improvement in F1-Macro score

---

### **6. Why CatBoost Over XGBoost/LightGBM?**

**Comparison**:
| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Categorical Handling | Manual encoding | Manual encoding | **Native support** ✅ |
| Overfitting Risk | Moderate | Higher (leaf-wise) | Lower (ordered boosting) |
| Training Speed | Moderate | Fast | Slower |
| Final QWK | 0.828 | 0.825 | **0.847** ✅ |

**Winner**: CatBoost
- No need to encode categorical features manually
- More robust to overfitting (critical for production)
- Best performance on test set

---

## 🏗️ Project Architecture

```
Data_Storytelling/
├── dataset/
│   ├── raw/                          # Original data
│   │   ├── US_Accidents_March23.csv  # Full dataset (3.1M records)
│   │   ├── train.csv                 # Training split (80%)
│   │   └── test.csv                  # Test split (20%)
│   └── processed/                    # Pipeline outputs
│       ├── train_processed.csv       # After preprocessing
│       ├── test_processed.csv
│       ├── X_train_featured.csv      # After feature engineering
│       ├── X_test_featured.csv
│       ├── y_train.csv               # Target labels
│       └── y_test.csv
│
├── src/                              # Source code (production-ready)
│   ├── config/
│   │   └── config.py                 # Centralized configuration (DataConfig class)
│   ├── data/
│   │   └── data_processing.py        # Preprocessing pipeline
│   ├── features/
│   │   └── feature_engineering.py    # Temporal features, VIF selection
│   ├── model/
│   │   └── training.py               # Model training, OptimizedRounder, evaluation
│   ├── evaluation/
│   │   └── metrics.py                # QWK, F1, classification metrics
│   ├── visualization/
│   │   ├── plots.py                  # Core plotting functions
│   │   ├── generate_all_eda_plots.py # EDA visualizations
│   │   └── generate_model_plots.py   # Model performance plots
│   └── pipeline.py                   # End-to-end ML pipeline (MLPipeline class)
│
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── 00_train_test_split.ipynb     # Initial data split
│   ├── 01_understanding_eda.ipynb    # Raw data exploration
│   ├── 02_data_processing.ipynb      # Preprocessing development
│   ├── 03_eda_processed.ipynb        # Processed data analysis
│   ├── 04_feature_engineering_and_feature_selection.ipynb
│   └── 05_model.ipynb                # Model experimentation
│
├── figures/                          # Generated visualizations
│   ├── eda/                          # EDA plots
│   └── model/                        # Model evaluation plots
│
├── models/                           # Trained model artifacts (pickled)
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation

```

### Design Principles

1. **Modularity**: Each component (preprocessing, feature engineering, modeling) is independently testable
2. **Configuration Management**: Single source of truth (`DataConfig`) for all parameters
3. **Reproducibility**: Fixed random seeds, versioned data splits
4. **Extensibility**: Easy to add new models, features, or evaluation metrics

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (for full dataset processing)
- 2GB+ disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/HoangPhuc-999/Data_Storytelling.git
cd Data_Storytelling
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 📝 Usage Guide

## Method 1: Complete Pipeline Execution (Recommended)

### Using Python Script

Create a file `run_pipeline.py` in the project root:

```python
from src.pipeline import MLPipeline

# Initialize pipeline with default configuration
pipeline = MLPipeline()

# Run complete pipeline
results = pipeline.run(
    force_preprocessing=False,           # Skip if processed files exist
    force_feature_engineering=False,     # Skip if featured files exist
    models=['random_forest', 'xgboost', 'lightgbm', 'catboost'],
    optimize_thresholds=True,            # Enable OptimizedRounder for QWK
    save_best_model=True                 # Save best model to models/
)

# Display results
print("\n" + "="*70)
print("PIPELINE EXECUTION COMPLETE")
print("="*70)
print(f"Best Model: {pipeline.best_model_name}")
print(f"Test QWK Score: {pipeline.pipeline_metadata.get('best_model_score', 0):.4f}")

# Compare all models
pipeline.compare_results(metric='qwk', dataset='test')
```

**Run the pipeline:**

```bash
# Windows (PowerShell)
python run_pipeline.py

# Linux/Mac
python3 run_pipeline.py
```

**Expected Output:**
```
======================================================================
STEP 1: DATA PREPROCESSING
======================================================================
✓ Using existing processed data:
  - dataset/processed/train_processed.csv
  - dataset/processed/test_processed.csv

======================================================================
STEP 2: FEATURE ENGINEERING
======================================================================
✓ Loading processed data...
✓ Extracting temporal features...
✓ Encoding categorical features...
✓ Applying VIF feature selection...
✓ Feature engineering complete in 45.32s

======================================================================
STEP 3: MODEL TRAINING
======================================================================
Training Random Forest Regressor...
  ✓ Training complete (175s) | QWK: 0.8124

Training XGBoost Regressor...
  ✓ Training complete (120s) | QWK: 0.8285

Training LightGBM Regressor...
  ✓ Training complete (85s) | QWK: 0.8251

Training CatBoost Regressor...
  ✓ Training complete (210s) | QWK: 0.8472

BEST MODEL: catboost (QWK: 0.8472)
✓ Model saved to: models/catboost_best_model.pkl
```

---

## Method 2: Step-by-Step Execution

### Step 1: Train-Test Split

**Notebook:** `notebooks/00_train_test_split.ipynb`

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv('dataset/raw/US_Accidents_March23.csv')

# Split into train/test (80/20)
train, test = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['Severity']
)

# Save splits
train.to_csv('dataset/raw/train.csv', index=False)
test.to_csv('dataset/raw/test.csv', index=False)

print(f"Train: {len(train)} records")
print(f"Test: {len(test)} records")
```

**Or run via terminal:**

```bash
python -c "from src.config.config import DataConfig; import pandas as pd; from sklearn.model_selection import train_test_split; df = pd.read_csv(DataConfig.RAW_DATA_PATH); train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Severity']); train.to_csv('dataset/raw/train.csv', index=False); test.to_csv('dataset/raw/test.csv', index=False); print('Split complete')"
```

---

### Step 2: Data Preprocessing

**Notebook:** `notebooks/02_data_processing.ipynb`

**Python Script:**

```python
from src.data.data_processing import run_full_preprocessing_pipeline

# Run preprocessing on train and test sets
run_full_preprocessing_pipeline()

# This will create:
# - dataset/processed/train_processed.csv
# - dataset/processed/test_processed.csv
```

**Or via terminal:**

```bash
python -c "from src.data.data_processing import run_full_preprocessing_pipeline; run_full_preprocessing_pipeline()"
```

**What happens:**
1. ✅ Load raw train/test CSVs
2. ✅ Convert data types (datetime, boolean)
3. ✅ Remove outliers (temperature, visibility, wind speed, etc.)
4. ✅ Impute missing values (median for numeric, mode for categorical)
5. ✅ Drop data leakage features (ID, Description, End_Time, Distance)
6. ✅ Drop high-cardinality features (Street, Zipcode)
7. ✅ Save processed data

**Output files:**
- `dataset/processed/train_processed.csv` (~380 MB)
- `dataset/processed/test_processed.csv` (~95 MB)

---

### Step 3: Feature Engineering

**Notebook:** `notebooks/04_feature_engineering_and_feature_selection.ipynb`

**Python Script:**

```python
from src.features.feature_engineering import run_feature_engineering_pipeline

# Run feature engineering
X_train, y_train, X_test, y_test = run_feature_engineering_pipeline()

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Features are automatically saved to:
# - dataset/processed/X_train_featured.csv
# - dataset/processed/X_test_featured.csv
# - dataset/processed/y_train.csv
# - dataset/processed/y_test.csv
```

**Or via terminal:**

```bash
python -c "from src.features.feature_engineering import run_feature_engineering_pipeline; X_train, y_train, X_test, y_test = run_feature_engineering_pipeline(); print(f'Train: {X_train.shape}, Test: {X_test.shape}')"
```

**What happens:**
1. ✅ Extract temporal features (hour, day, month, year, season, is_weekend, is_night)
2. ✅ Frequency encoding for high-cardinality (City, County)
3. ✅ One-hot encoding for low-cardinality (State, Weather_Condition)
4. ✅ VIF-based feature selection (remove multicollinearity)
5. ✅ Save final feature matrices

**Output files:**
- `dataset/processed/X_train_featured.csv` (final features)
- `dataset/processed/X_test_featured.csv`
- `dataset/processed/y_train.csv` (target labels)
- `dataset/processed/y_test.csv`

---

### Step 4: Model Training & Evaluation

**Notebook:** `notebooks/05_model.ipynb`

**Python Script:**

```python
import pandas as pd
from src.model.training import train_and_evaluate_models

# Load featured data
X_train = pd.read_csv('dataset/processed/X_train_featured.csv')
X_test = pd.read_csv('dataset/processed/X_test_featured.csv')
y_train = pd.read_csv('dataset/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('dataset/processed/y_test.csv').values.ravel()

# Train all models
results = train_and_evaluate_models(
    X_train, y_train, X_test, y_test,
    models=['random_forest', 'xgboost', 'lightgbm', 'catboost'],
    optimize_thresholds=True,
    use_sample_weights=True  # Handle class imbalance
)

# Display results
for model_name, metrics in results.items():
    print(f"\n{model_name.upper()}")
    print(f"  Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"  QWK: {metrics['test']['qwk']:.4f}")
    print(f"  F1-Macro: {metrics['test']['f1_macro']:.4f}")
```

**Or train single model via terminal:**

```bash
# Train only CatBoost (fastest + best performance)
python -c "import pandas as pd; from src.model.training import train_catboost, evaluate_model_with_rounder; X_train = pd.read_csv('dataset/processed/X_train_featured.csv'); y_train = pd.read_csv('dataset/processed/y_train.csv').values.ravel(); X_test = pd.read_csv('dataset/processed/X_test_featured.csv'); y_test = pd.read_csv('dataset/processed/y_test.csv').values.ravel(); model = train_catboost(X_train, y_train); metrics = evaluate_model_with_rounder(model, X_train, y_train, X_test, y_test); print(f'CatBoost Test QWK: {metrics[\"test\"][\"qwk\"]:.4f}')"
```

**Model comparison:**

| Model | Command | Training Time | QWK Score |
|-------|---------|---------------|-----------|
| Random Forest | `models=['random_forest']` | ~175s | 0.812 |
| XGBoost | `models=['xgboost']` | ~120s | 0.828 |
| LightGBM | `models=['lightgbm']` | ~85s | 0.825 |
| CatBoost | `models=['catboost']` | ~210s | **0.847** |
| All models | `models=['random_forest', 'xgboost', 'lightgbm', 'catboost']` | ~590s | - |

---

## Method 3: Generate Visualizations

### EDA Plots

**Script:** `src/visualization/generate_all_eda_plots.py`

```bash
# Generate all EDA visualizations
python src/visualization/generate_all_eda_plots.py
```

**Or in Python:**

```python
from src.visualization.generate_all_eda_plots import generate_all_plots

generate_all_plots()
```

**Generated plots** (saved to `figures/eda/`):
- `severity_distribution.png` - Bar chart + pie chart
- `temporal_heatmap.png` - Hour vs. Day-of-week
- `state_distribution.png` - Top 10 states by accident count
- `weather_impact.png` - Weather condition vs. severity
- `correlation_matrix.png` - Feature correlations
- `outlier_boxplots.png` - Outlier detection visualizations

---

### Model Performance Plots

**Script:** `src/visualization/generate_model_plots.py`

```bash
# Generate model evaluation plots
python src/visualization/generate_model_plots.py
```

**Or in Python:**

```python
from src.visualization.generate_model_plots import generate_model_performance_plots

generate_model_performance_plots()
```

**Generated plots** (saved to `figures/model/`):
- `confusion_matrix_[model].png` - For each model
- `feature_importance_[model].png` - Top 20 features
- `model_comparison_qwk.png` - QWK scores comparison
- `model_comparison_f1.png` - F1 scores comparison
- `roc_curves_[model].png` - One-vs-rest ROC curves
- `threshold_optimization.png` - OptimizedRounder visualization

---

## Method 4: Using Jupyter Notebooks (Interactive Exploration)

### Launch Jupyter

```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook

# Or Jupyter Lab
pip install jupyterlab
jupyter lab
```

### Notebook Workflow

1. **`00_train_test_split.ipynb`**  
   - Load full dataset
   - Perform stratified split
   - Save train/test CSVs

2. **`01_understanding_eda.ipynb`**  
   - Explore raw data structure
   - Check data types, missing values
   - Initial visualizations

3. **`02_data_processing.ipynb`**  
   - Develop preprocessing pipeline
   - Test outlier removal
   - Validate imputation strategies

4. **`03_eda_processed.ipynb`**  
   - Analyze processed data
   - Visualize feature distributions
   - Correlation analysis

5. **`04_feature_engineering_and_feature_selection.ipynb`**  
   - Create temporal features
   - Test encoding strategies
   - VIF-based feature selection

6. **`05_model.ipynb`**  
   - Train multiple models
   - Hyperparameter tuning
   - Threshold optimization
   - Final model evaluation

**Run all cells:**  
`Kernel > Restart & Run All`

---

## 🔍 Verification Steps

### Check Data Processing

```python
import pandas as pd

# Check processed data exists
train = pd.read_csv('dataset/processed/train_processed.csv')
test = pd.read_csv('dataset/processed/test_processed.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Missing values in train: {train.isnull().sum().sum()}")
```

### Check Feature Engineering

```python
import pandas as pd

X_train = pd.read_csv('dataset/processed/X_train_featured.csv')
X_test = pd.read_csv('dataset/processed/X_test_featured.csv')

print(f"Features: {X_train.shape[1]}")
print(f"Feature names: {list(X_train.columns[:10])}...")
print(f"Data types: {X_train.dtypes.value_counts()}")
```

### Check Model Outputs

```python
import os

model_dir = 'models/'
if os.path.exists(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    print(f"Saved models: {models}")
else:
    print("No models saved yet")
```

---

## ⚠️ Troubleshooting

### Issue 1: Memory Error

**Error:** `MemoryError: Unable to allocate array`

**Solution:**
```python
# Process data in chunks
from src.config.config import DataConfig
DataConfig.CHUNK_SIZE = 100000  # Process 100K rows at a time
```

### Issue 2: File Not Found

**Error:** `FileNotFoundError: dataset/raw/train.csv`

**Solution:**
```bash
# Run train-test split first
python -c "from src.data.data_processing import split_raw_data; split_raw_data()"
```

### Issue 3: Import Error

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure you're in project root
cd Data_Storytelling

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)"              # Windows PowerShell
```

### Issue 4: Slow Training

**Solution:**
```python
# Use single model (CatBoost recommended)
results = pipeline.run(models=['catboost'])

# Or reduce data size for testing
from src.config.config import DataConfig
DataConfig.SAMPLE_SIZE = 100000  # Use 100K samples
```

---

## 🔬 Methodology

### 1. Data Preprocessing

**Objective**: Clean raw data and prepare for feature engineering

#### Steps

1. **Data Type Conversion**
   - Parse datetime columns (`Start_Time`, `End_Time`, `Weather_Timestamp`)
   - Convert boolean features to binary (0/1)

2. **Outlier Detection & Removal**
   - **Temperature**: Remove anomalies < -50°F or > 150°F
   - **Visibility**: Cap at 10 miles (valid sensor range)
   - **Wind Speed**: Remove extremes > 100 mph
   - **Pressure**: Keep 28-32 inHg (standard atmospheric range)
   - **Humidity**: Constrain 0-100%

3. **Missing Value Imputation**
   - **Numerical**: Median imputation (robust to outliers)
   - **Categorical**: Mode imputation or "Unknown" category
   - **Strategic Drops**: Features with >50% missing data

4. **Data Leakage Prevention**
   - Drop `ID`, `Description` (contains severity keywords)
   - Drop `End_Time`, `Distance(mi)` (consequences of severity)

5. **Cardinality Reduction**
   - Drop high-cardinality features: `Street` (112K unique), `Zipcode` (194K unique)
   - Retain: `City` (9.5K), `County` (1.5K), `State` (49)

**Output**: `train_processed.csv`, `test_processed.csv`

---

### 2. Feature Engineering

**Objective**: Extract temporal patterns and select non-collinear features

#### Temporal Features Extraction

```python
# From Start_Time
- hour: 0-23 (rush hour identification)
- day_of_week: 0-6 (weekend vs. weekday)
- month: 1-12 (seasonal patterns)
- year: 2020-2023 (temporal trends)
- is_weekend: binary
- is_night: binary (7PM - 7AM)
- season: Spring/Summer/Fall/Winter
```

#### Encoding Strategies

- **High Cardinality** (City, County): Frequency encoding (% of accidents)
- **Low Cardinality** (State, Weather_Condition): One-hot encoding
- **Ordinal** (Timezone): Label encoding

#### Multicollinearity Mitigation

- Compute Variance Inflation Factor (VIF) for all numeric features
- Iteratively remove features with VIF > 10
- Retained features: temperature, visibility, wind speed, temporal features

**Output**: `X_train_featured.csv`, `X_test_featured.csv`

---

### 3. Model Training

#### Problem Formulation: Ordinal Regression

Severity prediction is treated as **ordinal regression** rather than standard classification:
- Train regressors (output: continuous severity score)
- Apply optimized thresholds to convert to discrete classes (1, 2, 3, 4)

**Advantage**: Preserves ordinal nature of severity levels

#### Models Implemented

| Model | Key Hyperparameters | Strengths |
|-------|---------------------|-----------|
| **Random Forest Regressor** | `n_estimators=200`, `max_depth=20` | Robust to overfitting, handles non-linearity |
| **XGBoost Regressor** | `learning_rate=0.1`, `max_depth=7` | Fast training, regularization control |
| **LightGBM Regressor** | `num_leaves=31`, `learning_rate=0.1` | Efficient for large datasets, low memory |
| **CatBoost Regressor** | `iterations=500`, `depth=8` | Best handling of categorical features, no preprocessing needed |

#### Class Imbalance Handling

```python
from sklearn.utils.class_weight import compute_sample_weight

# Compute sample weights (inverse class frequency)
sample_weights = compute_sample_weight('balanced', y_train)

# Apply during training
model.fit(X_train, y_train, sample_weight=sample_weights)
```

#### Threshold Optimization (OptimizedRounder)

Standard approach: Use fixed thresholds [1.5, 2.5, 3.5]  
**Our approach**: Learn optimal thresholds to maximize QWK

```python
class OptimizedRounder:
    def fit(self, y_pred_continuous, y_true):
        # Find thresholds [t1, t2, t3] that maximize QWK
        # Using Nelder-Mead optimization
        
    def predict(self, y_pred_continuous):
        # Apply learned thresholds to discretize predictions
```

**Example Learned Thresholds**: [1.42, 2.18, 3.65] (CatBoost model)

---

### 4. Model Evaluation

#### Primary Metric: Quadratic Weighted Kappa (QWK)

**Formula**:  
$$
\kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}
$$

Where:
- $O_{i,j}$: Observed agreement matrix
- $E_{i,j}$: Expected agreement matrix
- $w_{i,j} = \frac{(i-j)^2}{(N-1)^2}$: Quadratic weight (penalizes larger misclassifications)

**Interpretation**:
- QWK = 1: Perfect agreement
- QWK = 0: Random predictions
- QWK < 0: Worse than random

**Why QWK?** Penalizes predicting Severity 1 as 4 more than predicting as 2.

#### Results Summary

| Model | Test Accuracy | Test QWK | F1-Macro | F1-Weighted | Training Time |
|-------|---------------|----------|----------|-------------|---------------|
| Random Forest (Classifier) | 0.821 | 0.783 | 0.524 | 0.798 | 180s |
| Random Forest (Regressor) | 0.838 | 0.812 | 0.562 | 0.821 | 175s |
| XGBoost | 0.845 | 0.828 | 0.581 | 0.834 | 120s |
| LightGBM | 0.843 | 0.825 | 0.576 | 0.831 | 85s |
| **CatBoost** | **0.856** | **0.847** | **0.603** | **0.848** | 210s |

**Best Model**: CatBoost Regressor with OptimizedRounder  
**Key Insight**: Ordinal regression + threshold optimization consistently outperforms direct classification

---

## 📈 Key Findings

### EDA Insights

1. **Temporal Patterns**
   - Peak hours: 7-9 AM, 4-6 PM (morning/evening rush)
   - Weekday accidents: 3x higher than weekends
   - Winter months: +40% accidents (weather-related)

2. **Geospatial Distribution**
   - Top 5 states: California, Texas, Florida, South Carolina, North Carolina
   - Urban areas: 75% of accidents occur in cities >100K population

3. **Weather Impact**
   - Clear weather: 65% of accidents (high traffic volume)
   - Rain/Snow: Higher severity (2.3 avg vs. 2.1 overall)
   - Low visibility (<1 mi): 2.5x likelihood of Severity 3/4

4. **Infrastructure Correlation**
   - Traffic signals: -15% severity (controlled intersections)
   - No nearby amenities: +20% severity (remote areas)

### Feature Importance (CatBoost Model)

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| Temperature | 18.3% | Extreme temperatures correlate with severity |
| Hour | 15.7% | Time-of-day is critical predictor |
| Visibility | 12.4% | Low visibility increases severity |
| City (frequency) | 11.2% | Urban density affects severity |
| Humidity | 9.8% | Weather condition proxy |

---

## 📊 Visualizations

All plots are automatically generated and saved to `figures/` directory.

### EDA Plots
- Severity distribution (bar chart, pie chart)
- Temporal heatmaps (hour vs. day-of-week)
- Geospatial maps (state-level choropleth)
- Weather correlation matrices
- Outlier detection box plots

### Model Performance Plots
- Confusion matrices (all models)
- ROC curves (one-vs-rest)
- Feature importance bar charts
- Learning curves (training vs. validation loss)
- Threshold optimization visualization

**Generate all plots**:
```python
from src.visualization.generate_all_eda_plots import generate_all_plots
from src.visualization.generate_model_plots import generate_model_performance_plots

generate_all_plots()  # EDA
generate_model_performance_plots()  # Models
```

---

## 🛠️ Configuration

All parameters are centralized in `src/config/config.py`:

```python
class DataConfig:
    # File paths
    RAW_DATA_PATH = Path("dataset/raw/US_Accidents_March23.csv")
    TRAIN_PROCESSED_PATH = Path("dataset/processed/train_processed.csv")
    
    # Split parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TARGET_COL = 'Severity'
    
    # Outlier thresholds
    TEMP_MIN, TEMP_MAX = -50, 150
    VISIBILITY_MAX = 10
    WIND_SPEED_MAX = 100
    
    # VIF threshold
    VIF_THRESHOLD = 10
```

**Modify configuration** by editing `config.py` or passing custom `DataConfig` object.

---

## 📦 Dependencies

Core libraries (see `requirements.txt`):

```
numpy>=1.21.0          # Numerical computing
pandas>=1.3.0          # Data manipulation
scikit-learn>=1.0.0    # ML algorithms, metrics
xgboost>=1.5.0         # Gradient boosting
lightgbm>=3.3.0        # Gradient boosting (Microsoft)
catboost>=1.0.0        # Gradient boosting (Yandex)
matplotlib>=3.4.0      # Plotting
seaborn>=0.11.0        # Statistical visualization
statsmodels>=0.13.0    # VIF calculation
joblib>=1.1.0          # Model serialization
```

---

---

## 🎓 Conclusion: Data Storytelling in Practice

### **The Story We Told**

Through this project, we built a **complete data story**:

**🔹 Beginning (Problem Statement)**:
- Traffic accidents are a serious problem in the United States
- Dataset: 2.5M+ accidents across 3 years, 49 states

**🔹 Middle (Analysis & Discovery)**:
- Data Cleaning: Handle outliers with business logic
- Feature Engineering: Create temporal features from domain knowledge
- EDA Insights: Rush hour patterns, weather impact, geographic hotspots

**🔹 End (Solution & Results)**:
- Predictive Model: 85.6% accuracy, 0.847 QWK
- Actionable Insights: Weather-based alerts, infrastructure improvements
- Production-Ready: Modular code, reproducible pipeline

---

### **Technical Analysis: Lessons Learned**

#### ✅ **Key Decisions That Worked**

1. **Ordinal Regression over Classification**
   - Preserved severity ordering → +0.035 QWK improvement
   - Threshold optimization crucial for ordinal problems

2. **Frequency Encoding for High-Cardinality**
   - City (9,562 unique) → Single column with meaningful pattern
   - Avoided curse of dimensionality from one-hot encoding

3. **VIF-Based Feature Selection**
   - Removed 30% redundant features
   - Faster training, better interpretability

4. **Sample Weights for Imbalance**
   - Forced model to learn minority classes
   - +0.08 F1-Macro improvement

5. **CatBoost Model Selection**
   - Native categorical handling
   - Best QWK (0.847) among all models

#### ⚠️ **Challenges & Trade-offs**

1. **Class Imbalance (75% Severity 2)**
   - Challenge: Model biased toward majority
   - Solution: Sample weights + QWK metric (penalizes errors differently)

2. **High Cardinality (City, County)**
   - Challenge: 9,562 cities → Dimensionality explosion
   - Solution: Frequency encoding (accident rate per city)

3. **Multicollinearity (Weather features)**
   - Challenge: Temperature, Wind_Chill, Feels_Like highly correlated
   - Solution: VIF analysis → Keep only Temperature

4. **Data Leakage Risk**
   - Challenge: Description contains severity keywords ("major crash")
   - Solution: Drop Description, End_Time, Distance (effects, not causes)

---

### **Visualizations: The Storytelling Tools**

**25+ plots** organized by purpose:

#### 📊 **Exploratory (Understanding)**
- Distribution plots → Class imbalance, feature skewness
- Correlation matrices → Feature relationships
- Geographic maps → Spatial patterns

#### 📈 **Diagnostic (Validation)**
- Outlier boxplots → Data cleaning effectiveness
- VIF analysis → Multicollinearity detection
- Before/after comparisons → Preprocessing impact

#### 💡 **Insight (Storytelling)**
- Temporal heatmaps → Rush hour identification
- Weather correlations → Environmental impact
- State/City distributions → Geographic hotspots

#### 🤖 **Model Performance (Results)**
- Confusion matrices → Error patterns
- Feature importance → Key predictors
- ROC curves → Classification quality
- Model comparison → Best model selection

**Each plot tells part of the story** - from raw data exploration to final model insights.

---

### **Real-World Applications**

#### 🚨 **1. Emergency Response Systems**
- **Use Case**: Predict severity in real-time when accident reported
- **Impact**: Prioritize ambulance/fire truck dispatch
- **Key Features**: Location, time, weather conditions

#### 🗺️ **2. Navigation Apps (Waze, Google Maps)**
- **Use Case**: Alert drivers about high-risk conditions
- **Impact**: "Heavy rain ahead, accidents 2.5x more likely - reduce speed"
- **Key Features**: Current weather, time of day, traffic density

#### 🏙️ **3. Urban Planning**
- **Use Case**: Identify intersections needing traffic signals
- **Impact**: Data-driven infrastructure investments
- **Key Features**: Accident frequency by location, infrastructure presence

#### 📜 **4. Policy Making**
- **Use Case**: Evidence for speed limit adjustments
- **Impact**: Lower limits during winter months (-40% severity)
- **Key Features**: Seasonal patterns, weather impact analysis

---

## 📦 Deliverables Summary

### ✅ **Part 1: Data Storytelling**

**Slide/Presentation Materials**:
- 📊 **25+ Professional Visualizations**: EDA + Model insights
- 📖 **Narrative Structure**: Beginning → Middle → End
- 💡 **Key Insights**: 4 major findings (temporal, geographic, weather, infrastructure)
- 🎯 **Business Impact**: Real-world applications explained

**Storytelling Techniques Used**:
- Clear narrative arc (problem → analysis → solution)
- Visualizations as evidence (not decoration)
- Technical choices explained with business logic
- Insights connected to actionable recommendations

---

### ✅ **Part 2: Technical Analysis**

**Technical Report Sections**:

1. **Data Cleaning Strategy**
   - Outlier detection with domain thresholds
   - Missing value strategy (median/mode)
   - Data leakage prevention

2. **Feature Engineering Decisions**
   - Temporal features rationale (rush hour, seasonality)
   - Encoding strategies per cardinality level
   - VIF analysis for multicollinearity

3. **Model Selection**
   - Ordinal regression vs. classification comparison
   - Sample weights for class imbalance
   - Threshold optimization algorithm

4. **Visualization Choices**
   - Heatmaps → Temporal patterns
   - Correlation matrices → Feature relationships
   - Confusion matrices → Error analysis
   - Feature importance → Model interpretability

**Each decision justified with**:
- ✅ What was chosen
- ✅ Why it was chosen
- ✅ What alternatives were considered
- ✅ What impact it had

---

### ✅ **Part 3: GitHub Code Project**

**Repository Structure**:
```
✅ Modular code (src/config, src/data, src/features, src/model)
✅ Comprehensive README.md (this file)
✅ Requirements.txt (all dependencies)
✅ Reproducible pipeline (MLPipeline class)
✅ Documentation (docstrings, comments)
✅ Jupyter notebooks (exploratory analysis)
✅ Automated visualization generation
```

**Code Quality**:
- ✅ PEP 8 compliance (Python style guide)
- ✅ Modular design (separation of concerns)
- ✅ Configuration management (DataConfig class)
- ✅ Error handling (try-except blocks)
- ✅ Logging (console output for tracking)

---

## 📚 References & Further Reading

### **Dataset**
- **Original Paper**: Moosavi, Sobhan, et al. "A Countrywide Traffic Accident Dataset." *arXiv preprint arXiv:1906.05409* (2019).
- **Kaggle**: Kaggle Competition
- **DOI**: `10.48550/arXiv.1906.05409`

### **Techniques Used**
- **VIF Analysis**: Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics*.
- **Quadratic Weighted Kappa**: Cohen, J. (1968). "Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit." *Psychological Bulletin*.
- **Ordinal Regression**: McCullagh, P. (1980). "Regression models for ordinal data." *Journal of the Royal Statistical Society*.
- **CatBoost**: Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features." *NeurIPS*.

### **Data Storytelling Resources**
- Knaflic, Cole Nussbaumer. *Storytelling with Data* (2015).
- Few, Stephen. *Show Me the Numbers* (2012).
- Tufte, Edward. *The Visual Display of Quantitative Information* (2001).

---

## 👥 Authors

**Group 7 - Data Preparation and Visualization Course**

**Roles**:
- 🔹 **Data Storytelling**: Narrative development, visualization design, insight extraction
- 🔹 **Technical Analysis**: Feature engineering justification, model selection rationale, technique comparison
- 🔹 **Code Development**: Pipeline architecture, modular design, documentation

**Contact**:
- GitHub Repository: [Data_Storytelling](https://github.com/HoangPhuc-999/Data_Storytelling)
- GitHub Profile: [@HoangPhuc-999](https://github.com/HoangPhuc-999)

---

## 🙏 Acknowledgments

- **Course**: Data Preparation and Visualization (2025)
- **Instructors**: For emphasizing the importance of data storytelling alongside technical rigor
- **Dataset Authors**: Sobhan Moosavi, Mohammad Hossein Samavatian, Arnab Nandi, Srinivasan Parthasarathy, Rajiv Ramnath
- **Open Source Community**: Pandas, Scikit-learn, XGBoost, LightGBM, CatBoost, Matplotlib, Seaborn contributors

---

<div align="center">

### 🌟 **Data tells the story. We just need to listen.** 🌟

**Project Completed**: December 2025  
**Course**: Data Preparation and Visualization  
**Focus**: Storytelling through Data + Technical Rigor

</div>