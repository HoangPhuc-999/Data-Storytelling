# US Traffic Accidents - Severity Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

Predictive model for US traffic accident severity (2020-2023) with comprehensive data preparation and machine learning pipeline. Combines rigorous data cleaning, feature engineering, and CatBoost ordinal regression for actionable accident severity predictions.

## Overview

- **Dataset**: 2.5M accidents from 49 US states (2020-2023)
- **Objective**: Predict accident severity levels (1-4)
- **Model**: CatBoost regressor with optimized thresholds
- **Architecture**: Modular, reproducible, production-ready

## Installation

```bash
# 1. Clone and setup
git clone <repo-url> && cd Data_Storytelling
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate              # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Google Drive → dataset/raw/US_Accidents_March23.csv
# 4. Generate train/test split
python -c "import pandas as pd; from sklearn.model_selection import train_test_split; \
df = pd.read_csv('dataset/raw/US_Accidents_March23.csv'); \
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Severity']); \
train.to_csv('dataset/raw/train.csv', index=False); \
test.to_csv('dataset/raw/test.csv', index=False); \
print('✓ Train/test split complete')"
```

## Quick Start

```bash
# Complete pipeline (all steps)
python -c "from src.pipeline import MLPipeline; pipeline = MLPipeline(); pipeline.run()"

# Or step-by-step
python -c "from src.data.data_processing import run_full_preprocessing_pipeline; run_full_preprocessing_pipeline()"
python -c "from src.features.feature_engineering import run_feature_engineering_pipeline; run_feature_engineering_pipeline()"
jupyter notebook notebooks/05_model.ipynb  # Train models interactively
```

## Project Structure

```
src/                          # Production code
├── pipeline.py               # End-to-end pipeline
├── config/config.py          # Centralized config
├── data/data_processing.py   # Cleaning & preprocessing
├── features/
│   └── feature_engineering.py # Temporal, weather, encoding features
├── model/training.py         # CatBoost, threshold optimization
├── evaluation/metrics.py      # QWK, F1, accuracy
└── visualization/            # EDA & model plots

notebooks/                    # Jupyter notebooks for exploration
├── 00_train_test_split.ipynb
├── 01_understanding_eda.ipynb
├── 02_data_processing.ipynb
├── 03_eda_processed.ipynb
├── 04_feature_engineering_and_feature_selection.ipynb
└── 05_model.ipynb

dataset/raw/                  # Original data (gitignored)
dataset/processed/            # Pipeline outputs
models/                       # Trained model artifacts
figures/eda/                  # Exploratory visualizations
figures/model/                # Model evaluation plots
```

## Data Preparation Pipeline

### 1. Data Cleaning
- **Outlier Removal**: Domain-driven bounds (Temp: -50-150°F, Visibility: 0-10 mi)
- **Missing Values**: Median imputation (numerical), Mode imputation (categorical)
- **Leakage Prevention**: Removed consequence-based features (End_Time, Distance, Description)
- **Cardinality Reduction**: Dropped high-cardinality (Street, Zipcode) and redundant features (City, County)

### 2. Feature Engineering
- **Temporal**: Hour, day_of_week, month, season, is_night, is_weekend → Captures rush hour patterns
- **Weather**: Rain, snow, fog flags from categorical Weather_Condition → Enables weather impact analysis
- **Geographic**: Smoothed target encoding for State (49 unique, m=20 regularization)
- **Infrastructure**: Road context score from amenity/junction/signal features
- **Selection**: Random Forest importance filtering (retain top 50% features)

### 3. Ordinal Regression
- **Approach**: Train regressor → Apply optimized thresholds → Discrete classes (1,2,3,4)
- **Metric**: Quadratic Weighted Kappa (QWK) - penalizes distant predictions
- **Optimization**: Nelder-Mead algorithm maximizes QWK on validation set

### 4. Class Imbalance
- 75% of data is Severity 2 (major imbalance)
- Solution: Sample weights = inverse class frequency applied during training

## Models

| Model | Status | Key Feature |
|-------|--------|------------|
| CatBoost | ✅ Production | Native categorical handling |
| XGBoost | ✅ Implemented | Gradient boosting |
| LightGBM | ✅ Implemented | Fast training |
| Random Forest | ✅ Baseline | Feature importance analysis |

**Evaluation**: Models compared on QWK, accuracy, macro-F1 scores.

## Usage Examples

### Method 1: Jupyter Notebooks (Interactive)
```bash
jupyter notebook
# Open notebooks/05_model.ipynb for model training
```

### Method 2: Python Script
```python
from src.data.data_processing import run_full_preprocessing_pipeline
from src.features.feature_engineering import run_feature_engineering_pipeline
from src.model.training import train_and_evaluate_models
import pandas as pd

# Preprocessing
run_full_preprocessing_pipeline()

# Feature engineering
X_train, y_train, X_test, y_test = run_feature_engineering_pipeline()

# Model training
results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
```

### Method 3: Command Line
```bash
# Generate EDA plots
python src/visualization/generate_all_eda_plots.py

# Generate model plots (after training)
python src/visualization/generate_model_plots.py
```

## Configuration

Edit [src/config/config.py](src/config/config.py) to customize:
- Random seeds for reproducibility
- Imputation strategies
- Feature selection thresholds
- Model hyperparameters
- Output directories

## Key Insights

**EDA Findings:**
- Rush hour (7-9 AM, 4-6 PM): 3x higher accident frequency
- Winter months (Dec-Feb): 40% increase in severity
- Geographic hotspots: CA, TX, FL account for 48% of incidents
- Traffic signals: -15% average severity reduction
- Weather impact: Rain/snow conditions +20% severity vs. clear

**Model Features (Expected):**
- Temporal: Hour, day_of_week, season
- Environmental: Temperature, visibility, humidity
- Geographic: State (target-encoded)
- Weather: Rain, snow, fog flags
- Infrastructure: Road context score

## Dependencies

- pandas, numpy: Data manipulation
- scikit-learn: Preprocessing, Random Forest, metrics
- catboost, xgboost, lightgbm: Models
- matplotlib, seaborn: Visualization

See [requirements.txt](requirements.txt) for versions.

## Reproducibility

- ✅ Fixed random seeds (src/config/config.py)
- ✅ Versioned data splits (stratified train/test)
- ✅ Configuration-driven parameters
- ✅ Complete feature engineering pipeline

## Technical Decisions

**Why CatBoost?**
- Native categorical feature support (no manual encoding)
- Strong performance on tabular data with mixed feature types
- Handles class imbalance well

**Why Ordinal Regression?**
- Severity is ordered (1 < 2 < 3 < 4)
- QWK metric properly penalizes distant misclassifications
- Threshold optimization captures ordinal nature better than classification

**Why Smoothed Target Encoding?**
- Captures target relationship for State feature
- Regularization (m=20) prevents overfitting on rare states
- More interpretable than one-hot encoding

**Why Random Forest for Feature Selection?**
- Captures non-linear relationships (vs. VIF which is linear only)
- Target-aware selection based on predictive power
- Reduces dimensionality by ~50% with minimal information loss

## Troubleshooting

**Memory Error**: Reduce chunk size in config.py → `CHUNK_SIZE = 100000`

**Missing Dataset**: Download US_Accidents_March23.csv from Google Drive → `dataset/raw/`

**Import Error**: Run from project root: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**Slow Training**: Use single model: `pipeline.run(models=['catboost'])`

## Next Steps

1. Run complete pipeline: `python -c "from src.pipeline import MLPipeline; MLPipeline().run()"`
2. Review EDA plots: `figures/eda/`
3. Check model results: `report/report.txt`
4. Adapt configuration in `src/config/config.py` for your use case

## References

- US Accidents Dataset: Kaggle US-Accidents (2016-2023)
- Ordinal Regression Approach: Optimized Rounder for threshold selection
- EDA Techniques: Data storytelling through visualization
- Feature Engineering: Domain-driven temporal and categorical feature extraction

