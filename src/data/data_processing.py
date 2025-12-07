"""
Data Processing Module for US Accidents Dataset
Version: 1.0.0
"""

import os
import sys
import warnings
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from src.config.config import DataConfig, PreprocessingConfig

warnings.filterwarnings('ignore')


# ============================================================================
# CUSTOM TRANSFORMERS
# ============================================================================

class VarianceThresholdSelector(BaseEstimator, TransformerMixin):
    """Remove features with zero or low variance."""
    
    def __init__(self, threshold: float = 0):
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.numeric_cols = None
        self.retained_cols = None
        self.dropped_cols = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the variance threshold selector."""
        self.numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        self.selector.fit(X[self.numeric_cols])
        
        # Save retained and dropped columns
        self.retained_cols = self.numeric_cols[self.selector.get_support()]
        self.dropped_cols = [col for col in self.numeric_cols 
                            if col not in self.retained_cols]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by removing low-variance features."""
        cols_to_keep = list(self.retained_cols) + list(X.columns.difference(self.numeric_cols))
        return X[cols_to_keep]


class ConstantAndDuplicateRemover(BaseEstimator, TransformerMixin):
    """Remove constant columns and duplicate rows."""
    
    def __init__(self):
        self.constant_cols = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """Identify constant columns."""
        self.constant_cols = [col for col in X.columns if X[col].nunique() == 1]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove constant columns and duplicate rows."""
        X_cleaned = X.drop(columns=self.constant_cols, errors="ignore")
        
        initial_rows = len(X_cleaned)
        X_cleaned = X_cleaned.drop_duplicates()
        removed_rows = initial_rows - len(X_cleaned)
        
        return X_cleaned


class SunriseSunsetImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing Sunrise_Sunset values based on Start_Time.
    Rule: Hour 6-18 = 'Day', otherwise = 'Night'
    """
    
    def __init__(self):
        self.n_missing_ = 0
    
    def fit(self, X: pd.DataFrame, y=None):
        """Record number of missing values."""
        if 'Sunrise_Sunset' in X.columns:
            self.n_missing_ = X['Sunrise_Sunset'].isnull().sum()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply time-based imputation rule."""
        X_copy = X.copy()
        
        if 'Sunrise_Sunset' in X_copy.columns and 'Start_Time' in X_copy.columns:
            mask = X_copy['Sunrise_Sunset'].isnull()
            
            if mask.sum() > 0:
                X_copy.loc[mask, 'Sunrise_Sunset'] = X_copy.loc[mask, 'Start_Time'].apply(
                    lambda x: 'Day' if PreprocessingConfig.DAY_HOUR_START <= x.hour < PreprocessingConfig.DAY_HOUR_END else 'Night'
                )
        
        return X_copy


class WeatherConditionImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing Weather_Condition values using meteorological rules.
    Groups conditions into 5 categories: Fog, Snow, Rain, Cloudy, Clear
    """
    
    def __init__(self):
        self.n_missing_ = 0
        self.train_stats_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Calculate median statistics from training data."""
        if 'Weather_Condition' in X.columns:
            self.n_missing_ = X['Weather_Condition'].isnull().sum()
            
            # Calculate median values for fallback
            self.train_stats_ = {
                'Visibility(mi)': X['Visibility(mi)'].median() if 'Visibility(mi)' in X.columns else 10.0,
                'Temperature(F)': X['Temperature(F)'].median() if 'Temperature(F)' in X.columns else 60.0,
                'Humidity(%)': X['Humidity(%)'].median() if 'Humidity(%)' in X.columns else 50.0
            }
        
        return self
    
    def _apply_weather_rules(self, row: pd.Series) -> str:
        """Apply meteorological rules to determine weather condition."""
        if pd.notna(row['Weather_Condition']):
            return row['Weather_Condition']
        
        # Get values with fallback to training statistics
        vis = row['Visibility(mi)'] if pd.notna(row['Visibility(mi)']) else self.train_stats_['Visibility(mi)']
        temp = row['Temperature(F)'] if pd.notna(row['Temperature(F)']) else self.train_stats_['Temperature(F)']
        humid = row['Humidity(%)'] if pd.notna(row['Humidity(%)']) else self.train_stats_['Humidity(%)']
        
        # Apply meteorological decision rules using PreprocessingConfig
        if vis < PreprocessingConfig.FOG_VISIBILITY_THRESHOLD:
            return 'Fog'
        if temp < PreprocessingConfig.SNOW_TEMP_THRESHOLD and humid > PreprocessingConfig.SNOW_HUMIDITY_THRESHOLD:
            return 'Snow'
        if vis < PreprocessingConfig.FOG_VISIBILITY_THRESHOLD_2 and humid > PreprocessingConfig.FOG_HUMIDITY_THRESHOLD:
            return 'Fog'
        if PreprocessingConfig.CLOUDY_HUMIDITY_MIN < humid <= PreprocessingConfig.CLOUDY_HUMIDITY_MAX:
            return 'Cloudy'
        
        return 'Clear'
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply weather condition imputation."""
        X_copy = X.copy()
        
        if 'Weather_Condition' in X_copy.columns:
            X_copy['Weather_Condition'] = X_copy.apply(self._apply_weather_rules, axis=1)
        
        return X_copy


class DomainBasedOutlierCapper(BaseEstimator, TransformerMixin):
    """
    Cap outliers using physical domain constraints instead of statistical methods.
    """
    
    def __init__(self, 
                 temp_bounds: Tuple[float, float] = DataConfig.TEMPERATURE_BOUNDS,
                 humid_bounds: Tuple[float, float] = DataConfig.HUMIDITY_BOUNDS):
        self.temp_bounds = temp_bounds
        self.humid_bounds = humid_bounds
        self.capped_stats_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit is a no-op for this transformer."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply domain-based outlier capping."""
        X_copy = X.copy()
        
        # Temperature capping
        if 'Temperature(F)' in X_copy.columns:
            X_copy['Temperature(F)'] = X_copy['Temperature(F)'].clip(*self.temp_bounds)
        
        # Humidity capping
        if 'Humidity(%)' in X_copy.columns:
            X_copy['Humidity(%)'] = X_copy['Humidity(%)'].clip(*self.humid_bounds)
        
        return X_copy


class SequentialProcessor(BaseEstimator, TransformerMixin):
    """
    Sequential preprocessing pipeline that orchestrates:
    1. Categorical missing value imputation
    2. Numerical missing value imputation  
    3. Domain-based outlier capping
    """
    
    def __init__(self, numeric_features: List[str]):
        self.numeric_features = numeric_features
        
        # Initialize component transformers
        self.sunrise_imputer = SunriseSunsetImputer()
        self.weather_imputer = WeatherConditionImputer()
        self.numeric_imputer = SimpleImputer(strategy=PreprocessingConfig.NUMERIC_IMPUTE_STRATEGY)
        self.outlier_capper = DomainBasedOutlierCapper()
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit all component transformers sequentially."""
        X_temp = X.copy()
        
        # Fit categorical imputers
        self.sunrise_imputer.fit(X_temp)
        X_temp = self.sunrise_imputer.transform(X_temp)
        
        self.weather_imputer.fit(X_temp)
        X_temp = self.weather_imputer.transform(X_temp)
        
        # Fit numerical imputer
        if len(self.numeric_features) > 0:
            available_numeric = [col for col in self.numeric_features if col in X_temp.columns]
            if available_numeric:
                self.numeric_imputer.fit(X_temp[available_numeric])
        
        # Fit outlier capper
        self.outlier_capper.fit(X_temp)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations sequentially."""
        X_result = X.copy()
        
        # Step 1: Categorical imputation
        X_result = self.sunrise_imputer.transform(X_result)
        X_result = self.weather_imputer.transform(X_result)
        
        # Step 2: Numerical imputation
        if len(self.numeric_features) > 0:
            available_numeric = [col for col in self.numeric_features if col in X_result.columns]
            if available_numeric:
                X_result[available_numeric] = self.numeric_imputer.transform(X_result[available_numeric])
        
        # Step 3: Outlier capping
        X_result = self.outlier_capper.transform(X_result)
        
        return X_result


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Load raw accident data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing raw accident data
        
    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def stratified_train_test_split_data(
    df: pd.DataFrame,
    target_col: str = DataConfig.TARGET_COL,
    test_size: float = DataConfig.TEST_SIZE,
    random_state: int = DataConfig.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets with stratification.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion of test set (0-1)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Perform stratified split
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[[target_col]]
    )
    
    return train, test


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with converted types
    """
    df_copy = df.copy()
    
    # Convert time columns to datetime
    for col in DataConfig.TIME_COLUMNS:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    
    # Convert boolean columns to object
    for col in DataConfig.BOOL_COLUMNS:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype('object')
    
    return df_copy


def drop_missing_city(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Drop rows with missing City values (low missing rate, critical feature).
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target
        
    Returns:
        Tuple of cleaned (X_train, y_train, X_test, y_test)
    """
    # Train set
    city_missing_train = X_train['City'].isnull()
    n_missing_train = city_missing_train.sum()
    
    if n_missing_train > 0:
        X_train_clean = X_train[~city_missing_train].reset_index(drop=True)
        y_train_clean = y_train[~city_missing_train].reset_index(drop=True)
    else:
        X_train_clean, y_train_clean = X_train, y_train
    
    # Test set
    city_missing_test = X_test['City'].isnull()
    n_missing_test = city_missing_test.sum()
    
    if n_missing_test > 0:
        X_test_clean = X_test[~city_missing_test].reset_index(drop=True)
        y_test_clean = y_test[~city_missing_test].reset_index(drop=True)
    else:
        X_test_clean, y_test_clean = X_test, y_test
    
    return X_train_clean, y_train_clean, X_test_clean, y_test_clean


def remove_feature_leakage_and_redundancy(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove features that cause data leakage or are redundant.
    
    Args:
        X_train, X_test: Training and test features
        
    Returns:
        Tuple of cleaned (X_train, X_test)
    """
    cols_to_drop = [col for col in DataConfig.get_all_drop_columns() 
                    if col in X_train.columns]
    
    if cols_to_drop:
        X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
        X_test = X_test.drop(columns=cols_to_drop, errors='ignore')
    
    return X_train, X_test


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical features.
    
    Args:
        X: Input DataFrame
        
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numeric_features, categorical_features


def validate_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Validate processed data quality.
    
    Args:
        X_train, X_test: Processed features
        y_train, y_test: Target variables
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'train_missing': X_train.isnull().sum().sum(),
        'test_missing': X_test.isnull().sum().sum(),
        'feature_consistency': list(X_train.columns) == list(X_test.columns),
        'target_classes': sorted(y_train.unique())
    }
    
    return validation_results


def save_processed_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = Path("dataset/processed")
) -> None:
    """
    Save processed data to CSV files.
    
    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target
        output_dir: Output directory path
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target
    train_final = X_train.copy()
    train_final[DataConfig.TARGET_COL] = y_train
    
    test_final = X_test.copy()
    test_final[DataConfig.TARGET_COL] = y_test
    
    # Save to CSV
    train_path = output_dir / "train_processed.csv"
    test_path = output_dir / "test_processed.csv"
    
    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def run_full_preprocessing_pipeline(
    raw_data_path: Path = DataConfig.RAW_DATA_PATH,
    save_output: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Execute the complete data preprocessing pipeline.
    
    This function orchestrates all preprocessing steps:
    1. Load raw data
    2. Train-test split with stratification
    3. Type conversion
    4. Feature cleaning (variance, duplicates)
    5. Feature dropping (leakage, redundancy)
    6. Missing value handling
    7. Outlier capping
    8. Validation and saving
    
    Args:
        raw_data_path: Path to raw data CSV file
        save_output: Whether to save processed data to disk
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Step 1: Load raw data
    df = load_raw_data(raw_data_path)
    
    # Step 2: Train-test split
    train_df, test_df = stratified_train_test_split_data(df)
    
    # Save raw splits
    if save_output:
        DataConfig.TRAIN_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(DataConfig.TRAIN_RAW_PATH, index=False)
        test_df.to_csv(DataConfig.TEST_RAW_PATH, index=False)
    
    # Step 3: Convert data types
    train_df = convert_data_types(train_df)
    test_df = convert_data_types(test_df)
    
    # Step 4: Separate features and target
    X_train = train_df.drop(DataConfig.TARGET_COL, axis=1)
    y_train = train_df[DataConfig.TARGET_COL]
    X_test = test_df.drop(DataConfig.TARGET_COL, axis=1)
    y_test = test_df[DataConfig.TARGET_COL]
    
    # Step 5: Remove zero-variance features and duplicates
    cleaning_pipeline = Pipeline(steps=[
        ("var_thresh", VarianceThresholdSelector(threshold=0)),
        ("const_dup", ConstantAndDuplicateRemover())
    ])
    
    X_train = cleaning_pipeline.fit_transform(X_train)
    X_test = cleaning_pipeline.transform(X_test)
    
    # Step 6: Drop features causing data leakage
    X_train, X_test = remove_feature_leakage_and_redundancy(X_train, X_test)
    
    # Step 7: Identify feature types
    numeric_features, categorical_features = identify_feature_types(X_train)
    
    # Step 8: Drop rows with missing City
    X_train, y_train, X_test, y_test = drop_missing_city(X_train, y_train, X_test, y_test)
    
    # Step 9: Apply sequential preprocessing (imputation + outlier capping)
    preprocessor = SequentialProcessor(numeric_features=numeric_features)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    # Step 10: Validate processed data
    validation_results = validate_processed_data(X_train, X_test, y_train, y_test)
    
    # Step 11: Save processed data
    if save_output:
        save_processed_data(X_train, y_train, X_test, y_test)
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """Execute the preprocessing pipeline when run as a script."""
    
    X_train, y_train, X_test, y_test = run_full_preprocessing_pipeline(
        raw_data_path=DataConfig.RAW_DATA_PATH,
        save_output=True
    )
    
    print(f"✓ Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"✓ Test:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
