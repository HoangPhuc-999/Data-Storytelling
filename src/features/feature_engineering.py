"""
Feature Engineering Module
===========================
Transform processed data into model-ready features.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import DataConfig

warnings.filterwarnings('ignore')


# ============================================================================
# MEMORY OPTIMIZATION
# ============================================================================

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        verbose: Print memory usage info
        
    Returns:
        DataFrame with optimized dtypes
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f'Initial memory usage: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f'Memory after optimization: {end_mem:.2f} MB (Reduced by {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    
    return df


# ============================================================================
# TIME FEATURES
# ============================================================================

def extract_time_features(df: pd.DataFrame, time_col: str = 'Start_Time') -> pd.DataFrame:
    """
    Extract time-based features from datetime column.
    
    Args:
        df: Input DataFrame
        time_col: Name of datetime column
        
    Returns:
        DataFrame with extracted time features
    """
    df = df.copy()
    
    if time_col not in df.columns:
        return df
    
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    df['Year'] = df[time_col].dt.year.astype('int16')
    df['Month'] = df[time_col].dt.month.astype('int8')
    df['Day'] = df[time_col].dt.day.astype('int8')
    df['Hour'] = df[time_col].dt.hour.astype('int8')
    df['DayOfWeek'] = df[time_col].dt.dayofweek.astype('int8')
    df['Quarter'] = df[time_col].dt.quarter.astype('int8')
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype('int8')
    
    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical (sin/cos) features for temporal variables.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cyclical features
    """
    df = df.copy()
    
    if 'Month' in df.columns:
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12).astype('float32')
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12).astype('float32')
    
    if 'Hour' in df.columns:
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24).astype('float32')
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24).astype('float32')
    
    if 'DayOfWeek' in df.columns:
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7).astype('float32')
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7).astype('float32')
    
    return df


# ============================================================================
# WEATHER FEATURES
# ============================================================================

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather interaction and derived features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with weather features
    """
    df = df.copy()
    
    if 'Temperature(F)' in df.columns and 'Humidity(%)' in df.columns:
        df['Temp_Humidity_Interaction'] = (df['Temperature(F)'] * df['Humidity(%)']).astype('float32')
    
    if 'Visibility(mi)' in df.columns and 'Humidity(%)' in df.columns:
        df['Visibility_Humidity_Ratio'] = (df['Visibility(mi)'] / (df['Humidity(%)'] + 1)).astype('float32')
    
    if 'Temperature(F)' in df.columns:
        df['Temp_Squared'] = (df['Temperature(F)'] ** 2).astype('float32')
    
    return df


def simplify_weather_condition(df: pd.DataFrame, col: str = 'Weather_Condition') -> pd.DataFrame:
    """
    Group weather conditions into broader categories.
    
    Args:
        df: Input DataFrame
        col: Weather condition column name
        
    Returns:
        DataFrame with simplified weather categories
    """
    df = df.copy()
    
    if col not in df.columns:
        return df
    
    weather_mapping = {
        'Clear': 'Clear',
        'Cloudy': 'Cloudy',
        'Rain': 'Rain',
        'Snow': 'Snow',
        'Fog': 'Fog'
    }
    
    df[col + '_Grouped'] = df[col].map(weather_mapping).fillna('Other')
    
    return df


# ============================================================================
# INFRASTRUCTURE FEATURES
# ============================================================================

def create_infrastructure_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create road infrastructure complexity score.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with infrastructure score
    """
    df = df.copy()
    
    infra_cols = ['Amenity', 'Crossing', 'Junction', 'Railway', 
                  'Station', 'Stop', 'Traffic_Signal']
    
    available_cols = [col for col in infra_cols if col in df.columns]
    
    if available_cols:
        for col in available_cols:
            df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0)
        
        df['Road_Context_Score'] = df[available_cols].sum(axis=1).astype('int8')
    
    return df


# ============================================================================
# ENCODING
# ============================================================================

def encode_categorical_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    high_cardinality_cols: list = None,
    label_encode_cols: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using appropriate strategies.
    
    Args:
        X_train: Training features
        X_test: Test features
        high_cardinality_cols: Columns for frequency encoding
        label_encode_cols: Columns for label encoding
        
    Returns:
        Tuple of (X_train_encoded, X_test_encoded, encoders_dict)
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    encoders = {}
    
    high_cardinality_cols = high_cardinality_cols or ['City', 'County', 'State']
    label_encode_cols = label_encode_cols or ['Sunrise_Sunset', 'Weather_Condition']
    
    # Frequency encoding for high cardinality
    for col in high_cardinality_cols:
        if col in X_train.columns:
            freq_map = X_train[col].value_counts(normalize=True).to_dict()
            X_train[col + '_encoded'] = X_train[col].map(freq_map).fillna(0).astype('float32')
            X_test[col + '_encoded'] = X_test[col].map(freq_map).fillna(0).astype('float32')
            X_train.drop(col, axis=1, inplace=True)
            X_test.drop(col, axis=1, inplace=True)
    
    # Label encoding for ordinal/low cardinality
    for col in label_encode_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col + '_encoded'] = le.fit_transform(X_train[col].astype(str))
            X_test[col + '_encoded'] = le.transform(X_test[col].astype(str))
            encoders[col] = le
            X_train.drop(col, axis=1, inplace=True)
            X_test.drop(col, axis=1, inplace=True)
    
    return X_train, X_test, encoders


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: str = 'median',
    n_estimators: int = 100
) -> Tuple[list, RandomForestClassifier]:
    """
    Select important features using Random Forest.
    
    Args:
        X_train: Training features
        y_train: Training target
        threshold: Importance threshold ('median', 'mean', or float)
        n_estimators: Number of trees
        
    Returns:
        Tuple of (selected_features, trained_model)
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf.fit(X_train, y_train)
    
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    return selected_features, rf


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_feature_engineering_pipeline(
    train_path: Path,
    test_path: Path,
    save_output: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Execute complete feature engineering pipeline.
    
    Args:
        train_path: Path to processed training data
        test_path: Path to processed test data
        save_output: Whether to save engineered features
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['Severity'])
    y_train = train_df['Severity']
    X_test = test_df.drop(columns=['Severity'])
    y_test = test_df['Severity']
    
    # Time features
    X_train = extract_time_features(X_train)
    X_test = extract_time_features(X_test)
    
    X_train = create_cyclical_features(X_train)
    X_test = create_cyclical_features(X_test)
    
    # Drop original Start_Time
    X_train.drop(columns=['Start_Time'], inplace=True, errors='ignore')
    X_test.drop(columns=['Start_Time'], inplace=True, errors='ignore')
    
    # Weather features
    X_train = create_weather_features(X_train)
    X_test = create_weather_features(X_test)
    
    X_train = simplify_weather_condition(X_train)
    X_test = simplify_weather_condition(X_test)
    
    # Infrastructure features
    X_train = create_infrastructure_score(X_train)
    X_test = create_infrastructure_score(X_test)
    
    # Encoding
    X_train, X_test, encoders = encode_categorical_features(X_train, X_test)
    
    # Memory optimization
    X_train = reduce_mem_usage(X_train, verbose=False)
    X_test = reduce_mem_usage(X_test, verbose=False)
    
    # Save
    if save_output:
        output_dir = DataConfig.PROCESSED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        X_train.to_csv(output_dir / 'X_train_featured.csv', index=False)
        X_test.to_csv(output_dir / 'X_test_featured.csv', index=False)
        y_train.to_csv(output_dir / 'y_train.csv', index=False)
        y_test.to_csv(output_dir / 'y_test.csv', index=False)
        
        print(f"✓ Saved featured data to {output_dir}")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = run_feature_engineering_pipeline(
        train_path=DataConfig.TRAIN_PROCESSED_PATH,
        test_path=DataConfig.TEST_PROCESSED_PATH,
        save_output=True
    )
