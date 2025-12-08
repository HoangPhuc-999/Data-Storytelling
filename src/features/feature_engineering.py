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

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from Start_Time including cyclical features.
    Matches notebook implementation exactly.
    
    Args:
        df: Input DataFrame with Start_Time column
        
    Returns:
        DataFrame with time features (Start_Time dropped)
    """
    df = df.copy()
    
    if 'Start_Time' not in df.columns:
        return df
    
    # Convert to datetime
    if df['Start_Time'].dtype == 'object':
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    
    dt = df['Start_Time'].dt
    
    # 1. Basic attributes
    df['Hour'] = dt.hour.astype('int8')
    df['Month'] = dt.month.astype('int8')
    df['DayOfWeek'] = dt.dayofweek.astype('int8')
    
    # 2. Is_Weekend (Sat=5, Sun=6)
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype('int8')
    
    # 3. Rush Hour (7-9 AM and 5-7 PM)
    df['Is_Rush_Hour'] = 0
    mask_rush = ((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 17) & (df['Hour'] <= 19))
    df.loc[mask_rush, 'Is_Rush_Hour'] = 1
    df['Is_Rush_Hour'] = df['Is_Rush_Hour'].astype('int8')

    # 4. Season
    df['Season'] = ((df['Month'] % 12 + 3) // 3).astype('int8')

    # 5. CYCLICAL FEATURES
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24).astype('float32')
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24).astype('float32')
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12).astype('float32')
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12).astype('float32')

    # Drop the original Start_Time column
    return df.drop(columns=['Start_Time'])


# ============================================================================
# WEATHER FEATURES
# ============================================================================

def create_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather flag features from Weather_Condition.
    Matches notebook implementation exactly.
    
    Args:
        df: Input DataFrame with Weather_Condition column
        
    Returns:
        DataFrame with weather flags (Weather_Condition dropped)
    """
    df = df.copy()
    
    if 'Weather_Condition' not in df.columns:
        return df
    
    w_cond = df['Weather_Condition'].fillna('').astype(str).str.lower()
    
    # 1. Weather Types
    df['Is_Rain'] = w_cond.str.contains('rain|storm|shower', regex=True).astype('int8')
    df['Is_Snow'] = w_cond.str.contains('snow|blizzard|sleet', regex=True).astype('int8')
    df['Is_Fog'] = w_cond.str.contains('fog|mist|haze', regex=True).astype('int8')
    df['Is_Clear'] = w_cond.str.contains('clear', regex=True).astype('int8')
    df['Is_Cloudy'] = w_cond.str.contains('cloud', regex=True).astype('int8')

    # 2. Threshold checks
    if 'Visibility(mi)' in df.columns:
        df['Low_Visibility'] = (df['Visibility(mi)'] < 3).astype('int8')
    if 'Temperature(F)' in df.columns:
        df['Low_Temp'] = (df['Temperature(F)'] < 40).astype('int8')
    if 'Humidity(%)' in df.columns:
        df['High_Humidity'] = (df['Humidity(%)'] > 90).astype('int8')

    # 3. Compound Flag: Bad_Weather
    df['Bad_Weather'] = (
        df['Is_Rain'] | df['Is_Snow'] | df['Is_Fog'] | 
        df['Low_Visibility'] | df['High_Humidity']
    ).astype('int8')

    return df.drop(columns=['Weather_Condition'])


# ============================================================================
# INFRASTRUCTURE FEATURES
# ============================================================================

def create_road_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create road infrastructure complexity score.
    Matches notebook implementation exactly.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with Road_Context_Score
    """
    df = df.copy()
    
    poi_cols = ['Amenity', 'Crossing', 'Junction', 'Railway', 
                'Station', 'Stop', 'Traffic_Signal']
    
    # Convert all POI columns to int8 (0/1)
    for col in poi_cols:
        if col in df.columns:
            # Handle cases where data is 'True'/'False' string or boolean
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).map({'True': 1, 'False': 0, 'true': 1, 'false': 0}).fillna(0)
            df[col] = df[col].astype(bool).astype('int8')
    
    # Calculate the total number of POIs
    available_cols = [col for col in poi_cols if col in df.columns]
    if available_cols:
        df['Road_Context_Score'] = df[available_cols].sum(axis=1).astype('int8')
    
    return df


# ============================================================================
# INTERACTION FEATURES
# ============================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features.
    Matches notebook implementation exactly.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction features (Sunrise_Sunset dropped)
    """
    df = df.copy()
    
    # 1. Day/Night
    if 'Sunrise_Sunset' in df.columns:
        df['Is_Night'] = (df['Sunrise_Sunset'] == 'Night').astype('int8')
        df.drop(columns=['Sunrise_Sunset'], inplace=True)
    
    # 2. Interaction: Night + Rain + Junction
    if all(col in df.columns for col in ['Is_Night', 'Is_Rain', 'Junction']):
        df['Night_Rain_Junction'] = (
            df['Is_Night'] & df['Is_Rain'] & df['Junction']
        ).astype('int8')
    
    return df


# ============================================================================
# ENCODING
# ============================================================================

def encode_boolean_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    bool_features: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode boolean features using Label Encoding.
    
    Args:
        X_train: Training features
        X_test: Test features
        bool_features: List of boolean feature names
        
    Returns:
        Tuple of (X_train_encoded, X_test_encoded)
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    le = LabelEncoder()
    
    for col in bool_features:
        if col in X_train.columns:
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
    
    return X_train, X_test


def encode_target_with_smoothing(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cols: list,
    m: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform smoothed target encoding for high cardinality features.
    Matches notebook implementation exactly.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        cols: Columns to encode
        m: Smoothing parameter
        
    Returns:
        Tuple of (X_train_encoded, X_test_encoded)
    """
    global_mean = y_train.mean()
    
    # Copy to avoid SettingWithCopyWarning
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    for col in cols:
        if col not in X_train.columns:
            continue
            
        # Calculate statistics on the TRAIN set
        agg = pd.DataFrame({'feature': X_train[col], 'target': y_train})
        stats = agg.groupby('feature')['target'].agg(['count', 'mean'])
        
        # Smoothing formula
        smooth = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        
        # Map smoothed values to Train and Test
        X_train_enc[col + '_encoded'] = X_train_enc[col].map(smooth).fillna(global_mean).astype('float32')
        X_test_enc[col + '_encoded'] = X_test_enc[col].map(smooth).fillna(global_mean).astype('float32')
        
        # Drop original string/object column to save memory
        X_train_enc.drop(columns=[col], inplace=True)
        X_test_enc.drop(columns=[col], inplace=True)
    
    return X_train_enc, X_test_enc


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    threshold: str = 'median',
    n_estimators: int = 100,
    max_depth: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform feature selection using Random Forest feature importance.
    Matches notebook implementation exactly.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        threshold: Importance threshold ('median', 'mean', or float)
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, feature_importance_df)
    """
    print("\n" + "="*80)
    print("🔍 PERFORMING FEATURE SELECTION...")
    print("="*80)
    
    # 1. Initialize model to evaluate feature importance
    selector_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("🌲 Training Random Forest to evaluate feature importance...")
    selector_model.fit(X_train, y_train)
    print("✅ Training completed!")
    
    # 2. Get feature importance
    importances = selector_model.feature_importances_
    feature_names = X_train.columns
    
    # Create DataFrame for easier visualization
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Print summary statistics
    print(f"\n📈 Feature Importance Statistics:")
    print(f"   • Median importance: {feature_imp_df['Importance'].median():.6f}")
    print(f"   • Mean importance: {feature_imp_df['Importance'].mean():.6f}")
    print(f"   • Max importance: {feature_imp_df['Importance'].max():.6f}")
    print(f"   • Min importance: {feature_imp_df['Importance'].min():.6f}")
    
    # 3. Feature selection using threshold
    print(f"\n🔍 Selecting features with importance > {threshold}...")
    
    sfm = SelectFromModel(selector_model, threshold=threshold, prefit=True)
    
    # Transform the data
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)
    
    # Get selected feature names
    selected_features = feature_names[sfm.get_support()]
    
    # Convert back to DataFrame (preserve index)
    X_train_final = pd.DataFrame(
        X_train_selected,
        columns=selected_features,
        index=X_train.index
    )
    X_test_final = pd.DataFrame(
        X_test_selected,
        columns=selected_features,
        index=X_test.index
    )
    
    # Calculate removed features
    removed_features = list(set(feature_names) - set(selected_features))
    
    # Print results
    print(f"\n✅ FEATURE SELECTION RESULTS:")
    print(f"   • Original features: {X_train.shape[1]}")
    print(f"   • Selected features: {X_train_final.shape[1]} ({X_train_final.shape[1]/X_train.shape[1]*100:.1f}%)")
    print(f"   • Removed features: {len(removed_features)} ({len(removed_features)/X_train.shape[1]*100:.1f}%)")
    
    print(f"\n📋 Selected Features ({len(selected_features)}):")
    for i, feat in enumerate(sorted(selected_features), 1):
        imp = feature_imp_df[feature_imp_df['Feature'] == feat]['Importance'].values[0]
        print(f"   {i:2d}. {feat:<30} (importance: {imp:.6f})")
    
    if removed_features:
        print(f"\n🗑️  Removed Features ({len(removed_features)}):")
        for i, feat in enumerate(sorted(removed_features), 1):
            imp = feature_imp_df[feature_imp_df['Feature'] == feat]['Importance'].values[0]
            print(f"   {i:2d}. {feat:<30} (importance: {imp:.6f})")
    
    print("\n" + "="*80)
    
    return X_train_final, X_test_final, feature_imp_df


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
    Matches notebook implementation exactly.
    
    Args:
        train_path: Path to processed training data
        test_path: Path to processed test data
        save_output: Whether to save engineered features
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print("\n" + "="*80)
    print("🚀 STARTING FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Load data
    print("\n📂 Loading processed data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['Severity'])
    y_train = train_df['Severity']
    X_test = test_df.drop(columns=['Severity'])
    y_test = test_df['Severity']
    
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Convert boolean columns to object for encoding
    print("\n🔄 Converting boolean features to object type...")
    bool_col = ['Amenity', 'Crossing', 'Junction', 'Railway', 'Station', 'Stop', 'Traffic_Signal']
    for col in bool_col:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('object')
            X_test[col] = X_test[col].astype('object')
    
    # Memory optimization - Step 1
    print("\n📉 Memory optimization - Step 1...")
    X_train = reduce_mem_usage(X_train, verbose=True)
    X_test = reduce_mem_usage(X_test, verbose=True)
    
    # Encode boolean features
    print("\n🔢 Encoding boolean features...")
    X_train, X_test = encode_boolean_features(X_train, X_test, bool_col)
    print(f"   ✅ Boolean features encoded (False=0, True=1)")
    
    # Feature Engineering
    print("\n🛠️  Applying feature engineering functions...")
    print("   ⏳ Creating Time Features...")
    X_train = extract_time_features(X_train)
    X_test = extract_time_features(X_test)
    
    print("   ☁️  Creating Weather Flags...")
    X_train = create_weather_flags(X_train)
    X_test = create_weather_flags(X_test)
    
    print("   🚦 Creating Road Context features...")
    X_train = create_road_context(X_train)
    X_test = create_road_context(X_test)
    
    print("   🔗 Creating Interaction & Density features...")
    X_train = create_interaction_features(X_train)
    X_test = create_interaction_features(X_test)
    
    # Drop City and County (keep only State for target encoding)
    print("\n🗑️  Dropping City and County columns...")
    X_train.drop(columns=['City', 'County'], inplace=True, errors='ignore')
    X_test.drop(columns=['City', 'County'], inplace=True, errors='ignore')
    
    # Target encoding for State only
    print("\n🎯 Applying smoothed target encoding for State...")
    high_cardinality_features = ['State']
    X_train, X_test = encode_target_with_smoothing(
        X_train, X_test, y_train, high_cardinality_features, m=20
    )
    
    print("\n📊 Feature Engineering Summary:")
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    print(f"   Total features: {X_train.shape[1]}")
    
    # Feature Selection
    print("\n🔍 Running feature selection...")
    X_train, X_test, importance_df = select_features_rf(
        X_train, y_train, X_test,
        threshold='median',
        n_estimators=100,
        max_depth=20
    )
    
    # Final memory optimization
    print("\n📉 Final memory optimization...")
    X_train = reduce_mem_usage(X_train, verbose=True)
    X_test = reduce_mem_usage(X_test, verbose=True)
    
    # Save
    if save_output:
        output_dir = DataConfig.PROCESSED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving feature-engineered datasets...")
        X_train.to_csv(output_dir / 'X_train_featured.csv', index=False)
        X_test.to_csv(output_dir / 'X_test_featured.csv', index=False)
        y_train.to_csv(output_dir / 'y_train.csv', index=False)
        y_test.to_csv(output_dir / 'y_test.csv', index=False)
        
        print(f"✅ Feature-engineered datasets saved successfully!")
        print(f"   • {output_dir / 'X_train_featured.csv'}")
        print(f"   • {output_dir / 'X_test_featured.csv'}")
        print(f"   • {output_dir / 'y_train.csv'}")
        print(f"   • {output_dir / 'y_test.csv'}")
    
    print("\n" + "="*80)
    print("✅ FEATURE ENGINEERING PIPELINE COMPLETED!")
    print("="*80)
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = run_feature_engineering_pipeline(
        train_path=DataConfig.TRAIN_PROCESSED_PATH,
        test_path=DataConfig.TEST_PROCESSED_PATH,
        save_output=True
    )
