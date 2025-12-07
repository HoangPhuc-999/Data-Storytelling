"""
Configuration Module for Data Processing Pipeline
=================================================

Centralized configuration for the data preprocessing pipeline.

Version: 1.0.0
"""

from pathlib import Path
from typing import List, Tuple


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

class DataConfig:
    """
    Configuration class for data processing parameters.
    
    This class contains all configuration parameters for the preprocessing pipeline,
    including file paths, split parameters, feature lists, and physical constraints.
    """
    
    # ========================================================================
    # FILE PATHS
    # ========================================================================
    
    # Project root
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Raw data paths
    RAW_DATA_PATH = Path("dataset/raw/US_Accidents_March23.csv")
    TRAIN_RAW_PATH = Path("dataset/raw/train.csv")
    TEST_RAW_PATH = Path("dataset/raw/test.csv")
    
    # Processed data paths
    PROCESSED_DIR = Path("dataset/processed")
    TRAIN_PROCESSED_PATH = PROCESSED_DIR / "train_processed.csv"
    TEST_PROCESSED_PATH = PROCESSED_DIR / "test_processed.csv"
    
    # Featured data paths (after feature engineering)
    X_TRAIN_FEATURED_PATH = PROCESSED_DIR / "X_train_featured.csv"
    X_TEST_FEATURED_PATH = PROCESSED_DIR / "X_test_featured.csv"
    Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
    Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"
    
    # Output directories
    MODELS_DIR = PROJECT_ROOT / "models"
    FIGURES_DIR = PROJECT_ROOT / "figures"
    
    # ========================================================================
    # TRAIN-TEST SPLIT PARAMETERS
    # ========================================================================
    
    TEST_SIZE = 0.2           # 20% for test set
    RANDOM_STATE = 42         # For reproducibility
    TARGET_COL = 'Severity'   # Target column name
    
    # ========================================================================
    # FEATURE CATEGORIES
    # ========================================================================
    
    # Time-related columns (convert to datetime)
    TIME_COLUMNS = [
        "Start_Time",
        "End_Time",
        "Weather_Timestamp"
    ]
    
    # Boolean columns (convert to object/categorical)
    BOOL_COLUMNS = [
        'Amenity',
        'Crossing',
        'Junction',
        'Railway',
        'Station',
        'Stop',
        'Traffic_Signal'
    ]
    
    # ========================================================================
    # FEATURES TO DROP
    # ========================================================================
    
    # Data leakage features (contain target information)
    DROP_LEAKAGE = [
        'ID',            # Unique identifier, no predictive value
        'Description',   # Text contains severity keywords
        'End_Time',      # Duration correlates with severity
        'Distance(mi)'   # Affected road length is consequence of severity
    ]
    
    # Redundant location features (high cardinality or too granular)
    DROP_LOCATION = [
        'Street',       # Too specific, captured by City/County
        'Zipcode',      # Overly granular
        'Latitude',     # Exact coordinates add noise
        'Longitude'     # Aggregate location is sufficient
    ]
    
    # Metadata features (no predictive value)
    DROP_METADATA = [
        'Airport_Code',      # Weather station identifier
        'Timezone',          # Redundant with State
        'Weather_Timestamp'  # Timestamp not needed, values are
    ]
    
    # ========================================================================
    # MISSING VALUE HANDLING
    # ========================================================================
    
    # Columns to handle missing values
    CITY_COLUMN = 'City'                    # Drop rows if missing
    SUNRISE_SUNSET_COLUMN = 'Sunrise_Sunset'  # Time-based imputation
    WEATHER_CONDITION_COLUMN = 'Weather_Condition'  # Meteorological rules
    
    # Numerical columns for median imputation
    NUMERIC_IMPUTE_COLUMNS = [
        'Temperature(F)',
        'Humidity(%)',
        'Visibility(mi)',
        'Wind_Speed(mph)',
        'Pressure(in)',
        'Precipitation(in)',
        'Wind_Chill(F)'
    ]
    
    # ========================================================================
    # OUTLIER HANDLING
    # ========================================================================
    
    # Columns to check for outliers
    OUTLIER_COLUMNS = [
        'Temperature(F)',
        'Humidity(%)',
        'Visibility(mi)'
    ]
    
    # Physical constraints for outlier capping
    TEMPERATURE_BOUNDS: Tuple[float, float] = (-50, 130)  # Fahrenheit
    HUMIDITY_BOUNDS: Tuple[float, float] = (0, 100)       # Percentage
    # Note: Visibility is NOT capped (low visibility is real weather condition)
    
    # ========================================================================
    # FEATURE ENGINEERING (for future use)
    # ========================================================================
    
    # Cyclical features
    CYCLICAL_FEATURES = [
        'Month',
        'Hour',
        'Day_of_Week'
    ]
    
    # Interaction features
    WEATHER_FEATURES = [
        'Temperature(F)',
        'Visibility(mi)',
        'Humidity(%)',
        'Wind_Speed(mph)'
    ]
    
    INFRASTRUCTURE_FEATURES = [
        'Amenity',
        'Crossing',
        'Junction',
        'Traffic_Signal'
    ]
    
    # ========================================================================
    # VARIANCE THRESHOLD
    # ========================================================================
    
    VARIANCE_THRESHOLD = 0  # Remove zero-variance features
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @classmethod
    def get_all_drop_columns(cls) -> List[str]:
        """
        Get all columns to drop (leakage + location + metadata).
        
        Returns:
            List of column names to drop
        """
        return cls.DROP_LEAKAGE + cls.DROP_LOCATION + cls.DROP_METADATA
    
    @classmethod
    def get_all_paths(cls) -> dict:
        """
        Get all file paths as a dictionary.
        
        Returns:
            Dictionary with path names and Path objects
        """
        return {
            'raw_data': cls.RAW_DATA_PATH,
            'train_raw': cls.TRAIN_RAW_PATH,
            'test_raw': cls.TEST_RAW_PATH,
            'train_processed': cls.TRAIN_PROCESSED_PATH,
            'test_processed': cls.TEST_PROCESSED_PATH,
            'processed_dir': cls.PROCESSED_DIR
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration parameters."""
        if not 0 < cls.TEST_SIZE < 1:
            raise ValueError(f"TEST_SIZE must be between 0 and 1, got {cls.TEST_SIZE}")
        
        if cls.TEMPERATURE_BOUNDS[0] >= cls.TEMPERATURE_BOUNDS[1]:
            raise ValueError(f"Invalid TEMPERATURE_BOUNDS: {cls.TEMPERATURE_BOUNDS}")
        
        if cls.HUMIDITY_BOUNDS != (0, 100):
            raise ValueError(f"HUMIDITY_BOUNDS should be (0, 100), got {cls.HUMIDITY_BOUNDS}")
        
        return True


# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

class PreprocessingConfig:
    """
    Configuration for preprocessing transformers.
    
    This class contains parameters specific to each preprocessing step.
    """
    
    # Imputation strategies
    NUMERIC_IMPUTE_STRATEGY = 'median'  # Options: 'mean', 'median', 'most_frequent'
    
    # Sunrise/Sunset imputation rules
    DAY_HOUR_START = 6   # 6 AM
    DAY_HOUR_END = 18    # 6 PM
    
    # Weather condition imputation thresholds
    FOG_VISIBILITY_THRESHOLD = 1.0      # miles
    FOG_VISIBILITY_THRESHOLD_2 = 5.0    # miles with high humidity
    FOG_HUMIDITY_THRESHOLD = 85         # percentage
    SNOW_TEMP_THRESHOLD = 32            # Fahrenheit
    SNOW_HUMIDITY_THRESHOLD = 70        # percentage
    CLOUDY_HUMIDITY_MIN = 60            # percentage
    CLOUDY_HUMIDITY_MAX = 85            # percentage
    
    # IQR outlier detection (for analysis only, not used in capping)
    IQR_MULTIPLIER = 1.5


# ============================================================================
# VALIDATION
# ============================================================================

def validate_all_configs() -> bool:
    """Validate all configuration classes."""
    try:
        DataConfig.validate_config()
        return True
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    """Validate configuration when run as a script."""
    if validate_all_configs():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration is invalid")
