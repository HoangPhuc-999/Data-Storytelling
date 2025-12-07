"""
Data processing module for US Accidents dataset.
"""

# Import configuration from config module
from ..config.config import DataConfig, PreprocessingConfig

from .data_processing import (
    # Custom Transformers
    VarianceThresholdSelector,
    ConstantAndDuplicateRemover,
    SunriseSunsetImputer,
    WeatherConditionImputer,
    DomainBasedOutlierCapper,
    SequentialProcessor,
    
    # Main Pipeline
    run_full_preprocessing_pipeline,
    
    # Utility Functions
    load_raw_data,
    stratified_train_test_split_data,
    convert_data_types,
    drop_missing_city,
    remove_feature_leakage_and_redundancy,
    identify_feature_types,
    validate_processed_data,
    save_processed_data,
)

__all__ = [
    # Configuration (imported from config module)
    'DataConfig',
    'PreprocessingConfig',
    
    # Transformers
    'VarianceThresholdSelector',
    'ConstantAndDuplicateRemover',
    'SunriseSunsetImputer',
    'WeatherConditionImputer',
    'DomainBasedOutlierCapper',
    'SequentialProcessor',
    
    # Main Pipeline
    'run_full_preprocessing_pipeline',
    
    # Utilities
    'load_raw_data',
    'stratified_train_test_split_data',
    'convert_data_types',
    'drop_missing_city',
    'remove_feature_leakage_and_redundancy',
    'identify_feature_types',
    'validate_processed_data',
    'save_processed_data',
]

__version__ = '1.0.0'
