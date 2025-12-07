"""
US Accidents Severity Prediction
=================================
Production ML pipeline for predicting accident severity.
"""

__version__ = '1.0.0'
__author__ = 'Data Science Team'

from .config import DataConfig, PreprocessingConfig
from .data import data_processing
from .features import feature_engineering
from .model import training
from .evaluation import metrics
from .visualization import plots
from .pipeline import (
    MLPipeline,
    run_quick_pipeline,
    run_fast_pipeline,
    run_custom_pipeline
)

__all__ = [
    'DataConfig',
    'PreprocessingConfig',
    'data_processing',
    'feature_engineering',
    'training',
    'metrics',
    'plots',
    'MLPipeline',
    'run_quick_pipeline',
    'run_fast_pipeline',
    'run_custom_pipeline'
]
