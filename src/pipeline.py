"""
ML Pipeline Module
==================
Flexible and configurable end-to-end machine learning pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings
import time
import pickle
import json
from datetime import datetime

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import DataConfig
from src.data.data_processing import run_full_preprocessing_pipeline
from src.features.feature_engineering import run_feature_engineering_pipeline
from src.model.training import (
    train_and_evaluate_models,
    compute_sample_weights
)
from src.evaluation.metrics import compare_models, print_evaluation_report

warnings.filterwarnings('ignore')


class MLPipeline:
    """
    End-to-end ML pipeline for US Accidents Severity Prediction.
    
    Manages data preprocessing, feature engineering, model training,
    and evaluation with configurable steps and model persistence.
    """
    
    def __init__(
        self,
        config: Optional[DataConfig] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Data configuration object
            output_dir: Directory for saving models and results
        """
        self.config = config or DataConfig()
        self.output_dir = output_dir or Path("models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.pipeline_metadata = {
            'created_at': datetime.now().isoformat(),
            'steps_completed': []
        }
    
    def run_preprocessing(self, force: bool = False) -> Tuple[Path, Path]:
        """
        Execute data preprocessing step.
        
        Args:
            force: Force reprocessing even if files exist
            
        Returns:
            Tuple of (train_path, test_path)
        """
        print("\n" + "="*70)
        print("STEP 1: DATA PREPROCESSING")
        print("="*70)
        
        train_path = self.config.TRAIN_PROCESSED_PATH
        test_path = self.config.TEST_PROCESSED_PATH
        
        if not force and train_path.exists() and test_path.exists():
            print(f"✓ Using existing processed data:")
            print(f"  - {train_path}")
            print(f"  - {test_path}")
            self.pipeline_metadata['steps_completed'].append('preprocessing_skipped')
            return train_path, test_path
        
        start_time = time.time()
        run_full_preprocessing_pipeline()
        elapsed = time.time() - start_time
        
        print(f"✓ Preprocessing complete in {elapsed:.2f}s")
        self.pipeline_metadata['steps_completed'].append('preprocessing')
        self.pipeline_metadata['preprocessing_time'] = elapsed
        
        return train_path, test_path
    
    def run_feature_engineering(self, force: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Execute feature engineering step.
        
        Args:
            force: Force re-engineering even if files exist
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        print("\n" + "="*70)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*70)
        
        X_train_path = self.config.PROCESSED_DIR / 'X_train_featured.csv'
        y_train_path = self.config.PROCESSED_DIR / 'y_train.csv'
        X_test_path = self.config.PROCESSED_DIR / 'X_test_featured.csv'
        y_test_path = self.config.PROCESSED_DIR / 'y_test.csv'
        
        if not force and all(p.exists() for p in [X_train_path, y_train_path, X_test_path, y_test_path]):
            print(f"✓ Using existing featured data:")
            print(f"  - {X_train_path}")
            print(f"  - {X_test_path}")
            
            X_train = pd.read_csv(X_train_path)
            y_train = pd.read_csv(y_train_path).squeeze()
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).squeeze()
            
            self.pipeline_metadata['steps_completed'].append('feature_engineering_skipped')
            return X_train, y_train, X_test, y_test
        
        start_time = time.time()
        X_train, y_train, X_test, y_test = run_feature_engineering_pipeline(
            train_path=self.config.TRAIN_PROCESSED_PATH,
            test_path=self.config.TEST_PROCESSED_PATH,
            save_output=True
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Feature engineering complete in {elapsed:.2f}s")
        self.pipeline_metadata['steps_completed'].append('feature_engineering')
        self.pipeline_metadata['feature_engineering_time'] = elapsed
        
        return X_train, y_train, X_test, y_test
    
    def run_model_training(
        self,
        models_to_train: Optional[List[str]] = None,
        save_models: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute model training and evaluation step.
        
        Args:
            models_to_train: List of model names to train
            save_models: Save trained models to disk
            
        Returns:
            Dictionary of model results
        """
        print("\n" + "="*70)
        print("STEP 3: MODEL TRAINING & EVALUATION")
        print("="*70)
        
        if models_to_train is None:
            models_to_train = ['rf_regressor', 'xgboost', 'lightgbm', 'catboost']
        
        start_time = time.time()
        self.results = train_and_evaluate_models(
            X_train_path=self.config.PROCESSED_DIR / 'X_train_featured.csv',
            y_train_path=self.config.PROCESSED_DIR / 'y_train.csv',
            X_test_path=self.config.PROCESSED_DIR / 'X_test_featured.csv',
            y_test_path=self.config.PROCESSED_DIR / 'y_test.csv',
            models_to_train=models_to_train
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Model training complete in {elapsed:.2f}s")
        self.pipeline_metadata['steps_completed'].append('model_training')
        self.pipeline_metadata['training_time'] = elapsed
        
        # Find best model
        self._select_best_model()
        
        # Save models
        if save_models:
            self._save_models()
        
        return self.results
    
    def _select_best_model(self, metric: str = 'qwk'):
        """
        Select best model based on test set metric.
        
        Args:
            metric: Metric to compare ('qwk', 'f1_macro', 'f1_weighted')
        """
        best_score = -np.inf
        
        for model_name, model_results in self.results.items():
            test_metrics = model_results.get('test_metrics', {})
            score = test_metrics.get(metric, -np.inf)
            
            if score > best_score:
                best_score = score
                self.best_model = model_results['model']
                self.best_model_name = model_name
        
        self.pipeline_metadata['best_model'] = self.best_model_name
        self.pipeline_metadata['best_model_score'] = best_score
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {self.best_model_name} (QWK: {best_score:.4f})")
        print(f"{'='*70}")
    
    def _save_models(self):
        """Save all trained models to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = self.output_dir / f"run_{timestamp}"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_results in self.results.items():
            model = model_results['model']
            model_path = models_dir / f"{model_name}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"✓ Saved {model_name} to {model_path}")
        
        # Save metadata
        metadata_path = models_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.pipeline_metadata, f, indent=2)
        
        print(f"✓ Saved metadata to {metadata_path}")
        
        self.pipeline_metadata['models_saved_to'] = str(models_dir)
    
    def run(
        self,
        skip_preprocessing: bool = False,
        skip_feature_engineering: bool = False,
        models_to_train: Optional[List[str]] = None,
        save_models: bool = True,
        force_rerun: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute full pipeline.
        
        Args:
            skip_preprocessing: Skip preprocessing step
            skip_feature_engineering: Skip feature engineering step
            models_to_train: List of models to train
            save_models: Save trained models
            force_rerun: Force rerun all steps even if outputs exist
            
        Returns:
            Dictionary of model results
        """
        pipeline_start = time.time()
        
        print("\n" + "#"*70)
        print("# US ACCIDENTS SEVERITY PREDICTION - ML PIPELINE")
        print("#"*70)
        
        # Step 1: Preprocessing
        if not skip_preprocessing:
            self.run_preprocessing(force=force_rerun)
        
        # Step 2: Feature Engineering
        if not skip_feature_engineering:
            self.run_feature_engineering(force=force_rerun)
        
        # Step 3: Model Training
        self.run_model_training(models_to_train, save_models)
        
        # Pipeline summary
        pipeline_elapsed = time.time() - pipeline_start
        self.pipeline_metadata['total_time'] = pipeline_elapsed
        
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        print(f"Total time: {self.pipeline_metadata.get('total_time', 0):.2f}s")
        print(f"Steps completed: {len(self.pipeline_metadata['steps_completed'])}")
        print(f"  {', '.join(self.pipeline_metadata['steps_completed'])}")
        print(f"Best model: {self.best_model_name}")
        print(f"Best QWK score: {self.pipeline_metadata.get('best_model_score', 0):.4f}")
        
        if 'models_saved_to' in self.pipeline_metadata:
            print(f"Models saved to: {self.pipeline_metadata['models_saved_to']}")
        
        print("="*70 + "\n")
    
    def compare_results(self, metric: str = 'qwk', dataset: str = 'test'):
        """
        Compare all trained models.
        
        Args:
            metric: Metric to compare
            dataset: 'train' or 'test'
        """
        if not self.results:
            print("No models trained yet. Run pipeline first.")
            return
        
        return compare_models(self.results, metric, dataset)
    
    def get_best_model(self):
        """Get the best trained model."""
        if self.best_model is None:
            print("No models trained yet. Run pipeline first.")
            return None
        
        return self.best_model
    
    def predict(self, X: pd.DataFrame, use_best_model: bool = True, model_name: Optional[str] = None):
        """
        Make predictions using trained model.
        
        Args:
            X: Features to predict
            use_best_model: Use best model (default)
            model_name: Specific model to use (if use_best_model=False)
            
        Returns:
            Predictions array
        """
        if use_best_model:
            if self.best_model is None:
                raise ValueError("No model trained. Run pipeline first.")
            model = self.best_model
        else:
            if model_name not in self.results:
                raise ValueError(f"Model {model_name} not found.")
            model = self.results[model_name]['model']
        
        return model.predict(X)
    
    def save_pipeline(self, filepath: Path):
        """
        Save entire pipeline to disk.
        
        Args:
            filepath: Path to save pipeline
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"✓ Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: Path):
        """
        Load pipeline from disk.
        
        Args:
            filepath: Path to pipeline file
            
        Returns:
            Loaded MLPipeline object
        """
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        
        print(f"✓ Pipeline loaded from {filepath}")
        return pipeline


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_quick_pipeline(models: Optional[List[str]] = None) -> MLPipeline:
    """
    Quick pipeline execution with default settings.
    
    Args:
        models: List of models to train (default: all)
        
    Returns:
        Completed MLPipeline object
    """
    pipeline = MLPipeline()
    pipeline.run(
        skip_preprocessing=False,
        skip_feature_engineering=False,
        models_to_train=models,
        save_models=True,
        force_rerun=False
    )
    return pipeline


def run_fast_pipeline(models: Optional[List[str]] = None) -> MLPipeline:
    """
    Fast pipeline using existing preprocessed data.
    
    Args:
        models: List of models to train (default: all)
        
    Returns:
        Completed MLPipeline object
    """
    pipeline = MLPipeline()
    pipeline.run(
        skip_preprocessing=True,
        skip_feature_engineering=True,
        models_to_train=models,
        save_models=True,
        force_rerun=False
    )
    return pipeline


def run_custom_pipeline(
    preprocessing: bool = True,
    feature_engineering: bool = True,
    models: Optional[List[str]] = None,
    save: bool = True
) -> MLPipeline:
    """
    Custom pipeline with flexible step control.
    
    Args:
        preprocessing: Run preprocessing
        feature_engineering: Run feature engineering
        models: List of models to train
        save: Save models
        
    Returns:
        Completed MLPipeline object
    """
    pipeline = MLPipeline()
    pipeline.run(
        skip_preprocessing=not preprocessing,
        skip_feature_engineering=not feature_engineering,
        models_to_train=models,
        save_models=save,
        force_rerun=False
    )
    return pipeline


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example 1: Full pipeline with all models
    print("\n### EXAMPLE 1: Full Pipeline ###")
    pipeline = run_quick_pipeline()
    
    # Example 2: Fast pipeline (skip preprocessing)
    # print("\n### EXAMPLE 2: Fast Pipeline ###")
    # pipeline = run_fast_pipeline(models=['xgboost', 'lightgbm'])
    
    # Example 3: Custom pipeline
    # print("\n### EXAMPLE 3: Custom Pipeline ###")
    # pipeline = run_custom_pipeline(
    #     preprocessing=False,
    #     feature_engineering=True,
    #     models=['catboost'],
    #     save=True
    # )
    
    # Compare results
    pipeline.compare_results(metric='qwk', dataset='test')
    
    # Get best model
    best_model = pipeline.get_best_model()
    print(f"\nBest model obtained: {pipeline.best_model_name}")
