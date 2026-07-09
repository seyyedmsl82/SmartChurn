"""
Model training module for churn prediction
"""
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Handle model training with experiment tracking"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 experiment_name: str = "churn_prediction"):
        """
        Initialize model trainer
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for MLflow experiment
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_params = {}
        self.metrics = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Set random seed
        self.random_state = self.config.get('random_state', 42)
        np.random.seed(self.random_state)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_models(self) -> Dict[str, Any]:
        """Get dictionary of models to train"""
        model_configs = self.config.get('models', {})
        
        models = {}
        
        # Logistic Regression
        if model_configs.get('logistic_regression', {}).get('enabled', True):
            models['LogisticRegression'] = LogisticRegression(
                C=model_configs.get('logistic_regression', {}).get('C', 1.0),
                max_iter=model_configs.get('logistic_regression', {}).get('max_iter', 1000),
                class_weight='balanced',
                random_state=self.random_state
            )
        
        # Random Forest
        if model_configs.get('random_forest', {}).get('enabled', True):
            models['RandomForest'] = RandomForestClassifier(
                n_estimators=model_configs.get('random_forest', {}).get('n_estimators', 100),
                max_depth=model_configs.get('random_forest', {}).get('max_depth', 10),
                min_samples_split=model_configs.get('random_forest', {}).get('min_samples_split', 5),
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # XGBoost
        if model_configs.get('xgboost', {}).get('enabled', True):
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=model_configs.get('xgboost', {}).get('n_estimators', 100),
                learning_rate=model_configs.get('xgboost', {}).get('learning_rate', 0.1),
                max_depth=model_configs.get('xgboost', {}).get('max_depth', 6),
                scale_pos_weight=model_configs.get('xgboost', {}).get('scale_pos_weight', 5),
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        # LightGBM
        if model_configs.get('lightgbm', {}).get('enabled', True):
            models['LightGBM'] = LGBMClassifier(
                n_estimators=model_configs.get('lightgbm', {}).get('n_estimators', 100),
                learning_rate=model_configs.get('lightgbm', {}).get('learning_rate', 0.1),
                num_leaves=model_configs.get('lightgbm', {}).get('num_leaves', 31),
                class_weight='balanced',
                random_state=self.random_state
            )
        
        return models
    
    def train_single_model(self, 
                          model: Any, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          X_val: pd.DataFrame,
                          y_val: pd.Series,
                          model_name: str) -> Dict[str, float]:
        """
        Train a single model and evaluate
        
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'val_accuracy': accuracy_score(y_val, y_pred_val),
            'val_precision': precision_score(y_val, y_pred_val, average='weighted'),
            'val_recall': recall_score(y_val, y_pred_val, average='weighted'),
            'val_f1': f1_score(y_val, y_pred_val, average='weighted')
        }
        
        if y_proba_val is not None:
            metrics['val_roc_auc'] = roc_auc_score(y_val, y_proba_val)
            metrics['val_pr_auc'] = average_precision_score(y_val, y_proba_val)
        
        self.models[model_name] = model
        self.metrics[model_name] = metrics
        
        logger.info(f"  {model_name} - Val F1: {metrics['val_f1']:.4f}")
        
        return metrics
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series) -> Dict[str, Dict]:
        """Train all configured models"""
        
        models = self.get_models()
        results = {}
        
        for name, model in models.items():
            metrics = self.train_single_model(
                model, X_train, y_train, X_val, y_val, name
            )
            results[name] = metrics
        
        # Find best model
        best_name = max(results, key=lambda x: results[x]['val_f1'])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        logger.info(f"\nBest model: {best_name} (F1: {results[best_name]['val_f1']:.4f})")
        
        return results
    
    def cross_validate_model(self,
                            model: Any,
                            X: pd.DataFrame,
                            y: pd.Series,
                            cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation for a model"""
        
        logger.info(f"Cross-validating {model.__class__.__name__}")
        
        # Use stratified K-fold for classification
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Metrics to track
        metrics_list = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model
            model_clone = self._clone_model(model)
            model_clone.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model_clone.predict(X_val)
            y_proba = model_clone.predict_proba(X_val)[:, 1] if hasattr(model_clone, 'predict_proba') else None
            
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred))
            metrics_list['recall'].append(recall_score(y_val, y_pred))
            metrics_list['f1'].append(f1_score(y_val, y_pred))
            if y_proba is not None:
                metrics_list['roc_auc'].append(roc_auc_score(y_val, y_proba))
        
        # Calculate mean and std
        results = {}
        for metric, values in metrics_list.items():
            if values:
                results[f'{metric}_mean'] = np.mean(values)
                results[f'{metric}_std'] = np.std(values)
        
        logger.info(f"  CV F1: {results['f1_mean']:.4f} (+/- {results['f1_std']:.4f})")
        
        return results
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model with the same parameters"""
        if hasattr(model, '__class__'):
            params = model.get_params() if hasattr(model, 'get_params') else {}
            return model.__class__(**params, random_state=self.random_state)
        return model
    
    def hyperparameter_tuning(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             model_type: str = 'xgboost',
                             n_trials: int = 30) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model ('xgboost', 'random_forest', 'lightgbm')
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters
        """
        logger.info(f"Hyperparameter tuning for {model_type}")
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
                }
                model = xgb.XGBClassifier(
                    **params,
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_float('max_features', 0.1, 1.0)
                }
                model = RandomForestClassifier(
                    **params,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                )
            
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
                }
                model = LGBMClassifier(
                    **params,
                    class_weight='balanced',
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Cross-validation score
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=3, scoring='f1_weighted'
            )
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")
        
        self.best_params = best_params
        
        return best_params
    
    def save_model(self, model: Any, model_name: str, save_path: str = 'models'):
        """Save model to disk"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = save_path / f'{model_name}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics.get(model_name, {}),
            'params': model.get_params() if hasattr(model, 'get_params') else {}
        }
        
        meta_file = save_path / f'{model_name}_metadata.json'
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Metadata saved to {meta_file}")
        
        return model_file
    
    def load_model(self, model_path: str) -> Any:
        """Load a saved model"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
