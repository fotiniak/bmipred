#!/usr/bin/env python3
# bmipred/modeling/model_trainer.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
import importlib
import logging


def create_model_from_config(model_config: Dict[str, Any]) -> BaseEstimator:
    """
    Create a model instance from configuration.
    
    Args:
        model_config: Model configuration with class and params
        
    Returns:
        Model instance
    """
    module_name, class_name = model_config['class'].rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class(**model_config['params'])


def train_single_model(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    scoring: str = 'roc_auc',
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a single model with hyperparameter tuning.
    
    Args:
        model: Model instance
        param_grid: Parameter grid for tuning
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of CV folds
        scoring: Scoring metric
        random_state: Random seed
        n_jobs: Number of parallel jobs
        
    Returns:
        (best_model, cv_results)
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }


def train_all_models(
    model_configs: Dict[str, Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    training_config: Dict[str, Any]
) -> Dict[str, Tuple[BaseEstimator, Dict[str, Any]]]:
    """
    Train all models with hyperparameter tuning.
    
    Args:
        model_configs: Dictionary of model configurations
        X_train: Training features
        y_train: Training labels
        training_config: Training configuration
        
    Returns:
        Dictionary of model_name -> (trained_model, cv_results)
    """
    trained_models = {}
    logger = logging.getLogger(__name__)
    
    for name, config in model_configs.items():
        logger.info(f"Training {name}...")
        
        model = create_model_from_config(config)
        param_grid = config.get('param_grid', {})
        
        trained_model, cv_results = train_single_model(
            model, param_grid, X_train, y_train,
            cv_folds=training_config['cv_folds'],
            scoring=training_config['scoring'],
            n_jobs=training_config['n_jobs']
        )
        
        trained_models[name] = (trained_model, cv_results)
        
    return trained_models