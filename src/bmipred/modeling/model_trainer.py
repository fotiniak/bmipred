#!/usr/bin/env python3
# bmipred/modeling/model_trainer.py

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
import importlib
import logging


def create_model_from_config(model_config: Dict[str, Any]) -> BaseEstimator:
    # Create a model instance from configuration.
    module_name, class_name = model_config['class'].rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class(**model_config['params'])


MODEL_SELECTION_SCORING = "roc_auc"

CV_SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
}


def train_single_model(model: BaseEstimator,
                       param_grid: Dict[str, Any],
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       sample_weight: Optional[np.ndarray] = None,
                       cv_folds: int = 5,
                       random_state: int = 42,
                       n_jobs: int = -1,) -> Tuple[BaseEstimator, Dict[str, Any]]:
    # Train a single model with hyperparameter tuning using multi-metric scoring.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=CV_SCORING,
        n_jobs=n_jobs,
        refit=MODEL_SELECTION_SCORING,
        return_train_score=False,
        pre_dispatch="n_jobs",
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight

    grid_search.fit(X_train, y_train, **fit_kwargs)

    best_idx = grid_search.best_index_
    cv_res = grid_search.cv_results_

    return grid_search.best_estimator_, {
        'best_params': grid_search.best_params_,
        'best_index': best_idx,
        'cv_results': cv_res,
        'CV Average Precision': cv_res['mean_test_average_precision'][best_idx],
        'CV ROC AUC': cv_res['mean_test_roc_auc'][best_idx],
        'CV Balanced Accuracy': cv_res['mean_test_balanced_accuracy'][best_idx],
    }


def train_all_models(model_configs: Dict[str, Dict[str, Any]],
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     training_config: Dict[str, Any],
                     sample_weight: Optional[np.ndarray] = None) -> Dict[str, Tuple[BaseEstimator, Dict[str, Any]]]:
    # Train all models with hyperparameter tuning.
    trained_models = {}
    logger = logging.getLogger(__name__)
    
    for name, config in model_configs.items():
        logger.info(f"Training {name}...")
        
        model = create_model_from_config(config)
        param_grid = config.get('param_grid', {})
        
        trained_model, cv_results = train_single_model(
            model, param_grid, X_train, y_train,
            sample_weight=sample_weight,
            cv_folds=training_config['cv_folds'],
            n_jobs=training_config['n_jobs'],
        )
        
        trained_models[name] = (trained_model, cv_results)
        
    return trained_models