#!/usr/bin/env python3

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import xgboost as xgb
import inspect
import logging

from bmipred.modeling.metrics import sensitivity, specificity, ppv, npv, find_best_f1_threshold


def evaluate_single_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluate a single trained model.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Dictionary of metrics
    """
    # Find optimal threshold on training data
    y_train_prob = model.predict_proba(X_train)[:, 1]
    best_threshold, _ = find_best_f1_threshold(y_train, y_train_prob)
    
    # Predictions on test data
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob > best_threshold).astype(int)
    
    return {
        'Model': model_name,
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Test F1 Score (positive)': f1_score(y_test, y_test_pred, average='binary', pos_label=1),
        'Test F1 Score (weighted)': f1_score(y_test, y_test_pred, average='weighted'),
        'Test F1 Score (macro)': f1_score(y_test, y_test_pred, average='macro'),
        'Test ROC AUC': roc_auc_score(y_test, y_test_prob),
        'Test Sensitivity (Recall)': sensitivity(y_test, y_test_pred),
        'Test Specificity': specificity(y_test, y_test_pred),
        'Test PPV (Precision)': ppv(y_test, y_test_pred),
        'Test NPV': npv(y_test, y_test_pred),
        'Test Matthews Correlation Coefficient': matthews_corrcoef(y_test, y_test_pred),
        'Optimal Threshold (F1)': best_threshold
    }


def compute_shap_values(
    model: BaseEstimator,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    max_samples: int = 100
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for a model.
    
    Args:
        model: Trained model
        X_train: Training features for background
        X_test: Test features to explain
        feature_names: List of feature names
        max_samples: Maximum samples for background
        
    Returns:
        (shap_values, importance_df)
    """
    logger = logging.getLogger(__name__)
    
    if X_test.shape[1] == 0 or len(feature_names) == 0:
        logger.warning("No features for SHAP computation")
        return np.zeros((X_test.shape[0], 0)), pd.DataFrame(columns=["feature", "mean_abs_shap"])
    
    TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, xgb.XGBClassifier)
    estimator = model.named_steps['classifier'] if isinstance(model, Pipeline) else model
    
    background = shap.utils.sample(X_train, min(max_samples, X_train.shape[0]), random_state=1993)
    
    try:
        if isinstance(estimator, TREE_MODELS):
            if "check_additivity" in inspect.signature(shap.TreeExplainer).parameters:
                explainer = shap.TreeExplainer(estimator, data=background, model_output="probability", check_additivity=False)
                raw_shap = explainer.shap_values(X_test)
            else:
                explainer = shap.TreeExplainer(estimator, data=background, model_output="probability")
                raw_shap = explainer.shap_values(X_test, check_additivity=False)
        elif hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, background)
            raw_shap = explainer.shap_values(X_test)
        else:
            logger.warning(f"SHAP not supported for {type(estimator)}")
            return np.zeros((X_test.shape[0], len(feature_names))), pd.DataFrame(columns=["feature", "mean_abs_shap"])
        
        # Convert to 2D array
        if isinstance(raw_shap, list):
            shap_values = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
        else:
            shap_array = np.asarray(raw_shap)
            if shap_array.ndim == 3:
                shap_values = shap_array[:, :, 1] if shap_array.shape[-1] >= 2 else shap_array[:, :, 0]
            else:
                shap_values = shap_array
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)
        
        return shap_values, importance_df
        
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return np.zeros((X_test.shape[0], len(feature_names))), pd.DataFrame(columns=["feature", "mean_abs_shap"])


def evaluate_all_models(
    trained_models: Dict[str, Tuple[BaseEstimator, Dict[str, Any]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """
    Evaluate all trained models.
    
    Args:
        trained_models: Dictionary of trained models
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        
    Returns:
        (metrics_df, shap_values_dict, shap_importance_dict)
    """
    metrics = []
    shap_values = {}
    shap_importance = {}
    
    for name, (model, _) in trained_models.items():
        # Evaluate model
        model_metrics = evaluate_single_model(model, X_train, y_train, X_test, y_test, name)
        metrics.append(model_metrics)
        
        # Compute SHAP
        shap_vals, shap_imp = compute_shap_values(model, X_train, X_test, feature_names)
        shap_values[name] = shap_vals
        shap_importance[name] = shap_imp
    
    return pd.DataFrame(metrics), shap_values, shap_importance