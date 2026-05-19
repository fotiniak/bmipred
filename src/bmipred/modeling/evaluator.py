#!/usr/bin/env python3

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, roc_auc_score,
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
from bmipred.modeling.data_utils import make_stratified_cv, out_of_fold_positive_prob


def evaluate_single_model(model: BaseEstimator,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          model_name: str,
                          cv_results: Dict[str, Any] = None,
                          cv_folds: int = 5,
                          sample_weight: Optional[np.ndarray] = None,
                          random_state: int = 7000,) -> Dict[str, Any]:
    # Evaluate a single trained model. Finds optimal F1 threshold via out-of-fold predictions and computes test metrics at that threshold.
    # Find optimal threshold using out-of-fold probabilities
    threshold_cv = make_stratified_cv(y_train, cv_folds, random_state=random_state)
    y_oof_prob = out_of_fold_positive_prob(
        model, X_train, y_train, threshold_cv, sample_weight=sample_weight
    )
    best_threshold, best_oof_f1 = find_best_f1_threshold(y_train, y_oof_prob)

    # Predictions on test data
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    result = {
        'Model': model_name,
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Test Balanced Accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'Test F1 Score (positive)': f1_score(y_test, y_test_pred, average='binary', pos_label=1, zero_division=0),
        'Test F1 Score (weighted)': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'Test F1 Score (macro)': f1_score(y_test, y_test_pred, average='macro', zero_division=0),
        'Test ROC AUC': roc_auc_score(y_test, y_test_prob),
        'Test Average Precision': average_precision_score(y_test, y_test_prob),
        'Test Sensitivity (Recall)': sensitivity(y_test, y_test_pred),
        'Test Specificity': specificity(y_test, y_test_pred),
        'Test PPV (Precision)': ppv(y_test, y_test_pred),
        'Test NPV': npv(y_test, y_test_pred),
        'Test Matthews Correlation Coefficient': matthews_corrcoef(y_test, y_test_pred),
        'Test Brier Score': brier_score_loss(y_test, y_test_prob),
        'Optimal Threshold (OOF F1)': best_threshold,
        'OOF F1 at Threshold': best_oof_f1,
    }

    # Include CV metrics from GridSearchCV if provided
    if cv_results is not None:
        result['CV Average Precision'] = cv_results.get('CV Average Precision', np.nan)
        result['CV ROC AUC'] = cv_results.get('CV ROC AUC', np.nan)
        result['CV Balanced Accuracy'] = cv_results.get('CV Balanced Accuracy', np.nan)
        result['Best Parameters'] = cv_results.get('best_params', {})

    return result


def compute_shap_values(model: BaseEstimator,
                        X_train: np.ndarray,
                        X_test: np.ndarray,
                        feature_names: List[str],
                        max_samples: int = 100) -> Tuple[np.ndarray, pd.DataFrame]:
    # Compute SHAP values for a model.

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
    feature_names: List[str],
    cv_folds: int = 5,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 7000,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    # Evaluate all trained models.

    metrics = []
    shap_values = {}
    shap_importance = {}

    for name, (model, cv_results) in trained_models.items():
        model_metrics = evaluate_single_model(
            model, X_train, y_train, X_test, y_test, name,
            cv_results=cv_results,
            cv_folds=cv_folds,
            sample_weight=sample_weight,
            random_state=random_state,
        )
        metrics.append(model_metrics)

        shap_vals, shap_imp = compute_shap_values(model, X_train, X_test, feature_names)
        shap_values[name] = shap_vals
        shap_importance[name] = shap_imp

    return pd.DataFrame(metrics), shap_values, shap_importance