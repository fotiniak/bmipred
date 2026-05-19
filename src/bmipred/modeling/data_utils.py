#!/usr/bin/env python3
# bmipred/modeling/data_utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from typing import Tuple, Dict, List, Optional


def has_two_classes(arr: np.ndarray) -> bool:
    # Check if array has exactly two unique values.
    return len(np.unique(arr)) == 2


def stratified_split(data: pd.DataFrame, 
                     target_col: str, 
                     test_size: float, 
                     random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Split data ensuring patients don't appear in both train/test sets.
    # Sample one row per patient for stratification
    unique = (
        data.groupby('PatientDurableKey', group_keys=False)
        .sample(n=1, random_state=random_state)
        .reset_index(drop=True)
    )
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(unique, unique[target_col]))
    
    train_keys = unique.iloc[train_idx]['PatientDurableKey']
    test_keys = unique.iloc[test_idx]['PatientDurableKey']
    
    train = data[data['PatientDurableKey'].isin(train_keys)]
    test = data[data['PatientDurableKey'].isin(test_keys)]
    
    # Validate both classes present
    if not (has_two_classes(train[target_col]) and has_two_classes(test[target_col])):
        raise RuntimeError("Both train and test must contain both classes")
    
    return train, test


def detect_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str], List[str]]:
    # Detect numeric, binary, and categorical features automatically.
    exclude = [target_col] + [c for c in df.columns if 'id' in c.lower() or 'key' in c.lower()]
    columns = [c for c in df.columns if c not in exclude]
    
    numeric, binary, categorical = [], [], []
    
    for col in columns:
        unique_vals = df[col].dropna().unique()

        if set(unique_vals).issubset({0, 1}):
            binary.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical.append(col)
        else:
            if len(unique_vals) <= 10:
                categorical.append(col)
            else:
                numeric.append(col)
    
    return numeric, binary, categorical


def get_global_categorical_levels(df: pd.DataFrame, target_col: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    # Get all possible categories per categorical feature for consistency across splits.

    _, _, categorical = detect_feature_types(df, target_col)
    categorical_levels = {col: df[col].dropna().unique() for col in categorical}
    return categorical, categorical_levels


def convert_timedelta_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Convert timedelta columns to days, returning a copy when needed.
    td_cols = [c for c in df.select_dtypes(['timedelta64']).columns if c != target_col]
    if not td_cols:
        return df
    out = df.copy()
    for col in td_cols:
        out[col] = out[col].dt.days
    return out


def encoder_categories(series: pd.Series) -> np.ndarray:
    # Return stable category levels for one encoder column.
    levels = series.dropna().unique()
    if isinstance(levels, pd.Categorical):
        levels = levels.to_numpy()
    else:
        levels = np.asarray(levels)
    if pd.api.types.is_numeric_dtype(levels):
        levels = np.sort(levels)
    return levels


def prettify_feature_name(feature: str, column_mapping: Dict[str, str]) -> str:
    # Map raw feature names to a prettified display name when possible.
    if feature in column_mapping:
        return column_mapping[feature]
    for raw_name, pretty_name in column_mapping.items():
        prefix = f"{raw_name}_"
        if feature.startswith(prefix):
            return f"{pretty_name}: {feature[len(prefix):]}"
    return feature


def make_stratified_cv(y: np.ndarray, desired_folds: int, random_state: int, logger=None) -> StratifiedKFold:
    # Return a valid StratifiedKFold for the smallest class count.
    _, counts = np.unique(y, return_counts=True)
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Stratified cross-validation requires at least two samples in each class.")
    requested = max(2, int(desired_folds))
    n_splits = min(requested, int(counts.min()))
    if n_splits < requested and logger is not None:
        logger.warning(
            "Reducing CV folds from %s to %s because the minority class has %s samples.",
            requested, n_splits, counts.min()
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def out_of_fold_positive_prob(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Fit cloned estimators on CV folds and return out-of-fold positive probabilities.
    y = np.asarray(y)
    oof_prob = np.full(y.shape[0], np.nan, dtype=float)
    sample_weight = None if sample_weight is None else np.asarray(sample_weight)

    for train_idx, valid_idx in cv.split(X, y):
        fold_model = clone(estimator)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[train_idx]
        fold_model.fit(X[train_idx], y[train_idx], **fit_kwargs)
        oof_prob[valid_idx] = fold_model.predict_proba(X[valid_idx])[:, 1]

    if np.isnan(oof_prob).any():
        raise RuntimeError("Failed to compute out-of-fold probabilities for all rows.")
    return oof_prob


def align_shap_arrays(
    shap_list: List[np.ndarray],
    all_feature_names: List[str],
    split_feat_names: List[List[str]],
) -> List[np.ndarray]:
    # Align SHAP arrays to a shared union of features across splits.
    aligned = []
    feature_idx = {feature: i for i, feature in enumerate(all_feature_names)}
    for shap_arr, feat_names in zip(shap_list, split_feat_names):
        arr_aligned = np.zeros((shap_arr.shape[0], len(all_feature_names)))
        for i, feat in enumerate(feat_names):
            arr_aligned[:, feature_idx[feat]] = shap_arr[:, i]
        aligned.append(arr_aligned)
    return aligned


def create_output_directory(table_name: str, base_out_dir: str, split_id: int) -> str:
    # Create output directory structure for a table split.
    
    import os
    
    out = os.path.join(base_out_dir, table_name, f"split_{split_id}")
    for sub in ("models", "plots", "results"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out