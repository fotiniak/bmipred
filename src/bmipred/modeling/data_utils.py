#!/usr/bin/env python3
# bmipred/modeling/data_utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, Dict, List


def has_two_classes(arr: np.ndarray) -> bool:
    """Check if array has exactly two unique values."""
    return len(np.unique(arr)) == 2


def stratified_train_test_split(
    data: pd.DataFrame, 
    target_col: str, 
    test_size: float, 
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data ensuring patients don't appear in both train/test sets.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        (train_df, test_df)
    """
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
    """
    Detect numeric, binary, and categorical features automatically.
    
    Args:
        df: Input DataFrame
        target_col: Target column to exclude
        
    Returns:
        (numeric_cols, binary_cols, categorical_cols)
    """
    exclude = [target_col] + [c for c in df.columns if 'id' in c.lower() or 'key' in c.lower()]
    columns = [c for c in df.columns if c not in exclude]
    
    numeric, binary, categorical = [], [], []
    
    for col in columns:
        unique_vals = df[col].dropna().unique()
        
        if pd.api.types.is_numeric_dtype(df[col]) and len(unique_vals) > 2:
            numeric.append(col)
        elif set(unique_vals).issubset({0, 1}):
            binary.append(col)
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical.append(col)
        else:
            if len(unique_vals) <= 10:
                categorical.append(col)
            else:
                numeric.append(col)
    
    return numeric, binary, categorical


def get_global_categorical_levels(df: pd.DataFrame, target_col: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Get all possible categories per categorical feature for consistency across splits.
    
    Args:
        df: Input DataFrame
        target_col: Target column to exclude
        
    Returns:
        (categorical_columns, categorical_levels_dict)
    """
    _, _, categorical = detect_feature_types(df, target_col)
    categorical_levels = {col: df[col].dropna().unique() for col in categorical}
    return categorical, categorical_levels


def create_output_directory(table_name: str, base_out_dir: str, split_id: int) -> str:
    """
    Create output directory structure for a table split.
    
    Args:
        table_name: Name of the table
        base_out_dir: Root output directory
        split_id: Split identifier
        
    Returns:
        Path to created directory
    """
    import os
    
    out = os.path.join(base_out_dir, table_name, f"split_{split_id}")
    for sub in ("models", "plots", "results"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out