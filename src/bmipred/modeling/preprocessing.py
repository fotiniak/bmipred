#!/usr/bin/env python3
# bmipred/modeling/preprocessing.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .data_utils import detect_feature_types


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    categorical_levels: Optional[Dict[str, np.ndarray]] = None,
    categorical_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], ColumnTransformer]:
    """
    Preprocess training and test data with scaling, encoding, and imputation.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_col: Target column name
        categorical_levels: Predefined categorical levels for consistency
        categorical_columns: Predefined categorical column names
        
    Returns:
        (X_train, X_test, feature_names, pretty_feature_names, preprocessor)
    """
    
    numeric, binary, detected_categorical = detect_feature_types(train_df, target_col)
    categorical = categorical_columns if categorical_columns is not None else detected_categorical
    
    # Handle timedelta columns
    td_cols = [c for c in train_df.select_dtypes(['timedelta64']).columns if c != target_col]
    for col in td_cols:
        for df in (train_df, test_df):
            df[col] = df[col].dt.days
        numeric = list(set(numeric) | set(td_cols))
    
    # Setup categorical encoder
    if categorical_levels is not None and categorical:
        cat_ohe = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            categories=[categorical_levels[c] for c in categorical]
        )
    else:
        cat_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Create preprocessor
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), numeric),
        ('bin', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent'))
        ]), binary),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', cat_ohe)
        ]), categorical)
    ], remainder='drop')
    
    # Transform data
    X_train = preprocessor.fit_transform(train_df.drop(columns=[target_col], errors='ignore'))
    X_test = preprocessor.transform(test_df.drop(columns=[target_col], errors='ignore'))
    
    # Generate feature names
    feat_names = []
    if numeric:
        feat_names += numeric
    if binary:
        feat_names += binary
    if categorical:
        cats = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical)
        feat_names += list(cats)

    return X_train, X_test, feat_names, preprocessor