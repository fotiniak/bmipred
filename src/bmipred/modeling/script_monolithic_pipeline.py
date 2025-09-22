#!/usr/bin/env python3


import os
import sys
import datetime
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
from typing import Dict
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, matthews_corrcoef
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import contextlib

# --- CONFIGURATION ---
config = {
    'target_col': 'target',
    'columns_to_drop': ['PatientDurableKey', 'CreateInstant', 'RateOfBMIChange_month'],
    'test_size': 0.2,
    'random_state': 1993,
    'n_repeats': 10,
    'cv_folds': 10,
    'models': {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=1993),
        'RandomForest': RandomForestClassifier(random_state=1993),
        'DecisionTree': DecisionTreeClassifier(random_state=1993),
        'GradientBoosting': GradientBoostingClassifier(random_state=1993),
        'XGBoost': xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=1993)
    },
    'param_grids': {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1],
            'penalty': ['l2', 'elasticnet'],
            'l1_ratio': [0.1, 0.5, 0.9],
            'solver': ['saga'],
            'max_iter': [5000]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 20],
            'max_features': ['sqrt', 0.2],
            'min_samples_split': [2, 10, 30]
        },
        'DecisionTree': {
            'max_depth': [5, 10, 20],
            'min_samples_split': [5, 10, 30],
            'max_features': ['sqrt', 0.2]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'max_features': ['sqrt', 0.2]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.2, 0.5, 1.0],
            'reg_alpha': [0, 0.5, 1],
            'reg_lambda': [1, 3]
        }
    },
    'plots': {
        'roc_curve': {'xlabel': 'False Positive Rate', 'ylabel': 'True Positive Rate', 'title': 'ROC Curve', 'figsize': (5, 5)},
        'confusion_matrix': {'xlabel': 'Predicted', 'ylabel': 'Actual', 'title': 'Confusion Matrix', 'figsize': (5, 5)},
        'precision_recall_curve': {'xlabel': 'Recall', 'ylabel': 'Precision', 'title': 'Precision-Recall Curve', 'figsize': (5, 5)},
        'calibration_curve': {'xlabel': 'Mean predicted value', 'ylabel': 'Fraction of positives', 'title': 'Calibration Curve', 'figsize': (5, 5)}
    },
    'plot_params': {'figure.figsize': (7, 7), 'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'savefig.dpi': 500}
}

plt.rcParams.update(config['plot_params'])

# --- FILE PATHS ---
table_paths: Dict[str, str] = {
    "cohort_on_olanzapine_first": "/home/azureuser/cloudfiles/code/Users/foteini.aktypi/bmipred/data/processed/cohort_on_olanzapine_first_refined.parquet",
    "cohort_on_olanzapine_last": "/home/azureuser/cloudfiles/code/Users/foteini.aktypi/bmipred/data/processed/cohort_on_olanzapine_last_refined.parquet",
    #"cohort_on_olanzapine_first_mini": "/home/azureuser/cloudfiles/code/Users/foteini.aktypi/bmipred/data/processed/cohort_on_olanzapine_first_mini.parquet",
    #"cohort_on_olanzapine_last_mini": "/home/azureuser/cloudfiles/code/Users/foteini.aktypi/bmipred/data/processed/cohort_on_olanzapine_last_mini.parquet",
    #"cohort_on_olanzapine_first_mini10": "/home/azureuser/cloudfiles/code/Users/foteini.aktypi/bmipred/data/processed/cohort_on_olanzapine_first_mini10.parquet",
    #"cohort_on_olanzapine_last_mini10": "/home/azureuser/cloudfiles/code/Users/foteini.aktypi/bmipred/data/processed/cohort_on_olanzapine_last_mini10.parquet",
    "cohort_on_olanzapine_first_mini3": "/home/azureuser/cloudfiles/code/Users/foteini.aktypi/bmipred/data/processed/cohort_on_olanzapine_first_mini3.parquet",
    "cohort_on_olanzapine_last_mini3": "/home/azureuser/cloudfiles/code/Users/foteini.aktypi/bmipred/data/processed/cohort_on_olanzapine_last_mini3.parquet"
}


column_mapping = {
    'HeightInCentimeters_median': 'Height (cm)',
    'WeightInGrams': 'Weight (g)',
    'BodyMassIndex_recalc': 'BMI (kg/m²)',
    'healthAssesment_age': 'Age (years)',
    'bmi_diff_from_next': 'BMI difference from next measurement (kg/m²)',
    'bmi_diff_timepass_from_next': 'Time until next BMI measurement (days)',
    'BMI_Variance_History': 'BMI variance in patient history ((kg/m²)²)',
    'bmi_diff_from_first_measurement': 'BMI difference from first measurement (kg/m²)',
    'bmi_diff_from_first_measurement_perc': 'Percentage % BMI change from first measurement',
    'max_bmi_difference': 'Max BMI deviation from baseline (kg/m²)',
    'min_bmi_difference': 'Min BMI deviation from baseline (kg/m²)',
    'Labanalysis_ALBUMINP': 'Albumin levels',
    'Labanalysis_BASISKFOSFATASEP': 'Basic phosphate levels',
    'Labanalysis_BASOFILOCYTTERB': 'Basophils levels',
    'Labanalysis_CREAKTIVTPROTEINCRPP': 'C-reactive protein levels',
    'Labanalysis_EGFR173MCKDEPINYRE': 'Estimated glomerular filtration rate levels',
    'Labanalysis_EOSINOFILOCYTTERB': 'Eosinophils levels',
    'Labanalysis_ERYTROCYTTERB': 'Erythrocytes levels',
    'Labanalysis_ERYTROCYTTERVOLFRB': 'Erythrocyte volume levels',
    'Labanalysis_ERYTROCYTVOLMIDDELMCVERCB': 'Mean corpuscular volume (MCV) levels',
    'Labanalysis_GLUKOSEP': 'Glucose levels',
    'Labanalysis_HMOGLOBINB': 'Hemoglobin levels',
    'Labanalysis_HMOGLOBININDHOLDMCHERCB': 'Hemoglobin content levels',
    'Labanalysis_HMOGLOBINMCHCERCB': 'Mean corpuscular hemoglobin concentration (MCHC) levels',
    'Labanalysis_KOAGULATIONSFAKTORIIVIIXINRP': 'Coagulation factor VII levels',
    'Labanalysis_KREATININP': 'Creatinine levels',
    'Labanalysis_LAKTATDEHYDROGENASEP': 'Lactate dehydrogenase levels',
    'Labanalysis_LEUKOCYTTERB': 'Leukocytes levels',
    'Labanalysis_LYMFOCYTTERB': 'Lymphocytes levels',
    'Labanalysis_METAMYELOMYELOPROMYELOCYTTERB': 'Metamyelocytes/Promyelocytes levels',
    'Labanalysis_NATRIUMP': 'Sodium levels',
    'Labanalysis_NEUTROFILOCYTTERB': 'Neutrophils levels',
    'Labanalysis_TROMBOCYTTERB': 'Platelets levels',
    'NumberOfTotalHospitalizations': 'Total number of hospital admissions',
    'TotalHospitalizationDays': 'Total days hospitalized',
    'Relative_with_Alkoholmisbrug': 'Relative with alcohol abuse',
    'Relative_with_Angstsygdom': 'Relative with anxiety disorder',
    'Relative_with_Cancer': 'Relative with cancer',
    'Relative_with_Depression': 'Relative with depression',
    'Relative_with_Diabetesmellitus': 'Relative with diabetes mellitus',
    'Relative_with_Hjertesygdom': 'Relative with heart disease',
    'Sex': 'Sex (Female)',
    'RateOfBMIChange_classification': 'Rate of BMI change (positive/negative)',
    'IsCurrentlyHospitalized': 'Currently hospitalized',
    'antipsychotic_medications': 'Antipsychotic medications',
    'olanzapine_medication': 'Olanzapine medication',
    'schizophrenia_diagnosis': 'Schizophrenia diagnosis',
    'cancer_diagnosis': 'Cancer diagnosis',
    'StrengthNumeric': 'Olanzapine dosage',
    'DailyDosage_weighted_mean_nafilled': 'Olanzapine dosage',
    'MedicationDuration': 'Duration of olanzapine medication',
    'NumberOfUniqueMedicationsPerscribed_BeforeNow': 'Number of different medications prescribed',
    'NumberOfTimesOnAntipsychotics': 'Times on antipsychotics prescription',
    'NumberOfTimesOnAntidepressants': 'Times on antidepressants prescription',
    'NumberOfTimesOnAnxiolytics': 'Times on anxiolytics prescription',
    'TimesCurrentATCTaken_BeforeNow': 'Times olanzapine was perscribed before',
    'NumberOfOverlappingATC': 'Number of currently overlapping medications',
    'NumberOfDaysOnMedication': 'Days on olanzapine medication until next visit',
    'NumberOfDaysOffMedication': 'Days off olanzapine medication in next visit',
    'bmi_diff_percent': 'Percent BMI change in next visit',
    'BMI_distance_from_StartInstant': 'BMI distance from olanzapine medication start',
    'target': 'BMI increase >5% in the next visit',
    'ATC_N05C': 'ATC_N05C (Sedatives/Hypnotics)',
    'ATC_N06A': 'ATC_N06A (Antidepressants)',
    'ATC_A02A': 'ATC_A02A (Antacids)',
    'SKS_DZ71': 'SKS_DZ71 (Healthcare System Contact)',
    'StillOnOlanzapine': 'Olanzapine status in the next BMI assessment'
}

# ── performance metrics for final reporting ────────────────────────────────
KEEP_METRICS = {
    "Test ROC AUC"                    : "AUROC",
    "Test F1 Score (positive)"        : "F1 (positive)",
    "Test F1 Score (weighted)"        : "F1 (weighted)",
    "Test F1 Score (macro)"           : "F1 (macro)",
    "Test Accuracy"                   : "Accuracy",
    "Test Sensitivity (Recall)"       : "Sensitivity",
    "Test Specificity"                : "Specificity",
    "Test PPV (Precision)"            : "PPV",
    "Test NPV"                        : "NPV",
    "Test Matthews Correlation Coefficient" : "MCC",
}

# --- LOGGING ---
def setup_global_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
setup_global_logging()

# --- UTILS ---
def pretty(name: str) -> str:
    """Return the human-readable label if it exists, else the original name."""
    return column_mapping.get(name, name)

def has_two_classes(arr):
    uniq = np.unique(arr)
    return len(uniq) == 2

def find_best_f1_threshold(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-8)
    idx = np.argmax(f1)
    return t[idx], f1[idx]

def sens(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan

def spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

def ppv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fp) if (tp + fp) > 0 else np.nan

def npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else np.nan

def create_output_directory_from_table(table_name: str, base_out_dir: str, split_id: int) -> str:
    out = os.path.join(base_out_dir, table_name, f"split_{split_id}")
    for sub in ("models", "plots", "results"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out

def save_plot(fig, pdf: PdfPages, plots_dir: str, fname: str):
    pdf.savefig(fig)
    fig.savefig(os.path.join(plots_dir, f"{fname}.png"))
    plt.close(fig)

def detect_feature_types(df, target_col):
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


# Get all possible categories per categorical feature, per table
def get_global_categorical_levels(df, target_col):
    _, _, categorical = detect_feature_types(df, target_col)
    categorical_levels = {col: df[col].dropna().unique() for col in categorical}
    return categorical, categorical_levels


def preprocess_data(train_df, test_df, categorical_levels=None, categorical_columns=None):
    target = config['target_col']
    numeric, binary, detected_categorical = detect_feature_types(train_df, target)
    # now always use the global categorical_columns if passed
    categorical = categorical_columns if categorical_columns is not None else detected_categorical
    td_cols = [c for c in train_df.select_dtypes(['timedelta64']).columns if c != target]
    for col in td_cols:
        for df in (train_df, test_df):
            df[col] = df[col].dt.days
        numeric = list(set(numeric) | set(td_cols))
    if categorical_levels is not None and categorical:
        cat_ohe = OneHotEncoder(
            handle_unknown='ignore', sparse_output=False,
            categories=[categorical_levels[c] for c in categorical]
        )
    else:
        cat_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    pre = ColumnTransformer([
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
    X_train = pre.fit_transform(train_df.drop(columns=[target], errors='ignore'))
    X_test = pre.transform(test_df.drop(columns=[target], errors='ignore'))
    feat_names = []
    if numeric: feat_names += numeric
    if binary: feat_names += binary
    if categorical:
        cats = pre.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical)
        feat_names += list(cats)
    pretty_feat_names = [pretty(f) for f in feat_names]
    return X_train, X_test, feat_names, pretty_feat_names, pre

#-----UTILS - Helpers to create summary statistics------------------------------

def _is_binary(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    s = series.dropna().unique()
    return set(s).issubset({0, 1})

def _pseudo_stats(arr: np.ndarray) -> tuple[float, float, float]:
    n = len(arr)
    if n < 5:
        return (np.nan, np.nan, np.nan)
    arr = np.sort(arr)
    pmin   = arr[:5].mean()
    centre = arr[max(0, n // 2 - 2) : max(0, n // 2 - 2) + 5].mean()
    pmax   = arr[-5:].mean()
    return (pmin, centre, pmax)

def summarise_column(col: pd.Series) -> dict:
    total_n   = len(col)
    miss_n    = col.isna().sum()
    out = {"N": total_n, "Missing_%": round(miss_n / total_n * 100, 2)}

    if total_n == miss_n:          # everything missing
        return out

    if _is_binary(col):
        t = ((col == 1) | (col == True)).sum()
        f = ((col == 0) | (col == False)).sum()
        out.update(
            True_count        = int(t),
            False_count       = int(f),
            True_percentage   = round(t / (total_n - miss_n) * 100, 2),
            False_percentage  = round(f / (total_n - miss_n) * 100, 2),
        )
    else:
        num = pd.to_numeric(col, errors="coerce").dropna().to_numpy()
        pmin, pmed, pmax = _pseudo_stats(num)
        out.update(
            Min    = round(pmin, 2),
            Max    = round(pmax, 2),
            Mean   = round(num.mean(), 2),
            Median = round(pmed, 2),
        )
    return out

def save_feature_summary(df: pd.DataFrame, csv_path: str):
    rows = []
    for col in df.columns:
        stats = summarise_column(df[col])
        stats["Variable"] = pretty(col)
        rows.append(stats)

    summary = pd.DataFrame(rows)[[
        "Variable", "N", "Missing_%",
        "True_count", "False_count", "True_percentage", "False_percentage",
        "Min", "Max", "Mean", "Median"
    ]]
    summary.to_csv(csv_path, index=False)
    logging.info(f"Feature summary saved → {csv_path}")


# --- Compact metrics helper ------------------------------------------

# ── compact-metrics helper ─────────────────────────────────────────────────
def _collapse_ci(df: pd.DataFrame, base: str) -> pd.Series:
    """Return strings like '0.734 (0.700–0.765)' for one metric."""
    return (
        df[f"{base} Mean"].round(3).astype(str)
        + " (" + df[f"{base} 95% CI Lower"].round(3).astype(str)
        + "–"  + df[f"{base} 95% CI Upper"].round(3).astype(str) + ")"
    )

def save_compact_metrics(wide_csv_path: str):
    """
    Read the *_summary_metrics.csv* produced for a table
    and write *_summary_metrics_compact.csv* next to it.
    """
    wide = pd.read_csv(wide_csv_path)

    # build the compact dataframe ------------------------------------------------
    compact_cols = ["Table", "Model"]
    for base, pretty in KEEP_METRICS.items():
        wide[pretty] = _collapse_ci(wide, base)
        compact_cols.append(pretty)

    compact = wide[compact_cols].copy()

    # (optional) highlight best AUROC per table -----------------------------------
    for tbl, grp in compact.groupby("Table"):
        idx = grp["AUROC"].str.extract(r'^([\d.]+)')[0].astype(float).idxmax()
        compact.loc[idx, "AUROC"] = "**" + compact.loc[idx, "AUROC"] + "**"

    # write -----------------------------------------------------------------------
    out_file = wide_csv_path.replace("_summary_metrics.csv",
                                     "_summary_metrics_compact.csv")
    compact.to_csv(out_file, index=False)
    logging.info(f"Compact metrics saved → {out_file}")


# --- SUBGROUP ROC FOR ONE SPLIT --------------------------------------
def compute_subgroup_roc_auc(
        model,
        preprocessor,
        test_df: pd.DataFrame,
        positive_label,
        subgroup_col: str,
        *,
        bins=None,
        labels=None,
        plots_dir: str,
        pdf: PdfPages | None,
        split_id: int,
        model_name: str
):
    """
    Draw & save a ROC curve for every level of `subgroup_col`
    (e.g. Sex or age bins) **for the current split / model**.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    df = test_df.copy()
    if bins is not None and labels is not None:
        df['__subgroup__'] = pd.cut(df[subgroup_col], bins=bins, labels=labels)
    else:
        df['__subgroup__'] = df[subgroup_col]

    # drop rows without subgroup information
    df = df.dropna(subset=['__subgroup__'])
    if df.empty:
        return

    for grp, grp_df in df.groupby('__subgroup__'):
        # need at least 2 classes present
        if len(grp_df) < 3 or grp_df[config['target_col']].nunique() < 2:
            continue

        Xg = preprocessor.transform(grp_df.drop(columns=[config['target_col']]))
        yg = (grp_df[config['target_col']] == positive_label).astype(int).values
        prob = model.predict_proba(Xg)[:, 1]
        auc_val = roc_auc_score(yg, prob)
        fpr, tpr, _ = roc_curve(yg, prob)

        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC={auc_val:.2f}")
        plt.plot([0, 1], [0, 1], "--", linewidth=1)
        plt.title(f"ROC – {subgroup_col}: {grp}\nmodel={model_name} | split={split_id}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()

        fname = f"roc_{model_name}_split{split_id}_{subgroup_col}_{grp}.png"
        plt.savefig(os.path.join(plots_dir, fname))
        if pdf is not None:
            pdf.savefig()
        plt.close()

def plot_roc_auc_all_subgroups(
    model,
    preprocessor,
    test_df: pd.DataFrame,
    positive_label,
    subgroup_col: str,
    group_labels: list,
    plots_dir: str,
    pdf: PdfPages | None,
    split_id: int,
    model_name: str,
    bins=None,
    labels=None,
):
    """
    Plot ROC curves for all subgroups (e.g., all age groups or all sexes) in a single plot for the current split/model.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    df = test_df.copy()
    if bins is not None and labels is not None:
        df['__subgroup__'] = pd.cut(df[subgroup_col], bins=bins, labels=labels)
    else:
        df['__subgroup__'] = df[subgroup_col]

    # drop rows without subgroup information
    df = df.dropna(subset=['__subgroup__'])
    if df.empty:
        return

    plt.figure(figsize=(6, 6))
    for grp in group_labels:
        grp_df = df[df['__subgroup__'] == grp]
        # need at least 2 classes present
        if len(grp_df) < 3 or grp_df[config['target_col']].nunique() < 2:
            continue
        Xg = preprocessor.transform(grp_df.drop(columns=[config['target_col']]))
        yg = (grp_df[config['target_col']] == positive_label).astype(int).values
        prob = model.predict_proba(Xg)[:, 1]
        auc_val = roc_auc_score(yg, prob)
        fpr, tpr, _ = roc_curve(yg, prob)
        plt.plot(fpr, tpr, label=f"{grp} (AUC={auc_val:.2f})")

    plt.plot([0, 1], [0, 1], "--", linewidth=1, color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC by {subgroup_col} – {model_name} | split={split_id}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fname = f"roc_{model_name}_split{split_id}_all_{subgroup_col}.png"
    plt.savefig(os.path.join(plots_dir, fname))
    if pdf is not None:
        pdf.savefig()
    plt.close()


# --- MEAN ROC PER SUBGROUP (ACROSS SPLITS) ---------------------------
def plot_mean_roc_per_subgroup(
        preds: list[dict],
        test_dfs: list[pd.DataFrame],
        *,
        subgroup_col: str,
        group_labels: list,
        out_path: str,
        title: str | None = None
):
    """
    Aggregate ROC curves over all splits **for each subgroup level** and
    draw the mean curve with a 95 % band.
    """
    from sklearn.metrics import roc_curve, auc

    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(8, 8))

    for label in group_labels:
        tprs, aucs = [], []
        for i, entry in enumerate(preds):
            y_true_full = np.asarray(entry['y_test'])
            y_prob_full = np.asarray(entry['y_test_prob'])
            df_split    = test_dfs[i]

            if subgroup_col == 'healthAssesment_age':
                # convert numeric age to bins
                bins = [0, 30, 50, 70, 120]
                df_split = df_split.copy()
                df_split['age_group'] = pd.cut(df_split['healthAssesment_age'],
                                               bins=bins,
                                               labels=["18-29", "30-49", "50-69", "70+"])
                mask = df_split['age_group'] == label
            else:
                mask = df_split[subgroup_col] == label

            if mask.sum() < 3 or len(np.unique(y_true_full[mask])) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true_full[mask], y_prob_full[mask])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        if not tprs:
            continue

        mean_tpr = np.mean(tprs, axis=0)
        lower    = np.percentile(tprs, 2.5, axis=0)
        upper    = np.percentile(tprs, 97.5, axis=0)
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr,
                 label=f"{label} (AUC={mean_auc:.2f}±{std_auc:.2f})")
        plt.fill_between(mean_fpr, lower, upper, alpha=0.20)

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title or f"Mean ROC per {subgroup_col} Across Splits")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_mean_roc_across_models(
        preds_by_model: dict[str, list[dict]],
        out_path: str,
        title: str = "Mean ROC per Model Across Splits"):
    """
    preds_by_model :  {"ModelName": [ {y_test, y_test_prob}, … ], … }
                     Each list element must come from a **different split**.
    Produces one curve (mean ± 95 % band) per model in the same figure.
    """
    from sklearn.metrics import roc_curve, auc

    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(8, 8))

    for model_name, entries in preds_by_model.items():
        if not entries:
            continue

        tprs, aucs = [], []
        for entry in entries:
            fpr, tpr, _ = roc_curve(entry["y_test"], entry["y_test_prob"])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        lower    = np.percentile(tprs, 2.5, axis=0)
        upper    = np.percentile(tprs, 97.5, axis=0)
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr,
                 label=f"{model_name} (AUC={mean_auc:.2f}±{std_auc:.2f})")
        plt.fill_between(mean_fpr, lower, upper, alpha=0.15)

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_auc_all_models(
    fitted_models: dict,
    preprocessor,
    test_df: pd.DataFrame,
    positive_label,
    feats,
    class_names,
    split_id: int,
    plots_dir: str,
    pdf: PdfPages | None = None
):
    """
    Plot ROC curves for all models on the same split/test set, in one figure.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    y_test = (test_df[config['target_col']] == positive_label).astype(int).values
    X_test = preprocessor.transform(test_df.drop(columns=[config['target_col']], errors='ignore'))

    plt.figure(figsize=(6, 6))
    for model_name, model in fitted_models.items():
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_val = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.2f})")
    plt.plot([0, 1], [0, 1], "--", linewidth=1, color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – All Models | split={split_id}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fname = f"roc_all_models_split{split_id}.png"
    plt.savefig(os.path.join(plots_dir, fname))
    if pdf is not None:
        pdf.savefig()
    plt.close()

def plot_shap_importance_across_splits(shap_arrays, feature_names_per_split, features_per_split, out_path, model_name, tbl_out, tbl, max_display=15):
    # get all unique features across splits
    all_feature_names = []
    
    for feats in feature_names_per_split:
        for f in feats:
            if f not in all_feature_names:
                all_feature_names.append(f)
    pretty_feature_names = [pretty(f) for f in all_feature_names]
    
    # align SHAP arrays and features
    
    def align(arr, names):
        arr_aligned = np.zeros((arr.shape[0], len(all_feature_names)))
        for i, f in enumerate(names):
            idx = all_feature_names.index(f)
            arr_aligned[:, idx] = arr[:, i]
        return arr_aligned
    
    aligned_shap = [align(arr, names) for arr, names in zip(shap_arrays, feature_names_per_split)]
    aligned_features = [align(arr, names) for arr, names in zip(features_per_split, feature_names_per_split)]
    all_shap = np.concatenate(aligned_shap, axis=0)
    all_features = np.concatenate(aligned_features, axis=0)

    # saving the summarized shap values
    df_shap = pd.DataFrame(all_shap, columns=all_feature_names)
    df_shap.to_csv(os.path.join(tbl_out, f"{tbl}_shap_values_{model_name}_all_splits.csv"), index=False)
    summary = pd.DataFrame({
        "feature": all_feature_names,
        "mean_abs_shap": np.abs(all_shap).mean(axis=0),
        "std_abs_shap": np.abs(all_shap).std(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    summary.to_csv(os.path.join(tbl_out, f"{tbl}_shap_summary_{model_name}_all_splits.csv"), index=False)

    
    # plot the summarized shap plot
    import shap
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, max_display * 0.4 + 1))
    shap.summary_plot(
        all_shap,
        all_features,
        feature_names=all_feature_names,
        plot_type="layered_violin",
        max_display=max_display,
        show=False
    )
    plt.title(f"Mean |SHAP| across splits: {model_name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()





# --- STRATIFIED TRAIN/TEST SPLIT WITH GROUPING ---
def stratified_train_test_split(data: pd.DataFrame, target_col: str, test_size: float, random_state: int):
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
    ytr, yte = train[target_col], test[target_col]
    if not (has_two_classes(ytr) and has_two_classes(yte)):
        raise RuntimeError("Both train and test must contain both classes.")
    return train, test



# --- EVALUATE AND PLOT FOR EACH SPLIT ---
def evaluate_models_and_plots(
    models, param_grids,
    X_train, y_train, X_test, y_test,
    class_names, plots_dir, pdf, feats,
    pretty_feats,
    split_id, results_dir, logger
):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        precision_recall_curve, roc_curve,
        confusion_matrix, matthews_corrcoef
    )
    from sklearn.calibration import calibration_curve
    
    TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier,
                   DecisionTreeClassifier, xgb.XGBClassifier)

    metrics = []
    shap_imp_per_model, shap_vals_per_model = {}, {}
    fitted_models = {}

    for model_name, base_model in models.items():
        logger.info(f"[Split {split_id}] [{model_name}] hyper-parameter tuning…")
        tuner = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[model_name],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            refit=True
        )
        tuner.fit(X_train, y_train)
        best_model = tuner.best_estimator_
        fitted_models[model_name] = best_model

        y_train_prob = best_model.predict_proba(X_train)[:, 1]
        best_thr, _ = find_best_f1_threshold(y_train, y_train_prob)
        y_test_prob = best_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob > best_thr).astype(int)

        try:
            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        except ValueError:
            tn = fp = fn = tp = np.nan  # handle weird single-class edge case

        metrics.append({
            'Model': model_name,
            'Test Accuracy': accuracy_score(y_test, y_test_pred),
            'Test F1 Score (positive)': f1_score(y_test, y_test_pred, average='binary', pos_label=1),
            'Test F1 Score (weighted)': f1_score(y_test, y_test_pred, average='weighted'),
            'Test F1 Score (macro)': f1_score(y_test, y_test_pred, average='macro'),
            'Test ROC AUC': roc_auc_score(y_test, y_test_prob),
            'Test Sensitivity (Recall)': sens(y_test, y_test_pred),
            'Test Specificity': spec(y_test, y_test_pred),
            'Test PPV (Precision)': ppv(y_test, y_test_pred),
            'Test NPV': npv(y_test, y_test_pred),
            'Test Matthews Correlation Coefficient': matthews_corrcoef(y_test, y_test_pred),
            'Optimal Threshold (F1)': best_thr
        })

        # confusion matrix plot
        cm = pd.DataFrame([[tn, fp], [fn, tp]], index=class_names, columns=class_names)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set(title=f"Confusion Matrix - {model_name}", xlabel="Predicted", ylabel="Actual")
        save_plot(fig, pdf, plots_dir, f"confusion_matrix_{model_name}_split{split_id}")

        # ROC plot
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_test_prob):.2f}")
        ax.plot([0, 1], [0, 1], '--', linewidth=1)
        ax.set(title=f"ROC Curve - {model_name}", xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.legend(loc='lower right')
        save_plot(fig, pdf, plots_dir, f"roc_curve_{model_name}_split{split_id}")

        # Precision-recall plot
        prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(rec, prec)
        ax.set(title=f"Precision-Recall Curve - {model_name}", xlabel="Recall", ylabel="Precision")
        save_plot(fig, pdf, plots_dir, f"precision_recall_curve_{model_name}_split{split_id}")

        # Calibration plot
        prob_true, prob_pred = calibration_curve(y_test, y_test_prob, n_bins=10)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1)
        ax.plot([0, 1], [0, 1], '--', linewidth=1)
        ax.set(title=f"Calibration Curve - {model_name}", xlabel="Mean predicted value", ylabel="Fraction of positives")
        save_plot(fig, pdf, plots_dir, f"calibration_curve_{model_name}_split{split_id}")

        # SHAP importance
        logger.info(f"[Split {split_id}] [{model_name}] computing SHAP…")
        shap_vals, imp_df = compute_shap_and_importance(
            best_model, X_train, X_test, feats, pretty_feats,
            logger=logger, max_display=15, results_dir=results_dir,
            plots_dir=plots_dir, pdf=pdf, split_id=split_id, model_name=model_name
        )
        shap_imp_per_model[model_name] = imp_df
        shap_vals_per_model[model_name] = shap_vals

    return (pd.DataFrame(metrics),
            shap_imp_per_model,
            shap_vals_per_model,
            fitted_models)


def compute_shap_and_importance(
        model, X_train, X_test, feature_names,
        pretty_names,
        logger=None, max_display=15,
        results_dir=None, plots_dir=None, pdf=None,
        split_id=None, model_name=None):
    """
    compute SHAP values and mean |SHAP| importances for a fitted model.
    robust to zero features (skips if X_test or feature_names empty).
    """

    import shap, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    import xgboost as xgb
    import inspect

    # --- hard fail if zero features to avoid all downstream issues ---
    if X_test.shape[1] == 0 or len(feature_names) == 0:
        if logger is not None:
            logger.warning(f"SHAP: No features for {model_name} split {split_id}; skipping SHAP computation and plot.")
        # return empty arrays/dataframe as expected shape
        return np.zeros((X_test.shape[0], 0)), pd.DataFrame(columns=["feature", "mean_abs_shap"])

    TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, xgb.XGBClassifier)
    est = model.named_steps['classifier'] if isinstance(model, Pipeline) else model

    background = shap.utils.sample(X_train, min(100, X_train.shape[0]), random_state=1993)

    # choose explainer and get raw SHAP values
    if isinstance(est, TREE_MODELS):
        if "check_additivity" in inspect.signature(shap.TreeExplainer).parameters:
            explainer = shap.TreeExplainer(est, data=background, model_output="probability", check_additivity=False)
            raw_shap = explainer.shap_values(X_test)
        else:
            explainer = shap.TreeExplainer(est, data=background, model_output="probability")
            raw_shap = explainer.shap_values(X_test, check_additivity=False)
    elif hasattr(est, "coef_"):
        explainer = shap.LinearExplainer(est, background)
        raw_shap = explainer.shap_values(X_test)
    else:
        raise RuntimeError(f"SHAP not supported for {type(est)}")

    # convert to (n_samples, n_features)
    def _to_2d(vals):
        if isinstance(vals, list):
            return vals[1] if len(vals) > 1 else vals[0]
        arr = np.asarray(vals)
        if arr.ndim == 3:
            return arr[:, :, 1] if arr.shape[-1] >= 2 else arr[0]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Unexpected SHAP shape {arr.shape}")

    shap_2d = _to_2d(raw_shap)

    # --- check again here for edge cases from SHAP API ---
    if shap_2d.shape[1] == 0:
        if logger is not None:
            logger.warning(f"SHAP: Computed zero SHAP features for {model_name} split {split_id}; skipping plot.")
        return shap_2d, pd.DataFrame(columns=["feature", "mean_abs_shap"])
    if shap_2d.shape[1] != len(feature_names):
        raise ValueError(f"{shap_2d.shape[1]} SHAP columns vs {len(feature_names)} features")

    imp_df = pd.DataFrame({
        "feature": pretty_names,
        "mean_abs_shap": np.abs(shap_2d).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    if results_dir and model_name is not None and split_id is not None:
        pd.DataFrame(shap_2d, columns=feature_names).to_csv(
            os.path.join(results_dir, f"shap_values_{model_name}_split{split_id}.csv"), index=False)
        imp_df.to_csv(
            os.path.join(results_dir, f"shap_importance_{model_name}_split{split_id}.csv"), index=False)

    # --- safe plotting ---
    if plots_dir and model_name is not None and split_id is not None:
        try:
            if X_test.shape[1] > 0 and len(feature_names) > 0 and shap_2d.shape[1] > 0:
                shap.summary_plot(
                    shap_2d,
                    features=X_test,
                    feature_names=pretty_names,
                    max_display=max_display,
                    show=False,
                    plot_type="layered_violin"
                )
                plt.tight_layout()
                fig = plt.gcf()
                fig.savefig(os.path.join(
                    plots_dir, f"shap_summary_{model_name}_split{split_id}.png"), bbox_inches="tight")
                if pdf is not None:
                    pdf.savefig(fig)
                plt.close(fig)
            else:
                if logger is not None:
                    logger.warning(f"SHAP: No features to plot for {model_name} split {split_id}")
        except Exception as e:
            if logger is not None:
                logger.error(f"SHAP plotting failed for {model_name} split {split_id}: {e}")

    if logger is not None:
        logger.info(f"Computed SHAP for {model_name} split {split_id}")

    return shap_2d, imp_df





# --- MEAN ROC ACROSS SPLITS ---
def plot_mean_roc_across_splits(model_results, out_path, label="Mean ROC Curve (All Splits)"):
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    from sklearn.metrics import auc, roc_curve
    for entry in model_results:
        y_test, y_test_prob = entry['y_test'], entry['y_test_prob']
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    tpr_upper = np.percentile(tprs, 97.5, axis=0)
    tpr_lower = np.percentile(tprs, 2.5, axis=0)
    plt.figure(figsize=(8, 8))
    plt.plot(mean_fpr, mean_tpr, color='b', label=f"{label} (AUC={mean_auc:.2f}±{std_auc:.2f})")
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='b', alpha=0.2, label="95% CI")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curve Across Splits")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def align_shap_arrays(shap_list, all_feature_names, split_feat_names):
    """
    # shap_list: list of [n_samples, n_split_features] arrays (one per split)
    # all_feature_names: list of all features seen across splits (union)
    # split_feat_names: list of feature name lists for each split
    # Returns: list of [n_samples, n_total_features] arrays, with missing features filled as 0
    """
    aligned = []
    for shap_arr, feat_names in zip(shap_list, split_feat_names):
        arr_aligned = np.zeros((shap_arr.shape[0], len(all_feature_names)))
        for i, feat in enumerate(feat_names):
            j = all_feature_names.index(feat)
            arr_aligned[:, j] = shap_arr[:, i]
        aligned.append(arr_aligned)
    return aligned


# --- MAIN PIPELINE FOR ONE SPLIT -------------------------------------
def main(
    df: pd.DataFrame,
    table_name: str,
    out_dir: str,
    split_id: int,
    categorical_levels=None,
    categorical_columns=None
):
    out = create_output_directory_from_table(table_name, out_dir, split_id)
    results_dir = os.path.join(out, "results")
    plots_dir   = os.path.join(out, "plots")
    models_dir  = os.path.join(out, "models")

    logfile = os.path.join(results_dir, f"bmipred_split{split_id}.log")
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    try:
        # ---------- split data ----------
        train_df, test_df = stratified_train_test_split(
            df, config['target_col'],
            test_size=config['test_size'],
            random_state=config['random_state'] + split_id
        )
        train_df = train_df.drop(columns=config['columns_to_drop'], errors='ignore')
        test_df  = test_df.drop(columns=config['columns_to_drop'],  errors='ignore')

        # ---------- label mapping ----------
        target_uniques = train_df[config['target_col']].unique()
        if set(target_uniques) in [{True, False}, {False, True}]:
            POS, NEG = True, False
        elif set(target_uniques) in [{"Yes", "No"}, {"No", "Yes"}]:
            POS, NEG = "Yes", "No"
        elif set(target_uniques) in [{1, 0}, {0, 1}]:
            POS, NEG = 1, 0
        else:
            raise ValueError(f"unknown target values: {target_uniques}")

        class_names = [str(NEG), str(POS)]
        y_train = (train_df[config['target_col']] == POS).astype(int).values
        y_test  = (test_df[config['target_col']]  == POS).astype(int).values
        X_train, X_test, feats, pretty_feats, pre = preprocess_data(train_df, test_df, categorical_levels, categorical_columns)

        pdf_path = os.path.join(plots_dir, "model_evaluation.pdf")
        with PdfPages(pdf_path) as pdf:
            metrics_df, shap_imp_per_model, shap_vals_per_model, fitted_models = evaluate_models_and_plots(
                config['models'], config['param_grids'],
                X_train, y_train, X_test, y_test,
                class_names, plots_dir, pdf, feats,
                pretty_feats, 
                split_id, results_dir, logger
            )

            # ---------- NEW: all age groups in one ROC plot per split/model ----------
            for m_name, mdl in fitted_models.items():
                plot_roc_auc_all_subgroups(
                    mdl, pre, test_df, POS,
                    subgroup_col='healthAssesment_age',
                    group_labels=["18-29", "30-49", "50-69", "70+"],
                    plots_dir=plots_dir, pdf=pdf,
                    split_id=split_id, model_name=m_name,
                    bins=[0, 30, 50, 70, 120],
                    labels=["18-29", "30-49", "50-69", "70+"]
                )
                sex_labels = sorted(test_df['Sex'].dropna().unique())
                if sex_labels:
                    plot_roc_auc_all_subgroups(
                        mdl, pre, test_df, POS,
                        subgroup_col='Sex',
                        group_labels=sex_labels,
                        plots_dir=plots_dir, pdf=pdf,
                        split_id=split_id, model_name=m_name
                    )

            # ---------- NEW: all models on one ROC plot for this split ----------
            plot_roc_auc_all_models(
                fitted_models, pre, test_df, POS, feats, class_names,
                split_id, plots_dir, pdf=pdf
            )

        # ---------- persist artefacts ----------
        for name, mdl in fitted_models.items():
            joblib.dump(mdl, os.path.join(models_dir, f"{name}.joblib"))
        joblib.dump(pre, os.path.join(models_dir, "preprocessor.joblib"))
        joblib.dump(class_names, os.path.join(models_dir, "class_names.joblib"))

        metrics_df.to_csv(os.path.join(results_dir, "performance_metrics.csv"), index=False)
        for name in shap_imp_per_model:
            np.save(os.path.join(results_dir, f"shap_imp_{name}.npy"),
                    shap_imp_per_model[name]['mean_abs_shap'].values)

        y_test_probs_dict = {n: fitted_models[n].predict_proba(X_test)[:, 1]
                             for n in fitted_models}

        return (
            metrics_df,
            {k: v['mean_abs_shap'].values for k, v in shap_imp_per_model.items()},
            y_test,
            y_test_probs_dict,
            test_df,
            shap_vals_per_model,
            feats,
            X_test
        )

    finally:
        logger.removeHandler(file_handler)
        file_handler.close()





# --- ENTRY POINT ---
if __name__ == "__main__":
    import os, sys, datetime

    # ---------- run-specific output root ----------
    base_out_dir = "/home/azureuser/cloudfiles/code/Users/foteini.aktypi/bmipred/models/"
    now_str      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name  = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    out_dir      = os.path.join(base_out_dir, f"results_{script_name}_{now_str}")
    os.makedirs(out_dir, exist_ok=True)

    n_repeats   = config["n_repeats"]
    age_groups  = ["18-29", "30-49", "50-69", "70+"]

    for tbl, path in table_paths.items():
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logging.error(f"failed reading {path}: {e}")
            continue
        
        # ---------- directory for THIS table ----------
        tbl_root = os.path.join(out_dir, tbl)          # <── the “global” dir for the table
        os.makedirs(tbl_root, exist_ok=True)

        # ---------- NEW: save descriptive statistics there ----------
        save_feature_summary(
            df,
            csv_path=os.path.join(tbl_root, f"{tbl}_feature_summary_statistics.csv")
        )

        print(f"\n========== TABLE: {tbl} ==========")

        # ---- get categorical levels for OHE consistency ----
        categorical_columns, categorical_levels = get_global_categorical_levels(df, config['target_col'])

        # ---- containers for this table only ----
        tbl_metrics   : list[pd.DataFrame] = []
        tbl_test_dfs  : list[pd.DataFrame] = []
        tbl_preds     = {m: [] for m in config["models"]}
        tbl_shap_vals = {m: [] for m in config["models"]}
        tbl_feat_names = None
        tbl_feat_names_per_split = []
        tbl_features_per_split = []

        for split_id in range(n_repeats):
            print(f"--- split {split_id+1}/{n_repeats} ---")
            res = main(df, tbl, out_dir, split_id, categorical_levels=categorical_levels, categorical_columns=categorical_columns)
            if res is None:
                continue

            metrics_df, shap_imp_dict, y_test, y_prob_dict, test_df_split, shap_vals_per_model, feats, X_test_split = res
            metrics_df["Split"] = split_id
            tbl_metrics.append(metrics_df)
            tbl_test_dfs.append(test_df_split)
            tbl_features_per_split.append(X_test_split)
            tbl_feat_names_per_split.append(feats)

            for m in config["models"]:
                tbl_preds[m].append({
                    "Split":       split_id,
                    "y_test":      y_test,
                    "y_test_prob": y_prob_dict[m]
                })
                tbl_shap_vals[m].append(shap_vals_per_model[m])  # collecting SHAP arrays per split

            if tbl_feat_names is None:
                tbl_feat_names = feats  # same for all splits

        if not tbl_metrics:
            print(f"No successful splits for {tbl}")
            continue

        tbl_out = os.path.join(out_dir, tbl)
        os.makedirs(tbl_out, exist_ok=True)

        tbl_metrics_df = pd.concat(tbl_metrics, ignore_index=True)
        tbl_metrics_df.to_csv(
            os.path.join(tbl_out, f"{tbl}_all_splits_metrics.csv"),
            index=False
        )

        metric_cols = [
            "Test ROC AUC", 'Test F1 Score (positive)', 
            "Test F1 Score (weighted)", "Test F1 Score (macro)", "Test Accuracy",
            "Test Sensitivity (Recall)", "Test Specificity",
            "Test PPV (Precision)", "Test NPV",
            "Test Matthews Correlation Coefficient",
        ]
        summary_rows = []
        for model, grp in tbl_metrics_df.groupby("Model"):
            row = {"Table": tbl, "Model": model}
            for m in metric_cols:
                vals = grp[m].values
                row[f"{m} Mean"]          = np.mean(vals)
                row[f"{m} 95% CI Lower"], \
                row[f"{m} 95% CI Upper"]  = np.percentile(vals, [2.5, 97.5])
            summary_rows.append(row)
        
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(tbl_out, f"{tbl}_summary_metrics.csv"),
            index=False
        )
        
        save_compact_metrics(os.path.join(tbl_out, f"{tbl}_summary_metrics.csv"))

        for model_name in config["models"]:

            shap_arrays = tbl_shap_vals[model_name]
            if not shap_arrays:
                continue
            out_path = os.path.join(tbl_out, f"{tbl}_shap_importance_{model_name}_across_splits.png")
            plot_shap_importance_across_splits(shap_arrays, tbl_feat_names_per_split, tbl_features_per_split, out_path, model_name, tbl_out, tbl, max_display=15)


            preds_model = tbl_preds[model_name]
            if not preds_model:
                continue

            plot_mean_roc_across_splits(
                preds_model,
                out_path=os.path.join(
                    tbl_out, f"{tbl}_mean_roc_overall_{model_name}.png"),
                label=f"{tbl} – {model_name}"
            )

            plot_mean_roc_per_subgroup(
                preds_model, tbl_test_dfs,
                subgroup_col="healthAssesment_age",
                group_labels=age_groups,
                out_path=os.path.join(
                    tbl_out, f"{tbl}_mean_roc_age_{model_name}.png"),
                title=f"{tbl} – Mean ROC per Age – {model_name}"
            )

            sex_labels = sorted({v
                                 for df in tbl_test_dfs if "Sex" in df
                                 for v in df["Sex"].dropna().unique()})
            if sex_labels:
                plot_mean_roc_per_subgroup(
                    preds_model, tbl_test_dfs,
                    subgroup_col="Sex",
                    group_labels=sex_labels,
                    out_path=os.path.join(
                        tbl_out, f"{tbl}_mean_roc_sex_{model_name}.png"),
                    title=f"{tbl} – Mean ROC per Sex – {model_name}"
                )

        plot_mean_roc_across_models(
            tbl_preds,
            out_path=os.path.join(tbl_out,
                     f"{tbl}_mean_roc_all_models.png"),
            title=f"{tbl} – Mean ROC Across Models"
        )

        print(f"✓ finished aggregation for {tbl}  →  outputs in {tbl_out}")




