#!/usr/bin/env python3
# bmipred/modeling/metrics.py

import numpy as np
from typing import Tuple
from sklearn.metrics import confusion_matrix, precision_recall_curve


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sensitivity (True Positive Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float('nan')


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity (True Negative Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan')


def ppv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Positive Predictive Value (Precision)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float('nan')


def npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative Predictive Value."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tn / (tn + fn)) if (tn + fn) > 0 else float('nan')


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])