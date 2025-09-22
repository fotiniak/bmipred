#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Optional, Any
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import os


class ModelPlotter:
    """Handles all plotting functionality for model evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize plotter with configuration."""
        self.config = config
        plt.rcParams.update(config['visualization']['plot_params'])
    
    def save_plot(self, fig, pdf: Optional[PdfPages], plots_dir: str, fname: str):
        """Save plot to both PDF and PNG."""
        if pdf is not None:
            pdf.savefig(fig)
        fig.savefig(os.path.join(plots_dir, f"{fname}.png"), bbox_inches='tight')
        plt.close(fig)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], model_name: str, 
                            split_id: int, plots_dir: str, pdf: Optional[PdfPages] = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
        self.save_plot(fig, pdf, plots_dir, f"confusion_matrix_{model_name}_split{split_id}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      model_name: str, split_id: int, plots_dir: str, 
                      pdf: Optional[PdfPages] = None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
        ax.plot([0, 1], [0, 1], '--', linewidth=1, color='gray')
        ax.set_title(f"ROC Curve - {model_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc='lower right')
        
        self.save_plot(fig, pdf, plots_dir, f"roc_curve_{model_name}_split{split_id}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   model_name: str, split_id: int, plots_dir: str,
                                   pdf: Optional[PdfPages] = None):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(recall, precision)
        ax.set_title(f"Precision-Recall Curve - {model_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        
        self.save_plot(fig, pdf, plots_dir, f"precision_recall_curve_{model_name}_split{split_id}")
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                              model_name: str, split_id: int, plots_dir: str,
                              pdf: Optional[PdfPages] = None):
        """Plot calibration curve."""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1)
        ax.plot([0, 1], [0, 1], '--', linewidth=1, color='gray')
        ax.set_title(f"Calibration Curve - {model_name}")
        ax.set_xlabel("Mean predicted value")
        ax.set_ylabel("Fraction of positives")
        
        self.save_plot(fig, pdf, plots_dir, f"calibration_curve_{model_name}_split{split_id}")
    

    def plot_shap_summary(self, shap_values: np.ndarray, X_test: np.ndarray,
                         feature_names: List[str], model_name: str, split_id: int, 
                         plots_dir: str, pdf: Optional[PdfPages] = None):  # Remove pretty_names parameter
        """Plot SHAP summary."""
        if shap_values.shape[1] == 0:
            return
        
        try:
            max_display = self.config['visualization']['max_display_features']
            
            plt.figure(figsize=(8, max_display * 0.4 + 1))
            shap.summary_plot(
                shap_values,
                features=X_test,
                feature_names=feature_names,  # Use feature_names directly
                max_display=max_display,
                show=False,
                plot_type="layered_violin"
            )
            plt.tight_layout()
            fig = plt.gcf()
            
            self.save_plot(fig, pdf, plots_dir, f"shap_summary_{model_name}_split{split_id}")
            
        except Exception as e:
            print(f"SHAP plotting failed for {model_name}: {e}")


    def plot_mean_roc_across_splits(self, predictions: List[Dict], out_path: str,
                                   label: str = "Mean ROC Curve"):
        """Plot mean ROC curve across multiple splits."""
        from sklearn.metrics import auc
        
        mean_fpr = np.linspace(0, 1, 100)
        tprs, aucs = [], []
        
        for pred in predictions:
            fpr, tpr, _ = roc_curve(pred['y_test'], pred['y_test_prob'])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
        
        mean_tpr = np.mean(tprs, axis=0)
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
    
    def plot_all_models_roc(self, models: Dict, preprocessor, test_df: pd.DataFrame,
                           target_col: str, positive_label, split_id: int,
                           plots_dir: str, pdf: Optional[PdfPages] = None):
        """Plot ROC curves for all models on same plot."""
        y_test = (test_df[target_col] == positive_label).astype(int).values
        X_test = preprocessor.transform(test_df.drop(columns=[target_col], errors='ignore'))
        
        plt.figure(figsize=(6, 6))
        
        for model_name, (model, _) in models.items():
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc_val = roc_auc_score(y_test, y_score)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.2f})")
        
        plt.plot([0, 1], [0, 1], '--', linewidth=1, color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC - All Models | split={split_id}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        self.save_plot(plt.gcf(), pdf, plots_dir, f"roc_all_models_split{split_id}")