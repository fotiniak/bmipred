#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Optional, Any, Sequence
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import os


class ModelPlotter:
    # Plotting functionality for model evaluation.
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize plotting with configuration.
        self.config = config
        plt.rcParams.update(config['visualization']['plot_params'])
    
    def save_plot(self, fig, pdf: Optional[PdfPages], plots_dir: str, fname: str):
        # Save plot to both PDF and PNG.
        if pdf is not None:
            pdf.savefig(fig)
        fig.savefig(os.path.join(plots_dir, f"{fname}.png"), bbox_inches='tight')
        plt.close(fig)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], model_name: str, 
                            split_id: int, plots_dir: str, pdf: Optional[PdfPages] = None):
        # Plot confusion matrix.
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
        # Plot ROC curve.
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
        # Plot precision-recall curve with AP score and prevalence baseline.
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        prevalence = np.mean(y_true)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(recall, precision, label=f"AP={ap:.2f}")
        ax.axhline(y=prevalence, linestyle="--", linewidth=1, color="gray", label="Prevalence")
        ax.set_title(f"Precision-Recall Curve - {model_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")

        self.save_plot(fig, pdf, plots_dir, f"precision_recall_curve_{model_name}_split{split_id}")
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                              model_name: str, split_id: int, plots_dir: str,
                              pdf: Optional[PdfPages] = None):
        # Plot calibration curve with adaptive n_bins.
        n_bins = min(10, max(2, len(np.unique(y_prob))))
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

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
        # Plot SHAP summary.
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


    def plot_shap_density_violin(
        self,
        shap_values: np.ndarray,
        feature_names: Sequence[str],
        feature_values: Optional[np.ndarray] = None,
        max_display: int = 15,
        title: Optional[str] = None,
        min_bin_count: int = 5,
    ):
        # Plot SHAP distributions with optional feature-value coloring, matching the reference pipeline.
        shap_values = np.asarray(shap_values, dtype=float)
        feature_names = list(feature_names)
        if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names):
            raise ValueError("SHAP values must be a 2D array matching feature_names.")
        if feature_values is not None:
            feature_values = np.asarray(feature_values, dtype=float)
            if feature_values.shape != shap_values.shape:
                raise ValueError("feature_values must have the same shape as shap_values.")

        mean_abs = np.nanmean(np.abs(shap_values), axis=0)
        order = np.argsort(np.nan_to_num(mean_abs, nan=-np.inf))[::-1][:max_display]

        def density_values(vals):
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return None
            if vals.size == 1 or np.nanmin(vals) == np.nanmax(vals):
                val = float(vals[0])
                eps = max(abs(val) * 1e-9, 1e-12)
                vals = np.array([val - eps, val, val + eps])
            return vals

        records = []
        for idx in order:
            vals = density_values(shap_values[:, idx])
            if vals is None:
                continue
            record = {"label": feature_names[idx], "shap": vals, "feature": None}
            if feature_values is None:
                records.append(record)
                continue

            fvals = feature_values[:, idx]
            mask = np.isfinite(shap_values[:, idx]) & np.isfinite(fvals)
            if mask.sum() >= min_bin_count * 2 and np.nanmin(fvals[mask]) != np.nanmax(fvals[mask]):
                record = {"label": feature_names[idx], "shap": shap_values[mask, idx], "feature": fvals[mask]}
            records.append(record)

        if not records:
            raise ValueError("No finite SHAP values to plot.")

        fig, ax = plt.subplots(figsize=(8, max(len(records) * 0.4 + 1, 3)))
        cmap = plt.get_cmap("coolwarm")
        used_gradient = False

        def draw_neutral(vals, position):
            parts = ax.violinplot(
                [vals],
                positions=[position],
                vert=False,
                widths=0.38,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor("#6A6A6A")
                body.set_edgecolor("#444444")
                body.set_alpha(0.75)

        def draw_gradient(vals, fvals, position):
            nonlocal used_gradient
            x_min, x_max = np.nanmin(vals), np.nanmax(vals)
            f_min, f_max = np.nanpercentile(fvals, [5, 95])
            if x_min == x_max or f_min == f_max:
                draw_neutral(density_values(vals), position)
                return

            n_bins = min(60, max(12, vals.size // min_bin_count))
            edges = np.linspace(x_min, x_max, n_bins + 1)
            counts, _ = np.histogram(vals, bins=edges)
            valid = counts >= min_bin_count
            if valid.sum() < 2:
                draw_neutral(density_values(vals), position)
                return

            smooth_counts = np.convolve(counts.astype(float), [0.25, 0.5, 0.25], mode="same")
            widths = smooth_counts / smooth_counts[valid].max() * 0.38
            for i in np.where(valid)[0]:
                left, right = edges[i], edges[i + 1]
                if i == len(counts) - 1:
                    in_bin = (vals >= left) & (vals <= right)
                else:
                    in_bin = (vals >= left) & (vals < right)
                mean_f = np.nanmean(fvals[in_bin])
                color_pos = np.clip((mean_f - f_min) / (f_max - f_min), 0, 1)
                ax.fill_between(
                    [left, right],
                    position - widths[i],
                    position + widths[i],
                    color=cmap(color_pos),
                    linewidth=0,
                    alpha=0.9,
                )
            used_gradient = True

        for pos, record in enumerate(records):
            if record["feature"] is None:
                draw_neutral(record["shap"], pos)
            else:
                draw_gradient(record["shap"], record["feature"], pos)

        ax.axvline(0, color="gray", linewidth=1, linestyle="--")
        ax.set_yticks(np.arange(len(records)))
        ax.set_yticklabels([record["label"] for record in records])
        ax.invert_yaxis()
        ax.set_xlabel("SHAP value")
        if used_gradient:
            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(facecolor=cmap(0.05), edgecolor="none", label="Low feature value"),
                    Patch(facecolor=cmap(0.95), edgecolor="none", label="High feature value"),
                ],
                loc="lower right",
                frameon=False,
            )
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig


    def plot_shap_importance_across_splits(
        self,
        shap_arrays: List[np.ndarray],
        feature_names_per_split: List[List[str]],
        feature_arrays_per_split: List[np.ndarray],
        column_mapping: Dict[str, str],
        out_path: str,
        model_name: str,
        tbl_out: str,
        tbl: str,
        max_display: int = 15,
    ):
        # Aggregate SHAP values across splits and plot a unified density view.
        all_feature_names: List[str] = []
        for feats in feature_names_per_split:
            for feature in feats:
                if feature not in all_feature_names:
                    all_feature_names.append(feature)

        feature_idx = {feature: i for i, feature in enumerate(all_feature_names)}

        def align(arr, names):
            arr_aligned = np.zeros((arr.shape[0], len(all_feature_names)))
            for i, feature in enumerate(names):
                arr_aligned[:, feature_idx[feature]] = arr[:, i]
            return arr_aligned

        aligned_shap = [align(arr, names) for arr, names in zip(shap_arrays, feature_names_per_split)]
        aligned_features = [align(arr, names) for arr, names in zip(feature_arrays_per_split, feature_names_per_split)]
        all_shap = np.concatenate(aligned_shap, axis=0)
        all_features = np.concatenate(aligned_features, axis=0)

        pd.DataFrame(all_shap, columns=all_feature_names).to_csv(
            os.path.join(tbl_out, f"{tbl}_shap_values_{model_name}_all_splits.csv"), index=False
        )
        pd.DataFrame(
            {
                "feature": all_feature_names,
                "mean_abs_shap": np.abs(all_shap).mean(axis=0),
                "std_abs_shap": np.abs(all_shap).std(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False).to_csv(
            os.path.join(tbl_out, f"{tbl}_shap_summary_{model_name}_all_splits.csv"), index=False
        )

        display_feature_names = [
            column_mapping.get(feature, feature) for feature in all_feature_names
        ]
        fig = self.plot_shap_density_violin(
            all_shap,
            display_feature_names,
            feature_values=all_features,
            max_display=max_display,
            title=f"SHAP density across splits: {model_name}",
        )
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


    def compute_subgroup_roc_auc(
        self,
        model,
        preprocessor,
        test_df: pd.DataFrame,
        target_col: str,
        positive_label,
        subgroup_col: str,
        *,
        bins=None,
        labels=None,
        plots_dir: str,
        pdf: Optional[PdfPages],
        split_id: int,
        model_name: str,
    ):
        # Plot a subgroup ROC curve for each subgroup level.
        if subgroup_col not in test_df.columns:
            return

        df = test_df.copy()
        if bins is not None and labels is not None:
            df["__subgroup__"] = pd.cut(df[subgroup_col], bins=bins, labels=labels)
        else:
            df["__subgroup__"] = df[subgroup_col]

        df = df.dropna(subset=["__subgroup__"])
        if df.empty:
            return

        for grp, grp_df in df.groupby("__subgroup__"):
            if len(grp_df) < 3 or grp_df[target_col].nunique() < 2:
                continue

            Xg = preprocessor.transform(grp_df.drop(columns=[target_col]))
            yg = (grp_df[target_col] == positive_label).astype(int).values
            prob = model.predict_proba(Xg)[:, 1]
            auc_val = roc_auc_score(yg, prob)
            fpr, tpr, _ = roc_curve(yg, prob)

            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=f"AUC={auc_val:.2f}")
            plt.plot([0, 1], [0, 1], "--", linewidth=1)
            plt.title(f"ROC - {subgroup_col}: {grp}\nmodel={model_name} | split={split_id}")
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
        self,
        model,
        preprocessor,
        test_df: pd.DataFrame,
        target_col: str,
        positive_label,
        subgroup_col: str,
        group_labels: List,
        plots_dir: str,
        pdf: Optional[PdfPages],
        split_id: int,
        model_name: str,
        bins=None,
        labels=None,
        right=True,
    ):
        # Plot all subgroup ROC curves in a single figure.
        if subgroup_col not in test_df.columns:
            return

        df = test_df.copy()
        if bins is not None and labels is not None:
            df["__subgroup__"] = pd.cut(df[subgroup_col], bins=bins, labels=labels, right=right)
        else:
            df["__subgroup__"] = df[subgroup_col]

        df = df.dropna(subset=["__subgroup__"])
        if df.empty:
            return

        plt.figure(figsize=(6, 6))
        plotted = False
        for grp in group_labels:
            grp_df = df[df["__subgroup__"] == grp]
            if len(grp_df) < 3 or grp_df[target_col].nunique() < 2:
                continue
            Xg = preprocessor.transform(grp_df.drop(columns=[target_col]))
            yg = (grp_df[target_col] == positive_label).astype(int).values
            prob = model.predict_proba(Xg)[:, 1]
            auc_val = roc_auc_score(yg, prob)
            fpr, tpr, _ = roc_curve(yg, prob)
            plt.plot(fpr, tpr, label=f"{grp} (AUC={auc_val:.2f})")
            plotted = True

        if not plotted:
            plt.close()
            return

        plt.plot([0, 1], [0, 1], "--", linewidth=1, color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC by {subgroup_col} - {model_name} | split={split_id}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        fname = f"roc_{model_name}_split{split_id}_all_{subgroup_col}.png"
        plt.savefig(os.path.join(plots_dir, fname))
        if pdf is not None:
            pdf.savefig()
        plt.close()


    def plot_mean_roc_per_subgroup(
        self,
        preds: List[dict],
        test_dfs: List[pd.DataFrame],
        *,
        subgroup_col: str,
        group_labels: List,
        out_path: str,
        title: Optional[str] = None,
    ):
        # Aggregate ROC curves across splits and plot a mean curve with a 95% band per subgroup.
        from sklearn.metrics import auc

        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(8, 8))
        plotted = False

        for label in group_labels:
            tprs, aucs = [], []
            for i, entry in enumerate(preds):
                y_true_full = np.asarray(entry['y_test'])
                y_prob_full = np.asarray(entry['y_test_prob'])
                df_split = test_dfs[i]

                if subgroup_col not in df_split.columns:
                    continue

                if subgroup_col == 'healthAssesment_age':
                    bins = [0, 30, 50, 70, 120]
                    df_split = df_split.copy()
                    df_split['age_group'] = pd.cut(df_split['healthAssesment_age'], bins=bins, labels=["18-29", "30-49", "50-69", "70+"])
                    mask = df_split['age_group'] == label
                elif subgroup_col == 'days_to_next_bmi':
                    bins = [-0.001, 30, 180, 360, np.inf]
                    df_split = df_split.copy()
                    df_split['days_to_next_bmi_group'] = pd.cut(df_split['days_to_next_bmi'], bins=bins, labels=["0-30", "30-180", "180-360", "360+"])
                    mask = df_split['days_to_next_bmi_group'] == label
                elif subgroup_col == 'BodyMassIndex_recalc':
                    bins = [-np.inf, 18.5, 25, 30, np.inf]
                    df_split = df_split.copy()
                    df_split['bmi_group'] = pd.cut(df_split['BodyMassIndex_recalc'], bins=bins, labels=["Underweight", "Normal", "Overweight", "Obese"], right=False)
                    mask = df_split['bmi_group'] == label
                else:
                    mask = df_split[subgroup_col] == label

                mask = np.asarray(mask, dtype=bool)
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
            lower = np.percentile(tprs, 2.5, axis=0)
            upper = np.percentile(tprs, 97.5, axis=0)
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

            plt.plot(mean_fpr, mean_tpr, label=f"{label} (AUC={mean_auc:.2f}±{std_auc:.2f})")
            plt.fill_between(mean_fpr, lower, upper, alpha=0.20)
            plotted = True

        if not plotted:
            plt.close()
            return

        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title or f"Mean ROC per {subgroup_col} Across Splits")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


    def plot_mean_roc_across_models(
        self,
        preds_by_model: Dict[str, List[dict]],
        out_path: str,
        title: str = "Mean ROC per Model Across Splits",
    ):
        # Plot one mean ROC curve per model across splits.
        from sklearn.metrics import auc

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
            lower = np.percentile(tprs, 2.5, axis=0)
            upper = np.percentile(tprs, 97.5, axis=0)
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

            plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC={mean_auc:.2f}±{std_auc:.2f})")
            plt.fill_between(mean_fpr, lower, upper, alpha=0.15)

        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


    def plot_mean_roc_across_splits(self, predictions: List[Dict], out_path: str,
                                   label: str = "Mean ROC Curve"):
        # Plot mean ROC curve across multiple splits.
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
        # Plot ROC curves for all models on same plot.
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