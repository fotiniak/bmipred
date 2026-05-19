#!/usr/bin/env python3
# bmipred/modeling/pipeline.py

import os
import datetime
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.utils.class_weight import compute_sample_weight

from bmipred.modeling.data_utils import (stratified_split, get_global_categorical_levels, create_output_directory)
from bmipred.modeling.preprocessing import preprocess_data
from bmipred.modeling.model_trainer import train_all_models
from bmipred.modeling.evaluator import evaluate_all_models
from bmipred.modeling.plots import ModelPlotter
from bmipred.modeling.reports import save_feature_summary, save_all_reports


class BMIPredPipeline:
    # Main pipeline for BMI prediction modeling experiments.
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize pipeline with configuration.
        self.config = config
        self.logger = self._setup_logging()
        self.plotter = ModelPlotter(config)
    
    def _setup_logging(self) -> logging.Logger:
        # Setup logging configuration.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger(__name__)
    
    def _determine_labels(self, train_df: pd.DataFrame) -> tuple:
        # Determine positive and negative labels and class names.
        target_col = self.config['data']['target_col']
        target_uniques = train_df[target_col].unique()
        
        if set(target_uniques) in [{True, False}, {False, True}]:
            pos_label, neg_label = True, False
        elif set(target_uniques) in [{"Yes", "No"}, {"No", "Yes"}]:
            pos_label, neg_label = "Yes", "No"
        elif set(target_uniques) in [{1, 0}, {0, 1}]:
            pos_label, neg_label = 1, 0
        else:
            raise ValueError(f"Unknown target values: {target_uniques}")
        
        class_names = [str(neg_label), str(pos_label)]
        return pos_label, neg_label, class_names
    
    def run_single_split(
        self,
        df: pd.DataFrame,
        table_name: str,
        split_id: int,
        output_dir: str,
        categorical_levels: Optional[Dict] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        # Run pipeline for a single train/test split.
        self.logger.info(f"Running split {split_id} for {table_name}")
        
        try:
            # Create output directories
            split_dir = create_output_directory(table_name, output_dir, split_id)
            results_dir = os.path.join(split_dir, "results")
            plots_dir = os.path.join(split_dir, "plots")
            models_dir = os.path.join(split_dir, "models")
            
            # Split data
            train_df, test_df = stratified_split(
                df,
                self.config['data']['target_col'],
                test_size=self.config['split']['test_size'],
                random_state=self.config['split']['random_state']
            )
            
            # Remove specified columns
            columns_to_drop = self.config['data']['columns_to_drop']
            train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
            test_df = test_df.drop(columns=columns_to_drop, errors='ignore')
            
            # Determine labels
            pos_label, neg_label, class_names = self._determine_labels(train_df)
            
            # Preprocess data
            X_train, X_test, feature_names, pretty_feature_names, preprocessor = preprocess_data(
                train_df, test_df,
                self.config['data']['target_col'],
                self.config.get('column_mapping'),
                categorical_levels,
                categorical_columns
            )
            
            # Prepare labels
            y_train = (train_df[self.config['data']['target_col']] == pos_label).astype(int).values
            y_test = (test_df[self.config['data']['target_col']] == pos_label).astype(int).values
            sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
            
            # Train models
            trained_models = train_all_models(
                self.config['models'],
                X_train, y_train,
                self.config['training'],
                sample_weight=sample_weight
            )
            
            # Evaluate models
            metrics_df, shap_values, shap_importance = evaluate_all_models(
                trained_models, X_train, y_train, X_test, y_test, feature_names,
                cv_folds=self.config['training']['cv_folds'],
                sample_weight=sample_weight,
                random_state=self.config['experiment']['random_state'] + split_id + 7000,
            )
            
            # Create plots
            pdf_path = os.path.join(plots_dir, "model_evaluation.pdf")
            with PdfPages(pdf_path) as pdf:
                self._create_all_plots(
                    trained_models, X_train, y_train, X_test, y_test,
                    shap_values, class_names, pretty_feature_names,
                    preprocessor, test_df, pos_label, split_id,
                    plots_dir, pdf
                )

                for name, (model, _) in trained_models.items():
                    self.plotter.plot_roc_auc_all_subgroups(
                        model, preprocessor, test_df,
                        self.config['data']['target_col'], pos_label,
                        subgroup_col='healthAssesment_age',
                        group_labels=['18-29', '30-49', '50-69', '70+'],
                        plots_dir=plots_dir, pdf=pdf,
                        split_id=split_id, model_name=name,
                        bins=[0, 30, 50, 70, 120],
                        labels=['18-29', '30-49', '50-69', '70+']
                    )
                    self.plotter.plot_roc_auc_all_subgroups(
                        model, preprocessor, test_df,
                        self.config['data']['target_col'], pos_label,
                        subgroup_col='days_to_next_bmi',
                        group_labels=['0-30', '30-180', '180-360', '360+'],
                        plots_dir=plots_dir, pdf=pdf,
                        split_id=split_id, model_name=name,
                        bins=[-0.001, 30, 180, 360, np.inf],
                        labels=['0-30', '30-180', '180-360', '360+']
                    )
                    self.plotter.plot_roc_auc_all_subgroups(
                        model, preprocessor, test_df,
                        self.config['data']['target_col'], pos_label,
                        subgroup_col='BodyMassIndex_recalc',
                        group_labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                        plots_dir=plots_dir, pdf=pdf,
                        split_id=split_id, model_name=name,
                        bins=[-np.inf, 18.5, 25, 30, np.inf],
                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                        right=False
                    )
                    sex_labels = sorted(test_df['Sex'].dropna().unique()) if 'Sex' in test_df else []
                    if sex_labels:
                        self.plotter.plot_roc_auc_all_subgroups(
                            model, preprocessor, test_df,
                            self.config['data']['target_col'], pos_label,
                            subgroup_col='Sex',
                            group_labels=sex_labels,
                            plots_dir=plots_dir, pdf=pdf,
                            split_id=split_id, model_name=name
                        )
            
            # Save artifacts
            self._save_artifacts(
                trained_models, preprocessor, class_names,
                metrics_df, shap_values, shap_importance,
                models_dir, results_dir, split_id
            )
            
            # Prepare predictions for aggregation
            predictions = {}
            for name, (model, _) in trained_models.items():
                y_prob = model.predict_proba(X_test)[:, 1]
                predictions[name] = {
                    'Split': split_id,
                    'y_test': y_test,
                    'y_test_prob': y_prob
                }
            
            return {
                'split_id': split_id,
                'metrics': metrics_df,
                'predictions': predictions,
                'test_df': test_df,
                'shap_values': shap_values,
                'feature_names': feature_names,
                'pretty_feature_names': pretty_feature_names,
                'X_test': X_test
            }
            
        except Exception as e:
            self.logger.error(f"Split {split_id} failed: {e}")
            return None
    
    def _create_all_plots(self, trained_models, X_train, y_train, X_test, y_test,
                         shap_values, class_names, feature_names,
                         preprocessor, test_df, pos_label, split_id,
                         plots_dir, pdf):
        # Create all plots for a single split.
        for name, (model, _) in trained_models.items():
            # Get predictions
            y_prob = model.predict_proba(X_test)[:, 1]
            from bmipred.modeling.metrics import find_best_f1_threshold
            threshold, _ = find_best_f1_threshold(y_train, model.predict_proba(X_train)[:, 1])
            y_pred = (y_prob > threshold).astype(int)
            
            # Create individual plots
            self.plotter.plot_confusion_matrix(y_test, y_pred, class_names, name, split_id, plots_dir, pdf)
            self.plotter.plot_roc_curve(y_test, y_prob, name, split_id, plots_dir, pdf)
            self.plotter.plot_precision_recall_curve(y_test, y_prob, name, split_id, plots_dir, pdf)
            self.plotter.plot_calibration_curve(y_test, y_prob, name, split_id, plots_dir, pdf)
            self.plotter.plot_shap_summary(shap_values[name], X_test, feature_names, name, split_id, plots_dir, pdf)
        
        # Plot all models together
        self.plotter.plot_all_models_roc(
            trained_models, preprocessor, test_df,
            self.config['data']['target_col'], pos_label,
            split_id, plots_dir, pdf
        )
    
    def _save_artifacts(self, trained_models, preprocessor, class_names,
                       metrics_df, shap_values, shap_importance,
                       models_dir, results_dir, split_id):
        # Save all artifacts for a split.
        # Save models
        for name, (model, _) in trained_models.items():
            joblib.dump(model, os.path.join(models_dir, f"{name}.joblib"))
        
        joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
        joblib.dump(class_names, os.path.join(models_dir, "class_names.joblib"))
        
        # Save metrics
        metrics_df.to_csv(os.path.join(results_dir, "performance_metrics.csv"), index=False)
        
        # Save SHAP values
        for name, shap_vals in shap_values.items():
            np.save(os.path.join(results_dir, f"shap_values_{name}_split{split_id}.npy"), shap_vals)
            if name in shap_importance:
                shap_importance[name].to_csv(
                    os.path.join(results_dir, f"shap_importance_{name}_split{split_id}.csv"),
                    index=False
                )
    
    def run_full_experiment(self):
        # Run complete experiment for all tables.
        # Create timestamped output directory
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = "bmipred_experiment"
        output_dir = os.path.join(
            self.config['experiment']['output_dir'],
            f"results_{script_name}_{now_str}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        n_repeats = self.config['experiment']['n_repeats']
        
        for table_name, table_path in self.config['data']['tables'].items():
            self.logger.info(f"\n========== TABLE: {table_name} ==========")
            
            try:
                # Load data
                df = pd.read_parquet(table_path)
                
                # Create table output directory
                table_dir = os.path.join(output_dir, table_name)
                os.makedirs(table_dir, exist_ok=True)
                
                # Save feature summary
                save_feature_summary(
                    df,
                    os.path.join(table_dir, f"{table_name}_feature_summary_statistics.csv"),
                    column_mapping=self.config.get('column_mapping')
                )
                
                # Get global categorical levels for consistency
                categorical_columns, categorical_levels = get_global_categorical_levels(
                    df, self.config['data']['target_col']
                )
                
                # Run all splits
                all_results = []
                all_predictions = {name: [] for name in self.config['models'].keys()}
                all_test_dfs = []
                all_shap_values = {name: [] for name in self.config['models'].keys()}
                all_feature_names = []
                all_pretty_feature_names = []
                all_X_test = []
                
                for split_id in range(n_repeats):
                    result = self.run_single_split(
                        df, table_name, split_id, output_dir,
                        categorical_levels, categorical_columns
                    )
                    
                    if result is not None:
                        all_results.append(result['metrics'])
                        all_test_dfs.append(result['test_df'])
                        all_feature_names.append(result['feature_names'])
                        all_pretty_feature_names.append(result['pretty_feature_names'])
                        all_X_test.append(result['X_test'])
                        for model_name, pred in result['predictions'].items():
                            all_predictions[model_name].append(pred)
                            all_shap_values[model_name].append(result['shap_values'][model_name])
                
                if all_results:
                    # Create summary reports
                    save_all_reports(
                        table_name, all_results, table_dir,
                        self.config['metrics']['keep_metrics']
                    )
                    
                    # Create aggregated plots
                    for model_name, predictions in all_predictions.items():
                        if predictions:
                            try:
                                self.plotter.plot_mean_roc_across_splits(
                                    predictions,
                                    os.path.join(table_dir, f"{table_name}_mean_roc_{model_name}.png"),
                                    f"{table_name} - {model_name}"
                                )
                            except Exception as e:
                                self.logger.error(f"Failed to plot mean ROC across splits for {model_name}: {e}")

                            try:
                                self.plotter.plot_mean_roc_per_subgroup(
                                    predictions,
                                    all_test_dfs,
                                    subgroup_col="healthAssesment_age",
                                    group_labels=["18-29", "30-49", "50-69", "70+"],
                                    out_path=os.path.join(table_dir, f"{table_name}_mean_roc_age_{model_name}.png"),
                                    title=f"{table_name} - Mean ROC per Age - {model_name}"
                                )
                            except Exception as e:
                                self.logger.error(f"Failed to plot mean ROC per age for {model_name}: {e}")
                            
                            try:
                                self.plotter.plot_mean_roc_per_subgroup(
                                    predictions,
                                    all_test_dfs,
                                    subgroup_col="days_to_next_bmi",
                                    group_labels=["0-30", "30-180", "180-360", "360+"],
                                    out_path=os.path.join(table_dir, f"{table_name}_mean_roc_days_to_next_bmi_{model_name}.png"),
                                    title=f"{table_name} - Mean ROC per Days to Next BMI - {model_name}"
                                )
                            except Exception as e:
                                self.logger.error(f"Failed to plot mean ROC per days to next BMI for {model_name}: {e}")
                            
                            try:
                                self.plotter.plot_mean_roc_per_subgroup(
                                    predictions,
                                    all_test_dfs,
                                    subgroup_col="BodyMassIndex_recalc",
                                    group_labels=["Underweight", "Normal", "Overweight", "Obese"],
                                    out_path=os.path.join(table_dir, f"{table_name}_mean_roc_baseline_bmi_{model_name}.png"),
                                    title=f"{table_name} - Mean ROC per Baseline BMI - {model_name}"
                                )
                            except Exception as e:
                                self.logger.error(f"Failed to plot mean ROC per baseline BMI for {model_name}: {e}")
                            
                            try:
                                sex_labels = sorted({
                                    value
                                    for df_split in all_test_dfs if "Sex" in df_split
                                    for value in df_split["Sex"].dropna().unique()
                                })
                                if sex_labels:
                                    self.plotter.plot_mean_roc_per_subgroup(
                                        predictions,
                                        all_test_dfs,
                                        subgroup_col="Sex",
                                        group_labels=sex_labels,
                                        out_path=os.path.join(table_dir, f"{table_name}_mean_roc_sex_{model_name}.png"),
                                        title=f"{table_name} - Mean ROC per Sex - {model_name}"
                                    )
                            except Exception as e:
                                self.logger.error(f"Failed to plot mean ROC per sex for {model_name}: {e}")

                    try:
                        self.plotter.plot_mean_roc_across_models(
                            all_predictions,
                            out_path=os.path.join(table_dir, f"{table_name}_mean_roc_all_models.png"),
                            title=f"{table_name} - Mean ROC Across Models"
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to plot mean ROC across models: {e}")

                    for model_name, shap_arrays in all_shap_values.items():
                        if not shap_arrays:
                            continue
                        try:
                            self.plotter.plot_shap_importance_across_splits(
                                shap_arrays,
                                all_pretty_feature_names,
                                all_X_test,
                                self.config.get('column_mapping', {}),
                                os.path.join(table_dir, f"{table_name}_shap_importance_{model_name}_across_splits.png"),
                                model_name,
                                table_dir,
                                table_name,
                            max_display=self.config.get('visualization', {}).get('max_display_features', 15)
                        )
                        except Exception as e:
                            self.logger.error(f"Failed to plot SHAP importance for {model_name}: {e}")
                    
                    self.logger.info(f"✓ Completed {table_name} → {table_dir}")
                else:
                    self.logger.warning(f"No successful splits for {table_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {table_name}: {e}")
                continue
        
        self.logger.info(f"\n✓ Experiment completed. Results in {output_dir}")
