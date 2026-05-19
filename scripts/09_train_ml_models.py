#!/usr/bin/env python3
# Script to train and test machine learning models for BMI prediction.
# All configuration parameters are defined at the beginning of the script.

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bmipred.modeling.pipeline import BMIPredPipeline

# ==================== CONFIGURATION PARAMETERS ====================

BASE_DIR = Path(__file__).parent.parent / "data"

CONFIG = {
    # Data paths and settings
    "data": {
        "target_col": "target",
        "tables": {
            "cohort2": str(BASE_DIR / "preprocessed" / "cohort2_olanzapine_on_treatment_final.parquet"),
        },
        "columns_to_drop": [
            "PatientDurableKey",
            "CreateInstant",
            "RateOfBMIChange_month",
            "bmi_days_before_medication_stop",
        ],
    },
    
    # Experiment settings
    "experiment": {
        "output_dir": str(BASE_DIR / "results" / "ml_models"),
        "n_repeats": 5,
        "random_state": 101010,
    },
    
    # Train/test split settings
    "split": {
        "test_size": 0.2,
        "random_state": 101010,
    },
    
    # Training settings
    "training": {
        "cv_folds": 5,
        "n_jobs": -1,
    },
    
    # Visualization settings
    "visualization": {
        "max_display_features": 15,
        "plot_params": {
            "figure.figsize": (12, 8),
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    },
    
    # Model settings
    "models": {
        "LogisticRegression": {
            "class": "sklearn.linear_model.LogisticRegression",
            "params": {
                "max_iter": 1000,
                "random_state": 101010,
            },
            "param_grid": {
                "C": [0.01, 0.1, 1],
                "penalty": ["l2"],
            }
        },
        "XGBoost": {
            "class": "xgboost.XGBClassifier",
            "params": {
                "objective": "binary:logistic",
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 101010,
                "n_jobs": 1,
            },
            "param_grid": {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
            }
        },
        "GradientBoosting": {
            "class": "sklearn.ensemble.GradientBoostingClassifier",
            "params": {
                "random_state": 101010,
            },
            "param_grid": {
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
            }
        },
        "RandomForest": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "params": {
                "random_state": 101010,
            },
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [5, 10],
            }
        },
    },
    
    # Metrics configuration
    "metrics": {
        "keep_metrics": {
            "Test Accuracy": "Accuracy",
            "Test Balanced Accuracy": "Balanced Accuracy",
            "Test F1 Score (positive)": "F1 Score",
            "Test ROC AUC": "ROC AUC",
            "Test Average Precision": "Average Precision",
            "Test Sensitivity": "Sensitivity",
            "Test Specificity": "Specificity",
            "Test PPV": "PPV",
            "Test NPV": "NPV",
        }
    },
}

# ==================== END CONFIGURATION =======================


def main():
    """Main entry point for ML modeling pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting BMI prediction ML modeling pipeline")
    logger.info(f"Output directory: {CONFIG['experiment']['output_dir']}")
    
    # Create output directory
    Path(CONFIG['experiment']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run pipeline
    try:
        pipeline = BMIPredPipeline(CONFIG)
        logger.info("Pipeline initialized successfully!")
        
        # Run full experiment
        logger.info("Running full experiment...")
        pipeline.run_full_experiment()
        
        logger.info("Machine Learning pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()