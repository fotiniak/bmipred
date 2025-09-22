BMI Prediction Machine Learning Pipeline
A comprehensive machine learning pipeline for predicting BMI changes in patients on olanzapine treatment. This repository provides end-to-end functionality for data preprocessing, feature engineering, model training, and evaluation.

🎯 Overview
This project implements a modular ML pipeline to predict significant BMI increases (>5%) in patients receiving olanzapine treatment. The pipeline includes:

-Data preprocessing and feature engineering
-Multiple ML models (Logistic Regression, Random Forest, Decision Trees, Gradient Boosting, XGBoost)
-Comprehensive evaluation with SHAP explainability
-Automated visualization and reporting
-Configurable experiments via YAML files

📁 Repository Structure

bmipred/
├── data/                                    # Data directory
│   ├── external/                            # External code mapping datasets
│   ├── interim/                             # Intermediate cleaned data files
│   ├── plots/                               # Plots exploratory files
│   ├── processed/                           # Feature engineered data tables
│   ├── results/                             # Analysis results data files
│   └── synthetic/                           # Randomly generated data files
├── results/                                 # Model outputs and analysis results
├── scripts/                                 # Executable scripts
│   ├── config/                              # Configuration files for each script
│   │   └── train_evaluate_models.yaml       # configs (# named after the script they configure)
│   └── 22_train_evaluate_models.py          # Scripts numbered in order of execution steb by step
└── src/bmipred/                             # Source code of all the functions and utilities
    ├── analysis/                            # Statistical analysis 
    ├── cleaning/                            # Data cleaning 
    ├── feature_engineering/                 # Feature engineering functions
    ├── generate_synthetic/                  # Synthetic data generation
    ├── modeling/                            # ML pipeline components
    ├── preprocessing/                       # Data preprocessing
    └── visualization/                       # Exploratory plotting and visualization

🚀 Quick Start
Prerequisites

# Required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap pyyaml joblib

Clone the repository:
git clone https://github.com/your-username/bmipred.git
cd bmipred

Running a Machine Learning Training Experiment:
Configure your experiment in train_evaluate_models.yaml and run the corresponding script.

📈 Output Structure
Results are organized in timestamped directories:
results_bmipred_experiment_20231215_143022/
├── cohort_on_olanzapine_first/
│   ├── split_0/, split_1/, ..., split_9/
│   │   ├── models/                 # Trained model artifacts
│   │   ├── plots/                  # Evaluation plots
│   │   └── results/                # Metrics and SHAP values
│   ├── cohort_on_olanzapine_first_summary_metrics.csv
│   ├── cohort_on_olanzapine_first_feature_summary_statistics.csv
│   └── mean_roc_curves/
└── cohort_on_olanzapine_last/
    └── ...


