#!/usr/bin/env python3
# Script to create cohort statistics plots using the bmipred statistics modules.
# All configuration parameters are defined at the beginning of the script.

import sys
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import statistics modules
from bmipred.statistics import boxplots, distributions, correlations, trendline, summaries, statistics

# ==================== CONFIGURATION PARAMETERS ====================

BASE_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = BASE_DIR / "results"

COHORT_PATHS = {
    "cohort1": BASE_DIR / "preprocessed" / "cohort1_olanzapine_initiation_final.parquet",
    "cohort2": BASE_DIR / "preprocessed" / "cohort2_olanzapine_on_treatment_final.parquet",
}

# ==================== END CONFIGURATION =======================  

def main():
    # Create cohort statistics plots from final cohort parquet files using bmipred statistics modules.
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {RESULTS_DIR}")
    
    # Load cohort data
    print("\n--- Loading Cohort Data ---")
    try:
        cohort1 = pd.read_parquet(COHORT_PATHS["cohort1"])
        cohort2 = pd.read_parquet(COHORT_PATHS["cohort2"])
        print(f"Cohort1 shape: {cohort1.shape}")
        print(f"Cohort2 shape: {cohort2.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # ===== BOXPLOTS =====
    print("\n--- Creating Boxplots ---")
    if "BodyMassIndex_recalc" in cohort1.columns and "BodyMassIndex_recalc" in cohort2.columns:
        # Prepare on/off treatment groups
        df_combined, u_stat, p_val = boxplots.prepare_groups(
            on_df=cohort2,
            off_df=cohort1,
            value_col="BodyMassIndex_recalc",
            label_col="Treatment_Status",
            on_label="On Treatment",
            off_label="Initiation"
        )
        
        # Create boxplot (median to Q3)
        ann_text = f"Mann-Whitney U: p={'<0.001' if p_val < 0.001 else f'{p_val:.3f}'}"
        
        # Create boxplot
        boxplots.plot_box_full_range(
            df_combined,
            value_col="BodyMassIndex_recalc",
            label_col="Treatment_Status",
            title="BMI by Treatment Status (Full Range)",
            xlabel="Olanzapine Status",
            ylabel="BMI (kg/m²)",
            annotate_text=ann_text,
            out_path=str(RESULTS_DIR / "boxplot_bmi_full_range.png")
        )
        print("✓ Boxplots created")
    
    # ===== DISTRIBUTIONS =====
    print("\n--- Creating Distribution Plots ---")
    if "Sex" in cohort1.columns and "BodyMassIndex_recalc" in cohort1.columns:
        try:
            distributions.plot_histogram_two_cohorts_by_sex(
                cohort1,
                cohort2,
                value_col="BodyMassIndex_recalc",
                sex_col="Sex",
                label1="Initiation",
                label2="On Treatment",
                xlabel="BMI (kg/m²)",
                out_dir=str(RESULTS_DIR)
            )
            print("✓ Distribution plots created")
        except Exception as e:
            print(f"⚠ Distribution plot error: {e}")
    
    # ===== CORRELATIONS =====
    print("\n--- Creating Correlation Plots ---")
    if "target" in cohort2.columns:
        try:
            correlations.plot_top_correlations(
                cohort2,
                df_name="Cohort2_OnTreatment",
                out_dir=str(RESULTS_DIR)
            )
            print("✓ Correlation plots created")
        except Exception as e:
            print(f"⚠ Correlation plot error: {e}")
    
    # ===== TRENDLINES =====
    print("\n--- Creating Trendline Plots ---")
    # Check available date columns
    date_cols = [col for col in cohort1.columns if 'age' in col.lower() or 'instant' in col.lower() or 'date' in col.lower()]
    print(f"Available date columns in cohort1: {date_cols}")
    
    if all(col in cohort1.columns and col in cohort2.columns 
           for col in ["healthAssesment_age", "BodyMassIndex_recalc"]):
        try:
            trendline.plot_lowess_two(
                cohort1,
                cohort2,
                age_col="healthAssesment_age",
                value_col="BodyMassIndex_recalc",
                label1="Initiation",
                label2="On Treatment",
                frac=0.3,
                figsize=(8, 6),
                save_path=str(RESULTS_DIR / "trendline_bmi_comparison.png")
            )
            print("✓ Trendline plots created")
        except Exception as e:
            print(f"⚠ Trendline plot error: {e}")
    else:
        print("⚠ Trendline skipped: 'CreateInstant' column not found in both cohorts")
    
    # ===== SUMMARIES =====
    print("\n--- Creating Summary Statistics ---")
    
    # Cohort1 summary
    cohort1_summary = {}
    for col in cohort1.columns:
        if pd.api.types.is_numeric_dtype(cohort1[col]) or pd.api.types.is_bool_dtype(cohort1[col]):
            cohort1_summary[col] = summaries.summarise_column(cohort1[col])
    
    cohort1_summary_df = pd.DataFrame(cohort1_summary).T
    cohort1_summary_df.to_csv(RESULTS_DIR / "cohort1_summary_statistics.csv")
    print(f"✓ Cohort1 summary saved ({len(cohort1_summary)} columns)")
    
    # Cohort2 summary
    cohort2_summary = {}
    for col in cohort2.columns:
        if pd.api.types.is_numeric_dtype(cohort2[col]) or pd.api.types.is_bool_dtype(cohort2[col]):
            cohort2_summary[col] = summaries.summarise_column(cohort2[col])
    
    cohort2_summary_df = pd.DataFrame(cohort2_summary).T
    cohort2_summary_df.to_csv(RESULTS_DIR / "cohort2_summary_statistics.csv")
    print(f"✓ Cohort2 summary saved ({len(cohort2_summary)} columns)")
    
    # ===== CLUSTERED CORRELATION HEATMAPS =====
    print("\n--- Creating Clustered Correlation Heatmaps ---")
    if "target" in cohort2.columns:
        try:
            # Filter and prepare data for correlation
            df_filtered = correlations.get_filtered_df(cohort2)
            
            # Clustered heatmap for cohort2
            correlations.plot_clustered_heatmap(
                df_filtered,
                df_name="Cohort2_OnTreatment",
                out_dir=str(RESULTS_DIR)
            )
            print("✓ Clustered heatmap created")
        except Exception as e:
            print(f"⚠ Clustered heatmap error: {e}")
    
    # ===== LOWER TRIANGLE HEATMAP WITH SIGNIFICANCE =====
    print("\n--- Creating Lower Triangle Correlation Heatmap ---")
    if "target" in cohort2.columns:
        try:
            df_filtered = correlations.get_filtered_df(cohort2)
            corr_df, p_values_df = correlations.compute_pairwise_spearman(df_filtered)
            
            correlations.plot_lower_triangle_heatmap(
                corr_df,
                p_values_df,
                title="Spearman Correlation Heatmap (Lower Triangle)",
                out_path=str(RESULTS_DIR / "cohort2_lower_triangle_heatmap.png")
            )
            print("✓ Lower triangle heatmap created")
        except Exception as e:
            print(f"⚠ Lower triangle heatmap error: {e}")
    
    # ===== SINGLE COHORT HISTOGRAMS WITH MEDIANS =====
    print("\n--- Creating Histograms with Medians ---")
    if "Sex" in cohort1.columns and "BodyMassIndex_recalc" in cohort1.columns:
        try:
            distributions.plot_histogram_with_medians(
                cohort1,
                column="BodyMassIndex_recalc",
                title="BMI Distribution at Olanzapine Initiation (Cohort 1)",
                x_label="BMI (kg/m²)",
                save_path=str(RESULTS_DIR / "cohort1_bmi_histogram_medians.png")
            )
            print("✓ Cohort1 histogram created")
        except Exception as e:
            print(f"⚠ Cohort1 histogram error: {e}")
    
    if "Sex" in cohort2.columns and "BodyMassIndex_recalc" in cohort2.columns:
        try:
            distributions.plot_histogram_with_medians(
                cohort2,
                column="BodyMassIndex_recalc",
                title="BMI Distribution On Treatment (Cohort 2)",
                x_label="BMI (kg/m²)",
                save_path=str(RESULTS_DIR / "cohort2_bmi_histogram_medians.png")
            )
            print("✓ Cohort2 histogram created")
        except Exception as e:
            print(f"⚠ Cohort2 histogram error: {e}")
    
    # ===== BMI SLOPE ANALYSIS =====
    print("\n--- Analyzing BMI Slopes ---")
    if all(col in cohort2.columns for col in ["StartInstant", "CreateInstant", "BodyMassIndex_recalc", "PatientDurableKey"]):
        try:
            # Rename columns to match function expectations
            cohort2_slopes = cohort2[["PatientDurableKey", "CreateInstant", "StartInstant", "BodyMassIndex_recalc"]].copy()
            cohort2_slopes.columns = ["PatientDurableKey", "date_col", "start_col", "bmi_col"]
            
            slopes_df = trendline.compute_patient_bmi_slopes(
                cohort2_slopes.rename(columns={"date_col": "CreateInstant", "start_col": "StartInstant", "bmi_col": "BodyMassIndex_recalc"}),
                pre_days=365,
                post_days=365,
                patient_col="PatientDurableKey",
                start_col="StartInstant",
                date_col="CreateInstant",
                bmi_col="BodyMassIndex_recalc"
            )
            
            slopes_df.to_csv(RESULTS_DIR / "cohort2_bmi_slopes.csv", index=False)
            print(f"✓ BMI slopes calculated and saved ({len(slopes_df)} patients)")
        except Exception as e:
            print(f"⚠ BMI slope analysis error: {e}")
    
    # ===== STATISTICAL COMPARISONS =====
    print("\n--- Performing Statistical Comparisons ---")
    if "BodyMassIndex_recalc" in cohort1.columns and "BodyMassIndex_recalc" in cohort2.columns:
        try:
            comparison_df = statistics.compare_two_groups_with_normality_check(
                cohort1["BodyMassIndex_recalc"].dropna().values,
                cohort2["BodyMassIndex_recalc"].dropna().values,
                label1="Initiation",
                label2="On Treatment",
                alpha=0.05
            )
            
            comparison_df.to_csv(RESULTS_DIR / "bmi_statistical_comparison.csv", index=False)
            print("✓ Statistical comparison completed")
            print(comparison_df.to_string())
        except Exception as e:
            print(f"⚠ Statistical comparison error: {e}")
    
    # ===== PATIENT CHARACTERISTICS =====
    print("\n--- Cohort Characteristics ---")
    print(f"Cohort 1 (Initiation): {cohort1.shape[0]} observations, {cohort1['PatientDurableKey'].nunique()} unique patients")
    print(f"Cohort 2 (On Treatment): {cohort2.shape[0]} observations, {cohort2['PatientDurableKey'].nunique()} unique patients")
    
    print(f"\n✓ All plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
