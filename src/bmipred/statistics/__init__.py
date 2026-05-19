# src/bmipred/statistics/__init__.py

from .statistics import (
    compare_two_groups_with_normality_check,
    wilcoxon_paired_test,
    smd_continuous,
    smd_binary,
)

from .summaries import (
    pseudo_quantile,
    round5,
    build_publication_table,
    summarise_dataframe,
)

from .boxplots import (
    prepare_groups,
    plot_box_q2_to_q3,
    plot_box_full_range,
)

from .distributions import (
    aggregate_in_groups_of_k,
    pseudo_median_from_values,
    plot_histogram_two_cohorts_by_sex,
    compute_pseudo_stats,
    plot_histogram_with_medians,
    plot_bmi_over_age,
)

from .trendline import (
    plot_lowess_two,
    compute_patient_bmi_slopes,
    plot_bmi_slope_comparison,
    build_delta_bmi_trajectory,
    plot_delta_bmi_trajectory,
    plot_delta_bmi_trajectory_3panel,
)

from .correlations import (
    get_filtered_df,
    filter_sparse_columns_pairwise,
    compute_pairwise_spearman,
    plot_lower_triangle_heatmap,
    plot_top_correlations,
    plot_clustered_heatmap,
    plot_clustered_heatmap_with_pvalues,
)

__all__ = [
    # statistics
    "compare_two_groups_with_normality_check",
    "wilcoxon_paired_test",
    "smd_continuous",
    "smd_binary",
    # summaries
    "pseudo_quantile",
    "round5",
    "build_publication_table",
    "summarise_dataframe",
    # boxplots
    "prepare_groups",
    "plot_box_q2_to_q3",
    "plot_box_full_range",
    # distributions
    "aggregate_in_groups_of_k",
    "pseudo_median_from_values",
    "plot_histogram_two_cohorts_by_sex",
    "compute_pseudo_stats",
    "plot_histogram_with_medians",
    "plot_bmi_over_age",
    # trendline
    "plot_lowess_two",
    "compute_patient_bmi_slopes",
    "plot_bmi_slope_comparison",
    "build_delta_bmi_trajectory",
    "plot_delta_bmi_trajectory",
    "plot_delta_bmi_trajectory_3panel",
    # correlations
    "get_filtered_df",
    "filter_sparse_columns_pairwise",
    "compute_pairwise_spearman",
    "plot_lower_triangle_heatmap",
    "plot_top_correlations",
    "plot_clustered_heatmap",
    "plot_clustered_heatmap_with_pvalues",
]

