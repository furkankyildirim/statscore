"""statscore: A production-quality library for ANOVA, Regression, Testing, and Bayesian Inference.

Provides structured-output implementations of one-way and two-way ANOVA,
multiple comparison procedures (Bonferroni, Sidak, Scheffe, Tukey),
OLS-based multiple linear regression with simultaneous inference and prediction,
normal distribution significance tests, and Bayesian conjugate prior inference.
"""

from statscore.anova import (
    ANOVA2_MLE,
    ANOVA1_CI_linear_combs,
    ANOVA1_is_contrast,
    ANOVA1_is_orthogonal,
    ANOVA1_partition_TSS,
    ANOVA1_test_equality,
    ANOVA1_test_linear_combs,
    ANOVA2_partition_TSS,
    ANOVA2_test_equality,
    Bonferroni_correction,
    Sidak_correction,
)
from statscore.bayes import (
    bayes_normal_mean_known_var,
    bayes_normal_mean_unknown_var,
)
from statscore.diagnostics import (
    LeveneResult,
    MeanConfidenceIntervalResult,
    RegressionDiagnosticsResult,
    ShapiroWilkResult,
    levene_test,
    mean_confidence_interval,
    regression_diagnostics,
    shapiro_wilk_test,
)
from statscore.io import LoadedData, load_data
from statscore.plots import (
    plot_anova_groups,
    plot_posterior_normal,
    plot_qq,
    plot_regression,
    plot_residuals,
)
from statscore.regression import (
    Mult_LR_Least_squares,
    Mult_LR_partition_TSS,
    Mult_norm_LR_CR,
    Mult_norm_LR_is_in_CR,
    Mult_norm_LR_pred_CI,
    Mult_norm_LR_simul_CI,
    Mult_norm_LR_test_comp,
    Mult_norm_LR_test_general,
    Mult_norm_LR_test_linear_reg,
    RegressionSummaryResult,
    regression_summary,
)
from statscore.testing import (
    chi2_test_variance,
    f_test_variances,
    t_test_mean,
    t_test_paired,
    t_test_two_sample,
    z_test_mean,
)
from statscore.utils.enums import (
    AlternativeHypothesis,
    CorrectionMethod,
    PredictionMethod,
    TwoWayTestFactor,
)

__version__ = "0.0.2"

__all__ = [
    # Enums (type-safe categorical parameters)
    "AlternativeHypothesis",
    "CorrectionMethod",
    "PredictionMethod",
    "TwoWayTestFactor",
    # ANOVA
    "ANOVA1_partition_TSS",
    "ANOVA1_test_equality",
    "ANOVA1_is_contrast",
    "ANOVA1_is_orthogonal",
    "Bonferroni_correction",
    "Sidak_correction",
    "ANOVA1_CI_linear_combs",
    "ANOVA1_test_linear_combs",
    "ANOVA2_partition_TSS",
    "ANOVA2_MLE",
    "ANOVA2_test_equality",
    # Regression
    "Mult_LR_Least_squares",
    "Mult_LR_partition_TSS",
    "Mult_norm_LR_simul_CI",
    "Mult_norm_LR_CR",
    "Mult_norm_LR_is_in_CR",
    "Mult_norm_LR_test_general",
    "Mult_norm_LR_test_comp",
    "Mult_norm_LR_test_linear_reg",
    "Mult_norm_LR_pred_CI",
    "RegressionSummaryResult",
    "regression_summary",
    # Testing
    "z_test_mean",
    "t_test_mean",
    "chi2_test_variance",
    "t_test_two_sample",
    "t_test_paired",
    "f_test_variances",
    # Bayesian
    "bayes_normal_mean_known_var",
    "bayes_normal_mean_unknown_var",
    # I/O
    "LoadedData",
    "load_data",
    # Diagnostics
    "ShapiroWilkResult",
    "LeveneResult",
    "RegressionDiagnosticsResult",
    "MeanConfidenceIntervalResult",
    "shapiro_wilk_test",
    "levene_test",
    "regression_diagnostics",
    "mean_confidence_interval",
    # Plots
    "plot_regression",
    "plot_residuals",
    "plot_qq",
    "plot_anova_groups",
    "plot_posterior_normal",
]
