"""statscore: A production-quality library for ANOVA, Regression, Testing, and Bayesian Inference.

Provides structured-output implementations of one-way and two-way ANOVA,
multiple comparison procedures (Bonferroni, Sidak, Scheffe, Tukey),
OLS-based multiple linear regression with simultaneous inference and prediction,
normal distribution significance tests, and Bayesian conjugate prior inference.
"""

from __future__ import annotations

from statscore.anova import (
    anova1_ci_linear_combs,
    anova1_is_contrast,
    anova1_is_orthogonal,
    anova1_partition_tss,
    anova1_test_equality,
    anova1_test_linear_combs,
    anova2_mle,
    anova2_partition_tss,
    anova2_test_equality,
    bonferroni_correction,
    sidak_correction,
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
from statscore.utils.plots import (
    plot_anova_groups,
    plot_f_test,
    plot_qq,
    plot_regression,
    plot_residuals,
    plot_simultaneous_ci,
    plot_t_test,
)
from statscore.regression import (
    RegressionSummaryResult,
    mult_lr_least_squares,
    mult_lr_partition_tss,
    mult_norm_lr_cr,
    mult_norm_lr_is_in_cr,
    mult_norm_lr_pred_ci,
    mult_norm_lr_simul_ci,
    mult_norm_lr_test_comp,
    mult_norm_lr_test_general,
    mult_norm_lr_test_linear_reg,
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

__version__ = "0.0.3"

__all__ = [
    # Enums (type-safe categorical parameters)
    "AlternativeHypothesis",
    "CorrectionMethod",
    "PredictionMethod",
    "TwoWayTestFactor",
    # ANOVA
    "anova1_partition_tss",
    "anova1_test_equality",
    "anova1_is_contrast",
    "anova1_is_orthogonal",
    "bonferroni_correction",
    "sidak_correction",
    "anova1_ci_linear_combs",
    "anova1_test_linear_combs",
    "anova2_partition_tss",
    "anova2_mle",
    "anova2_test_equality",
    # Regression
    "mult_lr_least_squares",
    "mult_lr_partition_tss",
    "mult_norm_lr_simul_ci",
    "mult_norm_lr_cr",
    "mult_norm_lr_is_in_cr",
    "mult_norm_lr_test_general",
    "mult_norm_lr_test_comp",
    "mult_norm_lr_test_linear_reg",
    "mult_norm_lr_pred_ci",
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
    "plot_t_test",
    "plot_f_test",
    "plot_simultaneous_ci",
]
