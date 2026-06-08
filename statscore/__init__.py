"""statscore: A production-quality library for ANOVA, Regression, Testing, and Bayesian Inference.

Provides structured-output implementations of one-way and two-way ANOVA,
multiple comparison procedures (Bonferroni, Sidak, Scheffe, Tukey),
OLS-based multiple linear regression with simultaneous inference and prediction,
normal distribution significance tests, and Bayesian conjugate prior inference.
"""

from statscore.anova import (
    ANOVA1_CI_linear_combs,
    ANOVA1_is_contrast,
    ANOVA1_is_orthogonal,
    ANOVA1_partition_TSS,
    ANOVA1_print_table,
    ANOVA1_test_equality,
    ANOVA1_test_linear_combs,
    ANOVA2_MLE,
    ANOVA2_partition_TSS,
    ANOVA2_print_table,
    ANOVA2_test_equality,
    Bonferroni_correction,
    Sidak_correction,
)
from statscore.bayes import (
    bayes_normal_mean_known_var,
    bayes_normal_mean_unknown_var,
    bayes_normal_mean_unknown_var_summary,
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
    "ANOVA1_print_table",
    "ANOVA1_is_contrast",
    "ANOVA1_is_orthogonal",
    "Bonferroni_correction",
    "Sidak_correction",
    "ANOVA1_CI_linear_combs",
    "ANOVA1_test_linear_combs",
    "ANOVA2_partition_TSS",
    "ANOVA2_MLE",
    "ANOVA2_test_equality",
    "ANOVA2_print_table",
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
    "bayes_normal_mean_unknown_var_summary",
]
