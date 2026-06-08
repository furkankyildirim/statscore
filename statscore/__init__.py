"""statscore: A production-quality library for ANOVA and Multiple Linear Regression.

Provides structured-output implementations of one-way and two-way ANOVA,
multiple comparison procedures (Bonferroni, Sidak, Scheffe, Tukey), and
OLS-based multiple linear regression with simultaneous inference and prediction.
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
from statscore.utils.enums import (
    CorrectionMethod,
    PredictionMethod,
    TwoWayTestFactor,
)

__version__ = "0.0.1"

__all__ = [
    # Enums (type-safe categorical parameters)
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
]
