"""Strongly-typed enumerations for categorical parameters in statscore.

All categorical parameters in public-facing functions use these enums
instead of raw strings, enforcing compile-time type safety and enabling
IDE autocompletion.
"""

from enum import Enum


class CorrectionMethod(Enum):
    """Multiple comparison correction methods for simultaneous inference.

    Used by ANOVA1_CI_linear_combs, ANOVA1_test_linear_combs.
    """

    SCHEFFE = "Scheffe"
    TUKEY = "Tukey"
    BONFERRONI = "Bonferroni"
    SIDAK = "Sidak"
    BEST = "best"


class PredictionMethod(Enum):
    """Prediction interval construction methods.

    Used by Mult_norm_LR_pred_CI.
    """

    SCHEFFE = "Scheffe"
    BONFERRONI = "Bonferroni"
    BEST = "best"


class TwoWayTestFactor(Enum):
    """Factor to test in two-way ANOVA.

    Used by ANOVA2_test_equality.
    """

    A = "A"
    B = "B"
    AB = "AB"


class AlternativeHypothesis(Enum):
    """Alternative hypothesis direction for significance tests.

    Used by z_test_mean, t_test_mean, chi2_test_variance,
    t_test_two_sample, t_test_paired, f_test_variances.
    """

    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"
