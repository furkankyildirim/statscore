"""Strongly-typed enumerations for categorical parameters in stats_toolbox.

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
