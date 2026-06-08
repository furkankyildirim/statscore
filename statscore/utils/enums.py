"""Strongly-typed enumerations for categorical parameters in statscore.

All categorical parameters in public-facing functions use these enums
instead of raw strings, enforcing compile-time type safety and enabling
IDE autocompletion.
"""

from __future__ import annotations

from enum import Enum


class CorrectionMethod(Enum):
    """Multiple comparison correction methods for simultaneous inference.

    Used by anova1_ci_linear_combs, anova1_test_linear_combs.
    """

    SCHEFFE = "Scheffe"
    TUKEY = "Tukey"
    BONFERRONI = "Bonferroni"
    SIDAK = "Sidak"
    BEST = "best"


class PredictionMethod(Enum):
    """Prediction interval construction methods.

    Used by mult_norm_lr_pred_ci.
    """

    SCHEFFE = "Scheffe"
    BONFERRONI = "Bonferroni"
    BEST = "best"


class TwoWayTestFactor(Enum):
    """Factor to test in two-way ANOVA.

    Used by anova2_test_equality.
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
