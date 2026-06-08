from __future__ import annotations

from statscore.methods.regression._results import (
    ConfidenceRegionResult,
    HypothesisTestResult,
    PredictionCIResult,
    SimultaneousCIBetaResult,
)
from statscore.methods.regression.inference import (
    mult_norm_lr_cr,
    mult_norm_lr_is_in_cr,
    mult_norm_lr_simul_ci,
    mult_norm_lr_test_comp,
    mult_norm_lr_test_general,
    mult_norm_lr_test_linear_reg,
)
from statscore.methods.regression.least_squares import (
    LeastSquaresResult,
    mult_lr_least_squares,
    mult_lr_partition_tss,
)
from statscore.methods.regression.prediction import mult_norm_lr_pred_ci
from statscore.methods.regression.summary import RegressionSummaryResult, regression_summary

__all__ = [
    "mult_lr_least_squares",
    "mult_lr_partition_tss",
    "mult_norm_lr_simul_ci",
    "mult_norm_lr_cr",
    "mult_norm_lr_is_in_cr",
    "mult_norm_lr_test_general",
    "mult_norm_lr_test_comp",
    "mult_norm_lr_test_linear_reg",
    "mult_norm_lr_pred_ci",
    "LeastSquaresResult",
    "SimultaneousCIBetaResult",
    "ConfidenceRegionResult",
    "HypothesisTestResult",
    "PredictionCIResult",
    "RegressionSummaryResult",
    "regression_summary",
]
