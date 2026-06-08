"""Result dataclasses for regression inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from statscore.utils.enums import PredictionMethod


@dataclass
class SimultaneousCIBetaResult:
    """Simultaneous confidence intervals for regression coefficients."""

    beta_hat: np.ndarray
    intervals: list[tuple[float, float]]
    half_widths: np.ndarray
    method: PredictionMethod

    def summary(self) -> None:
        w = 64
        print("=" * w)
        print("  Simultaneous CIs for Regression Coefficients")
        print(f"  Method: {self.method.value}")
        print("=" * w)
        print(
            f"  {'Coefficient':<14} {'Estimate':>10} {'Half-Width':>12} {'Lower':>10} {'Upper':>10}"
        )
        print("-" * w)
        for i, (lo, hi) in enumerate(self.intervals):
            b = float(self.beta_hat[i])
            hw = float(self.half_widths[i])
            print(f"  {'beta_' + str(i):<14} {b:>10.4f} {hw:>12.4f} {lo:>10.4f} {hi:>10.4f}")
        print("=" * w)

    def plot(self) -> Figure:
        from statscore.plots import plot_simultaneous_ci

        labels = [f"beta_{i}" for i in range(len(self.beta_hat))]
        return plot_simultaneous_ci(
            point_estimates=self.beta_hat,
            intervals=self.intervals,
            method=self.method.value,
            labels=labels,
            title="Simultaneous CIs for Regression Coefficients",
        )


@dataclass
class ConfidenceRegionResult:
    """Confidence region (ellipsoid) specification for C*beta."""

    center: np.ndarray
    shape_matrix: np.ndarray
    radius_squared: float
    Se2: float
    F_critical: float
    r: int
    df_residual: int


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test in regression."""

    test_statistic: float
    F_critical: float
    p_value: float
    reject_H0: bool
    alpha: float
    df_numerator: int
    df_denominator: int

    def summary(self) -> None:
        w = 60
        decision = "Reject H0" if self.reject_H0 else "Fail to reject H0"
        print("=" * w)
        print("  Regression Hypothesis Test")
        print("=" * w)
        print(f"  df_numerator = {self.df_numerator}    df_denominator = {self.df_denominator}")
        print(f"  alpha = {self.alpha}")
        print("-" * w)
        print(f"  F-statistic: {self.test_statistic:.4f}    F-critical: {self.F_critical:.4f}")
        print(f"  p-value:     {self.p_value:.4f}")
        print(f"  Decision:    {decision}")
        print("=" * w)

    def plot(self) -> Figure:
        from statscore.plots import plot_f_test

        return plot_f_test(
            f_statistic=self.test_statistic,
            f_critical_low=0.0,
            f_critical_up=self.F_critical,
            df1=self.df_numerator,
            df2=self.df_denominator,
            alternative="greater",
            title="Regression Hypothesis Test",
        )


@dataclass
class PredictionCIResult:
    """Simultaneous prediction confidence intervals."""

    point_estimates: np.ndarray
    intervals: list[tuple[float, float]]
    half_widths: np.ndarray
    method_used: PredictionMethod

    def summary(self) -> None:
        w = 64
        print("=" * w)
        print("  Simultaneous Prediction Intervals")
        print(f"  Method: {self.method_used.value}")
        print("=" * w)
        print(f"  {'#':<5} {'Point Est':>10} {'Half-Width':>12} {'Lower':>10} {'Upper':>10}")
        print("-" * w)
        for i, (lo, hi) in enumerate(self.intervals):
            pe = float(self.point_estimates[i])
            hw = float(self.half_widths[i])
            print(f"  {i + 1:<5} {pe:>10.4f} {hw:>12.4f} {lo:>10.4f} {hi:>10.4f}")
        print("=" * w)

    def plot(self) -> Figure:
        from statscore.plots import plot_simultaneous_ci

        labels = [f"Pred {i+1}" for i in range(len(self.point_estimates))]
        return plot_simultaneous_ci(
            point_estimates=self.point_estimates,
            intervals=self.intervals,
            method=self.method_used.value,
            labels=labels,
            title="Simultaneous Prediction Intervals",
        )
