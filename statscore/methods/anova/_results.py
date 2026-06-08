"""Result dataclasses for ANOVA procedures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from statscore.utils.enums import CorrectionMethod, TwoWayTestFactor


@dataclass
class ANOVA1PartitionResult:
    """Result of one-way ANOVA sum-of-squares partitioning."""

    SS_total: float
    SS_within: float
    SS_between: float
    group_means: np.ndarray
    grand_mean: float
    group_sizes: np.ndarray


@dataclass
class ANOVA1TestResult:
    """Result of one-way ANOVA F-test for equality of means."""

    SS_total: float
    SS_within: float
    SS_between: float
    df_between: int
    df_within: int
    df_total: int
    MS_between: float
    MS_within: float
    F_statistic: float
    F_critical: float
    p_value: float
    reject_H0: bool
    alpha: float

    def summary(self) -> None:
        w = 58
        decision = "Reject H0" if self.reject_H0 else "Fail to reject H0"
        print("=" * w)
        print("  One-Way ANOVA Table")
        print("=" * w)
        print(f"  {'Source':<12} {'df':>5} {'SS':>12} {'MS':>12} {'F':>10}")
        print("-" * w)
        print(
            f"  {'Between':<12} {self.df_between:>5} {self.SS_between:>12.4f} {self.MS_between:>12.4f} {self.F_statistic:>10.4f}"
        )
        print(
            f"  {'Within':<12} {self.df_within:>5} {self.SS_within:>12.4f} {self.MS_within:>12.4f} {'':>10}"
        )
        print("-" * w)
        print(f"  {'Total':<12} {self.df_total:>5} {self.SS_total:>12.4f} {'':>12} {'':>10}")
        print("=" * w)
        print(f"  F critical (alpha={self.alpha}): {self.F_critical:.4f}")
        print(f"  p-value:                  {self.p_value:.4f}")
        print(f"  Decision:                 {decision}")
        print("=" * w)

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt
        from scipy import stats

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        bars = ax1.bar(["SS Between", "SS Within"], [self.SS_between, self.SS_within],
                       color=["steelblue", "coral"], edgecolor="k", linewidth=0.5)
        ax1.set_ylabel("Sum of Squares")
        ax1.set_title("Sum of Squares Decomposition")
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, [self.SS_between, self.SS_within]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10)

        ax2 = axes[1]
        x_max = max(self.F_statistic * 1.5, self.F_critical * 2, stats.f.ppf(0.999, self.df_between, self.df_within))
        x = np.linspace(0.01, x_max, 500)
        pdf = stats.f.pdf(x, self.df_between, self.df_within)
        ax2.plot(x, pdf, color="steelblue", linewidth=2, label=f"F({self.df_between},{self.df_within})")
        ax2.fill_between(x, pdf, where=x >= self.F_critical, alpha=0.3, color="crimson", label="Rejection region")
        ax2.axvline(self.F_critical, color="crimson", linestyle="--", linewidth=1.2)
        ax2.axvline(self.F_statistic, color="darkgreen", linestyle="-", linewidth=2, label=f"F = {self.F_statistic:.4f}")
        ax2.set_xlabel("F")
        ax2.set_ylabel("Density")
        ax2.set_title("F-Test")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("One-Way ANOVA", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


@dataclass
class ANOVA2PartitionResult:
    """Result of two-way ANOVA sum-of-squares partition."""

    SS_total: float
    SS_A: float
    SS_B: float
    SS_AB: float
    SS_E: float


@dataclass
class ANOVA2MLEResult:
    """Maximum likelihood estimates for two-way ANOVA parameters."""

    mu: float
    a: np.ndarray
    b: np.ndarray
    delta: np.ndarray


@dataclass
class ANOVA2TestResult:
    """Result of a two-way ANOVA F-test."""

    source: TwoWayTestFactor
    df: int
    SS: float
    MS: float
    F_statistic: float
    F_critical: float
    p_value: float
    reject_H0: bool
    full_table: dict[str, dict[str, float]]

    def summary(self) -> None:
        t = self.full_table
        w = 66
        decision = "Reject H0" if self.reject_H0 else "Fail to reject H0"

        def _row(label: str, entry: dict[str, float], f_str: str = "") -> str:
            df = int(entry["df"])
            ss = entry["SS"]
            ms = entry.get("MS", float("nan"))
            f = entry.get("F", float("nan"))
            ms_s = f"{ms:>12.4f}" if ms == ms else f"{'':>12}"
            f_s = f"{f:>10.4f}" if f == f else f"{'':>10}"
            return f"  {label:<14} {df:>5} {ss:>12.4f} {ms_s} {f_s}"

        print("=" * w)
        print("  Two-Way ANOVA Table")
        print("=" * w)
        print(f"  {'Source':<14} {'df':>5} {'SS':>12} {'MS':>12} {'F':>10}")
        print("-" * w)
        print(_row("Factor A", t["A"], "f"))
        print(_row("Factor B", t["B"], "f"))
        print(_row("Interaction AB", t["AB"], "f"))
        print(_row("Within (Error)", t["within"], ""))
        print("-" * w)
        df_tot = int(t["total"]["df"])
        ss_tot = t["total"]["SS"]
        print(f"  {'Total':<14} {df_tot:>5} {ss_tot:>12.4f} {'':>12} {'':>10}")
        print("=" * w)
        print(f"  Testing:          {self.source.value}")
        print(f"  F statistic:      {self.F_statistic:.4f}")
        print(f"  F critical:       {self.F_critical:.4f}")
        print(f"  p-value:          {self.p_value:.4f}")
        print(f"  Decision:         {decision}")
        print("=" * w)

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt
        from scipy import stats

        df_den = int(self.full_table["within"]["df"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        ss_labels = ["SS_A", "SS_B", "SS_AB", "SS_E"]
        ss_values = [self.full_table["A"]["SS"], self.full_table["B"]["SS"],
                     self.full_table["AB"]["SS"], self.full_table["within"]["SS"]]
        colors = ["steelblue", "coral", "mediumpurple", "gray"]
        bars = ax1.bar(ss_labels, ss_values, color=colors, edgecolor="k", linewidth=0.5)
        ax1.set_ylabel("Sum of Squares")
        ax1.set_title("Sum of Squares Decomposition")
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, ss_values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax2 = axes[1]
        x_max = max(self.F_statistic * 1.5, self.F_critical * 2, stats.f.ppf(0.999, self.df, df_den))
        x = np.linspace(0.01, x_max, 500)
        pdf = stats.f.pdf(x, self.df, df_den)
        ax2.plot(x, pdf, color="steelblue", linewidth=2, label=f"F({self.df},{df_den})")
        ax2.fill_between(x, pdf, where=x >= self.F_critical, alpha=0.3, color="crimson", label="Rejection region")
        ax2.axvline(self.F_critical, color="crimson", linestyle="--", linewidth=1.2)
        ax2.axvline(self.F_statistic, color="darkgreen", linestyle="-", linewidth=2, label=f"F = {self.F_statistic:.4f}")
        ax2.set_xlabel("F")
        ax2.set_ylabel("Density")
        ax2.set_title(f"F-Test (Source: {self.source.value})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Two-Way ANOVA", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


@dataclass
class OrthogonalityResult:
    """Result of orthogonality check between two contrasts."""

    is_orthogonal: bool
    c1_is_contrast: bool
    c2_is_contrast: bool
    warning: str | None


@dataclass
class SimultaneousCIResult:
    """Simultaneous confidence intervals for linear combinations."""

    intervals: list[tuple[float, float]]
    method_used: CorrectionMethod
    point_estimates: np.ndarray
    half_widths: np.ndarray

    def summary(self) -> None:
        w = 60
        print("=" * w)
        print("  Simultaneous Confidence Intervals")
        print(f"  Method: {self.method_used.value}")
        print("=" * w)
        print(
            f"  {'Interval':<10} {'Point Est':>10} {'Half-Width':>12} {'Lower':>10} {'Upper':>10}"
        )
        print("-" * w)
        for i, (lo, hi) in enumerate(self.intervals):
            pe = float(self.point_estimates[i])
            hw = float(self.half_widths[i])
            print(f"  CI_{i + 1:<7} {pe:>10.4f} {hw:>12.4f} {lo:>10.4f} {hi:>10.4f}")
        print("=" * w)

    def plot(self) -> Figure:
        from statscore.plots import plot_simultaneous_ci

        return plot_simultaneous_ci(
            point_estimates=self.point_estimates,
            intervals=self.intervals,
            method=self.method_used.value,
            title="Simultaneous Confidence Intervals (ANOVA)",
        )


@dataclass
class SimultaneousTestResult:
    """Results of simultaneous hypothesis tests controlling FWER."""

    test_statistics: np.ndarray
    critical_values: np.ndarray
    p_values: np.ndarray
    reject: np.ndarray
    method_used: CorrectionMethod
    FWER: float

    def summary(self) -> None:
        w = 68
        print("=" * w)
        print(f"  Simultaneous Hypothesis Tests   (FWER = {self.FWER})")
        print(f"  Method: {self.method_used.value}")
        print("=" * w)
        print(f"  {'#':<5} {'|T|':>10} {'Critical':>10} {'p-value':>10} {'Reject?':>10}")
        print("-" * w)
        for i in range(len(self.test_statistics)):
            ts = float(self.test_statistics[i])
            cv = float(self.critical_values[i])
            pv = float(self.p_values[i])
            rej = "Yes" if bool(self.reject[i]) else "No"
            print(f"  {i + 1:<5} {ts:>10.4f} {cv:>10.4f} {pv:>10.4f} {rej:>10}")
        print("=" * w)

    def plot(self) -> Figure:
        from matplotlib import pyplot as plt

        m = len(self.test_statistics)
        fig, ax = plt.subplots(figsize=(8, max(4, m * 0.7)))
        y_pos = np.arange(m)
        labels = [f"Test {i+1}" for i in range(m)]

        colors = ["crimson" if bool(self.reject[i]) else "steelblue" for i in range(m)]
        ax.barh(y_pos, self.test_statistics, color=colors, edgecolor="k", linewidth=0.5, alpha=0.7)

        for i in range(m):
            ax.plot(float(self.critical_values[i]), i, marker="|", color="black", markersize=20, markeredgewidth=2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("|T|")
        method = self.method_used.value
        subtitle = f"Method: {method}" if method else ""
        ax.set_title(f"Simultaneous Hypothesis Tests\n{subtitle}" if subtitle else "Simultaneous Hypothesis Tests")
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="crimson", lw=6, alpha=0.7, label="Reject H0"),
                plt.Line2D([0], [0], color="steelblue", lw=6, alpha=0.7, label="Fail to reject"),
                plt.Line2D([0], [0], marker="|", color="black", linestyle="None", markersize=12, markeredgewidth=2, label="Critical value"),
            ],
        )
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        return fig


__all__ = [
    "ANOVA1PartitionResult",
    "ANOVA1TestResult",
    "ANOVA2PartitionResult",
    "ANOVA2MLEResult",
    "ANOVA2TestResult",
    "OrthogonalityResult",
    "SimultaneousCIResult",
    "SimultaneousTestResult",
]
