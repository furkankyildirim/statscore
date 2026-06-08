"""Two-way ANOVA: partitioning, MLE estimation, and hypothesis tests."""

from dataclasses import dataclass

import numpy as np

from statscore.utils.distributions import f_critical, f_pvalue
from statscore.utils.enums import TwoWayTestFactor
from statscore.utils.validation import validate_two_way_data


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
            df  = int(entry["df"])
            ss  = entry["SS"]
            ms  = entry.get("MS", float("nan"))
            f   = entry.get("F", float("nan"))
            ms_s = f"{ms:>12.4f}" if ms == ms else f"{'':>12}"
            f_s  = f"{f:>10.4f}"  if f == f else f"{'':>10}"
            return f"  {label:<14} {df:>5} {ss:>12.4f} {ms_s} {f_s}"

        print("=" * w)
        print("  Two-Way ANOVA Table")
        print("=" * w)
        print(f"  {'Source':<14} {'df':>5} {'SS':>12} {'MS':>12} {'F':>10}")
        print("-" * w)
        print(_row("Factor A",      t["A"],      "f"))
        print(_row("Factor B",      t["B"],      "f"))
        print(_row("Interaction AB", t["AB"],    "f"))
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


def ANOVA2_partition_TSS(data: np.ndarray) -> ANOVA2PartitionResult:
    """Partition total sum of squares in a two-way ANOVA layout.

    Parameters
    ----------
    data : array-like of shape (I, J, K)
        X_{i,j,k} for i=1,...,I, j=1,...,J, k=1,...,K.

    Returns
    -------
    ANOVA2PartitionResult with SS_total, SS_A, SS_B, SS_AB, SS_E.
    """
    data = np.asarray(data, dtype=float)
    validate_two_way_data(data)
    I, J, K = data.shape

    X_bar: float = float(data.mean())
    X_bar_i: np.ndarray = data.mean(axis=(1, 2))
    X_bar_j: np.ndarray = data.mean(axis=(0, 2))
    X_bar_ij: np.ndarray = data.mean(axis=2)

    SS_A: float = float(J * K * np.sum((X_bar_i - X_bar) ** 2))
    SS_B: float = float(I * K * np.sum((X_bar_j - X_bar) ** 2))
    SS_AB: float = float(K * np.sum(
        (X_bar_ij - X_bar_i[:, None] - X_bar_j[None, :] + X_bar) ** 2
    ))
    SS_E: float = float(np.sum((data - X_bar_ij[:, :, None]) ** 2))
    SS_total: float = float(np.sum((data - X_bar) ** 2))

    return ANOVA2PartitionResult(
        SS_total=SS_total,
        SS_A=SS_A,
        SS_B=SS_B,
        SS_AB=SS_AB,
        SS_E=SS_E,
    )


def ANOVA2_MLE(data: np.ndarray) -> ANOVA2MLEResult:
    """Compute MLE for two-way ANOVA parameters mu, a_i, b_j, delta_ij.

    Parameters
    ----------
    data : array-like of shape (I, J, K)

    Returns
    -------
    ANOVA2MLEResult with mu_hat, a_hat, b_hat, delta_hat.
    """
    data = np.asarray(data, dtype=float)
    validate_two_way_data(data)

    X_bar: float = float(data.mean())
    X_bar_i: np.ndarray = data.mean(axis=(1, 2))
    X_bar_j: np.ndarray = data.mean(axis=(0, 2))
    X_bar_ij: np.ndarray = data.mean(axis=2)

    mu_hat: float = X_bar
    a_hat: np.ndarray = X_bar_i - X_bar
    b_hat: np.ndarray = X_bar_j - X_bar
    delta_hat: np.ndarray = X_bar_ij - X_bar_i[:, None] - X_bar_j[None, :] + X_bar

    return ANOVA2MLEResult(mu=mu_hat, a=a_hat, b=b_hat, delta=delta_hat)


def ANOVA2_test_equality(
    data: np.ndarray, alpha: float = 0.05, test: TwoWayTestFactor = TwoWayTestFactor.A
) -> ANOVA2TestResult:
    """Perform one of the three basic tests in two-way ANOVA.

    Parameters
    ----------
    data : array-like of shape (I, J, K)
    alpha : float
        Significance level.
    test : TwoWayTestFactor
        TwoWayTestFactor.A for H0: a1=...=aI=0,
        TwoWayTestFactor.B for H0: b1=...=bJ=0,
        TwoWayTestFactor.AB for H0: all delta_ij=0.

    Returns
    -------
    ANOVA2TestResult with ANOVA table, F-statistic, critical value, decision.
    """
    if isinstance(test, str):
        test = TwoWayTestFactor(test.upper())

    data = np.asarray(data, dtype=float)
    validate_two_way_data(data)
    I, J, K = data.shape

    partition: ANOVA2PartitionResult = ANOVA2_partition_TSS(data)

    df_A: int = I - 1
    df_B: int = J - 1
    df_AB: int = (I - 1) * (J - 1)
    df_E: int = I * J * (K - 1)
    df_total: int = I * J * K - 1

    MS_A: float = partition.SS_A / df_A
    MS_B: float = partition.SS_B / df_B
    MS_AB: float = partition.SS_AB / df_AB
    MS_E: float = partition.SS_E / df_E

    full_table: dict[str, dict[str, float]] = {
        "A": {"df": df_A, "SS": partition.SS_A, "MS": MS_A, "F": MS_A / MS_E},
        "B": {"df": df_B, "SS": partition.SS_B, "MS": MS_B, "F": MS_B / MS_E},
        "AB": {"df": df_AB, "SS": partition.SS_AB, "MS": MS_AB, "F": MS_AB / MS_E},
        "within": {"df": df_E, "SS": partition.SS_E, "MS": MS_E},
        "total": {"df": df_total, "SS": partition.SS_total},
    }

    if test == TwoWayTestFactor.A:
        F_stat: float = MS_A / MS_E
        df_num: int = df_A
    elif test == TwoWayTestFactor.B:
        F_stat = MS_B / MS_E
        df_num = df_B
    elif test == TwoWayTestFactor.AB:
        F_stat = MS_AB / MS_E
        df_num = df_AB
    else:
        raise ValueError(f"Invalid test factor: {test}")

    F_crit: float = f_critical(alpha, df_num, df_E)
    p_val: float = f_pvalue(F_stat, df_num, df_E)

    return ANOVA2TestResult(
        source=test,
        df=df_num,
        SS=full_table[test.value]["SS"],
        MS=full_table[test.value]["MS"],
        F_statistic=F_stat,
        F_critical=F_crit,
        p_value=p_val,
        reject_H0=(F_stat > F_crit),
        full_table=full_table,
    )
