"""Multiple testing procedures for one-way ANOVA:
contrasts, orthogonality, corrections, simultaneous CIs and tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from statscore.anova.one_way import ANOVA1_partition_TSS
from statscore.utils.distributions import (
    f_critical,
    studentized_range_critical,
    t_critical,
    t_pvalue,
)
from statscore.utils.enums import CorrectionMethod
from statscore.utils.validation import validate_contrast_matrix, validate_data_groups


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


def ANOVA1_is_contrast(c: np.ndarray) -> bool:
    """Check whether coefficients c define a contrast (sum to zero).

    Parameters
    ----------
    c : array-like
        Coefficients (c_1, ..., c_I).

    Returns
    -------
    True if sum(c_i) = 0 (within numerical tolerance).
    """
    c = np.asarray(c, dtype=float)
    return bool(np.abs(c.sum()) < 1e-10)


def ANOVA1_is_orthogonal(n: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> OrthogonalityResult:
    """Check whether two contrasts are orthogonal given group sizes.

    Orthogonality condition: sum(c1_i * c2_i / n_i) = 0.

    Parameters
    ----------
    n : array-like
        Group sizes (n_1, ..., n_I).
    c1 : array-like
        First set of coefficients.
    c2 : array-like
        Second set of coefficients.

    Returns
    -------
    OrthogonalityResult with flags and optional warning.
    """
    n = np.asarray(n, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)

    c1_is_contrast: bool = ANOVA1_is_contrast(c1)
    c2_is_contrast: bool = ANOVA1_is_contrast(c2)

    warning: str | None = None
    if not c1_is_contrast or not c2_is_contrast:
        parts: list[str] = []
        if not c1_is_contrast:
            parts.append("c1")
        if not c2_is_contrast:
            parts.append("c2")
        warning = (
            f"Warning: {' and '.join(parts)} is not a contrast (coefficients do not sum to 0)."
        )

    inner_product: float = float(np.sum(c1 * c2 / n))
    is_orthogonal: bool = bool(np.abs(inner_product) < 1e-10)

    return OrthogonalityResult(
        is_orthogonal=is_orthogonal,
        c1_is_contrast=c1_is_contrast,
        c2_is_contrast=c2_is_contrast,
        warning=warning,
    )


def Bonferroni_correction(alpha: float, m: int) -> float:
    """Compute Bonferroni-corrected significance level.

    Parameters
    ----------
    alpha : float
        Family-wise error rate (FWER).
    m : int
        Number of tests.

    Returns
    -------
    Corrected significance level alpha/m for each individual test.
    """
    if m < 1:
        raise ValueError("Number of tests m must be >= 1.")
    return alpha / m


def Sidak_correction(alpha: float, m: int) -> float:
    """Compute Sidak-corrected significance level.

    Parameters
    ----------
    alpha : float
        Family-wise error rate (FWER).
    m : int
        Number of tests.

    Returns
    -------
    Corrected significance level 1 - (1-alpha)^(1/m) for each test.
    """
    if m < 1:
        raise ValueError("Number of tests m must be >= 1.")
    return float(1.0 - (1.0 - alpha) ** (1.0 / m))


def _all_are_contrasts(C: np.ndarray) -> bool:
    """Check if all rows of C are contrasts."""
    return all(np.abs(C[j].sum()) < 1e-10 for j in range(C.shape[0]))


def _all_are_pairwise(C: np.ndarray) -> bool:
    """Check if all rows of C define pairwise comparisons (mu_i - mu_j)."""
    for row in C:
        nonzero: np.ndarray = row[row != 0]
        if len(nonzero) != 2:
            return False
        if not (np.isclose(nonzero[0], -nonzero[1])):
            return False
    return True


def _check_orthogonal_set(C: np.ndarray, n: np.ndarray) -> bool:
    """Check if all pairs of rows in C are orthogonal contrasts."""
    m: int = C.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            if np.abs(np.sum(C[i] * C[j] / n)) > 1e-10:
                return False
    return True


def _scheffe_df(I: int, all_contrasts: bool) -> int:
    """Return Scheffe degrees of freedom: I-1 for contrasts, I for general linear combinations."""
    return I - 1 if all_contrasts else I


def _compute_ci_half_width(
    method: CorrectionMethod,
    c_row: np.ndarray,
    n: np.ndarray,
    SS_w: float,
    df_w: int,
    I: int,
    m: int,
    alpha: float,
    n_equal: int | None,
    scheffe_numerator_df: int,
) -> float:
    """Compute the half-width for a single linear combination's CI."""
    se_term: float = float(np.sqrt((SS_w / df_w) * np.sum(c_row**2 / n)))

    if method == CorrectionMethod.SCHEFFE:
        M: float = float(
            np.sqrt(scheffe_numerator_df * f_critical(alpha, scheffe_numerator_df, df_w))
        )
        return M * se_term
    elif method == CorrectionMethod.BONFERRONI:
        t_crit: float = t_critical(alpha / (2 * m), df_w)
        return t_crit * se_term
    elif method == CorrectionMethod.SIDAK:
        alpha_corr: float = Sidak_correction(alpha, m)
        t_crit = t_critical(alpha_corr / 2, df_w)
        return t_crit * se_term
    elif method == CorrectionMethod.TUKEY:
        if n_equal is None:
            raise ValueError("Tukey method requires equal group sizes.")
        q_crit: float = studentized_range_critical(alpha, I, df_w)
        return float((q_crit / np.sqrt(2)) * np.sqrt((SS_w / df_w) * (2.0 / n_equal)))
    else:
        raise ValueError(f"Unknown method: {method}")


def _validate_method_for_data(
    method: CorrectionMethod,
    C: np.ndarray,
    n: np.ndarray,
    all_contrasts: bool,
    all_pairwise: bool,
) -> CorrectionMethod | list[CorrectionMethod]:
    """Validate and resolve method choice."""
    n_equal: int | None = int(n[0]) if np.all(n == n[0]) else None

    if method == CorrectionMethod.TUKEY:
        if not all_pairwise:
            raise ValueError("Tukey's method is valid only for pairwise comparisons.")
        if n_equal is None:
            raise ValueError("Tukey's method requires equal group sizes.")
        return CorrectionMethod.TUKEY
    elif method == CorrectionMethod.SCHEFFE:
        return CorrectionMethod.SCHEFFE
    elif method == CorrectionMethod.BONFERRONI:
        return CorrectionMethod.BONFERRONI
    elif method == CorrectionMethod.SIDAK:
        all_orthogonal: bool = all_contrasts and _check_orthogonal_set(C, n)
        if not all_orthogonal:
            raise ValueError(
                "Sidak correction requires orthogonal contrasts for guaranteed FWER control. "
                "Use Bonferroni for non-orthogonal combinations."
            )
        return CorrectionMethod.SIDAK
    elif method == CorrectionMethod.BEST:
        return _choose_best_method(C, n, all_contrasts, all_pairwise)
    else:
        raise ValueError(f"Unknown method '{method}'. Use CorrectionMethod enum.")


def _choose_best_method(
    C: np.ndarray, n: np.ndarray, all_contrasts: bool, all_pairwise: bool
) -> list[CorrectionMethod]:
    """Return candidate methods for narrowest CI selection."""
    candidates: list[CorrectionMethod] = []
    n_equal: int | None = int(n[0]) if np.all(n == n[0]) else None

    candidates.append(CorrectionMethod.SCHEFFE)
    candidates.append(CorrectionMethod.BONFERRONI)
    if all_contrasts and _check_orthogonal_set(C, n):
        candidates.append(CorrectionMethod.SIDAK)
    if all_pairwise and n_equal is not None:
        candidates.append(CorrectionMethod.TUKEY)

    return candidates


def ANOVA1_CI_linear_combs(
    data: Sequence[np.ndarray],
    alpha: float,
    C: np.ndarray,
    method: CorrectionMethod = CorrectionMethod.BEST,
) -> SimultaneousCIResult:
    """Compute simultaneous confidence intervals for linear combinations of group means.

    Parameters
    ----------
    data : Sequence of array-like
        Data for each group.
    alpha : float
        Significance level.
    C : array-like
        m x I matrix where each row defines a linear combination.
    method : CorrectionMethod
        One of CorrectionMethod.SCHEFFE, .TUKEY, .BONFERRONI, .SIDAK, .BEST.

    Returns
    -------
    SimultaneousCIResult with intervals holding simultaneously at 1-alpha.
    """
    if isinstance(method, str):
        method = CorrectionMethod(method) if method != "best" else CorrectionMethod.BEST

    validate_data_groups(data)
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape(1, -1)

    I: int = len(data)
    validate_contrast_matrix(C, I)
    m: int = C.shape[0]

    partition = ANOVA1_partition_TSS(data)
    n: np.ndarray = partition.group_sizes
    group_means: np.ndarray = partition.group_means
    SS_w: float = partition.SS_within
    n_total: int = int(n.sum())
    df_w: int = n_total - I
    n_equal: int | None = int(n[0]) if np.all(n == n[0]) else None

    all_contrasts: bool = _all_are_contrasts(C)
    all_pairwise: bool = _all_are_pairwise(C)
    s_df: int = _scheffe_df(I, all_contrasts)

    resolved = _validate_method_for_data(method, C, n, all_contrasts, all_pairwise)

    if isinstance(resolved, list):
        best_method: CorrectionMethod | None = None
        best_half_widths: np.ndarray | None = None
        for candidate in resolved:
            hws: np.ndarray = np.array(
                [
                    _compute_ci_half_width(
                        candidate, C[j], n, SS_w, df_w, I, m, alpha, n_equal, s_df
                    )
                    for j in range(m)
                ]
            )
            total: float = float(hws.sum())
            if best_half_widths is None or total < float(best_half_widths.sum()):
                best_half_widths = hws
                best_method = candidate
        final_method: CorrectionMethod = best_method  # type: ignore[assignment]
        half_widths: np.ndarray = best_half_widths  # type: ignore[assignment]
    else:
        final_method = resolved
        half_widths = np.array(
            [
                _compute_ci_half_width(resolved, C[j], n, SS_w, df_w, I, m, alpha, n_equal, s_df)
                for j in range(m)
            ]
        )

    point_estimates: np.ndarray = C @ group_means
    intervals: list[tuple[float, float]] = [
        (float(pe - hw), float(pe + hw)) for pe, hw in zip(point_estimates, half_widths)
    ]

    return SimultaneousCIResult(
        intervals=intervals,
        method_used=final_method,
        point_estimates=point_estimates,
        half_widths=half_widths,
    )


def ANOVA1_test_linear_combs(
    data: Sequence[np.ndarray],
    alpha: float,
    C: np.ndarray,
    d: np.ndarray,
    method: CorrectionMethod = CorrectionMethod.BEST,
) -> SimultaneousTestResult:
    """Test multiple linear combinations of group means controlling FWER.

    H0_j: c_{j,1}*mu_1 + ... + c_{j,I}*mu_I = d_j,  j=1,...,m.

    Parameters
    ----------
    data : Sequence of array-like
        Data for each group.
    alpha : float
        Family-wise error rate.
    C : array-like
        m x I matrix of coefficients.
    d : array-like
        m x 1 vector of hypothesized values.
    method : CorrectionMethod
        One of CorrectionMethod.SCHEFFE, .TUKEY, .BONFERRONI, .SIDAK, .BEST.

    Returns
    -------
    SimultaneousTestResult with test outcomes, p-values, and decisions.
    """
    if isinstance(method, str):
        method = CorrectionMethod(method) if method != "best" else CorrectionMethod.BEST

    validate_data_groups(data)
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    d = np.asarray(d, dtype=float).ravel()

    I: int = len(data)
    validate_contrast_matrix(C, I)
    m: int = C.shape[0]
    if len(d) != m:
        raise ValueError(f"d must have {m} elements, got {len(d)}.")

    partition = ANOVA1_partition_TSS(data)
    n: np.ndarray = partition.group_sizes
    group_means: np.ndarray = partition.group_means
    SS_w: float = partition.SS_within
    n_total: int = int(n.sum())
    df_w: int = n_total - I
    n_equal: int | None = int(n[0]) if np.all(n == n[0]) else None

    all_contrasts: bool = _all_are_contrasts(C)
    all_pairwise: bool = _all_are_pairwise(C)
    s_df: int = _scheffe_df(I, all_contrasts)

    resolved = _validate_method_for_data(method, C, n, all_contrasts, all_pairwise)

    if isinstance(resolved, list):
        best_method: CorrectionMethod | None = None
        best_crit: np.ndarray | None = None
        for candidate in resolved:
            hws: np.ndarray = np.array(
                [
                    _compute_ci_half_width(
                        candidate, C[j], n, SS_w, df_w, I, m, alpha, n_equal, s_df
                    )
                    for j in range(m)
                ]
            )
            if best_crit is None or float(hws.sum()) < float(best_crit.sum()):
                best_crit = hws
                best_method = candidate
        final_method: CorrectionMethod = best_method  # type: ignore[assignment]
    else:
        final_method = resolved

    point_estimates: np.ndarray = C @ group_means
    se_terms: np.ndarray = np.array(
        [np.sqrt((SS_w / df_w) * np.sum(C[j] ** 2 / n)) for j in range(m)]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        test_statistics: np.ndarray = np.abs((point_estimates - d) / se_terms)
    test_statistics = np.where(np.isfinite(test_statistics), test_statistics, 0.0)

    critical_values: np.ndarray = np.zeros(m)
    p_values: np.ndarray = np.zeros(m)

    if final_method == CorrectionMethod.SCHEFFE:
        M: float = float(np.sqrt(s_df * f_critical(alpha, s_df, df_w)))
        critical_values[:] = M
        from scipy import stats as _st

        for j in range(m):
            F_obs: float = float(test_statistics[j] ** 2 / s_df)
            p_values[j] = 1.0 - _st.f.cdf(F_obs, s_df, df_w)
    elif final_method == CorrectionMethod.BONFERRONI:
        t_crit: float = t_critical(alpha / (2 * m), df_w)
        critical_values[:] = t_crit
        for j in range(m):
            raw_p: float = t_pvalue(test_statistics[j], df_w)
            p_values[j] = min(raw_p * m, 1.0)
    elif final_method == CorrectionMethod.SIDAK:
        alpha_corr: float = Sidak_correction(alpha, m)
        t_crit = t_critical(alpha_corr / 2, df_w)
        critical_values[:] = t_crit
        for j in range(m):
            raw_p = t_pvalue(test_statistics[j], df_w)
            p_values[j] = 1.0 - (1.0 - raw_p) ** m
    elif final_method == CorrectionMethod.TUKEY:
        q_crit: float = studentized_range_critical(alpha, I, df_w)
        critical_values[:] = q_crit / np.sqrt(2)
        from scipy import stats as _st

        for j in range(m):
            q_obs: float = float(test_statistics[j] * np.sqrt(2))
            p_values[j] = 1.0 - _st.studentized_range.cdf(q_obs, I, df_w)

    reject: np.ndarray = test_statistics > critical_values

    return SimultaneousTestResult(
        test_statistics=test_statistics,
        critical_values=critical_values,
        p_values=p_values,
        reject=reject,
        method_used=final_method,
        FWER=alpha,
    )
