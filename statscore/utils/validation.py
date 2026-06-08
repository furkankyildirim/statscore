"""Input validation helpers for statscore."""

from collections.abc import Sequence

import numpy as np

from statscore.utils.enums import AlternativeHypothesis


def validate_positive(value: float, name: str) -> None:
    """Raise ValueError if value is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def validate_non_negative(value: float, name: str) -> None:
    """Raise ValueError if value is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def validate_1d_sample(x: np.ndarray, name: str = "x", min_obs: int = 2) -> None:
    """Raise ValueError if x is not a 1-D array with at least min_obs observations."""
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    if len(x) < min_obs:
        raise ValueError(f"{name} must have at least {min_obs} observations.")


def validate_alternative(alternative: AlternativeHypothesis) -> None:
    """Raise TypeError if alternative is not an AlternativeHypothesis enum member."""
    if not isinstance(alternative, AlternativeHypothesis):
        raise TypeError(
            f"alternative must be an AlternativeHypothesis enum member, "
            f"got {type(alternative).__name__!r}."
        )


def validate_design_matrix(X: np.ndarray) -> None:
    """Validate that X is a full-rank n x (k+1) design matrix with n > k+1."""
    if X.ndim != 2:
        raise ValueError("Design matrix X must be 2-dimensional.")
    n, p = X.shape
    if n <= p:
        raise ValueError(
            f"Design matrix must have more rows than columns (n={n}, k+1={p})."
        )
    rank: int = int(np.linalg.matrix_rank(X))
    if rank < p:
        raise ValueError(
            f"Design matrix must have full column rank. Rank={rank}, expected={p}."
        )


def validate_data_groups(data: Sequence[np.ndarray]) -> None:
    """Validate one-way ANOVA data: list of arrays, each with at least 1 obs."""
    if not isinstance(data, (list, tuple)):
        raise TypeError("Data must be a list of arrays (one per group).")
    if len(data) < 2:
        raise ValueError("Need at least 2 groups for ANOVA.")
    for i, group in enumerate(data):
        arr: np.ndarray = np.asarray(group, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"Group {i} must be a 1-D array.")
        if len(arr) < 1:
            raise ValueError(f"Group {i} must have at least 1 observation.")


def validate_two_way_data(data: np.ndarray) -> None:
    """Validate two-way ANOVA data: 3-D array of shape (I, J, K)."""
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("Two-way ANOVA data must be a 3-D array of shape (I, J, K).")
    I, J, K = data.shape
    if I < 2 or J < 2:
        raise ValueError("Need at least 2 levels for each factor.")
    if K < 2:
        raise ValueError(
            "Two-way ANOVA with replication requires at least 2 replicates per cell "
            "(K >= 2). With K=1 the error degrees of freedom are zero and MS_E cannot "
            "be estimated."
        )


def validate_contrast_matrix(C: np.ndarray, I: int) -> None:
    """Validate that C is an m x I matrix."""
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2:
        raise ValueError("C must be a 2-D matrix.")
    if C.shape[1] != I:
        raise ValueError(
            f"C must have {I} columns (number of groups), got {C.shape[1]}."
        )


def validate_C_matrix(C: np.ndarray, X: np.ndarray) -> None:
    """Validate C matrix for regression: r x (k+1), full row rank."""
    _, p = X.shape
    if C.ndim != 2:
        raise ValueError("C must be a 2-D matrix.")
    if C.shape[1] != p:
        raise ValueError(
            f"C must have {p} columns matching design matrix, got {C.shape[1]}."
        )
    r: int = C.shape[0]
    rank: int = int(np.linalg.matrix_rank(C))
    if rank < r:
        raise ValueError(
            f"C must have full row rank. Rank={rank}, expected={r}."
        )
