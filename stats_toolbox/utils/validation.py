"""Input validation helpers for stats_toolbox."""

from typing import Sequence

import numpy as np


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
    if K < 1:
        raise ValueError("Need at least 1 replicate per cell.")


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
