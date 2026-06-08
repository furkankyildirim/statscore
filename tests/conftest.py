"""Shared pytest fixtures for statscore tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_groups():
    """Three groups of observations for one-way ANOVA."""
    return [
        np.array([28, 23, 14, 27, 31], dtype=float),
        np.array([33, 36, 34, 29, 24], dtype=float),
        np.array([18, 21, 20, 22], dtype=float),
    ]


@pytest.fixture
def simple_xy():
    """Simple predictor/response pair for regression."""
    attend = np.array([1, 0.5, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.6, 0.1, 0, 0, 0.7, 0.8, 1])
    y = np.array([60, 65, 40, 70, 65, 70, 85, 70, 44, 20, 40, 30, 50, 77, 90], dtype=float)
    return attend, y


@pytest.fixture
def design_matrix(simple_xy):
    """Design matrix with intercept column for simple_xy."""
    attend, y = simple_xy
    X = np.column_stack([np.ones(len(attend)), attend])
    return X, y


@pytest.fixture
def normal_sample():
    """Small sample from a known N(10, 0.3²) distribution."""
    return np.array([10.2, 9.8, 10.1, 10.3, 9.9, 10.0, 10.4, 9.7])


@pytest.fixture
def fixtures_dir():
    """Path to the tests/fixtures directory."""
    return FIXTURES_DIR
