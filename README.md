# stats-toolbox

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-58%20passing-brightgreen.svg)]()

A production-quality Python library for **Analysis of Variance (ANOVA)** and **Multiple Linear Regression** with structured output, simultaneous inference, and family-wise error rate (FWER) control.

## Features

- **One-Way & Two-Way ANOVA** — sum-of-squares decomposition, F-tests, MLE estimation
- **Multiple Comparison Procedures** — Bonferroni, Šidák, Scheffé, and Tukey methods with automatic "best" selection
- **OLS Multiple Linear Regression** — matrix-based estimation, TSS partition, R²
- **Simultaneous Inference** — confidence intervals, confidence regions (ellipsoids), general hypothesis tests
- **Prediction Intervals** — Scheffé and Bonferroni simultaneous prediction CIs
- **Structured Output** — all functions return typed dataclass objects, not raw tuples
- **Minimal Dependencies** — only NumPy and SciPy required

## Installation

```bash
# From PyPI (once published):
pip install stats-toolbox

# From source (development):
git clone https://github.com/furkankyildirim/stats-toolbox.git
cd stats-toolbox
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.9, NumPy >= 1.21, SciPy >= 1.7

## Quick Start

### One-Way ANOVA

```python
import numpy as np
from stats_toolbox import ANOVA1_partition_TSS, ANOVA1_test_equality

data = [
    np.array([28, 23, 14, 27, 31]),
    np.array([33, 36, 34, 29, 24]),
    np.array([18, 21, 20, 22]),
]

partition = ANOVA1_partition_TSS(data)
print(f"SS_total = {partition.SS_total}")
print(f"SS_within = {partition.SS_within}")
print(f"SS_between = {partition.SS_between}")

result = ANOVA1_test_equality(data, alpha=0.05)
print(f"F = {result.F_statistic:.4f}, p = {result.p_value:.4f}")
print(f"Reject H0: {result.reject_H0}")
```

### Multiple Comparisons with FWER Control

```python
from stats_toolbox import ANOVA1_CI_linear_combs, ANOVA1_test_linear_combs

C = np.array([[1, -1, 0], [0, 1, -1], [1, 0, -1]])
d = np.zeros(3)

# Automatic method selection (picks narrowest valid intervals)
ci_result = ANOVA1_CI_linear_combs(data, alpha=0.05, C=C, method="best")
for i, (lo, hi) in enumerate(ci_result.intervals):
    print(f"CI_{i+1}: [{lo:.2f}, {hi:.2f}]")
```

### Multiple Linear Regression

```python
from stats_toolbox import (
    Mult_LR_Least_squares, Mult_norm_LR_test_general, Mult_norm_LR_pred_CI
)

X = np.column_stack([np.ones(n), x1, x2])

ols = Mult_LR_Least_squares(X, y)
print(f"beta_hat = {ols.beta_hat}")
print(f"Se^2 = {ols.sigma2_unbiased:.4f}")

# General hypothesis test: H0: C*beta = c0
C = np.array([[0, 1, -1]])
c0 = np.array([0.0])
result = Mult_norm_LR_test_general(X, y, C, c0, alpha=0.05)
print(f"F = {result.test_statistic:.4f}, p = {result.p_value:.4f}")

# Simultaneous prediction intervals
D = np.array([[1, 0.5, 1.0], [1, 1.0, 0.0]])
pred = Mult_norm_LR_pred_CI(X, y, D, alpha=0.05, method="best")
for i, (lo, hi) in enumerate(pred.intervals):
    print(f"Prediction {i+1}: {pred.point_estimates[i]:.2f} [{lo:.2f}, {hi:.2f}]")
```

## Package Structure

```
stats_toolbox/
├── __init__.py              # Top-level exports (20 public functions)
├── anova/
│   ├── one_way.py           # ANOVA1_partition_TSS, ANOVA1_test_equality
│   ├── two_way.py           # ANOVA2_partition_TSS, ANOVA2_MLE, ANOVA2_test_equality
│   └── multiple_tests.py    # Contrasts, orthogonality, corrections, CI, tests
├── regression/
│   ├── least_squares.py     # Mult_LR_Least_squares, Mult_LR_partition_TSS
│   ├── inference.py         # Simultaneous CI, CR, general/component/linear tests
│   └── prediction.py        # Mult_norm_LR_pred_CI
├── utils/
│   ├── distributions.py     # Critical values and p-values (F, t, chi2, q)
│   └── validation.py        # Input validation helpers
├── examples/
│   └── demo.py              # Full demonstration of all 20 functions
└── tests/                   # 58 unit tests (pytest)
```

## API Reference

### ANOVA Functions

| Function | Description |
|----------|-------------|
| `ANOVA1_partition_TSS(data)` | Partition SS_total into SS_within and SS_between |
| `ANOVA1_test_equality(data, alpha)` | F-test for equality of group means |
| `ANOVA1_is_contrast(c)` | Check if coefficients form a contrast |
| `ANOVA1_is_orthogonal(n, c1, c2)` | Check orthogonality of two contrasts |
| `Bonferroni_correction(alpha, m)` | Bonferroni-corrected significance level |
| `Sidak_correction(alpha, m)` | Šidák-corrected significance level |
| `ANOVA1_CI_linear_combs(data, alpha, C, method)` | Simultaneous CIs for linear combinations |
| `ANOVA1_test_linear_combs(data, alpha, C, d, method)` | Test multiple linear combinations (FWER) |
| `ANOVA2_partition_TSS(data)` | Two-way ANOVA sum of squares partition |
| `ANOVA2_MLE(data)` | MLE for μ, α_i, β_j, δ_{ij} |
| `ANOVA2_test_equality(data, alpha, test)` | Two-way ANOVA F-tests ("A", "B", "AB") |

### Regression Functions

| Function | Description |
|----------|-------------|
| `Mult_LR_Least_squares(X, y)` | OLS estimation: β̂, σ² MLE & unbiased |
| `Mult_LR_partition_TSS(X, y)` | TSS = RegSS + RSS decomposition |
| `Mult_norm_LR_simul_CI(X, y, alpha)` | Simultaneous CIs for all β_i |
| `Mult_norm_LR_CR(X, y, C, alpha)` | Confidence region (ellipsoid) for Cβ |
| `Mult_norm_LR_is_in_CR(X, y, C, c0, alpha)` | Test if c₀ is inside the CR |
| `Mult_norm_LR_test_general(X, y, C, c0, alpha)` | General test H₀: Cβ = c₀ |
| `Mult_norm_LR_test_comp(X, y, alpha, components)` | Test H₀: β_{j₁}=...=β_{jᵣ}=0 |
| `Mult_norm_LR_test_linear_reg(X, y, alpha)` | Test existence of linear regression |
| `Mult_norm_LR_pred_CI(X, y, D, alpha, method)` | Simultaneous prediction CIs |

## Mathematical Background

### One-Way ANOVA

Model: X_{ij} = μ + α_i + ε_{ij}, where ε_{ij} ~ N(0, σ²).

**Sum of squares decomposition:**
- SS_total = Σ_i Σ_j (X_{ij} - X̄)²
- SS_within = Σ_i Σ_j (X_{ij} - X̄_i)²
- SS_between = Σ_i n_i (X̄_i - X̄)²
- Identity: SS_total = SS_within + SS_between

**F-test:** F = [SS_b/(I-1)] / [SS_w/(n-I)] ~ F_{I-1, n-I} under H₀.

### Two-Way ANOVA

Model: X_{ijk} = μ + α_i + β_j + δ_{ij} + ε_{ijk}.

**Decomposition:** SS_total = SS_A + SS_B + SS_AB + SS_E

### Multiple Linear Regression

Model: Y = Xβ + ε, where ε ~ N(0, σ²I).

**OLS estimator:** β̂ = (X^T X)^{-1} X^T Y

**TSS partition:** TSS = RegSS + RSS

**General test:** H₀: Cβ = c₀, using F = [(Cβ̂ - c₀)^T [C(X^TX)^{-1}C^T]^{-1} (Cβ̂ - c₀)] / (r · S_e²)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check .

# Type checking
mypy stats_toolbox/
```

## Running the Demo

```bash
python examples/demo.py
```

This exercises all 20 functions with representative sample data.

## License

MIT
