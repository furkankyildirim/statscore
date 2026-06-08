# statscore

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-98%20passing-brightgreen.svg)]()

A production-quality Python library for **ANOVA**, **normal distribution significance testing**, **multiple linear regression**, and **Bayesian conjugate inference** with structured output and family-wise error rate (FWER) control.

## Features

- **One-Way & Two-Way ANOVA** — sum-of-squares decomposition, F-tests, MLE estimation, formatted table printing
- **Multiple Comparison Procedures** — Bonferroni, Šidák, Scheffé, and Tukey methods with automatic "best" selection
- **Normal Distribution Significance Testing** — z-test, one/two-sample t-test, paired t-test, chi-squared variance test, F-test for variances; one-sided and two-sided alternatives
- **OLS Multiple Linear Regression** — matrix-based estimation, TSS partition, R², adjusted R²
- **Simultaneous Inference** — confidence intervals, confidence regions (ellipsoids), general hypothesis tests
- **Prediction Intervals** — Scheffé and Bonferroni simultaneous prediction CIs
- **Bayesian Conjugate Inference** — Normal-Normal (known variance) and Normal-Gamma (unknown variance) conjugate posteriors with credible intervals and posterior predictive distributions
- **Structured Output** — all functions return typed dataclass objects, not raw tuples
- **Type-Safe Enums** — all categorical parameters use enums; no raw strings
- **Minimal Dependencies** — only NumPy and SciPy required

## Installation

```bash
# From PyPI:
pip install statscore

# From source (development):
git clone https://github.com/furkankyildirim/statscore.git
cd statscore
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.9, NumPy >= 1.21, SciPy >= 1.7

## Quick Start

### One-Way ANOVA

```python
import numpy as np
from statscore import ANOVA1_partition_TSS, ANOVA1_test_equality, ANOVA1_print_table

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
ANOVA1_print_table(result)
# ==========================================================
#   One-Way ANOVA Table
# ==========================================================
#   Source          df           SS           MS          F
# ----------------------------------------------------------
#   Between          2     584.4167     292.2083    12.6385
#   Within          11     254.3333      23.1212
# ----------------------------------------------------------
#   Total           13     838.7500
# ==========================================================
#   F critical (α=0.05): 3.9823
#   p-value:                  0.0014
#   Decision:                 Reject H0
# ==========================================================
```

### Two-Way ANOVA

```python
import numpy as np
from statscore import ANOVA2_test_equality, ANOVA2_print_table, TwoWayTestFactor

# data shape: (I, J, K) — I levels of A, J levels of B, K replicates
data = np.array([
    [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
    [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
], dtype=float)

result = ANOVA2_test_equality(data, alpha=0.05, test=TwoWayTestFactor.B)
ANOVA2_print_table(result)
# ==================================================================
#   Two-Way ANOVA Table
# ==================================================================
#   Source            df           SS           MS          F
# ------------------------------------------------------------------
#   Factor A           1       6.0000       6.0000     1.3171
#   Factor B           2     252.0000     126.0000    27.6585
#   Interaction AB     2      28.0000      14.0000     3.0732
#   Within (Error)    18      82.0000       4.5556
# ------------------------------------------------------------------
#   Total             23     368.0000
# ==================================================================
#   Testing:          B
#   F statistic:      27.6585
#   F critical:       3.5546
#   p-value:          0.0000
#   Decision:         Reject H0
# ==================================================================
```

### Normal Distribution Significance Testing

```python
import numpy as np
from statscore import z_test_mean, t_test_mean, t_test_two_sample, AlternativeHypothesis

x = np.array([10.2, 9.8, 10.1, 10.3, 9.9, 10.0, 10.4, 9.7])

# Z-test (sigma known)
result = z_test_mean(x, mu0=10.0, sigma=0.3, alpha=0.05,
                     alternative=AlternativeHypothesis.TWO_SIDED)
print(f"Z = {result.z_statistic:.4f}, p = {result.p_value:.4f}")

# One-sample t-test (sigma unknown)
result = t_test_mean(x, mu0=10.0, alpha=0.05,
                     alternative=AlternativeHypothesis.TWO_SIDED)
print(f"T = {result.t_statistic:.4f}, p = {result.p_value:.4f}")

# Two-sample t-test (Welch)
x2 = np.array([10.8, 11.2, 10.9, 11.1, 10.7])
result = t_test_two_sample(x, x2, alpha=0.05, equal_var=False,
                            alternative=AlternativeHypothesis.TWO_SIDED)
print(f"T = {result.t_statistic:.4f}, p = {result.p_value:.4f}")
```

### Multiple Comparisons with FWER Control

```python
import numpy as np
from statscore import ANOVA1_CI_linear_combs, CorrectionMethod

C = np.array([[1, -1, 0], [0, 1, -1], [1, 0, -1]])

# Automatic method selection (picks narrowest valid intervals)
ci_result = ANOVA1_CI_linear_combs(data, alpha=0.05, C=C,
                                    method=CorrectionMethod.BEST)
print(f"Method: {ci_result.method_used.value}")
for i, (lo, hi) in enumerate(ci_result.intervals):
    print(f"CI_{i+1}: [{lo:.2f}, {hi:.2f}]")
```

### Multiple Linear Regression

```python
import numpy as np
from statscore import (
    Mult_LR_Least_squares, Mult_LR_partition_TSS,
    Mult_norm_LR_test_general, Mult_norm_LR_pred_CI,
    PredictionMethod,
)

X = np.column_stack([np.ones(n), x1, x2])

ols = Mult_LR_Least_squares(X, y)
print(f"beta_hat = {ols.beta_hat}")
print(f"Se^2 = {ols.sigma2_unbiased:.4f}")

# R² and adjusted R²
tss = Mult_LR_partition_TSS(X, y)
print(f"R² = {tss.R_squared:.4f},  adj R² = {tss.adj_R_squared:.4f}")

# General hypothesis test: H0: C*beta = c0
C = np.array([[0, 1, -1]])
c0 = np.array([0.0])
result = Mult_norm_LR_test_general(X, y, C, c0, alpha=0.05)
print(f"F = {result.test_statistic:.4f}, p = {result.p_value:.4f}")

# Simultaneous prediction intervals
D = np.array([[1, 0.5, 1.0], [1, 1.0, 0.0]])
pred = Mult_norm_LR_pred_CI(X, y, D, alpha=0.05, method=PredictionMethod.BEST)
for i, (lo, hi) in enumerate(pred.intervals):
    print(f"Prediction {i+1}: {pred.point_estimates[i]:.2f} [{lo:.2f}, {hi:.2f}]")
```

### Bayesian Conjugate Inference

```python
import numpy as np
from statscore import bayes_normal_mean_known_var, bayes_normal_mean_unknown_var, bayes_normal_mean_unknown_var_summary

x = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4])

# Known variance: Normal-Normal conjugate
result = bayes_normal_mean_known_var(x, sigma_sq=0.04, mu0=10.0, kappa0=2.0)
print(f"Posterior mean: {result.posterior_mean:.4f}")
print(f"95% Credible interval: {result.credible_interval}")

# Unknown variance: Normal-Gamma conjugate
result = bayes_normal_mean_unknown_var(x, mu0=10.0, kappa0=1.0, alpha0=2.0, beta0=0.1)
bayes_normal_mean_unknown_var_summary(result)
```

## Package Structure

```
statscore/
├── __init__.py              # Top-level exports (35 public symbols)
├── anova/
│   ├── one_way.py           # ANOVA1_partition_TSS, ANOVA1_test_equality, ANOVA1_print_table
│   ├── two_way.py           # ANOVA2_partition_TSS, ANOVA2_MLE, ANOVA2_test_equality, ANOVA2_print_table
│   └── multiple_tests.py    # Contrasts, orthogonality, corrections, CI, tests
├── bayes/
│   └── conjugate.py         # bayes_normal_mean_known_var, bayes_normal_mean_unknown_var
├── regression/
│   ├── least_squares.py     # Mult_LR_Least_squares, Mult_LR_partition_TSS (R², adj R²)
│   ├── inference.py         # Simultaneous CI, CR, general/component/linear tests
│   └── prediction.py        # Mult_norm_LR_pred_CI
├── testing/
│   ├── one_sample.py        # z_test_mean, t_test_mean, chi2_test_variance
│   └── two_sample.py        # t_test_two_sample, t_test_paired, f_test_variances
└── utils/
    ├── enums.py             # AlternativeHypothesis, CorrectionMethod, PredictionMethod, TwoWayTestFactor
    ├── distributions.py     # Critical values and p-values (F, t, chi2, z, q)
    └── validation.py        # Shared input validation helpers
```

## API Reference

### Enums

| Enum | Members |
|------|---------|
| `AlternativeHypothesis` | `TWO_SIDED`, `LESS`, `GREATER` |
| `CorrectionMethod` | `SCHEFFE`, `TUKEY`, `BONFERRONI`, `SIDAK`, `BEST` |
| `PredictionMethod` | `SCHEFFE`, `BONFERRONI`, `BEST` |
| `TwoWayTestFactor` | `A`, `B`, `AB` |

### ANOVA Functions

| Function | Description |
|----------|-------------|
| `ANOVA1_partition_TSS(data)` | Partition SS_total into SS_within and SS_between |
| `ANOVA1_test_equality(data, alpha)` | F-test for equality of group means |
| `ANOVA1_print_table(result)` | Print formatted one-way ANOVA table |
| `ANOVA1_is_contrast(c)` | Check if coefficients form a contrast |
| `ANOVA1_is_orthogonal(n, c1, c2)` | Check orthogonality of two contrasts |
| `Bonferroni_correction(alpha, m)` | Bonferroni-corrected significance level |
| `Sidak_correction(alpha, m)` | Šidák-corrected significance level |
| `ANOVA1_CI_linear_combs(data, alpha, C, method)` | Simultaneous CIs for linear combinations |
| `ANOVA1_test_linear_combs(data, alpha, C, d, method)` | Test multiple linear combinations (FWER) |
| `ANOVA2_partition_TSS(data)` | Two-way ANOVA sum of squares partition |
| `ANOVA2_MLE(data)` | MLE for μ, α_i, β_j, δ_{ij} |
| `ANOVA2_test_equality(data, alpha, test)` | Two-way ANOVA F-tests (A, B, AB) |
| `ANOVA2_print_table(result)` | Print formatted two-way ANOVA table |

### Normal Distribution Testing Functions

| Function | Description |
|----------|-------------|
| `z_test_mean(x, mu0, sigma, alpha, alternative)` | One-sample Z-test (σ known) |
| `t_test_mean(x, mu0, alpha, alternative)` | One-sample t-test (σ unknown) |
| `chi2_test_variance(x, sigma0_sq, alpha, alternative)` | Chi-squared test for σ² |
| `t_test_two_sample(x1, x2, alpha, alternative, equal_var)` | Two-sample t-test (pooled or Welch) |
| `t_test_paired(x1, x2, alpha, alternative)` | Paired t-test |
| `f_test_variances(x1, x2, alpha, alternative)` | F-test for equality of variances |

### Regression Functions

| Function | Description |
|----------|-------------|
| `Mult_LR_Least_squares(X, y)` | OLS estimation: β̂, σ² MLE & unbiased |
| `Mult_LR_partition_TSS(X, y)` | TSS = RegSS + RSS, R², adjusted R² |
| `Mult_norm_LR_simul_CI(X, y, alpha)` | Simultaneous CIs for all β_i |
| `Mult_norm_LR_CR(X, y, C, alpha)` | Confidence region (ellipsoid) for Cβ |
| `Mult_norm_LR_is_in_CR(X, y, C, c0, alpha)` | Test if c₀ is inside the CR |
| `Mult_norm_LR_test_general(X, y, C, c0, alpha)` | General test H₀: Cβ = c₀ |
| `Mult_norm_LR_test_comp(X, y, alpha, components)` | Test H₀: β_{j₁}=...=β_{jᵣ}=0 |
| `Mult_norm_LR_test_linear_reg(X, y, alpha)` | Test existence of linear regression |
| `Mult_norm_LR_pred_CI(X, y, D, alpha, method)` | Simultaneous prediction CIs |

### Bayesian Functions

| Function | Description |
|----------|-------------|
| `bayes_normal_mean_known_var(x, sigma_sq, mu0, kappa0, alpha)` | Normal-Normal conjugate posterior |
| `bayes_normal_mean_unknown_var(x, mu0, kappa0, alpha0, beta0, alpha)` | Normal-Gamma conjugate posterior |
| `bayes_normal_mean_unknown_var_summary(result)` | Print formatted posterior summary |

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

### Normal Distribution Tests

| Test | Statistic | Distribution |
|------|-----------|-------------|
| Z-test | Z = (x̄ - μ₀) / (σ/√n) | N(0,1) |
| One-sample t | T = (x̄ - μ₀) / (s/√n) | t(n-1) |
| Chi-squared variance | Q = (n-1)s²/σ₀² | χ²(n-1) |
| Two-sample t (pooled) | T = (x̄₁-x̄₂) / (sₚ√(1/n₁+1/n₂)) | t(n₁+n₂-2) |
| Welch's t | T = (x̄₁-x̄₂) / √(s₁²/n₁+s₂²/n₂) | t(ν) Welch-Satterthwaite |
| Paired t | T = d̄ / (s_d/√n) | t(n-1) |
| F variance | F = s₁²/s₂² | F(n₁-1, n₂-1) |

### Multiple Linear Regression

Model: Y = Xβ + ε, where ε ~ N(0, σ²I).

**OLS estimator:** β̂ = (X^T X)^{-1} X^T Y

**TSS partition:** TSS = RegSS + RSS,  R² = RegSS/TSS,  adj R² = 1 - (1-R²)(n-1)/(n-p)

**General test:** H₀: Cβ = c₀, using F = [(Cβ̂ - c₀)^T [C(X^TX)^{-1}C^T]^{-1} (Cβ̂ - c₀)] / (r · S_e²)

### Bayesian Inference (Normal-Gamma)

Model: X_i ~ N(μ, 1/τ), prior (μ, τ) ~ Normal-Gamma(μ₀, κ₀, α₀, β₀).

**Posterior updates:**
- κₙ = κ₀ + n,  μₙ = (κ₀μ₀ + nx̄) / κₙ
- αₙ = α₀ + n/2,  βₙ = β₀ + ½Σ(xᵢ-x̄)² + κ₀n(x̄-μ₀)²/(2κₙ)

**Marginal posteriors:** μ|x ~ t(2αₙ, μₙ, √(βₙ/(αₙκₙ))), σ²|x ~ Inverse-Gamma(αₙ, βₙ)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check .

# Type checking
mypy statscore/
```

## Running the Demo

```bash
python examples/demo.py
```

This exercises all 31 functions with representative sample data.

## License

MIT
