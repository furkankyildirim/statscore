# statscore

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-205%20passing-brightgreen.svg)]()

A production-quality Python library for **ANOVA**, **normal distribution significance testing**, **multiple linear regression**, and **Bayesian conjugate inference** with structured output and family-wise error rate (FWER) control.

## Features

- **One-Way & Two-Way ANOVA** — sum-of-squares decomposition, F-tests, MLE estimation, formatted table printing
- **Multiple Comparison Procedures** — Bonferroni, Šidák, Scheffé, and Tukey methods with automatic "best" selection
- **Normal Distribution Significance Testing** — z-test, one/two-sample t-test, paired t-test, chi-squared variance test, F-test for variances; one-sided and two-sided alternatives
- **OLS Multiple Linear Regression** — matrix-based estimation, TSS partition, R², adjusted R²
- **Simultaneous Inference** — confidence intervals, confidence regions (ellipsoids), general hypothesis tests
- **Prediction Intervals** — Scheffé and Bonferroni simultaneous prediction CIs
- **Bayesian Conjugate Inference** — Normal-Normal (known variance) and Normal-Gamma (unknown variance) conjugate posteriors with credible intervals and posterior predictive distributions
- **Regression Summary** — full OLS summary table (coefficients, SE, t-stats, p-values, F-test) analogous to R's `summary(lm(...))`
- **Diagnostics** — Shapiro-Wilk normality test, Levene's variance homogeneity test, leverage/standardized residuals/Cook's D with influence flags, mean confidence intervals
- **Visualization** — scatter+fit, residuals vs. fitted, normal Q-Q, ANOVA box plots, Bayesian posterior plots; all return `matplotlib.Figure`
- **Data I/O** — load `.csv`, `.tsv`, `.xlsx`, `.json` into a `LoadedData` result via a single `load_data` call
- **Interactive CLI** — `statscore` command launches an 11-item text menu covering all major analyses
- **Structured Output** — all functions return typed dataclass objects, not raw tuples
- **Type-Safe Enums** — all categorical parameters use enums; no raw strings
- **Standard Dependencies** — NumPy, SciPy, pandas, matplotlib

## Installation

```bash
# From PyPI:
pip install statscore

# From source (development):
git clone https://github.com/furkankyildirim/statscore.git
cd statscore
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.9, NumPy >= 1.21, SciPy >= 1.7, pandas >= 1.3, matplotlib >= 3.5

## Quick Start

### One-Way ANOVA

```python
import numpy as np
from statscore import anova1_partition_tss, anova1_test_equality

data = [
    np.array([28, 23, 14, 27, 31]),
    np.array([33, 36, 34, 29, 24]),
    np.array([18, 21, 20, 22]),
]

partition = anova1_partition_tss(data)
print(f"SS_total = {partition.SS_total}")
print(f"SS_within = {partition.SS_within}")
print(f"SS_between = {partition.SS_between}")
# SS_total = 548.8571428571428
# SS_within = 272.75
# SS_between = 276.1071428571428

result = anova1_test_equality(data, alpha=0.05)
result.summary()
# ==========================================================
#   One-Way ANOVA Table
# ==========================================================
#   Source          df           SS           MS          F
# ----------------------------------------------------------
#   Between          2     276.1071     138.0536     5.5677
#   Within          11     272.7500      24.7955
# ----------------------------------------------------------
#   Total           13     548.8571
# ==========================================================
#   F critical (α=0.05): 3.9823
#   p-value:                  0.0214
#   Decision:                 Reject H0
# ==========================================================
```

### Two-Way ANOVA

```python
import numpy as np
from statscore import anova2_test_equality, TwoWayTestFactor

# data shape: (I, J, K) — I levels of A, J levels of B, K replicates
data = np.array([
    [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
    [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
], dtype=float)

result = anova2_test_equality(data, alpha=0.05, test=TwoWayTestFactor.B)
result.summary()
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
z_test_mean(x, mu0=10.0, sigma=0.3, alpha=0.05,
            alternative=AlternativeHypothesis.TWO_SIDED).summary()
# ============================================================
#   One-Sample Z-Test
# ============================================================
#   n = 8    x̄ = 10.0500    σ = 0.3000
#   H0: μ = 10.0    Alternative: two-sided
# ------------------------------------------------------------
#   Z-statistic: 0.4714    Z-critical: ±1.9600
#   p-value:     0.6374    alpha: 0.05
#   Decision:    Fail to reject H0
# ============================================================

# One-sample t-test (sigma unknown)
t_test_mean(x, mu0=10.0, alpha=0.05,
            alternative=AlternativeHypothesis.TWO_SIDED).summary()
# ============================================================
#   One-Sample t-Test
# ============================================================
#   n = 8    x̄ = 10.0500    s = 0.2449    df = 7
#   H0: μ = 10.0    Alternative: two-sided
# ------------------------------------------------------------
#   t-statistic: 0.5774    t-critical: ±2.3646
#   p-value:     0.5818    alpha: 0.05
#   Decision:    Fail to reject H0
# ============================================================

# Two-sample t-test (Welch)
x2 = np.array([10.8, 11.2, 10.9, 11.1, 10.7])
t_test_two_sample(x, x2, alpha=0.05, equal_var=False,
                  alternative=AlternativeHypothesis.TWO_SIDED).summary()
# ============================================================
#   Two-Sample t-Test  (Welch)
# ============================================================
#   n1 = 8    x̄1 = 10.0500    s1 = 0.2449
#   n2 = 5    x̄2 = 10.9400    s2 = 0.2074
#   df = 9    Alternative: two-sided
# ------------------------------------------------------------
#   t-statistic: -7.0142    t-critical: ±2.2622
#   p-value:     0.0001    alpha: 0.05
#   Decision:    Reject H0
# ============================================================
```

### Multiple Comparisons with FWER Control

```python
import numpy as np
from statscore import anova1_ci_linear_combs, CorrectionMethod

C = np.array([[1, -1, 0], [0, 1, -1], [1, 0, -1]])

# Automatic method selection (picks narrowest valid intervals)
anova1_ci_linear_combs(data, alpha=0.05, C=C,
                       method=CorrectionMethod.BEST).summary()
# ============================================================
#   Simultaneous Confidence Intervals
#   Method: Bonferroni
# ============================================================
#   Interval    Point Est   Half-Width      Lower      Upper
# ------------------------------------------------------------
#   CI_1          -6.6000       8.8812   -15.4812     2.2812
#   CI_2          10.9500       9.4199     1.5301    20.3699
#   CI_3           4.3500       9.4199    -5.0699    13.7699
# ============================================================
```

### Multiple Linear Regression

```python
import numpy as np
from statscore import (
    mult_lr_least_squares, mult_lr_partition_tss,
    mult_norm_lr_test_general, mult_norm_lr_pred_ci,
    PredictionMethod,
)

attend = np.array([1, 0.5, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.6, 0.1, 0, 0, 0.7, 0.8, 1])
homework = np.array([0.25, 1, 0.5, 1, 1, 0.75, 1, 0.25, 0, 0, 1, 0.5, 0.25, 0.75, 1])
y = np.array([60, 65, 40, 70, 65, 70, 85, 70, 44, 20, 40, 30, 50, 77, 90], dtype=float)
n = len(y)
X = np.column_stack([np.ones(n), attend, homework])

ols = mult_lr_least_squares(X, y)
print(f"beta_hat = {ols.beta_hat}")
print(f"Se^2 = {ols.sigma2_unbiased:.4f}")
# beta_hat = [14.3676 46.1957 30.4521]
# Se^2 = 19.8400

# R² and adjusted R²
tss = mult_lr_partition_tss(X, y)
print(f"R² = {tss.R_squared:.4f},  adj R² = {tss.adj_R_squared:.4f}")
# R² = 0.9588,  adj R² = 0.9520

# General hypothesis test: H0: C*beta = c0
C = np.array([[0, 1, -1]])
c0 = np.array([0.0])
mult_norm_lr_test_general(X, y, C, c0, alpha=0.05).summary()
# ============================================================
#   Regression Hypothesis Test
# ============================================================
#   df_numerator = 1    df_denominator = 12
#   alpha = 0.05
# ------------------------------------------------------------
#   F-statistic: 11.4796    F-critical: 4.7472
#   p-value:     0.0054
#   Decision:    Reject H0
# ============================================================

# Simultaneous prediction intervals
D = np.array([[1, 0.5, 1.0], [1, 1.0, 0.0]])
mult_norm_lr_pred_ci(X, y, D, alpha=0.05, method=PredictionMethod.BEST).summary()
# ================================================================
#   Simultaneous Prediction Intervals
#   Method: Bonferroni
# ================================================================
#   #      Point Est   Half-Width      Lower      Upper
# ----------------------------------------------------------------
#   1        67.9175       4.2340    63.6835    72.1515
#   2        60.5633       6.9735    53.5898    67.5368
# ================================================================
```

### Bayesian Conjugate Inference

```python
import numpy as np
from statscore import bayes_normal_mean_known_var, bayes_normal_mean_unknown_var

x = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4])

# Known variance: Normal-Normal conjugate
result = bayes_normal_mean_known_var(x, sigma_sq=0.04, mu0=10.0, kappa0=2.0)
result.summary()
# ============================================================
#   Bayesian Normal Posterior  (Known Variance)
# ============================================================
#   n = 8    x̄ = 10.050000    sigma² known
# ----------------------------------------------------------
#   Posterior hyperparameters:
#     mu_n    = 10.040000
#     kappa_n = 10.000000
# ----------------------------------------------------------
#   Posterior: μ | x ~ N(10.040000, 0.004000)
#     mean = 10.040000
#     std  = 0.063246
#   95% Credible interval: (9.916041, 10.163959)
# ----------------------------------------------------------
#   Posterior predictive:
#     mean = 10.040000
#     std  = 0.209762
#   95% Predictive interval: (9.628874, 10.451126)
# ============================================================

# Unknown variance: Normal-Gamma conjugate
result = bayes_normal_mean_unknown_var(x, mu0=10.0, kappa0=1.0, alpha0=2.0, beta0=0.1)
result.summary()
# ============================================================
#   Bayesian Normal-Gamma Posterior Summary
# ============================================================
#   Observations (n):          8
#   Sample mean (x_bar):       10.050000
# ------------------------------------------------------------
#   Posterior Hyperparameters:
#     mu_n    = 10.044444
#     kappa_n = 9.000000
#     alpha_n = 6.000000
#     beta_n  = 0.311111
# ------------------------------------------------------------
#   Posterior Summaries:
#     E[mu]      = 10.044444
#     E[tau]     = 19.285714
#     E[sigma^2] = 0.062222
# ------------------------------------------------------------
#   95% Credible Intervals:
#     mu:      (9.879065, 10.209824)
#     sigma^2: (0.026663, 0.141292)
# ============================================================
```

### Visualization

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts/CI
from statscore import (
    plot_regression, plot_residuals, plot_qq, plot_anova_groups,
    mult_lr_least_squares, bayes_normal_mean_known_var,
    anova1_test_equality, z_test_mean, AlternativeHypothesis,
)

attend = np.array([1, 0.5, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.6, 0.1, 0, 0, 0.7, 0.8, 1])
y = np.array([60, 65, 40, 70, 65, 70, 85, 70, 44, 20, 40, 30, 50, 77, 90], dtype=float)
X2 = np.column_stack([np.ones(len(y)), attend])
beta_hat = mult_lr_least_squares(X2, y).beta_hat
fitted = X2 @ beta_hat
residuals = y - fitted

# Scatter plot with regression line (simple regression)
fig = plot_regression(attend, y, beta_hat,
                      x_label="Attendance rate", y_label="Grade",
                      title="Grade vs Attendance")
fig.savefig("plot_regression.png", dpi=100)
# → saves plot_regression.png  (matplotlib Figure object returned)

# Residuals vs fitted values
fig = plot_residuals(fitted, residuals)
fig.savefig("plot_residuals.png", dpi=100)
# → saves plot_residuals.png

# Normal Q-Q plot
fig = plot_qq(residuals)
fig.savefig("plot_qq.png", dpi=100)
# → saves plot_qq.png

# ANOVA group box plots
groups = [
    np.array([28, 23, 14, 27, 31]),
    np.array([33, 36, 34, 29, 24]),
    np.array([18, 21, 20, 22]),
]
fig = plot_anova_groups(groups,
                        group_labels=["Toxin 1", "Toxin 2", "Control"],
                        y_label="Response", title="Toxin Experiment")
fig.savefig("plot_anova_groups.png", dpi=100)
# → saves plot_anova_groups.png

# Result-level plot() — every result dataclass has a .plot() method
x_meas = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4])
fig = bayes_normal_mean_known_var(x_meas, sigma_sq=0.04, mu0=10.0, kappa0=2.0).plot()
fig.savefig("plot_posterior_normal.png", dpi=100)
# → saves plot_posterior_normal.png

fig = anova1_test_equality(groups, alpha=0.05).plot()
fig.savefig("plot_anova1.png", dpi=100)
# → saves plot_anova1.png

fig = z_test_mean(x_meas, mu0=10.0, sigma=0.2,
                  alternative=AlternativeHypothesis.TWO_SIDED).plot()
fig.savefig("plot_z_test.png", dpi=100)
# → saves plot_z_test.png
```

All plot functions return a `matplotlib.figure.Figure` — use `.savefig()` to write to disk
or `.show()` in interactive sessions. Pass `matplotlib.use("Agg")` before importing
statscore for headless / CI environments.

### Data I/O

```python
from statscore import load_data

# CSV (comma or semicolon — separator is detected automatically)
result = load_data("data.csv")

# TSV
result = load_data("data.tsv")

# Excel
result = load_data("data.xlsx")

# JSON (records or columns orientation)
result = load_data("data.json")

# Extra keyword arguments are forwarded to the underlying pandas reader
result = load_data("data.csv", skiprows=2, usecols=["x", "y"])

# LoadedData fields
print(result.format)        # "csv" | "tsv" | "xlsx" | "json"
print(result.n_rows)        # number of rows
print(result.n_cols)        # number of columns
print(result.column_names)  # list of column name strings
print(result.path)          # resolved file path
df = result.df              # pandas DataFrame — use for downstream analysis
```

Supported extensions: `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`.
Any other extension raises `ValueError`. A missing file raises `FileNotFoundError`.

## Package Structure

```
statscore/
├── __init__.py              # Top-level exports
├── __main__.py              # python -m statscore entry point
├── cli.py                   # Interactive CLI (15-item menu)
├── diagnostics.py           # shapiro_wilk_test, levene_test, regression_diagnostics, mean_confidence_interval
├── io.py                    # load_data (csv/tsv/xlsx/json → LoadedData)
├── anova/
│   ├── one_way.py           # anova1_partition_tss, anova1_test_equality
│   ├── two_way.py           # anova2_partition_tss, anova2_mle, anova2_test_equality
│   └── multiple_tests.py    # Contrasts, orthogonality, corrections, CI, tests
├── bayes/
│   └── conjugate.py         # bayes_normal_mean_known_var, bayes_normal_mean_unknown_var
├── regression/
│   ├── least_squares.py     # mult_lr_least_squares, mult_lr_partition_tss (R², adj R²)
│   ├── inference.py         # Simultaneous CI, CR, general/component/linear tests
│   ├── prediction.py        # mult_norm_lr_pred_ci
│   └── summary.py           # regression_summary (full OLS summary table)
├── testing/
│   ├── one_sample.py        # z_test_mean, t_test_mean, chi2_test_variance
│   └── two_sample.py        # t_test_two_sample, t_test_paired, f_test_variances
└── utils/
    ├── enums.py             # AlternativeHypothesis, CorrectionMethod, PredictionMethod, TwoWayTestFactor
    ├── distributions.py     # Critical values and p-values (F, t, chi2, z, q)
    ├── plots.py             # Shared plot utilities (plot_regression, plot_qq, plot_t_test, …)
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
| `anova1_partition_tss(data)` | Partition SS_total into SS_within and SS_between |
| `anova1_test_equality(data, alpha)` | F-test for equality of group means |
| `anova1_is_contrast(c)` | Check if coefficients form a contrast |
| `anova1_is_orthogonal(n, c1, c2)` | Check orthogonality of two contrasts |
| `bonferroni_correction(alpha, m)` | Bonferroni-corrected significance level |
| `sidak_correction(alpha, m)` | Šidák-corrected significance level |
| `anova1_ci_linear_combs(data, alpha, C, method)` | Simultaneous CIs for linear combinations |
| `anova1_test_linear_combs(data, alpha, C, d, method)` | Test multiple linear combinations (FWER) |
| `anova2_partition_tss(data)` | Two-way ANOVA sum of squares partition |
| `anova2_mle(data)` | MLE for μ, α_i, β_j, δ_{ij} |
| `anova2_test_equality(data, alpha, test)` | Two-way ANOVA F-tests (A, B, AB) |

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
| `mult_lr_least_squares(X, y)` | OLS estimation: β̂, σ² MLE & unbiased |
| `mult_lr_partition_tss(X, y)` | TSS = RegSS + RSS, R², adjusted R² |
| `mult_norm_lr_simul_ci(X, y, alpha)` | Simultaneous CIs for all β_i |
| `mult_norm_lr_cr(X, y, C, alpha)` | Confidence region (ellipsoid) for Cβ |
| `mult_norm_lr_is_in_cr(X, y, C, c0, alpha)` | Test if c₀ is inside the CR |
| `mult_norm_lr_test_general(X, y, C, c0, alpha)` | General test H₀: Cβ = c₀ |
| `mult_norm_lr_test_comp(X, y, alpha, components)` | Test H₀: β_{j₁}=...=β_{jᵣ}=0 |
| `mult_norm_lr_test_linear_reg(X, y, alpha)` | Test existence of linear regression |
| `mult_norm_lr_pred_ci(X, y, D, alpha, method)` | Simultaneous prediction CIs |

### Bayesian Functions

| Function | Description |
|----------|-------------|
| `bayes_normal_mean_known_var(x, sigma_sq, mu0, kappa0, alpha)` | Normal-Normal conjugate posterior |
| `bayes_normal_mean_unknown_var(x, mu0, kappa0, alpha0, beta0, alpha)` | Normal-Gamma conjugate posterior |

### Regression Summary

| Function | Description |
|----------|-------------|
| `regression_summary(X, y, alpha, feature_names)` | Full OLS summary: β̂, SE, t-stats, p-values, CIs, R², F-test |

### Diagnostics Functions

| Function | Description |
|----------|-------------|
| `shapiro_wilk_test(x, alpha)` | Shapiro-Wilk normality test |
| `levene_test(data, alpha)` | Levene's test for homogeneity of variances |
| `regression_diagnostics(X, y)` | Leverage, standardized residuals, Cook's D |
| `mean_confidence_interval(x, alpha, sigma)` | CI for the mean (z or t) |

### Visualization Functions

Shared plot utilities (available at top-level):

| Function | Description |
|----------|-------------|
| `plot_regression(x, y, beta_hat, ...)` | Scatter plot with fitted regression line |
| `plot_residuals(fitted, residuals, ...)` | Residuals vs. fitted values |
| `plot_qq(x, ...)` | Normal Q-Q plot |
| `plot_anova_groups(data, group_labels, ...)` | Box plots with jittered points for ANOVA groups |
| `plot_t_test(t_statistic, t_critical, df, alternative, ...)` | t-distribution with rejection region |
| `plot_f_test(f_statistic, f_critical_low, f_critical_up, ...)` | F-distribution with rejection region |
| `plot_simultaneous_ci(point_estimates, intervals, ...)` | Simultaneous confidence intervals |

Result-level plots (call `.plot()` on any result object):

| Result class | `.plot()` output |
|---|---|
| `ZTestResult` | Standard normal with rejection region |
| `TTestOneSampleResult` | t-distribution with rejection region |
| `Chi2VarianceTestResult` | Chi-squared distribution with rejection region |
| `FTestVariancesResult` | F-distribution with rejection region |
| `ANOVA1TestResult` | SS bar chart + F-distribution |
| `ANOVA2TestResult` | SS bar chart + F-distribution |
| `SimultaneousCIResult` | Simultaneous CI forest plot |
| `SimultaneousTestResult` | Horizontal bar chart of test statistics |
| `RegressionSummaryResult` | Coefficient plot with CIs |
| `RegressionDiagnosticsResult` | Four-panel diagnostics (leverage, std residuals, Cook's D, residuals vs leverage) |
| `MeanConfidenceIntervalResult` | Single CI interval plot |
| `NormalMeanKnownVarResult` | Prior/posterior density with credible interval |
| `NormalMeanUnknownVarResult` | Marginal posterior (Student-t) with credible interval |
| `ShapiroWilkResult` | Normal Q-Q plot |

### I/O Functions

| Function | Description |
|----------|-------------|
| `load_data(path, **kwargs)` | Load `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json` → `LoadedData` |

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

This exercises all major functions with representative sample data.

## Interactive CLI

```bash
# After pip install:
statscore

# Or directly:
python -m statscore
```

The CLI provides an 11-item menu covering ANOVA, hypothesis tests, regression, diagnostics, and Bayesian inference. Data can be entered as space/comma-separated numbers or loaded from a file (`.csv`, `.tsv`, `.xlsx`, `.json`).

## License

MIT
