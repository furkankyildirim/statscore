# statscore — User Guide

This guide walks through every feature of statscore with worked examples, explains what each analysis does, when to use it, and how to interpret the output. It is written for someone who knows the statistics but is new to this library.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Loading Data](#2-loading-data)
3. [ANOVA](#3-anova)
   - 3.1 [One-Way ANOVA](#31-one-way-anova)
   - 3.2 [Two-Way ANOVA](#32-two-way-anova)
   - 3.3 [Multiple Comparisons](#33-multiple-comparisons-with-fwer-control)
4. [Significance Testing](#4-significance-testing)
   - 4.1 [Z-test](#41-z-test-for-the-mean-σ-known)
   - 4.2 [One-Sample t-test](#42-one-sample-t-test)
   - 4.3 [Two-Sample t-test](#43-two-sample-t-test)
   - 4.4 [Paired t-test](#44-paired-t-test)
   - 4.5 [Chi-Squared Variance Test](#45-chi-squared-variance-test)
   - 4.6 [F-test for Variances](#46-f-test-for-equality-of-variances)
5. [Multiple Linear Regression](#5-multiple-linear-regression)
   - 5.1 [OLS Estimation](#51-ols-estimation)
   - 5.2 [Regression Summary Table](#52-regression-summary-table)
   - 5.3 [Simultaneous CIs for β](#53-simultaneous-confidence-intervals-for-β)
   - 5.4 [General Hypothesis Tests](#54-general-hypothesis-test-h₀-cβ--c₀)
   - 5.5 [Prediction Intervals](#55-simultaneous-prediction-intervals)
   - 5.6 [Confidence Regions](#56-confidence-regions)
6. [Diagnostics](#6-diagnostics)
   - 6.1 [Shapiro-Wilk Normality Test](#61-shapiro-wilk-normality-test)
   - 6.2 [Levene's Variance Test](#62-levenes-variance-homogeneity-test)
   - 6.3 [Regression Diagnostics](#63-regression-diagnostics)
   - 6.4 [Mean Confidence Interval](#64-mean-confidence-interval)
7. [Bayesian Conjugate Inference](#7-bayesian-conjugate-inference)
   - 7.1 [Normal-Normal (Known Variance)](#71-normal-normal-conjugate-known-variance)
   - 7.2 [Normal-Gamma (Unknown Variance)](#72-normal-gamma-conjugate-unknown-variance)
   - 7.3 [Beta-Binomial](#73-beta-binomial)
   - 7.4 [Gamma-Poisson](#74-gamma-poisson)
8. [Bayesian MCMC](#8-bayesian-mcmc)
   - 8.1 [MCMC for Normal Data](#81-mcmc-for-normal-data)
   - 8.2 [MCMC for Linear Regression](#82-mcmc-for-linear-regression)
   - 8.3 [Custom Models with run_mcmc](#83-custom-models-with-run_mcmc)
9. [Visualization](#9-visualization)
10. [Interactive CLI Reference](#10-interactive-cli-reference)
11. [Browser UI Reference](#11-browser-ui-reference)
12. [Result Objects](#12-result-objects)
13. [Enums Reference](#13-enums-reference)
14. [Input Formats](#14-input-formats)
15. [Worked End-to-End Example](#15-worked-end-to-end-example)

---

## 1. Getting Started

```bash
pip install statscore
```

Import the functions you need from the top-level package:

```python
from statscore import (
    anova1_test_equality,
    z_test_mean,
    regression_summary,
    bayes_normal_mean_known_var,
    AlternativeHypothesis,
)
import numpy as np
```

All functions accept NumPy arrays. All functions return a result dataclass — never a plain tuple. Call `.summary()` to print and `.plot()` to visualise.

---

## 2. Loading Data

Use `load_data` to read tabular files into a `LoadedData` object:

```python
from statscore import load_data

result = load_data("experiment.csv")

print(result.format)        # "csv"
print(result.n_rows)        # e.g. 120
print(result.n_cols)        # e.g. 4
print(result.column_names)  # ["subject", "group", "score", "time"]
df = result.df              # pandas DataFrame — use for downstream analysis
```

### Supported formats

| Extension | Notes |
|---|---|
| `.csv` | Comma or semicolon separator detected automatically |
| `.tsv` | Tab-separated |
| `.xlsx`, `.xls` | Requires `openpyxl` (installed with statscore) |
| `.json` | Records or column orientation |

### Extracting arrays from a DataFrame

```python
result = load_data("scores.csv")
df = result.df

# Single column
scores = df["score"].to_numpy(dtype=float)

# Group split (for ANOVA)
groups = [df.loc[df["group"] == g, "score"].to_numpy(float)
          for g in df["group"].unique()]
```

---

## 3. ANOVA

### 3.1 One-Way ANOVA

**When to use:** Compare means across I ≥ 2 independent groups. Assumes normal data with equal variances.

**H₀:** μ₁ = μ₂ = … = μ_I

```python
import numpy as np
from statscore import anova1_partition_tss, anova1_test_equality

# Each array is one group
groups = [
    np.array([28, 23, 14, 27, 31]),   # Group 1 (n=5)
    np.array([33, 36, 34, 29, 24]),   # Group 2 (n=5)
    np.array([18, 21, 20, 22]),       # Group 3 (n=4)
]

# Step 1: decompose total variation
partition = anova1_partition_tss(groups)
print(f"SS_between = {partition.SS_between:.4f}")  # explained variation
print(f"SS_within  = {partition.SS_within:.4f}")   # unexplained variation
print(f"SS_total   = {partition.SS_total:.4f}")

# Step 2: F-test
result = anova1_test_equality(groups, alpha=0.05)
result.summary()
```

Output:
```
==========================================================
  One-Way ANOVA Table
==========================================================
  Source          df           SS           MS          F
----------------------------------------------------------
  Between          2     276.1071     138.0536     5.5677
  Within          11     272.7500      24.7955
----------------------------------------------------------
  Total           13     548.8571
==========================================================
  F critical (α=0.05): 3.9823
  p-value:             0.0214
  Decision:            Reject H0
==========================================================
```

**Interpreting the output:**
- **Between df** = I − 1 = 2, **Within df** = n − I = 11
- **F = 5.57** exceeds **F_critical = 3.98**, so we reject H₀ at α = 0.05
- At least one group mean is significantly different

```python
# Save the SS bar chart + F-distribution plot
result.plot().savefig("anova1.png", dpi=150)

# Save a box-plot of all groups
from statscore import plot_anova_groups
plot_anova_groups(groups, group_labels=["Toxin A", "Toxin B", "Control"]).savefig("groups.png")
```

---

### 3.2 Two-Way ANOVA

**When to use:** Two categorical factors A (I levels) and B (J levels) with K ≥ 2 replicates per cell. Tests main effects and interaction.

```python
from statscore import anova2_partition_tss, anova2_mle, anova2_test_equality, TwoWayTestFactor
import numpy as np

# data shape: (I, J, K)
data = np.array([
    [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
    [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
], dtype=float)  # I=2, J=3, K=4

# SS decomposition for all sources
partition = anova2_partition_tss(data)
print(f"SS_A={partition.SS_A:.2f}  SS_B={partition.SS_B:.2f}  "
      f"SS_AB={partition.SS_AB:.2f}  SS_E={partition.SS_E:.2f}")

# MLE estimates
mle = anova2_mle(data)
print(f"Grand mean μ̂ = {mle.mu:.4f}")
print(f"Factor A effects: {mle.alpha}")
print(f"Factor B effects: {mle.beta}")

# Test Factor B (change to TwoWayTestFactor.A or .AB as needed)
anova2_test_equality(data, alpha=0.05, test=TwoWayTestFactor.B).summary()
```

**Available test factors:**

| `TwoWayTestFactor` | H₀ |
|---|---|
| `A` | α₁ = α₂ = … = α_I = 0 |
| `B` | β₁ = β₂ = … = β_J = 0 |
| `AB` | δ_{ij} = 0 for all i, j |

---

### 3.3 Multiple Comparisons with FWER Control

**When to use:** After rejecting the overall ANOVA H₀, identify which group mean differences are significant while controlling the family-wise error rate.

**Key concept — contrast:** A vector c such that Σcᵢ = 0. Each row of the contrast matrix C defines one linear combination of group means (e.g. μ₁ − μ₂).

```python
from statscore import (
    anova1_ci_linear_combs, anova1_test_linear_combs,
    anova1_is_contrast, anova1_is_orthogonal,
    CorrectionMethod,
)
import numpy as np

# Verify that row vectors are valid contrasts
c1 = np.array([1, -1,  0])
c2 = np.array([0,  1, -1])
print(anova1_is_contrast(c1))  # True

# Check orthogonality (uses group sample sizes n = [5, 5, 4])
n = np.array([5, 5, 4])
print(anova1_is_orthogonal(n, c1, c2))

# Contrast matrix (m=3 contrasts, I=3 groups)
C = np.array([
    [1, -1,  0],   # μ1 − μ2
    [0,  1, -1],   # μ2 − μ3
    [1,  0, -1],   # μ1 − μ3
])

# Simultaneous confidence intervals
ci = anova1_ci_linear_combs(groups, alpha=0.05, C=C, method=CorrectionMethod.BEST)
ci.summary()
ci.plot().savefig("simul_ci.png")

# Simultaneous hypothesis tests: H0: Cμ = d
d = np.zeros(3)  # all differences = 0 under H0
tests = anova1_test_linear_combs(groups, alpha=0.05, C=C, d=d, method=CorrectionMethod.SCHEFFE)
tests.summary()
tests.plot().savefig("simul_tests.png")
```

**Method selection guide:**

| Method | Best when |
|---|---|
| `SCHEFFE` | Arbitrary contrasts (not just pairwise); conservative |
| `TUKEY` | All pairwise comparisons; equal group sizes |
| `BONFERRONI` | Small number of planned comparisons |
| `SIDAK` | Like Bonferroni but slightly less conservative |
| `BEST` | Automatically picks the narrowest valid intervals |

---

## 4. Significance Testing

All test functions accept an `alternative` parameter of type `AlternativeHypothesis`:

| Value | H₁ |
|---|---|
| `AlternativeHypothesis.TWO_SIDED` | μ ≠ μ₀ (default for most tests) |
| `AlternativeHypothesis.LESS` | μ < μ₀ |
| `AlternativeHypothesis.GREATER` | μ > μ₀ |

Every result has `.summary()` and `.plot()`. The plot draws the reference distribution with the rejection region shaded and the test statistic marked.

---

### 4.1 Z-test for the Mean (σ Known)

**Use when:** The population standard deviation σ is known (or n is very large).

```python
from statscore import z_test_mean, AlternativeHypothesis
import numpy as np

x = np.array([10.2, 9.8, 10.1, 10.3, 9.9, 10.0, 10.4, 9.7])

result = z_test_mean(x, mu0=10.0, sigma=0.3, alpha=0.05,
                     alternative=AlternativeHypothesis.TWO_SIDED)
result.summary()
result.plot().savefig("z_test.png")
```

**Key output fields:** `result.z_statistic`, `result.p_value`, `result.reject_h0`

---

### 4.2 One-Sample t-test

**Use when:** σ is unknown and must be estimated from the data.

```python
from statscore import t_test_mean

result = t_test_mean(x, mu0=10.0, alpha=0.05,
                     alternative=AlternativeHypothesis.TWO_SIDED)
result.summary()
```

---

### 4.3 Two-Sample t-test

**Use when:** Comparing means of two independent groups.

```python
from statscore import t_test_two_sample

x1 = np.array([10.2, 9.8, 10.1, 10.3, 9.9])
x2 = np.array([10.8, 11.2, 10.9, 11.1, 10.7])

# Welch's t-test (recommended when variances may differ)
result = t_test_two_sample(x1, x2, alpha=0.05, equal_var=False,
                           alternative=AlternativeHypothesis.TWO_SIDED)
result.summary()

# Pooled t-test (assumes equal variances)
result = t_test_two_sample(x1, x2, alpha=0.05, equal_var=True,
                           alternative=AlternativeHypothesis.TWO_SIDED)
result.summary()
```

**Tip:** Use Levene's test (Section 6.2) to check variance equality before choosing pooled vs Welch.

---

### 4.4 Paired t-test

**Use when:** Measurements are paired (before/after, matched subjects).

```python
from statscore import t_test_paired

before = np.array([85, 90, 78, 92, 88, 76, 83])
after  = np.array([88, 94, 80, 95, 91, 79, 86])

result = t_test_paired(before, after, alpha=0.05,
                       alternative=AlternativeHypothesis.LESS)
result.summary()
# Tests H1: after > before (i.e. before - after < 0)
```

---

### 4.5 Chi-Squared Variance Test

**Use when:** Testing whether the population variance equals a specific value σ₀².

```python
from statscore import chi2_test_variance

result = chi2_test_variance(x, sigma0_sq=0.09, alpha=0.05,
                            alternative=AlternativeHypothesis.TWO_SIDED)
result.summary()
```

**Key output fields:** `result.chi2_statistic`, `result.p_value`, `result.sample_variance`

---

### 4.6 F-test for Equality of Variances

**Use when:** Testing whether two populations have equal variance.

```python
from statscore import f_test_variances

result = f_test_variances(x1, x2, alpha=0.05,
                          alternative=AlternativeHypothesis.TWO_SIDED)
result.summary()
```

---

## 5. Multiple Linear Regression

statscore provides the full OLS inference pipeline for the model:

**Y = Xβ + ε, ε ~ N(0, σ²I)**

where X is an n×p design matrix (first column is typically all-ones for the intercept).

---

### 5.1 OLS Estimation

```python
from statscore import mult_lr_least_squares, mult_lr_partition_tss
import numpy as np

attend   = np.array([1, 0.5, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.6, 0.1, 0, 0, 0.7, 0.8, 1])
homework = np.array([0.25, 1, 0.5, 1, 1, 0.75, 1, 0.25, 0, 0, 1, 0.5, 0.25, 0.75, 1])
y        = np.array([60, 65, 40, 70, 65, 70, 85, 70, 44, 20, 40, 30, 50, 77, 90], dtype=float)
n        = len(y)

# Design matrix with intercept column
X = np.column_stack([np.ones(n), attend, homework])

ols = mult_lr_least_squares(X, y)
print(f"β̂ = {ols.beta_hat}")              # OLS estimates
print(f"σ² MLE = {ols.sigma2_mle:.4f}")   # biased estimator
print(f"σ² unbiased = {ols.sigma2_unbiased:.4f}")
print(f"Fitted values: {ols.fitted_values[:3]}")
print(f"Residuals: {ols.residuals[:3]}")

# R² and adjusted R²
tss = mult_lr_partition_tss(X, y)
print(f"R²     = {tss.R_squared:.4f}")
print(f"adj R² = {tss.adj_R_squared:.4f}")
print(f"RegSS  = {tss.RegSS:.4f},  RSS = {tss.RSS:.4f}")
```

---

### 5.2 Regression Summary Table

```python
from statscore import regression_summary

result = regression_summary(X, y, alpha=0.05)
result.summary(feature_names=["(Intercept)", "attendance", "homework"])
result.plot().savefig("coeff_plot.png")  # coefficient forest plot
```

The summary table includes:
- Coefficient estimates, standard errors, t-statistics, p-values
- Significance stars (*** p<0.001, ** p<0.01, * p<0.05)
- 95% confidence intervals for each coefficient
- Overall F-test for model significance
- R² and adjusted R²

---

### 5.3 Simultaneous Confidence Intervals for β

**Use when:** Making inference on all regression coefficients at once (controls FWER).

```python
from statscore import mult_norm_lr_simul_ci

result = mult_norm_lr_simul_ci(X, y, alpha=0.05)
result.summary()
result.plot().savefig("simul_ci_beta.png")
```

---

### 5.4 General Hypothesis Test H₀: Cβ = c₀

**Use when:** Testing any linear restriction on the coefficient vector.

```python
from statscore import mult_norm_lr_test_general
import numpy as np

# Test H0: β_attend = β_homework  (i.e. C·β = 0 with C = [0, 1, -1])
C  = np.array([[0, 1, -1]])
c0 = np.array([0.0])

result = mult_norm_lr_test_general(X, y, C, c0, alpha=0.05)
result.summary()
# F = 11.4796, p = 0.0054 → Reject H0 at α=0.05
```

**Examples of C matrices:**

| Goal | C | c₀ |
|---|---|---|
| Test β₁ = β₂ | `[[0, 1, -1]]` | `[0]` |
| Test β₁ = 0 and β₂ = 0 | `[[0,1,0],[0,0,1]]` | `[0,0]` |
| Test β₁ + β₂ = 1 | `[[0, 1, 1]]` | `[1]` |

See also `mult_norm_lr_test_comp` (test that a subset of coefficients are zero) and `mult_norm_lr_test_linear_reg` (overall F-test for model significance).

---

### 5.5 Simultaneous Prediction Intervals

**Use when:** Predicting responses for m new observations simultaneously.

```python
from statscore import mult_norm_lr_pred_ci, PredictionMethod
import numpy as np

# Two new observations (rows must match design matrix columns)
D = np.array([
    [1, 0.5, 1.0],  # intercept=1, attend=0.5, homework=1.0
    [1, 1.0, 0.0],  # intercept=1, attend=1.0, homework=0.0
])

result = mult_norm_lr_pred_ci(X, y, D, alpha=0.05, method=PredictionMethod.BEST)
result.summary()
# Obs 1: ŷ = 67.92 ± 4.23,  CI = (63.68, 72.15)
# Obs 2: ŷ = 60.56 ± 6.97,  CI = (53.59, 67.54)
```

| `PredictionMethod` | Notes |
|---|---|
| `SCHEFFE` | Valid for arbitrary D; conservative |
| `BONFERRONI` | Tighter for a small, fixed number of predictions |
| `BEST` | Automatically selects the narrower interval |

---

### 5.6 Confidence Regions

```python
from statscore import mult_norm_lr_cr, mult_norm_lr_is_in_cr
import numpy as np

# Confidence region for (β_attend, β_homework)
C = np.eye(2, 3, 1)  # selects β₁ and β₂ from [β₀, β₁, β₂]
cr = mult_norm_lr_cr(X, y, C, alpha=0.05)

# Test whether a specific point lies inside the CR
c0 = np.array([40.0, 25.0])
inside = mult_norm_lr_is_in_cr(X, y, C, c0, alpha=0.05)
print(f"[40, 25] inside 95% CR? {inside}")
```

---

## 6. Diagnostics

### 6.1 Shapiro-Wilk Normality Test

**Use when:** Checking whether a sample comes from a normal distribution. Required before t-tests or ANOVA on small samples.

```python
from statscore import shapiro_wilk_test
import numpy as np

x = np.array([2.3, 2.7, 3.1, 2.8, 3.5, 2.9, 3.0, 2.4, 2.6, 3.2])

result = shapiro_wilk_test(x, alpha=0.05)
result.summary()
# W = 0.9802, p = 0.9713 → Fail to reject H0 (consistent with normality)

result.plot().savefig("qq_normality.png")  # normal Q-Q plot
```

**Note:** Power is low for n < 10. For n > 5000 SciPy issues a warning about p-value accuracy.

---

### 6.2 Levene's Variance Homogeneity Test

**Use when:** Checking the equal-variance assumption required by ANOVA and pooled t-tests.

```python
from statscore import levene_test

groups = [
    np.array([1.2, 1.5, 1.3, 1.4, 1.1]),
    np.array([2.1, 2.4, 2.2, 2.0, 2.3]),
    np.array([1.8, 1.6, 1.9, 1.7]),
]

result = levene_test(groups, alpha=0.05)
result.summary()
# F = 0.1234, p = 0.8845 → Fail to reject H0 (variances appear equal)
```

---

### 6.3 Regression Diagnostics

**Use when:** Checking OLS assumptions after fitting a regression model. Identifies outliers, high-leverage points, and influential observations.

```python
from statscore import regression_diagnostics
import numpy as np

X = np.column_stack([np.ones(15), attend, homework])
diag = regression_diagnostics(X, y)
diag.summary()
```

Output includes for each observation:
- **Leverage** h_ii: values > 2p/n are flagged as high-leverage
- **Standardized residual**: values outside ±2 are flagged as outliers
- **Cook's D**: values > 4/n are flagged as influential

```python
# Four-panel diagnostics plot
diag.plot().savefig("diagnostics.png", dpi=150)
```

The four panels show:
1. Leverage vs observation index (with threshold line)
2. Standardized residuals vs observation index
3. Cook's D vs observation index
4. Standardized residuals vs leverage

---

### 6.4 Mean Confidence Interval

```python
from statscore import mean_confidence_interval
import numpy as np

x = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4])

# t-interval (sigma unknown)
result = mean_confidence_interval(x, alpha=0.05)
result.summary()

# z-interval (sigma known)
result = mean_confidence_interval(x, alpha=0.05, sigma=0.3)
result.summary()

result.plot().savefig("mean_ci.png")
```

---

## 7. Bayesian Conjugate Inference

Conjugate models have closed-form posterior distributions — no numerical sampling required. All four models follow the same interface: call the function, call `.summary()`, optionally call `.plot()`.

### 7.1 Normal-Normal Conjugate (Known Variance)

**Model:** Xᵢ | μ ~ N(μ, σ²) with σ² known. Prior: μ ~ N(μ₀, σ²/κ₀).

**Posterior:** μ | x ~ N(μₙ, σ²/κₙ) where κₙ = κ₀ + n, μₙ = (κ₀μ₀ + nx̄)/κₙ.

```python
from statscore import bayes_normal_mean_known_var
import numpy as np

x = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4])

result = bayes_normal_mean_known_var(
    x,
    sigma_sq=0.04,   # known population variance σ²
    mu0=10.0,        # prior mean
    kappa0=2.0,      # prior precision scaling (higher → stronger prior)
    alpha=0.05,
)
result.summary()
result.plot().savefig("posterior_known_var.png")
```

**Hyperparameter guidance:**
- `kappa0 = 1` — one prior observation (weakly informative)
- `kappa0 = n` — prior carries as much weight as the data
- `kappa0 = 0.001` — essentially flat prior

---

### 7.2 Normal-Gamma Conjugate (Unknown Variance)

**Model:** Xᵢ | μ,τ ~ N(μ, 1/τ). Prior: (μ,τ) ~ Normal-Gamma(μ₀, κ₀, α₀, β₀).

**Marginal posteriors:** μ|x ~ t(2αₙ), σ²|x ~ InvGamma(αₙ, βₙ).

```python
from statscore import bayes_normal_mean_unknown_var

result = bayes_normal_mean_unknown_var(
    x,
    mu0=10.0,     # prior mean for μ
    kappa0=1.0,   # prior sample count for μ
    alpha0=2.0,   # prior shape for precision τ
    beta0=0.1,    # prior rate for precision τ
    alpha=0.05,
)
result.summary()
result.plot().savefig("posterior_unknown_var.png")
```

**Output includes:**
- Posterior hyperparameters (μₙ, κₙ, αₙ, βₙ)
- E[μ], E[τ], E[σ²]
- 95% credible intervals for μ and σ²

---

### 7.3 Beta-Binomial

**Model:** k | p ~ Binomial(n, p). Prior: p ~ Beta(α₀, β₀). **Posterior:** p | k ~ Beta(α₀+k, β₀+n−k).

**Use when:** Estimating a success probability from trial data.

```python
from statscore import bayes_beta_binomial

# 14 successes in 20 trials, uniform prior
result = bayes_beta_binomial(
    n_trials=20, n_successes=14,
    alpha0=1.0,   # prior pseudo-successes (1 = uniform)
    beta0=1.0,    # prior pseudo-failures
    alpha=0.05,
)
result.summary()
result.plot().savefig("beta_binomial.png")
```

**Output:**
- Prior: Beta(1.00, 1.00)
- Posterior: Beta(15.00, 7.00)
- Posterior mean: 0.6818,  95% CI: (0.4610, 0.8579)

---

### 7.4 Gamma-Poisson

**Model:** xᵢ | λ ~ Poisson(λ). Prior: λ ~ Gamma(α₀, β₀). **Posterior:** λ|x ~ Gamma(α₀ + Σxᵢ, β₀ + n).

**Use when:** Estimating a count rate (events per unit time/space).

```python
from statscore import bayes_gamma_poisson
import numpy as np

counts = np.array([3, 5, 2, 4, 6, 3, 5, 4, 2, 3])  # daily event counts

result = bayes_gamma_poisson(
    counts,
    alpha0=1.0,   # prior shape (1 = weakly informative)
    beta0=1.0,    # prior rate
    alpha=0.05,
)
result.summary()
result.plot().savefig("gamma_poisson.png")
```

---

## 8. Bayesian MCMC

MCMC is used when no closed-form posterior is available or when you want a fully Bayesian treatment without conjugate assumptions. All MCMC functions return `MCMCResult` with posterior summaries and credible intervals.

**MCMC result attributes:**
- `result.posterior_mean` — dict of parameter name → posterior mean
- `result.posterior_std` — dict of parameter name → posterior std
- `result.credible_intervals` — dict of parameter name → (lower, upper)
- `result.acceptance_rate` — fraction of proposals accepted (target: 0.15–0.50)
- `result.chain` — full Markov chain (n_iter × n_params array, including burn-in)
- `result.posterior_samples` — post-burn-in samples

---

### 8.1 MCMC for Normal Data

**Model:** Xᵢ ~ N(μ, σ²), with Normal prior on μ and Inverse-Gamma prior on σ².

```python
from statscore import mcmc_normal_mean_unknown_var
import numpy as np

x = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4])

result = mcmc_normal_mean_unknown_var(
    x,
    mu_prior_mean=0.0,       # prior mean for μ (set near your expectation)
    mu_prior_std=10.0,       # prior std for μ (large = vague)
    sigma_prior_alpha=2.0,   # InvGamma shape for σ² (>1 to keep well-defined mean)
    sigma_prior_beta=1.0,    # InvGamma rate for σ²
    n_iter=12000,
    n_burnin=2000,
    alpha=0.05,
    seed=42,                 # for reproducibility
)
result.summary()

# Trace plots + KDE posterior density per parameter
result.plot().savefig("mcmc_normal.png", dpi=150)
```

**Choosing `n_iter` and `n_burnin`:**
- A minimum of n_iter=5000, n_burnin=1000 is usually sufficient for 2-parameter models
- Check the trace plot: if it looks like white noise (mixes well), the chain has converged
- Target acceptance rate: 0.15–0.50. If acceptance rate is < 0.1, decrease `proposal_std`

---

### 8.2 MCMC for Linear Regression

**Model:** Y = Xβ + ε, ε ~ N(0, σ²), with Normal priors on β and Inverse-Gamma on σ².

```python
from statscore import mcmc_linear_regression
import numpy as np

X = np.column_stack([np.ones(15), attend, homework])

result = mcmc_linear_regression(
    X, y,
    beta_prior_mean=0.0,     # shared prior mean for all β_j (or array of length p)
    beta_prior_std=10.0,     # prior std for each β_j (large = vague)
    sigma_prior_alpha=2.0,
    sigma_prior_beta=1.0,
    n_iter=15000,
    n_burnin=3000,
    alpha=0.05,
    seed=42,
)
result.summary()
# Parameters: beta_0, beta_1, beta_2, sigma
# Posterior means will be close to OLS estimates under vague priors

result.plot().savefig("mcmc_regression.png", dpi=150)
```

**Comparison to OLS:** Under vague priors (large `beta_prior_std`), MCMC posteriors will be centred near the OLS estimates. The Bayesian approach additionally provides proper uncertainty quantification even for small samples.

---

### 8.3 Custom Models with `run_mcmc`

For any model where you can write a log-posterior function:

```python
from statscore import run_mcmc
import numpy as np

# Example: Bayesian linear regression with custom priors
x_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_obs = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

def log_posterior(theta):
    """theta = [intercept, slope, log_sigma]"""
    b0, b1, log_sigma = theta
    sigma = np.exp(log_sigma)

    # Log-likelihood
    y_pred = b0 + b1 * x_obs
    ll = -len(x_obs) * log_sigma - 0.5 * np.sum((y_obs - y_pred)**2) / sigma**2

    # Log-priors (vague Normal priors for intercept and slope)
    lp = -0.5 * b0**2 / 100.0 - 0.5 * b1**2 / 100.0

    # Jacobian correction for log_sigma → sigma reparameterisation
    return ll + lp + log_sigma

result = run_mcmc(
    log_posterior=log_posterior,
    init=np.array([0.0, 2.0, 0.0]),   # starting values
    param_names=["intercept", "slope", "log_sigma"],
    n_iter=10000,
    n_burnin=2000,
    proposal_std=0.3,    # tune this for target acceptance rate
    alpha=0.05,
    model_name="Custom Linear Regression",
    seed=0,
)
result.summary()
result.plot().savefig("custom_mcmc.png")
```

**Important:** For parameters constrained to (0, ∞) (e.g. σ, λ, precision), sample in log-space and include the Jacobian term `log_sigma` (= log |∂σ/∂(log σ)|) in the log-posterior. This is already handled automatically in `mcmc_normal_mean_unknown_var` and `mcmc_linear_regression`.

---

## 9. Visualization

### Shared plot utilities

All shared utilities return `matplotlib.figure.Figure`. Use `.savefig(path, dpi=150)` to write to disk or `.show()` in interactive sessions.

```python
import matplotlib
matplotlib.use("Agg")  # required for headless environments (CI, scripts)
from statscore import (
    plot_regression, plot_residuals, plot_qq, plot_anova_groups,
    plot_t_test, plot_f_test, plot_simultaneous_ci, plot_posterior_normal,
    mult_lr_least_squares, bayes_normal_mean_known_var, AlternativeHypothesis,
)
import numpy as np

# Regression line on scatter plot
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2.1, 4.2, 5.8, 8.1, 9.9])
X = np.column_stack([np.ones(5), x])
ols = mult_lr_least_squares(X, y)

fig = plot_regression(x, y, ols.beta_hat,
                      x_label="x", y_label="y",
                      title="Linear Regression")
fig.savefig("regression.png", dpi=150)

# Residuals vs fitted
fig = plot_residuals(ols.fitted_values, ols.residuals)
fig.savefig("residuals.png", dpi=150)

# Normal Q-Q plot
fig = plot_qq(ols.residuals, title="Q-Q Plot of Residuals")
fig.savefig("qq.png", dpi=150)

# ANOVA group box plots
groups = [
    np.array([28, 23, 14, 27, 31]),
    np.array([33, 36, 34, 29, 24]),
    np.array([18, 21, 20, 22]),
]
fig = plot_anova_groups(groups,
                        group_labels=["Toxin A", "Toxin B", "Control"],
                        y_label="Response (mg/L)",
                        title="Toxin Experiment")
fig.savefig("groups.png", dpi=150)

# Bayesian prior vs posterior density
x_meas = np.array([9.8, 10.2, 10.1, 9.9, 10.3])
bayes_result = bayes_normal_mean_known_var(x_meas, sigma_sq=0.04, mu0=10.0, kappa0=2.0)
fig = plot_posterior_normal(bayes_result, title="Prior vs Posterior for μ")
fig.savefig("posterior.png", dpi=150)
```

### Result-level `.plot()` method

Every result dataclass also exposes its own `.plot()` method:

```python
from statscore import anova1_test_equality, z_test_mean, mcmc_normal_mean_unknown_var

# F-distribution with rejection region + SS bar chart
anova1_test_equality(groups, alpha=0.05).plot().savefig("anova_plot.png")

# Standard normal with rejection region
z_test_mean(x_meas, mu0=10.0, sigma=0.2,
            alternative=AlternativeHypothesis.TWO_SIDED).plot().savefig("z_plot.png")

# MCMC trace + KDE posterior panels
mcmc_normal_mean_unknown_var(x_meas, mu_prior_mean=0.0, mu_prior_std=10.0,
                              sigma_prior_alpha=2.0, sigma_prior_beta=1.0,
                              n_iter=8000, n_burnin=1000, seed=0).plot().savefig("mcmc.png")
```

---

## 10. Interactive CLI Reference

Launch with `statscore` or `python -m statscore`. Type the menu number and press Enter. Type `q` to quit.

### Menu structure

```
[1]  One-Way ANOVA
[2]  Two-Way ANOVA
[3]  Multiple Comparisons
[4]  Z-test for Mean
[5]  One-Sample t-test
[6]  Two-Sample t-test
[7]  Paired t-test
[8]  Chi-squared Test for Variance
[9]  F-test for Variances
[10] Simple Linear Regression
[11] Multiple Linear Regression (full inference suite)
[12] Regression Diagnostics
[13] Normality Check (Shapiro-Wilk)
[14] Variance Homogeneity (Levene)
[15] Confidence Interval for Mean
[16] Bayesian Inference — Normal (known variance)
[17] Bayesian Inference — Normal-Gamma (unknown variance)
[18] Bayesian Inference — Beta-Binomial
[19] Bayesian Inference — Gamma-Poisson
[20] Bayesian MCMC — Normal mean & variance
[21] Bayesian MCMC — Linear Regression
```

### Data input formats in the CLI

When the CLI prompts you for data, you can provide:

| Format | Example | Notes |
|---|---|---|
| Space-separated numbers | `1.2 3.4 5.6` | Standard flat input |
| Comma-separated numbers | `1.2, 3.4, 5.6` | Commas treated as spaces |
| File path | `/data/scores.csv` | Loads first numeric column |
| File path + column | `/data/scores.csv:score` | Loads named column |
| File path + `ALL` | `/data/matrix.csv:ALL` | Loads entire file as a 2-D matrix |
| Semicolon-separated rows | `1,2,3; 4,5,6; 7,8,9` | Creates a 3×3 matrix |
| `interactive` | type `interactive` | Enter rows one at a time until done |

### CLI walkthrough: Multiple Linear Regression

```
Select> 11

  Multiple Linear Regression
  ---------------------------
  Enter design matrix X (include a column of 1s for the intercept).

  How to enter the design matrix:
    - Type numbers separated by spaces/commas for a flat array
    - Separate rows with semicolons:  1,0.5,0.3; 1,0.8,0.7; ...
    - Type a file path to load from CSV/TSV/XLSX/JSON
    - Type 'interactive' to enter row by row

  Design matrix X: 1,1,0.25; 1,0.5,1; 1,0.2,0.5
  Enter response (y) data: 60 65 40
  Alpha [0.05]: 0.05
  Feature names (comma-separated, 3 names, or leave blank): (Intercept),attend,hw

  OLS Regression Summary
  ...

  Show simultaneous CIs for beta? (y/n) [n]: y
  Save CI plot? (y/n) [n]: y
  → Plot saved to simul_ci_beta.png

  Save diagnostic plots (residuals, Q-Q)? (y/n) [n]: n

  Run general hypothesis test H0: C*beta = c0? (y/n) [n]: n

  Compute prediction intervals for new observations? (y/n) [n]: n
```

### CLI walkthrough: MCMC Normal

```
Select> 20

  Enter sample data: 9.8 10.2 10.1 9.9 10.3 9.7 10.0 10.4
  Prior for mu: Normal(mu_prior_mean, mu_prior_std^2)
  mu_prior_mean [0.0]: 10.0
  mu_prior_std [10.0]:
  Prior for sigma^2: InvGamma(sigma_prior_alpha, sigma_prior_beta)
  sigma_prior_alpha [2.0]:
  sigma_prior_beta [1.0]:
  MCMC iterations [12000]:
  Burn-in [2000]:
  Alpha [0.05]:
  Running MCMC... (this may take a few seconds)

  MCMC Posterior Summary ...

  Show MCMC trace/posterior plots? (y/n) [n]: y
  → Plot saved to mcmc_normal.png
```

---

## 11. Browser UI Reference

```bash
pip install streamlit
streamlit run statscore/app/__init__.py
# Opens at http://localhost:8501
```

### Page-by-page overview

#### Data Input

Upload a file (CSV, TSV, XLSX, JSON) or paste raw numbers/a semicolon-separated matrix into the text area. Click **Load**. The dataset is shared across all pages via session state — you load it once and use it everywhere.

After loading, the page shows a live preview (first 10 rows) and a `describe()` statistical summary. Loaded dataset columns are available as selectors on other pages.

#### ANOVA

Two tabs:

- **One-Way:** Select one grouping column and one response column from the loaded dataset (or enter data manually). The page runs `anova1_test_equality`, displays the ANOVA table, group box-plot with jitter, and the F-distribution with rejection region shaded.

- **Two-Way:** Enter I, J, K and fill in all I×J cells. Each cell expects K space-separated values. The page tests all three sources (A, B, AB) and shows the full two-way ANOVA table plus F-distribution.

#### Significance Tests

Six tabs (Z, t one-sample, t two-sample, t paired, Chi², F-variance). Each tab:
1. Accepts data from the loaded dataset or manual entry
2. Shows parameter inputs (μ₀, α, alternative, etc.)
3. Displays metric cards (test statistic, p-value, decision)
4. Shows a distribution plot with the rejection region

#### Regression

Design matrix X and response y can be loaded from the dataset (select columns) or entered manually (supports semicolon-separated matrix format).

Features:
- Intercept toggle (auto-prepends a column of 1s)
- Coefficient table with significance stars
- R², adj-R², Sₑ, F-statistic metric cards
- Scatter+fit plot (simple regression) or coefficient forest plot (multiple)
- Residuals vs fitted values
- Normal Q-Q plot of residuals
- Four-panel Cook's D diagnostics

#### Bayesian Inference

Two models: Normal (known variance) and Normal-Gamma (unknown variance). Enter prior hyperparameters in the sidebar. The page shows posterior metric cards and a prior-vs-posterior density plot with the credible interval shaded.

#### Multiple Comparisons

1. Enter group data (select from loaded dataset or paste manually)
2. Enter the contrast matrix C as a semicolon-separated matrix (rows = contrasts, columns = groups)
3. Select method (Scheffé/Tukey/Bonferroni/Šidák/Best)
4. Choose: Simultaneous CIs (forest plot) or Simultaneous Tests (horizontal bar chart)

---

## 12. Result Objects

Every function returns a typed dataclass. All result objects provide:

| Method | Description |
|---|---|
| `.summary()` | Print a formatted table to stdout |
| `.plot()` | Return a `matplotlib.figure.Figure` |

You can also access individual fields directly:

```python
result = anova1_test_equality(groups, alpha=0.05)

# Access fields directly
print(result.F_statistic)   # F value
print(result.F_critical)    # critical value at alpha
print(result.p_value)       # p-value
print(result.reject_h0)     # True if H0 is rejected
print(result.df_between)    # degrees of freedom (numerator)
print(result.df_within)     # degrees of freedom (denominator)
```

Common fields across test results:
- `.test_statistic` (or `.z_statistic`, `.t_statistic`, `.f_statistic`, etc.)
- `.p_value`
- `.reject_h0` — boolean
- `.alpha`
- `.alternative` — `AlternativeHypothesis` enum value

---

## 13. Enums Reference

Import enums from the top-level package:

```python
from statscore import AlternativeHypothesis, CorrectionMethod, PredictionMethod, TwoWayTestFactor
```

### `AlternativeHypothesis`

| Value | H₁ | Use for |
|---|---|---|
| `TWO_SIDED` | θ ≠ θ₀ | Default for most tests |
| `LESS` | θ < θ₀ | One-sided lower test |
| `GREATER` | θ > θ₀ | One-sided upper test |

### `CorrectionMethod`

| Value | Description |
|---|---|
| `BONFERRONI` | Divide α by m; simple and conservative |
| `SIDAK` | 1 − (1−α)^{1/m}; slightly less conservative |
| `SCHEFFE` | Based on F-distribution; valid for all contrasts |
| `TUKEY` | Based on Studentised range; best for all pairwise |
| `BEST` | Automatically selects the method giving narrowest valid CIs |

### `PredictionMethod`

| Value | Description |
|---|---|
| `SCHEFFE` | Conservative; valid for arbitrary observation matrices |
| `BONFERRONI` | Better when m (number of new observations) is small |
| `BEST` | Selects the narrower method automatically |

### `TwoWayTestFactor`

| Value | Tests |
|---|---|
| `A` | H₀: α₁ = … = α_I = 0 (no main effect of A) |
| `B` | H₀: β₁ = … = β_J = 0 (no main effect of B) |
| `AB` | H₀: δ_{ij} = 0 for all i,j (no interaction) |

---

## 14. Input Formats

### 1-D data (samples, observations)

```python
import numpy as np

# Inline
x = np.array([1.2, 3.4, 5.6, 7.8])

# From a loaded CSV
from statscore import load_data
df = load_data("data.csv").df
x = df["measurement"].to_numpy(float)

# ANOVA groups from a long-format dataset
groups = [df.loc[df["group"] == g, "value"].to_numpy(float)
          for g in df["group"].unique()]
```

### 2-D matrices (design matrix X, contrast matrix C)

```python
# Inline
X = np.column_stack([np.ones(n), x1, x2])

# From a file (all columns)
from statscore import load_data
X = load_data("design.csv").df.to_numpy(float)

# From semicolon-separated string (as accepted by the CLI)
from statscore.cli._io import _parse_raw_string
C = _parse_raw_string("1,-1,0; 0,1,-1")  # returns np.ndarray of shape (2, 3)
```

### File path support in the CLI

The CLI data prompts accept:
- `/path/to/file.csv` — loads first numeric column
- `/path/to/file.csv:colname` — loads named column
- `/path/to/file.csv:ALL` — loads entire file as 2-D matrix (use for design matrices)

---

## 15. Worked End-to-End Example

**Scenario:** You have exam scores for 15 students, along with their lecture attendance rate and homework completion rate. You want to:
1. Check whether scores differ across three attendance tiers
2. Fit a multiple regression model
3. Assess model quality and check assumptions
4. Form a Bayesian estimate of the mean score

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
from statscore import (
    anova1_test_equality, anova1_ci_linear_combs,
    regression_summary, mult_norm_lr_test_general, mult_norm_lr_pred_ci,
    regression_diagnostics, shapiro_wilk_test, levene_test,
    bayes_normal_mean_unknown_var,
    plot_anova_groups, CorrectionMethod, PredictionMethod, AlternativeHypothesis,
)

# ── Dataset ──────────────────────────────────────────────────────────────────
attend   = np.array([1, 0.5, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.6, 0.1, 0, 0, 0.7, 0.8, 1])
homework = np.array([0.25, 1, 0.5, 1, 1, 0.75, 1, 0.25, 0, 0, 1, 0.5, 0.25, 0.75, 1])
y        = np.array([60, 65, 40, 70, 65, 70, 85, 70, 44, 20, 40, 30, 50, 77, 90], dtype=float)

# ── Step 1: One-Way ANOVA across attendance tiers ────────────────────────────
low    = y[attend <= 0.3]
medium = y[(attend > 0.3) & (attend <= 0.7)]
high   = y[attend > 0.7]
groups = [low, medium, high]

anova_result = anova1_test_equality(groups, alpha=0.05)
anova_result.summary()
plot_anova_groups(groups, group_labels=["Low", "Medium", "High"],
                 y_label="Score", title="Scores by Attendance Tier").savefig("anova.png")

# Post-hoc: pairwise comparisons
C = np.array([[1,-1,0], [0,1,-1], [1,0,-1]])
anova1_ci_linear_combs(groups, alpha=0.05, C=C, method=CorrectionMethod.TUKEY).summary()

# ── Step 2: Check ANOVA assumptions ──────────────────────────────────────────
for i, g in enumerate(groups, 1):
    r = shapiro_wilk_test(g, alpha=0.05)
    print(f"Group {i}: W={r.statistic:.4f}, p={r.p_value:.4f}")

levene_test(groups, alpha=0.05).summary()

# ── Step 3: Multiple linear regression ──────────────────────────────────────
X = np.column_stack([np.ones(15), attend, homework])
summary = regression_summary(X, y, alpha=0.05)
summary.summary(feature_names=["(Intercept)", "attendance", "homework"])
summary.plot().savefig("coeff_forest.png")

# Test: are both predictors equally important?
C_test = np.array([[0, 1, -1]])
mult_norm_lr_test_general(X, y, C_test, np.array([0.0]), alpha=0.05).summary()

# Predict scores for two hypothetical students
D = np.array([[1, 0.6, 0.8], [1, 0.9, 1.0]])
mult_norm_lr_pred_ci(X, y, D, alpha=0.05, method=PredictionMethod.BEST).summary()

# ── Step 4: Regression diagnostics ──────────────────────────────────────────
diag = regression_diagnostics(X, y)
diag.summary()
diag.plot().savefig("diagnostics.png")

# ── Step 5: Bayesian estimate of mean score ───────────────────────────────────
bayes = bayes_normal_mean_unknown_var(y, mu0=60.0, kappa0=1.0, alpha0=2.0, beta0=100.0)
bayes.summary()
bayes.plot().savefig("posterior.png")
```

This example demonstrates the full statscore workflow: load data, explore group differences with ANOVA, check assumptions with diagnostics, fit and test a regression model, generate predictions, and finish with a Bayesian posterior summary.

---

*For a concise API reference and package structure overview, see [README.md](README.md).*
