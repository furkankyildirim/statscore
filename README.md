# statscore

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-205%20passing-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-0.0.3-orange.svg)]()

**statscore** is a production-quality Python statistical analysis library built for correctness, usability, and transparency. It implements the full classical inference pipeline — ANOVA, normal distribution testing, multiple linear regression, and Bayesian inference — with a consistent structured-output design philosophy: every function returns a typed dataclass with a `.summary()` method for formatted console output and a `.plot()` method that returns a `matplotlib.Figure`.

The library is designed to be used three ways: as a **Python API** importable in scripts and notebooks, as an **interactive CLI** requiring no programming, and as a **browser UI** built on Streamlit. All three interfaces call the same underlying statistical engine, so results are always identical regardless of how you access them.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Statistical Methods](#statistical-methods)
- [Design Philosophy](#design-philosophy)
- [Installation](#installation)
- [Interfaces](#interfaces)
  - [Python API](#python-api)
  - [Interactive CLI](#interactive-cli)
  - [Browser UI](#browser-ui)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [API Reference](#api-reference)
- [Mathematical Background](#mathematical-background)
- [Development](#development)

---

## Project Overview

Statistical analysis software often falls into one of two camps: full-featured platforms (R, SPSS, SAS) that are hard to integrate into software pipelines, or Python libraries (statsmodels, scipy.stats) with inconsistent APIs, raw tuple return values, and outputs that require significant post-processing. statscore occupies the space between them.

**What it covers:**

| Domain | Methods |
|--------|---------|
| **Analysis of Variance** | One-way and two-way ANOVA; sum-of-squares decomposition; MLE estimation; F-tests |
| **Multiple Comparison Procedures** | Bonferroni, Šidák, Scheffé, Tukey corrections; simultaneous CIs; family-wise error rate (FWER) control |
| **Normal Distribution Testing** | Z-test, one/two-sample t-test, paired t-test, chi-squared variance test, F-test for variances; one- and two-sided alternatives |
| **Multiple Linear Regression** | OLS estimation; TSS partition; full regression summary table; simultaneous CIs; confidence regions; general hypothesis tests; prediction intervals |
| **Bayesian Conjugate Inference** | Normal-Normal, Normal-Gamma, Beta-Binomial, Gamma-Poisson; closed-form posteriors; credible intervals |
| **Bayesian MCMC** | General-purpose Metropolis-Hastings sampler; Normal data and linear regression models; trace plots; KDE posteriors; acceptance rate diagnostics |
| **Model Diagnostics** | Shapiro-Wilk normality test; Levene's variance homogeneity test; leverage, standardized residuals, Cook's D; mean confidence intervals |
| **Visualization** | Scatter+fit, residuals vs fitted, normal Q-Q, ANOVA box plots, CI forest plots, Bayesian density plots, MCMC trace+KDE panels |
| **Data I/O** | Load `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json` via a single `load_data` call |

**What makes it different:**

- Every function returns a **typed dataclass** — never a raw tuple, never a dict, never a plain number. You always know what you're working with.
- Every result has `.summary()` (formatted table to stdout) and `.plot()` (matplotlib Figure) built in.
- The **CLI** gives access to all 21 analyses without writing a single line of Python. Data can be entered inline or loaded from files.
- The **Streamlit browser UI** provides a point-and-click interface to the same functions, with live plots that update as parameters change.
- All categorical parameters use **enums** — `AlternativeHypothesis.TWO_SIDED`, not the string `"two-sided"`. Typos are caught at import time, not at runtime.
- The library is tested with **205 tests** covering edge cases, mathematical correctness, and CLI behaviour.

---

## Statistical Methods

### Analysis of Variance (ANOVA)

ANOVA partitions total variability in a dataset into components attributable to different sources and tests whether those sources explain a statistically significant amount of variation.

**One-Way ANOVA** tests whether I ≥ 2 group means are equal. The model is X_{ij} = μ + α_i + ε_{ij} where ε_{ij} ~ N(0, σ²). The total sum of squares SS_T = SS_Between + SS_Within is partitioned, and the F-statistic F = MS_Between / MS_Within follows an F(I−1, n−I) distribution under H₀.

**Two-Way ANOVA** extends this to two crossed factors A (I levels) and B (J levels) with K ≥ 2 replicates per cell. The model X_{ijk} = μ + α_i + β_j + δ_{ij} + ε_{ijk} decomposes SS_T into main effects SS_A, SS_B, the interaction SS_AB, and error SS_E. Each source is tested independently.

**Multiple Comparisons** control the family-wise error rate when testing multiple contrasts simultaneously. A contrast c is a linear combination of group means with Σcᵢ = 0. statscore implements Bonferroni (per-comparison α/m), Šidák (1−(1−α)^{1/m}), Scheffé (F-distribution based, valid for all contrasts), and Tukey (Studentised range, optimal for all pairwise). The `BEST` method automatically selects whichever gives the narrowest valid intervals.

### Normal Distribution Significance Testing

Six tests for normal-population parameters:

- **Z-test**: H₀: μ = μ₀ with σ known. Statistic Z = (x̄ − μ₀)/(σ/√n) ~ N(0,1).
- **One-sample t-test**: H₀: μ = μ₀ with σ unknown. Statistic T = (x̄ − μ₀)/(s/√n) ~ t(n−1).
- **Two-sample t-test**: Pooled (equal variances assumed) or Welch (unequal variances, Welch–Satterthwaite df).
- **Paired t-test**: T = d̄/(s_d/√n) ~ t(n−1) on pairwise differences.
- **Chi-squared variance test**: H₀: σ² = σ₀². Statistic Q = (n−1)s²/σ₀² ~ χ²(n−1).
- **F-test for variances**: H₀: σ₁² = σ₂². Statistic F = s₁²/s₂² ~ F(n₁−1, n₂−1).

All six tests support `TWO_SIDED`, `LESS`, and `GREATER` alternatives.

### Multiple Linear Regression

statscore implements the full OLS inference pipeline for Y = Xβ + ε, ε ~ N(0, σ²I):

- **Estimation**: β̂ = (X^TX)^{−1}X^Ty, σ̂² unbiased = RSS/(n−p)
- **Goodness of fit**: R² = RegSS/TSS, adj R² = 1 − (1−R²)(n−1)/(n−p)
- **Full summary table**: coefficients, standard errors, t-statistics, p-values, significance stars, CIs — equivalent to R's `summary(lm(...))`
- **Simultaneous CIs**: Scheffé-based confidence intervals for all β_i simultaneously (FWER control)
- **Confidence regions**: ellipsoidal CR for any linear transformation Cβ
- **General hypothesis test**: H₀: Cβ = c₀ using F = (Cβ̂−c₀)^T[C(X^TX)^{−1}C^T]^{−1}(Cβ̂−c₀)/(r·σ̂²) ~ F(r, n−p)
- **Simultaneous prediction intervals**: Scheffé or Bonferroni intervals for m new observations simultaneously

### Bayesian Inference — Conjugate Models

When the prior and likelihood belong to a conjugate family, the posterior has a known closed form. statscore implements four conjugate models:

| Model | Prior | Likelihood | Posterior |
|-------|-------|------------|-----------|
| **Normal-Normal** | μ ~ N(μ₀, σ²/κ₀), σ² known | X | N(μₙ, σ²/κₙ) |
| **Normal-Gamma** | (μ,τ) ~ NG(μ₀, κ₀, α₀, β₀) | X ~ N(μ,1/τ) | Normal-Gamma(μₙ, κₙ, αₙ, βₙ) |
| **Beta-Binomial** | p ~ Beta(α₀, β₀) | k | p ~ Bin(n,p) | Beta(α₀+k, β₀+n−k) |
| **Gamma-Poisson** | λ ~ Gamma(α₀, β₀) | xᵢ ~ Poisson(λ) | Gamma(α₀+Σxᵢ, β₀+n) |

All four return credible intervals and prior/posterior density plots.

### Bayesian Inference — MCMC

For models with no tractable conjugate posterior, statscore provides a Metropolis-Hastings sampler (`run_mcmc`) that accepts any log-posterior function. Two ready-made models are included:

- **`mcmc_normal_mean_unknown_var`**: joint posterior of (μ, σ) for Normal data with Normal prior on μ and Inverse-Gamma prior on σ². Parameters constrained to (0,∞) are sampled in log-space to keep proposals Gaussian.
- **`mcmc_linear_regression`**: posterior of (β₀,…,β_{p−1}, σ) for Y = Xβ + ε with Normal priors on β and Inverse-Gamma on σ². Chains are OLS-initialised to reduce burn-in.

Results include trace plots, KDE posterior density panels, posterior mean/std, and HPD credible intervals.

### Diagnostics

- **Shapiro-Wilk test**: W-statistic normality test, returns Q-Q plot
- **Levene's test**: Centre-of-residuals test for homogeneity of variance across groups
- **Regression diagnostics**: leverage h_{ii}, standardized residuals, Cook's D; flags high-leverage (h > 2p/n) and influential points (D > 4/n); four-panel diagnostic plot
- **Mean CI**: z-interval (σ known) or t-interval (σ unknown) for the population mean

---

## Design Philosophy

**Every function returns a typed dataclass.** This means you can always introspect fields with tab-completion, pass results between functions, or write generic code that works with any test result. There are no raw tuples, no dicts with magic string keys, and no output you have to parse.

**`.summary()` and `.plot()` are universal.** Every result object provides both. This makes the API consistent: once you know how to use one function, you know how to use all of them.

**Enums instead of strings.** All categorical parameters (`AlternativeHypothesis`, `CorrectionMethod`, `PredictionMethod`, `TwoWayTestFactor`) use enums. This gives you IDE autocomplete, makes typos a `TypeError` rather than a silent wrong result, and lets code be read without memorising magic strings.

**Strict layered architecture.** The codebase is organised into dependency layers (utils → methods → plots/io/cli) with no circular imports. Domain modules never import from each other unless explicitly justified. Plot logic lives inside `result.plot()` methods rather than as standalone functions (except for a handful of shared utilities).

**Three interfaces, one engine.** The Python API, CLI, and Streamlit browser UI all call the same `statscore.methods.*` functions. If a result looks different between interfaces, there is a bug. The consistency is enforced by the typed API — the CLI cannot accidentally pass wrong types.

---

## Installation

```bash
# From PyPI
pip install statscore

# From source (with dev dependencies)
git clone https://github.com/furkankyildirim/statscore.git
cd statscore
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9 · NumPy ≥ 1.21 · SciPy ≥ 1.7 · pandas ≥ 1.3 · matplotlib ≥ 3.5 · openpyxl ≥ 3.0

For the browser UI: `pip install "statscore[ui]"`

---

## Interfaces

### Python API

All public symbols are importable from the top-level package. Nothing requires knowing internal module paths.

```python
from statscore import (
    # ANOVA
    anova1_partition_tss, anova1_test_equality,
    anova1_ci_linear_combs, anova1_test_linear_combs,
    anova2_partition_tss, anova2_mle, anova2_test_equality,
    bonferroni_correction, sidak_correction,
    # Testing
    z_test_mean, t_test_mean, t_test_two_sample,
    t_test_paired, chi2_test_variance, f_test_variances,
    # Regression
    mult_lr_least_squares, mult_lr_partition_tss,
    regression_summary, mult_norm_lr_simul_ci,
    mult_norm_lr_test_general, mult_norm_lr_pred_ci,
    # Bayesian
    bayes_normal_mean_known_var, bayes_normal_mean_unknown_var,
    bayes_beta_binomial, bayes_gamma_poisson,
    run_mcmc, mcmc_normal_mean_unknown_var, mcmc_linear_regression,
    # Diagnostics
    shapiro_wilk_test, levene_test,
    regression_diagnostics, mean_confidence_interval,
    # I/O
    load_data,
    # Enums
    AlternativeHypothesis, CorrectionMethod,
    PredictionMethod, TwoWayTestFactor,
)
```

### Interactive CLI

```bash
statscore            # after pip install
python -m statscore  # or directly
```

A 21-item numbered menu covering all analyses. Data can be entered as space/comma-separated numbers, loaded from a file, or pasted as a semicolon-separated matrix (`1,2,3; 4,5,6; 7,8,9`). After every analysis, the CLI optionally saves plots to PNG files.

```
================================================================
  statscore — Statistical Toolbox  (v0.0.3)
  Type a number to select a method, or 'q' to quit.
================================================================
  ── ANOVA ──────────────────────────────────────────────────
  [1]  One-Way ANOVA (F-test, ANOVA table)
  [2]  Two-Way ANOVA (Factor A / B / Interaction)
  [3]  Multiple Comparisons (Bonferroni/Scheffe/Tukey/Sidak)
  ── Significance Tests ─────────────────────────────────────
  [4]  Z-test for Mean (sigma known)
  [5]  One-Sample t-test
  [6]  Two-Sample t-test (pooled or Welch)
  [7]  Paired t-test
  [8]  Chi-squared Test for Variance
  [9]  F-test for Variances
  ── Regression ─────────────────────────────────────────────
  [10] Simple Linear Regression
  [11] Multiple Linear Regression (full inference suite)
  [12] Regression Diagnostics (Leverage / Cook's D)
  ── Diagnostics ────────────────────────────────────────────
  [13] Normality Check (Shapiro-Wilk + Q-Q plot)
  [14] Variance Homogeneity (Levene)
  [15] Confidence Interval for Mean
  ── Bayesian Inference — Conjugate Priors ──────────────────
  [16] Bayesian Inference — Normal (known variance)
  [17] Bayesian Inference — Normal-Gamma (unknown variance)
  [18] Bayesian Inference — Beta-Binomial (success prob.)
  [19] Bayesian Inference — Gamma-Poisson (count rate)
  ── Bayesian Inference — MCMC ──────────────────────────────
  [20] Bayesian MCMC — Normal mean & variance
  [21] Bayesian MCMC — Linear Regression
================================================================
```

### Browser UI

The browser UI lives in `statscore/app/__init__.py` and is launched via the `statscore-ui` console script, which bakes in the correct Streamlit flags automatically (headless mode, XSRF protection, 50 MB upload cap, minimal toolbar).

```bash
# Recommended — install with UI extra and use the console script
pip install "statscore[ui]"
statscore-ui

# Or run directly with Streamlit (requires streamlit>=1.30)
pip install streamlit
streamlit run statscore/app/__init__.py
# Opens at http://localhost:8501
```

Six pages share a loaded dataset via session state:

| Page | Content |
|------|---------|
| **Data Input** | Upload CSV/TSV/XLSX/JSON or paste data; live `describe()` preview; shared across pages |
| **ANOVA** | One-way (ANOVA table, box plots, F-distribution); Two-way (I×J×K cell entry, full decomposition table) |
| **Significance Tests** | All 6 tests; metric cards; reject/fail banner; distribution plot with shaded rejection region |
| **Regression** | Design matrix + y; coefficient table with significance stars; R²/adj-R²/Sₑ/F cards; scatter+fit, residuals, Q-Q, Cook's D |
| **Bayesian Inference** | Normal-Normal and Normal-Gamma conjugate; prior/posterior density with credible interval |
| **Multiple Comparisons** | Contrast matrix input; Scheffé/Tukey/Bonferroni/Šidák/Best; CI forest plot and test chart |

---

## Quick Start

### One-Way ANOVA

```python
import numpy as np
from statscore import anova1_partition_tss, anova1_test_equality, plot_anova_groups

groups = [
    np.array([28, 23, 14, 27, 31]),   # Group A
    np.array([33, 36, 34, 29, 24]),   # Group B
    np.array([18, 21, 20, 22]),       # Group C (control)
]

# Partition total variability
p = anova1_partition_tss(groups)
print(f"SS_between={p.SS_between:.2f}  SS_within={p.SS_within:.2f}  SS_total={p.SS_total:.2f}")

# F-test for equality of group means
result = anova1_test_equality(groups, alpha=0.05)
result.summary()
# ═══════════════════════════════════════════════════════
#   One-Way ANOVA Table
# ═══════════════════════════════════════════════════════
#   Source     df        SS        MS         F
# ───────────────────────────────────────────────────────
#   Between     2    276.11    138.05      5.57
#   Within     11    272.75     24.80
# ───────────────────────────────────────────────────────
#   Total      13    548.86
# ═══════════════════════════════════════════════════════
#   F critical (α=0.05): 3.9823   p-value: 0.0214
#   Decision: Reject H0
# ═══════════════════════════════════════════════════════

result.plot().savefig("anova1.png")
plot_anova_groups(groups, group_labels=["A", "B", "Control"]).savefig("groups.png")
```

### Multiple Comparisons

```python
from statscore import anova1_ci_linear_combs, anova1_test_linear_combs, CorrectionMethod
import numpy as np

C = np.array([[1,-1,0], [0,1,-1], [1,0,-1]])  # pairwise contrasts

# Simultaneous CIs — auto-select narrowest valid method
ci = anova1_ci_linear_combs(groups, alpha=0.05, C=C, method=CorrectionMethod.BEST)
ci.summary()
ci.plot().savefig("simul_ci.png")

# Simultaneous hypothesis tests H0: Cμ = 0
tests = anova1_test_linear_combs(groups, alpha=0.05, C=C,
                                  d=np.zeros(3), method=CorrectionMethod.TUKEY)
tests.summary()
```

### Significance Testing

```python
from statscore import z_test_mean, t_test_two_sample, AlternativeHypothesis
import numpy as np

x1 = np.array([10.2, 9.8, 10.1, 10.3, 9.9, 10.0, 10.4, 9.7])
x2 = np.array([10.8, 11.2, 10.9, 11.1, 10.7])

# Z-test (σ known)
z_test_mean(x1, mu0=10.0, sigma=0.3, alpha=0.05,
            alternative=AlternativeHypothesis.TWO_SIDED).summary()

# Two-sample Welch's t-test (unequal variances)
result = t_test_two_sample(x1, x2, alpha=0.05, equal_var=False,
                           alternative=AlternativeHypothesis.TWO_SIDED)
result.summary()
result.plot().savefig("t_test.png")
```

### Multiple Linear Regression

```python
import numpy as np
from statscore import (
    mult_lr_least_squares, mult_lr_partition_tss, regression_summary,
    mult_norm_lr_simul_ci, mult_norm_lr_test_general, mult_norm_lr_pred_ci,
    PredictionMethod,
)

attend   = np.array([1, 0.5, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.6, 0.1, 0, 0, 0.7, 0.8, 1])
homework = np.array([0.25, 1, 0.5, 1, 1, 0.75, 1, 0.25, 0, 0, 1, 0.5, 0.25, 0.75, 1])
y        = np.array([60, 65, 40, 70, 65, 70, 85, 70, 44, 20, 40, 30, 50, 77, 90], dtype=float)
X        = np.column_stack([np.ones(15), attend, homework])

# OLS estimates
ols = mult_lr_least_squares(X, y)
print(f"β̂ = {ols.beta_hat}")         # [14.37, 46.20, 30.45]
print(f"Sₑ² = {ols.sigma2_unbiased:.4f}")  # 19.8400

# R² and adjusted R²
tss = mult_lr_partition_tss(X, y)
print(f"R²={tss.R_squared:.4f}  adj R²={tss.adj_R_squared:.4f}")
# R²=0.9588  adj R²=0.9520

# Full summary table (like R's summary(lm(...)))
regression_summary(X, y, alpha=0.05).summary(
    feature_names=["(Intercept)", "attendance", "homework"])

# Simultaneous CIs for all β_i
mult_norm_lr_simul_ci(X, y, alpha=0.05).summary()

# General hypothesis test: H0: β_attend = β_homework
C = np.array([[0, 1, -1]]); c0 = np.array([0.0])
mult_norm_lr_test_general(X, y, C, c0, alpha=0.05).summary()
# F = 11.4796, p = 0.0054 → Reject H0

# Simultaneous prediction intervals for new observations
D = np.array([[1, 0.5, 1.0], [1, 1.0, 0.0]])
mult_norm_lr_pred_ci(X, y, D, alpha=0.05, method=PredictionMethod.BEST).summary()
```

### Bayesian Conjugate Inference

```python
import numpy as np
from statscore import (
    bayes_normal_mean_known_var, bayes_normal_mean_unknown_var,
    bayes_beta_binomial, bayes_gamma_poisson,
)

x = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4])

# Normal-Normal (variance known)
r = bayes_normal_mean_known_var(x, sigma_sq=0.04, mu0=10.0, kappa0=2.0)
r.summary()   # 95% CI: (9.916, 10.164)
r.plot().savefig("posterior_normal.png")

# Normal-Gamma (variance unknown)
bayes_normal_mean_unknown_var(x, mu0=10.0, kappa0=1.0,
                               alpha0=2.0, beta0=0.1).summary()

# Beta-Binomial (success probability from 14/20 successes)
bayes_beta_binomial(n_trials=20, n_successes=14,
                    alpha0=1.0, beta0=1.0).summary()
# Posterior: Beta(15, 7)  mean=0.6818  95% CI=(0.461, 0.858)

# Gamma-Poisson (Poisson rate from count data)
counts = np.array([3, 5, 2, 4, 6, 3, 5, 4])
bayes_gamma_poisson(counts, alpha0=1.0, beta0=1.0).summary()
# Posterior: Gamma(33, 9)  mean=3.667
```

### Bayesian MCMC

```python
import numpy as np
from statscore import mcmc_normal_mean_unknown_var, mcmc_linear_regression

x = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4])

# MCMC for Normal data: joint posterior of (μ, σ)
result = mcmc_normal_mean_unknown_var(
    x,
    mu_prior_mean=0.0, mu_prior_std=10.0,
    sigma_prior_alpha=2.0, sigma_prior_beta=1.0,
    n_iter=12000, n_burnin=2000, alpha=0.05, seed=42,
)
result.summary()
# Param   Post. Mean  Post. Std   2.5%      97.5%
# mu      10.0476     0.0694    9.9116    10.1836
# sigma    0.2315     0.0536    0.1534     0.3590
result.plot().savefig("mcmc_trace.png")  # trace + KDE panels

# MCMC for linear regression
X = np.column_stack([np.ones(15), attend, homework])
mcmc_linear_regression(X, y, beta_prior_std=10.0,
                        sigma_prior_alpha=2.0, sigma_prior_beta=1.0,
                        n_iter=15000, n_burnin=3000, seed=42).summary()
```

### Diagnostics

```python
from statscore import (
    shapiro_wilk_test, levene_test,
    regression_diagnostics, mean_confidence_interval,
)

# Normality check before t-test / ANOVA
shapiro_wilk_test(x, alpha=0.05).summary()
# W = 0.9502, p = 0.7121 → Fail to reject H0 (consistent with normality)

# Variance homogeneity check before ANOVA / pooled t-test
levene_test(groups, alpha=0.05).summary()

# Regression diagnostics: influence analysis
diag = regression_diagnostics(X, y)
diag.summary()   # flags high-leverage and influential points
diag.plot().savefig("diagnostics.png")  # 4-panel: leverage, std resid, Cook's D

# Mean confidence interval
mean_confidence_interval(x, alpha=0.05).summary()
# 95% t-interval for μ: (9.845, 10.255)
```

### Data I/O

```python
from statscore import load_data

result = load_data("experiment.csv")
print(result.format)        # "csv"
print(result.n_rows)        # 120
print(result.column_names)  # ["subject", "group", "score"]
df = result.df              # pandas DataFrame

# Extract groups for ANOVA
groups = [df.loc[df["group"] == g, "score"].to_numpy(float)
          for g in df["group"].unique()]
```

### Visualization

All plot functions return `matplotlib.figure.Figure`. Use `.savefig()` for files or `.show()` in interactive sessions. Call `matplotlib.use("Agg")` before importing statscore in headless environments.

```python
import matplotlib; matplotlib.use("Agg")
from statscore import (
    plot_regression, plot_residuals, plot_qq, plot_anova_groups,
    plot_t_test, plot_f_test, plot_simultaneous_ci, plot_posterior_normal,
)
```

Shared utilities for building custom figures. Every result dataclass also exposes `.plot()` which draws a context-appropriate chart automatically.

---

## Package Structure

```
statscore/
├── __init__.py              # Public API — re-exports ~59 symbols; __version__
├── __main__.py              # python -m statscore entry point
├── app/
│   ├── __init__.py          # Streamlit browser UI — 6 pages
│   └── _launcher.py         # statscore-ui console script entry point (subprocess wrapper)
├── plots/
│   └── __init__.py          # Shared plot utilities — 8 functions
├── io/
│   └── __init__.py          # load_data → LoadedData
├── cli/
│   ├── __init__.py          # main(), _MENU_HANDLERS, _print_menu() — 21-item menu
│   ├── _anova.py            # _run_one_way_anova, _run_two_way_anova, _run_anova_multiple_comparisons
│   ├── _testing.py          # _run_z_test … _run_mean_ci (9 handlers)
│   ├── _regression.py       # _run_simple_regression … _run_mcmc_regression (9 handlers)
│   └── _io.py               # _parse_data_input, _parse_matrix_input, _parse_raw_string, _parse_groups_input
├── methods/
│   ├── anova/
│   │   ├── _results.py      # ANOVA1PartitionResult, ANOVA1TestResult, ANOVA2*, SimultaneousCIResult, SimultaneousTestResult
│   │   ├── one_way.py       # anova1_partition_tss, anova1_test_equality
│   │   ├── two_way.py       # anova2_partition_tss, anova2_mle, anova2_test_equality
│   │   └── multiple_tests.py  # anova1_is_contrast/orthogonal, bonferroni/sidak_correction, anova1_ci/test_linear_combs
│   ├── bayes/
│   │   ├── _results.py      # NormalMeanKnownVarResult, NormalMeanUnknownVarResult
│   │   ├── _mcmc_results.py # MCMCResult, ConjugateModelResult
│   │   ├── conjugate.py     # bayes_normal_mean_known_var, bayes_normal_mean_unknown_var
│   │   └── mcmc.py          # run_mcmc, mcmc_normal_mean_unknown_var, mcmc_linear_regression,
│   │                        # bayes_beta_binomial, bayes_gamma_poisson
│   ├── diagnostics/
│   │   ├── _results.py      # ShapiroWilkResult, LeveneResult, RegressionDiagnosticsResult, MeanConfidenceIntervalResult
│   │   └── __init__.py      # shapiro_wilk_test, levene_test, regression_diagnostics, mean_confidence_interval
│   ├── regression/
│   │   ├── _results.py      # OLSResult, PartitionTSSResult, SimultaneousCIBetaResult,
│   │   │                    # ConfidenceRegionResult, HypothesisTestResult, PredictionCIResult, RegressionSummaryResult
│   │   ├── least_squares.py # mult_lr_least_squares, mult_lr_partition_tss
│   │   ├── inference.py     # mult_norm_lr_simul_ci, mult_norm_lr_cr, mult_norm_lr_is_in_cr,
│   │   │                    # mult_norm_lr_test_general, mult_norm_lr_test_comp, mult_norm_lr_test_linear_reg
│   │   ├── prediction.py    # mult_norm_lr_pred_ci
│   │   └── summary.py       # regression_summary
│   └── testing/
│       ├── _results.py      # ZTestResult, TTestOneSampleResult, TTestTwoSampleResult,
│       │                    # TTestPairedResult, Chi2VarianceTestResult, FTestVariancesResult
│       ├── one_sample.py    # z_test_mean, t_test_mean, chi2_test_variance
│       └── two_sample.py    # t_test_two_sample, t_test_paired, f_test_variances
└── utils/
    ├── enums.py             # AlternativeHypothesis, CorrectionMethod, PredictionMethod, TwoWayTestFactor
    ├── distributions.py     # f_critical, t_ppf, chi2_ppf, norm_ppf, q_ppf, p-value helpers
    └── validation.py        # validate_positive, validate_1d_sample, validate_alternative, …
```

---

## API Reference

### Enums

| Enum | Values | Used by |
|------|--------|---------|
| `AlternativeHypothesis` | `TWO_SIDED`, `LESS`, `GREATER` | All 6 significance tests |
| `CorrectionMethod` | `SCHEFFE`, `TUKEY`, `BONFERRONI`, `SIDAK`, `BEST` | ANOVA multiple comparisons |
| `PredictionMethod` | `SCHEFFE`, `BONFERRONI`, `BEST` | Regression prediction CIs |
| `TwoWayTestFactor` | `A`, `B`, `AB` | Two-way ANOVA |

---

### ANOVA

| Function | Returns | Description |
|----------|---------|-------------|
| `anova1_partition_tss(data)` | `ANOVA1PartitionResult` | SS_between, SS_within, SS_total |
| `anova1_test_equality(data, alpha)` | `ANOVA1TestResult` | F-test, df, MS, F-critical, p-value |
| `anova1_is_contrast(c)` | `bool` | Checks Σcᵢ = 0 |
| `anova1_is_orthogonal(n, c1, c2)` | `bool` | Checks Σnᵢc₁ᵢc₂ᵢ = 0 |
| `bonferroni_correction(alpha, m)` | `float` | Per-comparison α = α/m |
| `sidak_correction(alpha, m)` | `float` | Per-comparison α = 1−(1−α)^{1/m} |
| `anova1_ci_linear_combs(data, alpha, C, method)` | `SimultaneousCIResult` | Simultaneous CIs for Cμ |
| `anova1_test_linear_combs(data, alpha, C, d, method)` | `SimultaneousTestResult` | Simultaneous tests H₀: Cμ = d |
| `anova2_partition_tss(data)` | `ANOVA2PartitionResult` | SS_A, SS_B, SS_AB, SS_E, SS_T |
| `anova2_mle(data)` | `ANOVA2MLEResult` | μ̂, α̂ᵢ, β̂ⱼ, δ̂ᵢⱼ MLE estimates |
| `anova2_test_equality(data, alpha, test)` | `ANOVA2TestResult` | F-test for A, B, or AB |

---

### Significance Testing

| Function | Returns | Description |
|----------|---------|-------------|
| `z_test_mean(x, mu0, sigma, alpha, alternative)` | `ZTestResult` | Z-test (σ known) |
| `t_test_mean(x, mu0, alpha, alternative)` | `TTestOneSampleResult` | One-sample t-test |
| `chi2_test_variance(x, sigma0_sq, alpha, alternative)` | `Chi2VarianceTestResult` | Chi-squared variance test |
| `t_test_two_sample(x1, x2, alpha, alternative, equal_var)` | `TTestTwoSampleResult` | Pooled or Welch t-test |
| `t_test_paired(x1, x2, alpha, alternative)` | `TTestPairedResult` | Paired t-test |
| `f_test_variances(x1, x2, alpha, alternative)` | `FTestVariancesResult` | F-test for σ₁² = σ₂² |

---

### Regression

| Function | Returns | Description |
|----------|---------|-------------|
| `mult_lr_least_squares(X, y)` | `OLSResult` | β̂, σ²_MLE, σ²_unbiased, fitted values, residuals |
| `mult_lr_partition_tss(X, y)` | `PartitionTSSResult` | RegSS, RSS, TSS, R², adj R² |
| `regression_summary(X, y, alpha, feature_names)` | `RegressionSummaryResult` | Full OLS table with SE, t, p, CIs, F-test |
| `mult_norm_lr_simul_ci(X, y, alpha)` | `SimultaneousCIBetaResult` | Scheffé simultaneous CIs for all β_i |
| `mult_norm_lr_cr(X, y, C, alpha)` | `ConfidenceRegionResult` | Ellipsoidal CR for Cβ |
| `mult_norm_lr_is_in_cr(X, y, C, c0, alpha)` | `bool` | Is c₀ inside the CR? |
| `mult_norm_lr_test_general(X, y, C, c0, alpha)` | `HypothesisTestResult` | F-test for H₀: Cβ = c₀ |
| `mult_norm_lr_test_comp(X, y, alpha, components)` | `HypothesisTestResult` | F-test for H₀: β_j = 0 (subset) |
| `mult_norm_lr_test_linear_reg(X, y, alpha)` | `HypothesisTestResult` | Overall F-test for regression significance |
| `mult_norm_lr_pred_ci(X, y, D, alpha, method)` | `PredictionCIResult` | Simultaneous prediction CIs for D |

---

### Bayesian Inference

| Function | Returns | Description |
|----------|---------|-------------|
| `bayes_normal_mean_known_var(x, sigma_sq, mu0, kappa0, alpha)` | `NormalMeanKnownVarResult` | Normal-Normal conjugate; posterior + predictive |
| `bayes_normal_mean_unknown_var(x, mu0, kappa0, alpha0, beta0, alpha)` | `NormalMeanUnknownVarResult` | Normal-Gamma conjugate; marginal CIs for μ, σ² |
| `bayes_beta_binomial(n_trials, n_successes, alpha0, beta0, alpha)` | `ConjugateModelResult` | Beta posterior for success probability |
| `bayes_gamma_poisson(x, alpha0, beta0, alpha)` | `ConjugateModelResult` | Gamma posterior for Poisson rate |
| `run_mcmc(log_posterior, init, param_names, n_iter, n_burnin, proposal_std, alpha, model_name, seed)` | `MCMCResult` | General Metropolis-Hastings sampler |
| `mcmc_normal_mean_unknown_var(x, mu_prior_mean, mu_prior_std, sigma_prior_alpha, sigma_prior_beta, n_iter, n_burnin, alpha, seed)` | `MCMCResult` | MCMC for Normal data |
| `mcmc_linear_regression(X, y, beta_prior_mean, beta_prior_std, sigma_prior_alpha, sigma_prior_beta, n_iter, n_burnin, alpha, seed)` | `MCMCResult` | MCMC for linear regression |

---

### Diagnostics

| Function | Returns | Description |
|----------|---------|-------------|
| `shapiro_wilk_test(x, alpha)` | `ShapiroWilkResult` | Shapiro-Wilk W-statistic; Q-Q plot |
| `levene_test(data, alpha)` | `LeveneResult` | Centre-of-residuals variance homogeneity test |
| `regression_diagnostics(X, y)` | `RegressionDiagnosticsResult` | Leverage h_{ii}, standardized residuals, Cook's D; influence flags |
| `mean_confidence_interval(x, alpha, sigma)` | `MeanConfidenceIntervalResult` | z-interval (σ known) or t-interval |

---

### Visualization

#### Shared utilities (`statscore.plots`)

| Function | Description |
|----------|-------------|
| `plot_regression(x, y, beta_hat, ...)` | Scatter with fitted regression line |
| `plot_residuals(fitted, residuals, ...)` | Residuals vs fitted values |
| `plot_qq(x, ...)` | Normal Q-Q plot |
| `plot_anova_groups(data, group_labels, ...)` | Box plots with jittered data points |
| `plot_t_test(t_stat, t_crit, df, alternative, ...)` | t-distribution with rejection region |
| `plot_f_test(f_stat, f_crit_low, f_crit_up, ...)` | F-distribution with rejection region |
| `plot_simultaneous_ci(point_estimates, intervals, ...)` | CI forest plot |
| `plot_posterior_normal(result, title)` | Prior vs posterior density (Normal-Normal) |

#### Result-level `.plot()` methods

| Result class | Plot output |
|---|---|
| `ZTestResult` | Standard normal with rejection region |
| `TTestOneSampleResult` | t-distribution with rejection region |
| `TTestTwoSampleResult` | t-distribution with rejection region |
| `TTestPairedResult` | t-distribution with rejection region |
| `Chi2VarianceTestResult` | χ²-distribution with rejection region |
| `FTestVariancesResult` | F-distribution with rejection region |
| `ANOVA1TestResult` | SS bar chart + F-distribution |
| `ANOVA2TestResult` | SS bar chart + F-distribution |
| `SimultaneousCIResult` | CI forest plot |
| `SimultaneousTestResult` | Horizontal bar chart of test statistics |
| `RegressionSummaryResult` | Coefficient forest plot with CIs |
| `RegressionDiagnosticsResult` | 4-panel: leverage, std residuals, Cook's D, residuals vs leverage |
| `MeanConfidenceIntervalResult` | Single CI interval plot |
| `NormalMeanKnownVarResult` | Prior/posterior density with credible interval |
| `NormalMeanUnknownVarResult` | Marginal posterior (t-distribution) with credible interval |
| `ShapiroWilkResult` | Normal Q-Q plot |
| `MCMCResult` | Trace plots + KDE posterior density panels per parameter |
| `ConjugateModelResult` | Prior vs posterior density (Beta or Gamma) |

---

### Data I/O

| Function | Returns | Description |
|----------|---------|-------------|
| `load_data(path, **kwargs)` | `LoadedData` | Loads `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`; kwargs forwarded to pandas |

`LoadedData` fields: `.df` (DataFrame), `.path`, `.format`, `.n_rows`, `.n_cols`, `.column_names`.

---

## Mathematical Background

### One-Way ANOVA

Model: X_{ij} = μ + α_i + ε_{ij}, with Σᵢ nᵢαᵢ = 0 and ε_{ij} ~ N(0,σ²).

| Source | Sum of Squares | df | Mean Square | F |
|--------|---------------|----|-----------|----|
| Between | SS_B = Σᵢ nᵢ(X̄ᵢ − X̄)² | I−1 | MS_B = SS_B/(I−1) | MS_B/MS_W |
| Within | SS_W = Σᵢ Σⱼ (Xᵢⱼ − X̄ᵢ)² | n−I | MS_W = SS_W/(n−I) | |
| Total | SS_T = SS_B + SS_W | n−1 | | |

Under H₀, F ~ F(I−1, n−I). Scheffé simultaneous CI half-width for contrast c: √(r · F_{r,n−I,α} · MS_W · c^T(Gram)^{−1}c).

### Two-Way ANOVA

Model: X_{ijk} = μ + αᵢ + βⱼ + δᵢⱼ + εᵢⱼₖ, K ≥ 2.

Decomposition: SS_T = SS_A + SS_B + SS_AB + SS_E

Each source tested by its own F-ratio against MS_E = SS_E/(IJ(K−1)).

### Normal Distribution Tests

| Test | H₀ | Statistic | Null distribution |
|------|-----|-----------|-------------------|
| Z-test | μ = μ₀ | (x̄ − μ₀)/(σ/√n) | N(0,1) |
| One-sample t | μ = μ₀ | (x̄ − μ₀)/(s/√n) | t(n−1) |
| Chi-squared σ² | σ² = σ₀² | (n−1)s²/σ₀² | χ²(n−1) |
| Two-sample t (pooled) | μ₁ = μ₂ | (x̄₁−x̄₂)/(sₚ√(1/n₁+1/n₂)) | t(n₁+n₂−2) |
| Welch's t | μ₁ = μ₂ | (x̄₁−x̄₂)/√(s₁²/n₁+s₂²/n₂) | t(ν), ν Welch–Satterthwaite |
| Paired t | μ_d = 0 | d̄/(s_d/√n) | t(n−1) |
| F variance | σ₁² = σ₂² | s₁²/s₂² | F(n₁−1, n₂−1) |

### Multiple Linear Regression

Model: **Y** = **X**β + **ε**, **ε** ~ N(**0**, σ²**I**), **X** is n×p with intercept column.

**Estimators:**
- β̂ = (**X**^T**X**)^{−1}**X**^T**y** (OLS)
- σ̂² = (**y** − **X**β̂)^T(**y** − **X**β̂)/(n−p) (unbiased)

**TSS partition:** SS_T = SS_Reg + SS_Res,  R² = SS_Reg/SS_T,  adj R² = 1 − (1−R²)(n−1)/(n−p)

**General test** H₀: **C**β = **c**₀ (rank r):

F = (**C**β̂−**c**₀)^T [**C**(**X**^T**X**)^{−1}**C**^T]^{−1} (**C**β̂−**c**₀) / (r·σ̂²) ~ F(r, n−p)

**Scheffé simultaneous CI** for β_i: β̂ᵢ ± √(p · F_{p,n−p,α} · σ̂² · [(**X**^T**X**)^{−1}]ᵢᵢ)

### Bayesian Conjugate Models

**Normal-Normal** (σ² known):
- Prior: μ ~ N(μ₀, σ²/κ₀)
- Posterior: μ|**x** ~ N(μₙ, σ²/κₙ),  κₙ = κ₀ + n,  μₙ = (κ₀μ₀ + nx̄)/κₙ
- Predictive: x_new|**x** ~ N(μₙ, σ²(1 + 1/κₙ))

**Normal-Gamma** (σ² unknown):
- Prior: (μ,τ) ~ NG(μ₀, κ₀, α₀, β₀)
- Updates: κₙ = κ₀+n, μₙ = (κ₀μ₀+nx̄)/κₙ, αₙ = α₀+n/2, βₙ = β₀+½Σ(xᵢ−x̄)²+κ₀n(x̄−μ₀)²/(2κₙ)
- Marginals: μ|**x** ~ t(2αₙ, μₙ, √(βₙ/(αₙκₙ))),  σ²|**x** ~ InvGamma(αₙ, βₙ)

**Beta-Binomial:** p|k ~ Beta(α₀+k, β₀+n−k)

**Gamma-Poisson:** λ|**x** ~ Gamma(α₀+Σxᵢ, β₀+n)

### Metropolis-Hastings MCMC

Algorithm (random-walk MH):
1. Propose θ\* = θᵗ + ε,  ε ~ N(**0**, σ²_proposal · **I**)
2. Compute log acceptance ratio: log r = log π(θ\*) − log π(θᵗ)
3. Set θᵗ⁺¹ = θ\* with probability min(1, eˡᵒᵍ ʳ), else θᵗ⁺¹ = θᵗ

**Positivity constraints:** Parameters in (0,∞) are sampled in log-space. The chain stores log σ; the log-posterior includes the Jacobian log|∂σ/∂(log σ)| = log σ. This keeps proposals Gaussian and avoids boundary issues.

**Initialisation:** Sample statistics (x̄, log s) for the Normal model; OLS solution for the regression model. Both strategies centre the chain near the posterior mode, reducing required burn-in.

**Diagnostics:** Acceptance rate (target 0.15–0.50), trace plots (check for mixing and stationarity), KDE posterior density (check for unimodality).

---

## Development

```bash
pip install -e ".[dev,ui]"

pytest tests/ -v               # 205 tests across 12 test files
ruff check .                   # linting
mypy statscore/                # static type checking
python examples/demo.py        # end-to-end function demos

statscore                      # interactive CLI
statscore-ui                   # launch browser UI at http://localhost:8501
```

### Test organisation

| Test file | Coverage |
|-----------|----------|
| `test_anova_one_way.py` | anova1_* functions, contrast utilities |
| `test_anova_two_way.py` | anova2_* functions, MLE estimates |
| `test_bayes_conjugate.py` | All 4 conjugate models |
| `test_cli.py` | CLI menu, handler imports, data parsing |
| `test_diagnostics.py` | shapiro_wilk_test, levene_test, regression_diagnostics, mean_ci |
| `test_io.py` + `test_io_fixtures.py` | load_data for all 5 formats |
| `test_plots.py` | All shared plot utilities |
| `test_regression.py` | OLS, TSS partition, inference, prediction |
| `test_regression_summary.py` | regression_summary table |
| `test_testing_one_sample.py` | z_test_mean, t_test_mean, chi2_test_variance |
| `test_testing_two_sample.py` | t_test_two_sample, t_test_paired, f_test_variances |

---

## License

MIT
