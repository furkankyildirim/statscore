# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] - 2026-06-08

### Changed

- **Domain modules consolidated under `statscore/methods/`** — `anova/`, `bayes/`, `diagnostics/`, `regression/`, and `testing/` subpackages have been moved to `statscore/methods/anova/`, `statscore/methods/bayes/`, etc. This groups all statistical computation into a single `methods/` namespace.
- **Canonical import paths updated** — all internal imports, CLI handlers, and tests now use `statscore.methods.*` paths. No shim/compatibility layers remain.
- **Circular import fix in `methods.regression`** — `inference.py`, `prediction.py`, and `summary.py` now import directly from `_results.py` and sibling submodules instead of from the package `__init__`, eliminating the circular dependency.
- **Plot logic inlined into `result.plot()` methods** — all single-use standalone plot functions (`plot_z_test`, `plot_chi2_test`, `plot_anova1_test`, `plot_anova2_test`, `plot_posterior_normal`, `plot_posterior_normal_gamma`, `plot_regression_summary`, `plot_regression_diagnostics`, `plot_confidence_interval`, `plot_simultaneous_tests`) have been removed from the public API. Their logic now lives directly inside the corresponding `plot()` method on each result dataclass, eliminating the indirection layer.
- **`plot()` methods use enum identity checks** — internal alternative-hypothesis branches inside `plot()` methods now compare `self.alternative is AlternativeHypothesis.TWO_SIDED` / `.GREATER` instead of string comparisons against `.value`.
- **Removed from `statscore.__all__`**: `plot_z_test`, `plot_chi2_test`, `plot_anova1_test`, `plot_anova2_test`, `plot_posterior_normal`, `plot_posterior_normal_gamma`, `plot_regression_summary`, `plot_regression_diagnostics`, `plot_confidence_interval`, `plot_simultaneous_tests`. The 7 shared/multi-use plot utilities (`plot_regression`, `plot_residuals`, `plot_qq`, `plot_anova_groups`, `plot_t_test`, `plot_f_test`, `plot_simultaneous_ci`) remain in the public API.
- **`utils/plots.py` trimmed** — now contains only shared plot utilities used by multiple result classes or directly by users.
- **`ARCHITECTURE.md` updated** — layer diagram, layer rules table, DAG, package tree, and "Adding New Modules" guide reflect the `methods/` structure.
- **`README.md` Package Structure updated** — tree diagram reflects new `methods/` layout.

## [0.0.2] - 2026-06-08

### Added

**Normal distribution significance testing** (`statscore.testing`):
- `z_test_mean` — one-sample Z-test for the mean (σ known)
- `t_test_mean` — one-sample t-test for the mean (σ unknown)
- `chi2_test_variance` — chi-squared test for the population variance
- `t_test_two_sample` — two-sample t-test (pooled and Welch variants)
- `t_test_paired` — paired t-test
- `f_test_variances` — F-test for equality of two variances
- All tests support `TWO_SIDED`, `LESS`, and `GREATER` alternatives via `AlternativeHypothesis` enum

**Bayesian conjugate inference** (`statscore.bayes`):
- `bayes_normal_mean_known_var` — Normal-Normal conjugate posterior with credible and predictive intervals
- `bayes_normal_mean_unknown_var` — Normal-Gamma conjugate posterior with marginal credible intervals for μ and σ²
- `bayes_normal_mean_unknown_var_summary` — formatted posterior summary table

**New enum**:
- `AlternativeHypothesis` (`TWO_SIDED`, `LESS`, `GREATER`) — replaces raw strings in all test functions

**Shared validation helpers** (`statscore.utils.validation`):
- `validate_positive`, `validate_non_negative`, `validate_1d_sample`, `validate_alternative`

**New distribution utilities** (`statscore.utils.distributions`):
- `t_cdf`, `t_pvalue_one_sided`, `z_pvalue`, `chi2_pvalue`
- `f_critical_lower`, `f_pvalue_lower`
- `norm_ppf`, `t_ppf`, `chi2_ppf`

**Formatted table printing**:
- `result.summary()` method on `ANOVA1TestResult` — prints a formatted one-way ANOVA table (Source / df / SS / MS / F) with F critical, p-value, and decision.
- `result.summary()` method on `ANOVA2TestResult` — prints a formatted two-way ANOVA table (Factor A, Factor B, Interaction AB, Within, Total rows) with the tested source, F statistic, F critical, p-value, and decision.

**Visualization** (`statscore.plots`):
- `plot_regression` — scatter plot with fitted regression line (simple regression)
- `plot_residuals` — residuals vs. fitted values plot
- `plot_qq` — normal Q-Q plot
- `plot_anova_groups` — side-by-side box plots with jittered data points for ANOVA groups
- `plot_posterior_normal` — prior/posterior density plot for Normal-Normal conjugate model; shades the credible interval
- All plot functions return a `matplotlib.figure.Figure` object

**Diagnostics** (`statscore.diagnostics`):
- `shapiro_wilk_test` — Shapiro-Wilk normality test returning `ShapiroWilkResult`
- `levene_test` — Levene's test for homogeneity of variances returning `LeveneResult`
- `regression_diagnostics` — leverage, standardized residuals, and Cook's D returning `RegressionDiagnosticsResult`; flags high-leverage (h > 2p/n) and influential (Cook's D > 4/n) observations
- `mean_confidence_interval` — z-interval (σ known) or t-interval (σ unknown) returning `MeanConfidenceIntervalResult`

**Data I/O** (`statscore.io`):
- `load_data` — loads tabular data from `.csv`, `.tsv`, `.xlsx`/`.xls`, `.json` via pandas; returns `LoadedData` with DataFrame, path, format, dimensions, and column names

**Regression summary** (`statscore.regression.summary`):
- `regression_summary` — full OLS summary analogous to R's `summary(lm(...))`: coefficient estimates, standard errors, t-statistics, p-values (significance stars), confidence intervals, R², adjusted R², overall F-test; returns `RegressionSummaryResult`

**Interactive CLI** (`statscore.cli`):
- `statscore` command-line entry point via `python -m statscore` or the installed `statscore` script
- Fifteen interactive menu items: One-Way/Two-Way ANOVA, Z-test, one/two-sample and paired t-tests, chi-squared variance test, F-test for variances, simple linear regression (with optional plot saving), regression diagnostics, Shapiro-Wilk normality check, Levene variance homogeneity check, mean confidence interval, and both Bayesian conjugate models
- Accepts inline numbers or file paths (`.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`) for all data inputs

### Fixed

- `validate_two_way_data`: K=1 now raises `ValueError` (K ≥ 2 required). With K=1, `df_E = I·J·(K-1) = 0`, making MS_E undefined; previously caused a silent division-by-zero.
- `mult_lr_partition_tss`: added `adj_R_squared` field to `PartitionTSSResult`. Formula: `1 - (1 - R²)(n-1)/(n-p)`.

### Changed

- **API rename — all public functions now use `snake_case` (PEP 8):** `ANOVA1_partition_TSS` → `anova1_partition_tss`, `ANOVA1_test_equality` → `anova1_test_equality`, `ANOVA1_is_contrast` → `anova1_is_contrast`, `ANOVA1_is_orthogonal` → `anova1_is_orthogonal`, `ANOVA1_CI_linear_combs` → `anova1_ci_linear_combs`, `ANOVA1_test_linear_combs` → `anova1_test_linear_combs`, `ANOVA2_partition_TSS` → `anova2_partition_tss`, `ANOVA2_MLE` → `anova2_mle`, `ANOVA2_test_equality` → `anova2_test_equality`, `Bonferroni_correction` → `bonferroni_correction`, `Sidak_correction` → `sidak_correction`, `Mult_LR_Least_squares` → `mult_lr_least_squares`, `Mult_LR_partition_TSS` → `mult_lr_partition_tss`, `Mult_norm_LR_simul_CI` → `mult_norm_lr_simul_ci`, `Mult_norm_LR_CR` → `mult_norm_lr_cr`, `Mult_norm_LR_is_in_CR` → `mult_norm_lr_is_in_cr`, `Mult_norm_LR_test_general` → `mult_norm_lr_test_general`, `Mult_norm_LR_test_comp` → `mult_norm_lr_test_comp`, `Mult_norm_LR_test_linear_reg` → `mult_norm_lr_test_linear_reg`, `Mult_norm_LR_pred_CI` → `mult_norm_lr_pred_ci`
- `pyproject.toml`: added `openpyxl>=3.0` to dependencies (required by pandas for `.xlsx` read/write)
- `utils.validation`: shared helpers centralised here; `one_sample.py`, `two_sample.py`, and `conjugate.py` now import from `utils` instead of defining local copies
- `statscore.__all__`: version bumped to `0.0.2`; added 9 testing functions, 3 Bayesian functions, `AlternativeHypothesis` enum, `LoadedData`, `load_data`, `ShapiroWilkResult`, `LeveneResult`, `RegressionDiagnosticsResult`, `MeanConfidenceIntervalResult`, `shapiro_wilk_test`, `levene_test`, `regression_diagnostics`, `mean_confidence_interval`, `plot_regression`, `plot_residuals`, `plot_qq`, `plot_anova_groups`, `plot_posterior_normal`, `RegressionSummaryResult`, `regression_summary`
- `statscore.regression.__all__`: added `RegressionSummaryResult`, `regression_summary`
- `pyproject.toml`: added `pandas>=1.3` and `matplotlib>=3.5` to dependencies; registered `statscore = "statscore.cli:main"` console script entry point; description updated to reflect new modules
- Test suite expanded from 58 to 205 tests; new files `test_testing_one_sample.py`, `test_testing_two_sample.py`, `test_bayes_conjugate.py`, `test_plots.py`, `test_diagnostics.py`, `test_io.py`, `test_io_fixtures.py`, `test_cli.py`, `test_regression_summary.py`; new cases `test_k1_raises`, `test_adj_r_squared`
- `examples/demo.py` extended from 20 to 29 demos covering all new functions; `examples/fixture_analysis.py` added with 7 end-to-end analyses using static fixture files
- `tests/fixtures/` added: `basic.csv`, `semicolon.csv`, `groups.tsv`, `records.json`, `measurements.xlsx` — static fixture files for I/O tests

## [0.0.1] - 2026-06-08

### Added
- One-way ANOVA: `ANOVA1_partition_TSS`, `ANOVA1_test_equality`
- Contrast utilities: `ANOVA1_is_contrast`, `ANOVA1_is_orthogonal`
- Multiple comparison corrections: `Bonferroni_correction`, `Sidak_correction`
- Simultaneous inference: `ANOVA1_CI_linear_combs`, `ANOVA1_test_linear_combs`
- Two-way ANOVA: `ANOVA2_partition_TSS`, `ANOVA2_MLE`, `ANOVA2_test_equality`
- OLS regression: `Mult_LR_Least_squares`, `Mult_LR_partition_TSS`
- Regression inference: `Mult_norm_LR_simul_CI`, `Mult_norm_LR_CR`, `Mult_norm_LR_is_in_CR`
- Hypothesis testing: `Mult_norm_LR_test_general`, `Mult_norm_LR_test_comp`, `Mult_norm_LR_test_linear_reg`
- Prediction: `Mult_norm_LR_pred_CI` with Scheffe and Bonferroni methods
- Automatic "best" method selection for simultaneous intervals
- Full test suite (58 tests)
- Demo script exercising all 20 functions
