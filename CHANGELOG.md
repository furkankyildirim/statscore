# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] - 2026-06-08

### Added

**Browser UI** (`statscore/app.py`):
- `streamlit run statscore/app.py` launches a 6-page interactive web application
- **Data Input page** — upload CSV / TSV / XLSX / JSON via file picker, or paste raw numbers / matrix into a text area; live DataFrame preview with `describe()` summary; loaded dataset is shared across all pages via session state
- **ANOVA page** — One-Way: ANOVA table, group box-plots with jitter, F-distribution plot; Two-Way: cell-by-cell I×J×K entry, full ANOVA table for all sources (A, B, AB, Error, Total), factor selector, F-distribution plot
- **Significance Tests page** — Z-test (σ known), one-sample t, two-sample t (pooled / Welch), paired t, Chi²-variance, F-variance; each shows a metric panel, reject/fail-to-reject banner, and a distribution plot with shaded rejection region
- **Regression page** — design matrix X and response y from loaded dataset or manual text entry; intercept toggle; coefficient table with significance stars (***/**/*/.); R², adj-R², Sₑ, overall F metric cards; scatter+fit (simple) or coefficient forest plot (multiple); residuals vs fitted; Q-Q plot; 4-panel Cook's D diagnostics
- **Bayesian Inference page** — Normal–known-variance (Normal-Normal conjugate) and Normal–unknown-variance (Normal-Gamma conjugate); prior hyperparameter inputs; posterior metric cards (mean, std, credible interval bounds); prior vs posterior density plot with shaded credible interval
- **Multiple Comparisons page** — group data from loaded dataset or manual entry; contrast matrix C entered as text (rows = comparisons, cols = groups); method selector (Scheffé / Tukey / Bonferroni / Šidák / Best); Simultaneous CIs with CI forest plot or Simultaneous Tests with horizontal bar chart

**Bayesian MCMC** (`statscore.methods.bayes.mcmc`):
- `run_mcmc` — general-purpose Metropolis-Hastings sampler; accepts any log-posterior callable; returns `MCMCResult` with chain, acceptance rate, posterior summaries, and credible intervals
- `mcmc_normal_mean_unknown_var` — MCMC for Normal data with Normal prior on mu and Inverse-Gamma prior on sigma²; unconstrained sampling via log-sigma parameterisation
- `mcmc_linear_regression` — MCMC for linear regression Y = Xβ + ε with independent Normal priors on β and Inverse-Gamma on σ²; OLS-initialised; returns per-parameter posterior summaries
- `MCMCResult` — dataclass with `.summary()` (posterior table) and `.plot()` (trace + KDE posterior panel per parameter)

**New conjugate Bayesian models** (`statscore.methods.bayes.mcmc`):
- `bayes_beta_binomial` — Beta(α₀, β₀) prior on success probability p, Binomial likelihood; posterior Beta(α₀+k, β₀+n−k); returns `ConjugateModelResult`
- `bayes_gamma_poisson` — Gamma(α₀, β₀) prior on Poisson rate λ, Poisson likelihood; posterior Gamma(α₀+Σx, β₀+n); returns `ConjugateModelResult`
- `ConjugateModelResult` — dataclass with `.summary()` (prior/posterior params, CI) and `.plot()` (prior vs posterior density)

**Visualization**:
- `plot_posterior_normal` — standalone prior-vs-posterior density plot for `NormalMeanKnownVarResult`; shades credible interval; added to public API and `__all__`

**CLI — new menu items** (menu now has 21 items, organized into sections):
- `[3]`  Multiple Comparisons — simultaneous CIs and hypothesis tests (Bonferroni/Scheffe/Tukey/Sidak/Best); contrast matrix input with semicolon-separated rows; CI forest plot and test statistics plot
- `[11]` Multiple Linear Regression — arbitrary X matrix input, full inference: OLS summary, simultaneous CIs for β, general hypothesis test H₀: Cβ = c₀, simultaneous prediction intervals, residual/Q-Q plots
- `[18]` Bayesian Inference — Beta-Binomial (success prob.)
- `[19]` Bayesian Inference — Gamma-Poisson (count rate)
- `[20]` Bayesian MCMC — Normal mean & variance
- `[21]` Bayesian MCMC — Linear Regression

**Documentation**:
- `USER_GUIDE.md` — comprehensive walkthrough of every feature with worked examples, interpretation guides, and CLI walkthroughs

### Fixed

- **Domain modules consolidated under `statscore/methods/`** — `anova/`, `bayes/`, `diagnostics/`, `regression/`, and `testing/` subpackages moved under `statscore/methods/`; all internal imports, CLI handlers, and tests updated; no compatibility shims remain
- **Circular import fix in `methods.regression`** — `inference.py`, `prediction.py`, and `summary.py` now import directly from `_results.py` and sibling submodules
- **Plot logic inlined into `result.plot()` methods** — single-use standalone plot functions removed from public API; logic lives in each result dataclass's `plot()` method
- `_run_bayes_known_var` — now calls `result.summary()` instead of manually printing three lines
- `_run_levene_check` — now calls `result.summary()`; added optional group box-plot save
- `_run_regression_diagnostics` — replaced row-by-row `input()` loop with `_parse_matrix_input()`; added optional diagnostics-plot save
- All CLI test handlers now call `result.summary()` and expose a `Show plot? (y/n)` prompt

### Changed

- **`plot()` methods use enum identity checks** — branches inside `plot()` methods compare `self.alternative is AlternativeHypothesis.TWO_SIDED` instead of string comparisons
- **Removed from `statscore.__all__`**: `plot_z_test`, `plot_chi2_test`, `plot_anova1_test`, `plot_anova2_test`, `plot_posterior_normal_gamma`, `plot_regression_summary`, `plot_regression_diagnostics`, `plot_confidence_interval`, `plot_simultaneous_tests`; 7 shared utilities remain public
- CLI menu renumbered into sections: ANOVA (1–3), Significance Tests (4–9), Regression (10–12), Diagnostics (13–15), Bayesian Conjugate (16–19), Bayesian MCMC (20–21)
- `_parse_data_input` now recognises semicolon-separated rows and returns a 2-D array; `_parse_raw_string` and `_parse_matrix_input` helpers added to `_io.py`
- `README.md` fully rewritten with table of contents, feature table, interface sections (Python API / CLI / Browser UI), extended Quick Start, complete API reference tables, and Mathematical Background section
- `statscore.__version__` set to `0.0.3`
- **Production Streamlit config** baked into `run()` via CLI flags: `headless=true`, `showErrorDetails=false`, `toolbarMode="minimal"`, `gatherUsageStats=false`, XSRF protection, 50 MB upload cap, steelblue light theme — no `.streamlit/config.toml` needed
- **`statscore-ui` console script** — `pip install statscore[ui]` + `statscore-ui` launches the app from any directory
- `streamlit>=1.30` added as `[project.optional-dependencies] ui` in `pyproject.toml`

### Fixed

- Removed `import traceback; st.code(traceback.format_exc())` dev leaks from three `except` blocks in `statscore/app/__init__.py`; errors surface only via `st.error()`
- **`fill_between(where=...)` type errors** — wrapped all numpy bool arrays with `.tolist()` in `statscore/plots/__init__.py`, `statscore/methods/testing/_results.py`, and `statscore/methods/bayes/_mcmc_results.py`; fixes mypy `arg-type` incompatibility with matplotlib's `Sequence[bool] | None` stub
- **`NormalMeanKnownVarResult` undefined in `plots/__init__.py`** — moved import to `TYPE_CHECKING` guard; removed unused lazy import alias `_NKV`
- **`ShapiroWilkResult.plot()` missing argument** — `_run_normality_check` in `statscore/cli/_testing.py` now passes the sample array `x` to `result.plot(x)`
- **Ruff lint fixes in `statscore/app/__init__.py`** — removed unused `io` and `tempfile` imports; expanded single-line `if`/`try` statements (E701/E702); sorted lazy import blocks (I001); replaced `try/except/pass` blocks with `contextlib.suppress(Exception)` (SIM105)
- **mypy override for `statscore.app`** — added `ignore_errors = true` override in `pyproject.toml`; also added `ignore_missing_imports` overrides for `streamlit.*` and `matplotlib.*`
- **Package restructure** — `statscore/plots.py` → `statscore/plots/__init__.py` and `statscore/app.py` → `statscore/app/__init__.py`; all import paths remain unchanged

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
- `result.summary()` on `ANOVA1TestResult` — formatted one-way ANOVA table
- `result.summary()` on `ANOVA2TestResult` — formatted two-way ANOVA table

**Visualization** (`statscore.plots`):
- `plot_regression` — scatter plot with fitted regression line
- `plot_residuals` — residuals vs. fitted values plot
- `plot_qq` — normal Q-Q plot
- `plot_anova_groups` — side-by-side box plots with jittered data points
- `plot_posterior_normal` — prior/posterior density plot for Normal-Normal conjugate model
- All plot functions return `matplotlib.figure.Figure`

**Diagnostics** (`statscore.diagnostics`):
- `shapiro_wilk_test` — Shapiro-Wilk normality test
- `levene_test` — Levene's test for homogeneity of variances
- `regression_diagnostics` — leverage, standardized residuals, Cook's D; flags high-leverage (h > 2p/n) and influential (Cook's D > 4/n) observations
- `mean_confidence_interval` — z-interval (σ known) or t-interval (σ unknown)

**Data I/O** (`statscore.io`):
- `load_data` — loads tabular data from `.csv`, `.tsv`, `.xlsx`/`.xls`, `.json` via pandas; returns `LoadedData`

**Regression summary** (`statscore.regression.summary`):
- `regression_summary` — full OLS summary analogous to R's `summary(lm(...))`: coefficients, SE, t-stats, p-values, significance stars, CIs, R², adjusted R², overall F-test

**Interactive CLI** (`statscore.cli`):
- `statscore` command-line entry point
- 15 interactive menu items covering ANOVA, significance tests, regression, diagnostics, Bayesian conjugate models
- Accepts inline numbers or file paths for all data inputs

### Fixed

- `validate_two_way_data`: K=1 now raises `ValueError` (K ≥ 2 required)
- `mult_lr_partition_tss`: added `adj_R_squared` field; formula: `1 - (1 - R²)(n-1)/(n-p)`

### Changed

- **API rename — all public functions now use `snake_case` (PEP 8)**
- `pyproject.toml`: added `openpyxl>=3.0`, `pandas>=1.3`, `matplotlib>=3.5` to dependencies; registered `statscore` console script entry point
- Test suite expanded from 58 to 205 tests
- `examples/demo.py` extended to 29 demos; `examples/fixture_analysis.py` added
- `tests/fixtures/` added with static fixture files for I/O tests

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
