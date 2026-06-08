# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

**New table printing functions**:
- `ANOVA1_print_table(result)` — prints a formatted one-way ANOVA table (Source / df / SS / MS / F) with F critical, p-value, and decision.
- `ANOVA2_print_table(result)` — prints a formatted two-way ANOVA table (Factor A, Factor B, Interaction AB, Within, Total rows) with the tested source, F statistic, F critical, p-value, and decision.

### Fixed

- `validate_two_way_data`: K=1 now raises `ValueError` (K ≥ 2 required). With K=1, `df_E = I·J·(K-1) = 0`, making MS_E undefined; previously caused a silent division-by-zero.
- `Mult_LR_partition_TSS`: added `adj_R_squared` field to `PartitionTSSResult`. Formula: `1 - (1 - R²)(n-1)/(n-p)`.

### Changed

- `utils.validation`: shared helpers centralised here; `one_sample.py`, `two_sample.py`, and `conjugate.py` now import from `utils` instead of defining local copies
- `__init__.py`: version bumped to `0.0.2`; 9 new testing functions, 3 Bayesian functions, and `AlternativeHypothesis` enum added to `__all__`
- `pyproject.toml`: description updated to reflect new modules
- Test suite expanded from 58 to 98 tests; new files `test_testing_one_sample.py`, `test_testing_two_sample.py`, `test_bayes_conjugate.py`; new cases `test_k1_raises`, `test_adj_r_squared`
- `examples/demo.py` extended from 20 to 29 demos

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
