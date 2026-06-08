# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-08

**PyPI:** https://pypi.org/project/stats-toolbox/1.0.0/

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
