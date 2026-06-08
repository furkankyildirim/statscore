"""
Demonstration script for statscore.

Exercises all public functions end-to-end with representative sample data.
"""

import os
import tempfile

import numpy as np

from statscore import (
    AlternativeHypothesis,
    ConjugateModelResult,
    CorrectionMethod,
    LeveneResult,
    LoadedData,
    MCMCResult,
    MeanConfidenceIntervalResult,
    PredictionMethod,
    RegressionDiagnosticsResult,
    RegressionSummaryResult,
    ShapiroWilkResult,
    TwoWayTestFactor,
    anova1_ci_linear_combs,
    anova1_is_contrast,
    anova1_is_orthogonal,
    anova1_partition_tss,
    anova1_test_equality,
    anova1_test_linear_combs,
    anova2_mle,
    anova2_partition_tss,
    anova2_test_equality,
    bayes_beta_binomial,
    bayes_gamma_poisson,
    bayes_normal_mean_known_var,
    bayes_normal_mean_unknown_var,
    bonferroni_correction,
    chi2_test_variance,
    f_test_variances,
    levene_test,
    load_data,
    mcmc_linear_regression,
    mcmc_normal_mean_unknown_var,
    mean_confidence_interval,
    mult_lr_least_squares,
    mult_lr_partition_tss,
    mult_norm_lr_cr,
    mult_norm_lr_is_in_cr,
    mult_norm_lr_pred_ci,
    mult_norm_lr_simul_ci,
    mult_norm_lr_test_comp,
    mult_norm_lr_test_general,
    mult_norm_lr_test_linear_reg,
    plot_anova_groups,
    plot_f_test,
    plot_posterior_normal,
    plot_qq,
    plot_regression,
    plot_residuals,
    plot_simultaneous_ci,
    plot_t_test,
    regression_diagnostics,
    regression_summary,
    run_mcmc,
    shapiro_wilk_test,
    sidak_correction,
    t_test_mean,
    t_test_paired,
    t_test_two_sample,
    z_test_mean,
)


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# =============================================================================
# ONE-WAY ANOVA DATA
# =============================================================================
# Toxin experiment with 4 groups (3 treatments + control)
toxin1 = np.array([28, 23, 14, 27, 31], dtype=float)
toxin2 = np.array([33, 36, 34, 29, 24], dtype=float)
toxin3 = np.array([18, 21, 20, 22], dtype=float)
control = np.array([11, 14, 11, 16], dtype=float)
anova_data = [toxin1, toxin2, toxin3, control]


# =============================================================================
# DEMO 1: anova1_partition_tss
# =============================================================================
separator("1. anova1_partition_tss")
print("Data: Toxin experiment (4 groups)")
print(f"  Toxin 1: {toxin1}")
print(f"  Toxin 2: {toxin2}")
print(f"  Toxin 3: {toxin3}")
print(f"  Control: {control}")

result = anova1_partition_tss(anova_data)
print("\nResults:")
print(f"  Group means: {result.group_means}")
print(f"  Grand mean:  {result.grand_mean:.4f}")
print(f"  SS_total:    {result.SS_total:.4f}")
print(f"  SS_within:   {result.SS_within:.4f}")
print(f"  SS_between:  {result.SS_between:.4f}")
print(f"  Check: SS_w + SS_b = {result.SS_within + result.SS_between:.4f} (should equal SS_total)")


# =============================================================================
# DEMO 2: anova1_test_equality
# =============================================================================
separator("2. anova1_test_equality")
print("Testing H0: mu_1 = mu_2 = mu_3 = mu_4 at alpha = 0.05")

result = anova1_test_equality(anova_data, alpha=0.05)
print("\nANOVA Table:")
print(f"  {'Source':<12} {'df':<5} {'SS':<12} {'MS':<12} {'F':<10}")
print(
    f"  {'Between':<12} {result.df_between:<5} {result.SS_between:<12.4f} {result.MS_between:<12.4f} {result.F_statistic:<10.4f}"
)
print(f"  {'Within':<12} {result.df_within:<5} {result.SS_within:<12.4f} {result.MS_within:<12.4f}")
print(f"  {'Total':<12} {result.df_total:<5} {result.SS_total:<12.4f}")
print(f"\n  F critical (alpha=0.05): {result.F_critical:.4f}")
print(f"  p-value: {result.p_value:.6f}")
print(f"  Decision: {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 3: anova1_is_contrast
# =============================================================================
separator("3. anova1_is_contrast")
c1 = np.array([1, -1, 0, 0])
c2 = np.array([1, 1, -1, -1])
c3 = np.array([1, 1, 1, 0])

print(f"  c = {c1} -> is contrast? {anova1_is_contrast(c1)}")
print(f"  c = {c2} -> is contrast? {anova1_is_contrast(c2)}")
print(f"  c = {c3} -> is contrast? {anova1_is_contrast(c3)}")


# =============================================================================
# DEMO 4: anova1_is_orthogonal
# =============================================================================
separator("4. anova1_is_orthogonal")
n = np.array([5, 5, 4, 4])
c1 = np.array([1, -1, 0, 0])
c2 = np.array([0, 0, 1, -1])
c3 = np.array([1, 0, -1, 0])

result12 = anova1_is_orthogonal(n, c1, c2)
result13 = anova1_is_orthogonal(n, c1, c3)

print(f"  Group sizes: {n}")
print(f"  c1 = {c1}, c2 = {c2}")
print(f"    Orthogonal? {result12.is_orthogonal}")
print(f"  c1 = {c1}, c3 = {c3}")
print(f"    Orthogonal? {result13.is_orthogonal}")

c_bad = np.array([1, 1, 0, 0])
result_bad = anova1_is_orthogonal(n, c_bad, c2)
print(f"\n  c_bad = {c_bad}, c2 = {c2}")
print(f"    Warning: {result_bad.warning}")


# =============================================================================
# DEMO 5: bonferroni_correction
# =============================================================================
separator("5. bonferroni_correction")
alpha = 0.05
m = 4
corrected = bonferroni_correction(alpha, m)
print(f"  FWER alpha = {alpha}, number of tests m = {m}")
print(f"  Individual significance level: alpha/m = {corrected:.6f}")


# =============================================================================
# DEMO 6: sidak_correction
# =============================================================================
separator("6. sidak_correction")
corrected = sidak_correction(alpha, m)
print(f"  FWER alpha = {alpha}, number of tests m = {m}")
print(f"  Individual significance level: 1-(1-alpha)^(1/m) = {corrected:.6f}")
print(f"  (Compare Bonferroni: {bonferroni_correction(alpha, m):.6f})")
print("  Sidak is less conservative (larger individual alpha).")


# =============================================================================
# DEMO 7: anova1_ci_linear_combs
# =============================================================================
separator("7. anova1_ci_linear_combs")
print("Simultaneous 95% confidence intervals for contrasts:")
print("  H01: mu_1 - mu_4 (Toxin 1 vs Control)")
print("  H02: mu_2 - mu_4 (Toxin 2 vs Control)")
print("  H03: mu_3 - mu_4 (Toxin 3 vs Control)")
print("  H04: (mu_1+mu_2+mu_3)/3 - mu_4 (Average toxin vs Control)")

C = np.array(
    [
        [1, 0, 0, -1],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [1 / 3, 1 / 3, 1 / 3, -1],
    ]
)

result = anova1_ci_linear_combs(anova_data, alpha=0.05, C=C, method=CorrectionMethod.BEST)
print(f"\n  Method used: {result.method_used.value}")
for i, (lo, hi) in enumerate(result.intervals):
    print(f"  CI_{i + 1}: [{lo:.4f}, {hi:.4f}]  (point est: {result.point_estimates[i]:.4f})")


# =============================================================================
# DEMO 8: anova1_test_linear_combs
# =============================================================================
separator("8. anova1_test_linear_combs")
print("Testing contrasts at FWER = 0.05:")
d = np.array([0.0, 0.0, 0.0, 0.0])

result = anova1_test_linear_combs(anova_data, alpha=0.05, C=C, d=d, method=CorrectionMethod.BEST)
print(f"  Method used: {result.method_used.value}")
print(f"\n  {'Hypothesis':<35} {'|T|':<8} {'Crit':<8} {'p-value':<10} {'Reject?'}")
labels = ["mu1 - mu4 = 0", "mu2 - mu4 = 0", "mu3 - mu4 = 0", "avg_toxin - mu4 = 0"]
for i in range(4):
    print(
        f"  {labels[i]:<35} {result.test_statistics[i]:<8.4f} {result.critical_values[i]:<8.4f} {result.p_values[i]:<10.6f} {result.reject[i]}"
    )


# =============================================================================
# DEMO 9: anova2_partition_tss
# =============================================================================
separator("9. anova2_partition_tss")
print("Two-way ANOVA: Detergent x Temperature experiment")
print("  Factor A (detergent): Super, Best (I=2)")
print("  Factor B (temperature): Cold, Warm, Hot (J=3)")
print("  K=4 replicates per cell")

data_2way = np.array(
    [
        [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
        [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
    ],
    dtype=float,
)

result = anova2_partition_tss(data_2way)
print(f"\n  SS_total: {result.SS_total:.4f}")
print(f"  SS_A:     {result.SS_A:.4f}")
print(f"  SS_B:     {result.SS_B:.4f}")
print(f"  SS_AB:    {result.SS_AB:.4f}")
print(f"  SS_E:     {result.SS_E:.4f}")
print(
    f"  Check: SS_A + SS_B + SS_AB + SS_E = {result.SS_A + result.SS_B + result.SS_AB + result.SS_E:.4f}"
)


# =============================================================================
# DEMO 10: anova2_mle
# =============================================================================
separator("10. anova2_mle")
mle = anova2_mle(data_2way)
print(f"  mu_hat = {mle.mu:.4f}")
print(f"  a_hat  = {mle.a}")
print(f"  b_hat  = {mle.b}")
print("  delta_hat =")
print(f"    {mle.delta}")


# =============================================================================
# DEMO 11: anova2_test_equality
# =============================================================================
separator("11. anova2_test_equality")
print("Testing at alpha = 0.05:\n")

for test_type in [TwoWayTestFactor.A, TwoWayTestFactor.B, TwoWayTestFactor.AB]:
    result = anova2_test_equality(data_2way, alpha=0.05, test=test_type)
    print(f"  Test for factor {test_type.value}:")
    print(f"    df={result.df}, SS={result.SS:.4f}, MS={result.MS:.4f}")
    print(f"    F={result.F_statistic:.4f}, F_crit={result.F_critical:.4f}, p={result.p_value:.6f}")
    print(f"    Decision: {'Reject H0' if result.reject_H0 else 'Do not reject H0'}\n")

result = anova2_test_equality(data_2way, alpha=0.05, test=TwoWayTestFactor.A)
print("  Full ANOVA table:")
print(f"  {'Source':<10} {'df':<5} {'SS':<12} {'MS':<12} {'F':<10}")
for src in ["A", "B", "AB"]:
    t = result.full_table[src]
    print(f"  {src:<10} {t['df']:<5} {t['SS']:<12.4f} {t['MS']:<12.4f} {t['F']:<10.4f}")
t = result.full_table["within"]
print(f"  {'Within':<10} {t['df']:<5} {t['SS']:<12.4f} {t['MS']:<12.4f}")
t = result.full_table["total"]
print(f"  {'Total':<10} {t['df']:<5} {t['SS']:<12.4f}")


# =============================================================================
# REGRESSION DATA
# =============================================================================
# Student performance data: attendance rate, homework completion, final grade
attend = np.array([1, 0.5, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.6, 0.1, 0, 0, 0.7, 0.8, 1])
homework = np.array([0.25, 1, 0.5, 1, 1, 0.75, 1, 0.25, 0, 0, 1, 0.5, 0.25, 0.75, 1])
grade = np.array([60, 65, 40, 70, 65, 70, 85, 70, 44, 20, 40, 30, 50, 77, 90], dtype=float)
n_obs = len(grade)
X = np.column_stack([np.ones(n_obs), attend, homework])
y = grade


# =============================================================================
# DEMO 12: mult_lr_least_squares
# =============================================================================
separator("12. mult_lr_least_squares")
print("Multiple linear regression: Grade = beta0 + beta1*Attend + beta2*Homework + e")

result = mult_lr_least_squares(X, y)
print(f"\n  beta_hat = {result.beta_hat}")
print(f"    beta_0 (intercept): {result.beta_hat[0]:.4f}")
print(f"    beta_1 (attend):    {result.beta_hat[1]:.4f}")
print(f"    beta_2 (homework):  {result.beta_hat[2]:.4f}")
print(f"  sigma^2 MLE:             {result.sigma2_mle:.4f}")
print(f"  sigma^2 unbiased (Se^2): {result.sigma2_unbiased:.4f}")
print(f"  Se = {np.sqrt(result.sigma2_unbiased):.4f}")


# =============================================================================
# DEMO 13: mult_lr_partition_tss
# =============================================================================
separator("13. mult_lr_partition_tss")
result = mult_lr_partition_tss(X, y)
print(f"  TSS   = {result.TSS:.4f}")
print(f"  RegSS = {result.RegSS:.4f}")
print(f"  RSS   = {result.RSS:.4f}")
print(f"  R^2   = {result.R_squared:.4f}")
print(f"  Check: RegSS + RSS = {result.RegSS + result.RSS:.4f} (should = TSS)")


# =============================================================================
# DEMO 14: mult_norm_lr_simul_ci
# =============================================================================
separator("14. mult_norm_lr_simul_ci")
result = mult_norm_lr_simul_ci(X, y, alpha=0.1)
print("  Simultaneous 90% confidence intervals for beta:")
print(f"  Method: {result.method.value}")
labels = ["beta_0 (intercept)", "beta_1 (attend)", "beta_2 (homework)"]
for i, (lo, hi) in enumerate(result.intervals):
    print(f"    {labels[i]:<22}: [{lo:.4f}, {hi:.4f}]")


# =============================================================================
# DEMO 15: mult_norm_lr_cr
# =============================================================================
separator("15. mult_norm_lr_cr")
C_full = np.eye(3)
cr = mult_norm_lr_cr(X, y, C_full, alpha=0.1)
print("  90% Confidence region for beta (full vector):")
print(f"  Center: {cr.center}")
print("  Shape matrix (C(X^TX)^-1 C^T):")
for row in cr.shape_matrix:
    print(f"    {row}")
print(f"  Radius^2: {cr.radius_squared:.4f}")
print(f"  F_critical: {cr.F_critical:.4f}")


# =============================================================================
# DEMO 16: mult_norm_lr_is_in_cr
# =============================================================================
separator("16. mult_norm_lr_is_in_cr")
c0_in = np.array([20, 40, 20])
c0_out = np.array([0, 0, 0])
print(
    f"  Testing if c0 = {c0_in} is in 90% CR: {mult_norm_lr_is_in_cr(X, y, C_full, c0_in, alpha=0.1)}"
)
print(
    f"  Testing if c0 = {c0_out} is in 90% CR: {mult_norm_lr_is_in_cr(X, y, C_full, c0_out, alpha=0.1)}"
)


# =============================================================================
# DEMO 17: mult_norm_lr_test_general
# =============================================================================
separator("17. mult_norm_lr_test_general")
print("Testing H0: beta_1 = beta_2 (attend has same effect as homework)")
C_eq = np.array([[0, 1, -1]])
c0_eq = np.array([0.0])
result = mult_norm_lr_test_general(X, y, C_eq, c0_eq, alpha=0.1)
print(f"  C = {C_eq[0]}, c0 = {c0_eq}")
print(f"  F statistic: {result.test_statistic:.4f}")
print(f"  F critical:  {result.F_critical:.4f}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 18: mult_norm_lr_test_comp
# =============================================================================
separator("18. mult_norm_lr_test_comp")
print("Testing H0: beta_1 = 0 (attending lectures has no effect)")
result = mult_norm_lr_test_comp(X, y, alpha=0.1, components=[1])
print(f"  F statistic: {result.test_statistic:.4f}")
print(f"  F critical:  {result.F_critical:.4f}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")

print("\nTesting H0: beta_2 = 0 (homework has no effect)")
result = mult_norm_lr_test_comp(X, y, alpha=0.1, components=[2])
print(f"  F statistic: {result.test_statistic:.4f}")
print(f"  F critical:  {result.F_critical:.4f}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 19: mult_norm_lr_test_linear_reg
# =============================================================================
separator("19. mult_norm_lr_test_linear_reg")
print("Testing H0: beta_1 = beta_2 = 0 (no linear regression at all)")
result = mult_norm_lr_test_linear_reg(X, y, alpha=0.1)
print(f"  F statistic: {result.test_statistic:.4f}")
print(f"  F critical:  {result.F_critical:.4f}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 20: mult_norm_lr_pred_ci
# =============================================================================
separator("20. mult_norm_lr_pred_ci")
print("Prediction: A student with full attendance and no homework (d = (1, 1.0, 0.0))")
print("Also predict: half attendance, full homework completion (d = (1, 0.5, 1.0))")

D = np.array(
    [
        [1, 1.0, 0.0],
        [1, 0.5, 1.0],
    ]
)
result = mult_norm_lr_pred_ci(X, y, D, alpha=0.1, method=PredictionMethod.BEST)
print(f"\n  Method: {result.method_used.value}")
print(f"  Point estimates: {result.point_estimates}")
for i, (lo, hi) in enumerate(result.intervals):
    print(f"  CI_{i + 1}: [{lo:.4f}, {hi:.4f}] (half-width: {result.half_widths[i]:.4f})")


# =============================================================================
# NORMAL DISTRIBUTION SIGNIFICANCE TESTING DATA
# =============================================================================
# Blood pressure readings from two groups of patients
bp_treatment = np.array([138, 142, 130, 145, 137, 140, 133, 148, 135, 141], dtype=float)
bp_control = np.array([155, 162, 149, 158, 160, 153, 157, 151, 164, 156], dtype=float)


# =============================================================================
# DEMO 21: z_test_mean
# =============================================================================
separator("21. z_test_mean")
print("Testing H0: mu = 150 for blood pressure (sigma = 10 known)")
result = z_test_mean(
    bp_treatment,
    mu0=150.0,
    sigma=10.0,
    alpha=0.05,
    alternative=AlternativeHypothesis.TWO_SIDED,
)
print(f"  n = {result.n}, x_bar = {result.x_bar:.4f}")
print(f"  Z statistic:  {result.z_statistic:.4f}")
print(f"  Z critical:   ±{result.z_critical:.4f}")
print(f"  p-value:      {result.p_value:.6f}")
print(f"  Decision:     {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")

print("\nOne-sided test: H0: mu >= 150 vs H1: mu < 150")
result_less = z_test_mean(
    bp_treatment,
    mu0=150.0,
    sigma=10.0,
    alpha=0.05,
    alternative=AlternativeHypothesis.LESS,
)
print(f"  Z statistic: {result_less.z_statistic:.4f}, p-value: {result_less.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result_less.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 22: t_test_mean
# =============================================================================
separator("22. t_test_mean")
print("Testing H0: mu = 150 for blood pressure (sigma unknown)")
result = t_test_mean(
    bp_treatment,
    mu0=150.0,
    alpha=0.05,
    alternative=AlternativeHypothesis.TWO_SIDED,
)
print(f"  n = {result.n}, x_bar = {result.x_bar:.4f}, s = {result.s:.4f}")
print(f"  T statistic:  {result.t_statistic:.4f}")
print(f"  T critical:   ±{result.t_critical:.4f}  (df = {result.df})")
print(f"  p-value:      {result.p_value:.6f}")
print(f"  Decision:     {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 23: chi2_test_variance
# =============================================================================
separator("23. chi2_test_variance")
print("Testing H0: sigma^2 = 100 for blood pressure measurements")
result = chi2_test_variance(
    bp_treatment,
    sigma0_sq=100.0,
    alpha=0.05,
    alternative=AlternativeHypothesis.TWO_SIDED,
)
print(f"  n = {result.n}, s^2 = {result.s2:.4f}")
print(f"  Chi2 statistic: {result.chi2_statistic:.4f}")
print(
    f"  Critical region: < {result.chi2_critical_lower:.4f} or > {result.chi2_critical_upper:.4f}"
)
print(f"  p-value:        {result.p_value:.6f}")
print(f"  Decision:       {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 24: t_test_two_sample
# =============================================================================
separator("24. t_test_two_sample")
print("Testing H0: mu_treatment = mu_control (two-sample t-test)")

print("\n  Equal variance (pooled):")
result = t_test_two_sample(
    bp_treatment,
    bp_control,
    alpha=0.05,
    alternative=AlternativeHypothesis.TWO_SIDED,
    equal_var=True,
)
print(f"  x1_bar = {result.x1_bar:.2f}, x2_bar = {result.x2_bar:.2f}")
print(f"  T statistic: {result.t_statistic:.4f}, df = {result.df}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")

print("\n  Welch's t-test (unequal variance):")
result_welch = t_test_two_sample(
    bp_treatment,
    bp_control,
    alpha=0.05,
    alternative=AlternativeHypothesis.TWO_SIDED,
    equal_var=False,
)
print(f"  T statistic: {result_welch.t_statistic:.4f}, df = {result_welch.df}")
print(f"  p-value:     {result_welch.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result_welch.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 25: t_test_paired
# =============================================================================
separator("25. t_test_paired")
print("Paired t-test: blood pressure before vs. after treatment")
bp_before = np.array([158, 163, 151, 162, 155, 160, 157, 165, 153, 159], dtype=float)
bp_after = np.array([138, 142, 130, 145, 137, 140, 133, 148, 135, 141], dtype=float)

result = t_test_paired(
    bp_before,
    bp_after,
    alpha=0.05,
    alternative=AlternativeHypothesis.GREATER,
)
print(f"  n = {result.n}, d_bar = {result.d_bar:.4f} (mean reduction)")
print(f"  T statistic: {result.t_statistic:.4f}, df = {result.df}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")
print("  (H1: before > after, i.e., treatment reduces blood pressure)")


# =============================================================================
# DEMO 26: f_test_variances
# =============================================================================
separator("26. f_test_variances")
print("Testing H0: sigma^2_treatment = sigma^2_control")
result = f_test_variances(
    bp_treatment,
    bp_control,
    alpha=0.05,
    alternative=AlternativeHypothesis.TWO_SIDED,
)
print(f"  s1^2 = {result.s1_sq:.4f} (treatment), s2^2 = {result.s2_sq:.4f} (control)")
print(f"  F statistic: {result.f_statistic:.4f}")
print(f"  Critical region: < {result.f_critical_lower:.4f} or > {result.f_critical_upper:.4f}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# BAYESIAN DATA
# =============================================================================
# Lab measurement data with known/unknown variance scenarios
measurements = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4, 9.6, 10.1], dtype=float)


# =============================================================================
# DEMO 27: bayes_normal_mean_known_var
# =============================================================================
separator("27. bayes_normal_mean_known_var")
print("Bayesian inference: sensor measurement with known variance sigma^2 = 0.04")
print("Prior: mu ~ N(10.0, 0.04 / 2)  =>  kappa0 = 2, mu0 = 10.0")

result = bayes_normal_mean_known_var(
    measurements,
    sigma_sq=0.04,
    mu0=10.0,
    kappa0=2.0,
    alpha=0.05,
)
print(f"\n  n = {result.n}, x_bar = {result.x_bar:.4f}")
print(f"  Posterior: mu | x ~ N({result.mu_n:.4f}, {result.posterior_variance:.6f})")
print(f"  Posterior mean:    {result.posterior_mean:.6f}")
print(f"  Posterior std:     {result.posterior_std:.6f}")
print(
    f"  95% Credible interval: ({result.credible_interval[0]:.4f}, {result.credible_interval[1]:.4f})"
)
print("\n  Posterior predictive:")
print(f"  Predictive mean:   {result.predictive_mean:.6f}")
print(f"  Predictive std:    {result.predictive_std:.6f}")
print(
    f"  95% Predictive interval: ({result.predictive_interval[0]:.4f}, {result.predictive_interval[1]:.4f})"
)


# =============================================================================
# DEMO 28: bayes_normal_mean_unknown_var
# =============================================================================
separator("28. bayes_normal_mean_unknown_var")
print("Bayesian inference: sensor measurement with unknown variance")
print("Prior: (mu, tau) ~ Normal-Gamma(mu0=10.0, kappa0=1, alpha0=2, beta0=0.1)")

result = bayes_normal_mean_unknown_var(
    measurements,
    mu0=10.0,
    kappa0=1.0,
    alpha0=2.0,
    beta0=0.1,
    alpha=0.05,
)
print(f"\n  n = {result.n}, x_bar = {result.x_bar:.4f}")
print("\n  Posterior hyperparameters:")
print(f"    mu_n    = {result.mu_n:.6f}")
print(f"    kappa_n = {result.kappa_n:.6f}")
print(f"    alpha_n = {result.alpha_n:.6f}")
print(f"    beta_n  = {result.beta_n:.6f}")
print("\n  Posterior summaries:")
print(f"    E[mu]      = {result.posterior_mean_mu:.6f}")
print(f"    E[tau]     = {result.posterior_mean_precision:.6f}")
if result.posterior_mean_variance is not None:
    print(f"    E[sigma^2] = {result.posterior_mean_variance:.6f}")
print("\n  95% Credible intervals:")
print(f"    mu:      ({result.mu_credible_interval[0]:.4f}, {result.mu_credible_interval[1]:.4f})")
print(
    f"    sigma^2: ({result.variance_credible_interval[0]:.4f}, {result.variance_credible_interval[1]:.4f})"
)


# =============================================================================
# DEMO 29: bayes_normal_mean_unknown_var — .summary()
# =============================================================================
separator("29. bayes_normal_mean_unknown_var — result.summary()")
print("Formatted posterior summary (same result as Demo 28):\n")
result.summary()


# =============================================================================
# DEMO 30: regression_summary
# =============================================================================
separator("30. regression_summary")
print("Full OLS summary for Grade ~ Attend + Homework:\n")
summary = regression_summary(X, y, alpha=0.05, feature_names=["Attend", "Homework"])
summary.summary(feature_names=["Attend", "Homework"])
assert isinstance(summary, RegressionSummaryResult)


# =============================================================================
# DEMO 31: regression_diagnostics
# =============================================================================
separator("31. regression_diagnostics")
print("Leverage, standardized residuals, and Cook's D:\n")
diag = regression_diagnostics(X, y)
diag.summary()
assert isinstance(diag, RegressionDiagnosticsResult)


# =============================================================================
# DEMO 32: shapiro_wilk_test
# =============================================================================
separator("32. shapiro_wilk_test")
print("Testing normality of blood pressure treatment group:\n")
sw = shapiro_wilk_test(bp_treatment, alpha=0.05)
sw.summary()
assert isinstance(sw, ShapiroWilkResult)


# =============================================================================
# DEMO 33: levene_test
# =============================================================================
separator("33. levene_test")
print("Testing variance homogeneity across ANOVA groups:\n")
lev = levene_test(anova_data, alpha=0.05)
lev.summary()
assert isinstance(lev, LeveneResult)


# =============================================================================
# DEMO 34: mean_confidence_interval
# =============================================================================
separator("34. mean_confidence_interval")
print("95% t-interval for blood pressure treatment group (sigma unknown):")
ci_t = mean_confidence_interval(bp_treatment, alpha=0.05)
ci_t.summary()
assert isinstance(ci_t, MeanConfidenceIntervalResult)

print("\n95% z-interval for blood pressure treatment group (sigma = 10 known):")
ci_z = mean_confidence_interval(bp_treatment, alpha=0.05, sigma=10.0)
ci_z.summary()


# =============================================================================
# DEMO 35: load_data
# =============================================================================
separator("35. load_data")
print("Writing a temporary CSV and loading it back:\n")
csv_content = "x,y\n1.0,2.1\n2.0,4.0\n3.0,6.2\n4.0,8.1\n5.0,10.3\n"
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write(csv_content)
    tmp_path = f.name

loaded = load_data(tmp_path)
os.unlink(tmp_path)
assert isinstance(loaded, LoadedData)
print(f"  Path:    {loaded.path}")
print(f"  Format:  {loaded.format}")
print(f"  Shape:   {loaded.n_rows} rows x {loaded.n_cols} cols")
print(f"  Columns: {loaded.column_names}")
print(f"  Data:\n{loaded.df}")


# =============================================================================
# DEMO 36: plot_regression
# =============================================================================
separator("36. plot_regression")
print("Scatter + fitted line for simple regression (attend → grade):")
ols_simple = mult_lr_least_squares(np.column_stack([np.ones(n_obs), attend]), y)
fig = plot_regression(attend, y, ols_simple.beta_hat, x_label="Attendance", y_label="Grade")
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 37: plot_residuals
# =============================================================================
separator("37. plot_residuals")
ols_full = mult_lr_least_squares(X, y)
fig = plot_residuals(ols_full.fitted_values, ols_full.residuals)
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 38: plot_qq
# =============================================================================
separator("38. plot_qq")
fig = plot_qq(ols_full.residuals, title="Q-Q Plot of Regression Residuals")
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 39: plot_anova_groups
# =============================================================================
separator("39. plot_anova_groups")
fig = plot_anova_groups(
    anova_data,
    group_labels=["Toxin 1", "Toxin 2", "Toxin 3", "Control"],
    title="Toxin Experiment",
)
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 40: plot_posterior_normal
# =============================================================================
separator("40. plot_posterior_normal")
bayes_result = bayes_normal_mean_known_var(
    measurements,
    sigma_sq=0.04,
    mu0=10.0,
    kappa0=2.0,
    alpha=0.05,
)
fig = plot_posterior_normal(bayes_result, title="Posterior — Sensor Mean (known var)")
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 41: bayes_beta_binomial
# =============================================================================
separator("41. bayes_beta_binomial")
print("Drug trial: 47 responders out of 60 patients")
print("Prior: p ~ Beta(5, 5)  (weakly informative, centered at 0.5)")

result_bb = bayes_beta_binomial(
    n_trials=60,
    n_successes=47,
    alpha0=5.0,
    beta0=5.0,
    alpha=0.05,
)
result_bb.summary()
assert isinstance(result_bb, ConjugateModelResult)
print(f"  Posterior alpha_n = {result_bb.posterior_params['alpha_n']}")
print(f"  Posterior beta_n  = {result_bb.posterior_params['beta_n']}")

fig = result_bb.plot()
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 42: bayes_gamma_poisson
# =============================================================================
separator("42. bayes_gamma_poisson")
print("Call centre: daily arrival counts over 10 days")
print("Prior: lambda ~ Gamma(2, 0.2)  (prior mean = 10 calls/day)")

arrivals = np.array([8, 12, 7, 10, 9, 11, 13, 8, 10, 9], dtype=float)
result_gp = bayes_gamma_poisson(arrivals, alpha0=2.0, beta0=0.2, alpha=0.05)
result_gp.summary()
assert isinstance(result_gp, ConjugateModelResult)
print(f"  Posterior mean lambda = {result_gp.posterior_mean:.4f}")
print(f"  MLE lambda (x_bar)    = {result_gp.data_summary['x_bar']:.4f}")

fig = result_gp.plot()
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 43: mcmc_normal_mean_unknown_var
# =============================================================================
separator("43. mcmc_normal_mean_unknown_var")
print("MCMC for sensor data: Normal(mu, sigma²) with weakly informative priors")

result_mcmc_n = mcmc_normal_mean_unknown_var(
    measurements,
    mu_prior_mean=10.0,
    mu_prior_std=5.0,
    sigma_prior_alpha=2.0,
    sigma_prior_beta=0.5,
    n_iter=12_000,
    n_burnin=2_000,
    alpha=0.05,
)
result_mcmc_n.summary()
assert isinstance(result_mcmc_n, MCMCResult)
print(f"  Acceptance rate: {result_mcmc_n.acceptance_rate:.3f}")
print(f"  Posterior samples shape: {result_mcmc_n.posterior_samples.shape}")

fig = result_mcmc_n.plot()
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 44: mcmc_linear_regression
# =============================================================================
separator("44. mcmc_linear_regression")
print("Bayesian regression via MCMC: Grade ~ Attend + Homework")
print("  Prior: beta_j ~ N(0, 20²),  sigma ~ InvGamma(2, 1)")

result_mcmc_lr = mcmc_linear_regression(
    X,
    y,
    beta_prior_std=20.0,
    sigma_prior_alpha=2.0,
    sigma_prior_beta=1.0,
    n_iter=15_000,
    n_burnin=3_000,
    alpha=0.05,
)
result_mcmc_lr.summary()
assert isinstance(result_mcmc_lr, MCMCResult)
print(f"  Parameters: {result_mcmc_lr.param_names}")
print(f"  Acceptance rate: {result_mcmc_lr.acceptance_rate:.3f}")

fig = result_mcmc_lr.plot()
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 45: run_mcmc  (custom log-posterior)
# =============================================================================
separator("45. run_mcmc — custom Exponential rate model")
print("Custom MCMC: x_i ~ Exp(lambda),  prior lambda ~ Gamma(2, 1)")
print("  Reparameterised as log_lambda for unconstrained sampling")

exp_data = np.array([1.2, 0.8, 2.1, 0.5, 1.7, 0.9, 1.4, 2.3, 0.6, 1.1], dtype=float)
n_exp  = len(exp_data)
sum_x  = float(exp_data.sum())
a0, b0 = 2.0, 1.0  # Gamma prior hyper-parameters


def log_post_exp(params: np.ndarray) -> float:
    """Log-posterior for Exp(lambda) with Gamma(a0, b0) prior (log_lambda parameterisation)."""
    log_lam = float(params[0])
    lam = np.exp(log_lam)
    ll   = n_exp * log_lam - lam * sum_x
    lp   = (a0 - 1) * log_lam - b0 * lam
    return float(ll + lp + log_lam)  # Jacobian for change of variables


result_custom = run_mcmc(
    log_posterior=log_post_exp,
    init=np.array([np.log(1.0 / exp_data.mean())]),
    param_names=["log_lambda"],
    n_iter=10_000,
    n_burnin=2_000,
    proposal_std=np.array([0.3]),
    alpha=0.05,
    model_name="Exponential(lambda)",
)
result_custom.summary()
assert isinstance(result_custom, MCMCResult)

lam_samples = np.exp(result_custom.posterior_samples[:, 0])
print("\n  Back-transformed posterior (lambda = exp(log_lambda)):")
print(f"    Posterior mean:  {lam_samples.mean():.4f}")
print(f"    Posterior std:   {lam_samples.std(ddof=1):.4f}")
print(f"    MLE lambda:      {1.0 / exp_data.mean():.4f}  (1 / x_bar)")

fig = result_custom.plot()
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 46: plot_t_test
# =============================================================================
separator("46. plot_t_test")
t_res = t_test_mean(
    bp_treatment, mu0=150.0, alpha=0.05, alternative=AlternativeHypothesis.TWO_SIDED
)
print(f"  T statistic: {t_res.t_statistic:.4f},  df={t_res.df},  "
      f"t_crit=±{t_res.t_critical:.4f}")

fig = plot_t_test(
    t_statistic=t_res.t_statistic,
    t_critical=t_res.t_critical,
    df=t_res.df,
    alternative=t_res.alternative.value,
    title="One-Sample t-test — Blood Pressure",
)
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 47: plot_f_test
# =============================================================================
separator("47. plot_f_test")
f_res = f_test_variances(
    bp_treatment, bp_control, alpha=0.05, alternative=AlternativeHypothesis.TWO_SIDED
)
print(f"  F statistic:  {f_res.f_statistic:.4f}")
print(f"  F critical:   [{f_res.f_critical_lower:.4f}, {f_res.f_critical_upper:.4f}]")
print(f"  p-value:      {f_res.p_value:.6f}")

fig = plot_f_test(
    f_statistic=f_res.f_statistic,
    f_critical_low=f_res.f_critical_lower,
    f_critical_up=f_res.f_critical_upper,
    df1=f_res.df1,
    df2=f_res.df2,
    alternative=f_res.alternative.value,
    title="F-test for Variances — Blood Pressure",
)
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 48: plot_simultaneous_ci
# =============================================================================
separator("48. plot_simultaneous_ci")
print("Simultaneous CI forest plot for toxin contrasts vs control:")

C = np.array(
    [
        [1, 0, 0, -1],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [1 / 3, 1 / 3, 1 / 3, -1],
    ]
)
ci_result_plot = anova1_ci_linear_combs(anova_data, alpha=0.05, C=C, method=CorrectionMethod.BEST)
labels = ["Toxin1 − Ctrl", "Toxin2 − Ctrl", "Toxin3 − Ctrl", "Mean − Ctrl"]

fig = plot_simultaneous_ci(
    point_estimates=ci_result_plot.point_estimates,
    intervals=ci_result_plot.intervals,
    method=ci_result_plot.method_used.value,
    labels=labels,
    title="Simultaneous CIs — Toxin vs Control",
)
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
print(f"  Method used: {ci_result_plot.method_used.value}")
fig.clf()


# =============================================================================
separator("DEMO COMPLETE")
print("All statscore functions demonstrated successfully.")
print()
