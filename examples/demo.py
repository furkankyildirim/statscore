"""
Demonstration script for statscore.

Exercises all public functions end-to-end with representative sample data.
"""

import os
import tempfile

import numpy as np

from statscore import (
    ANOVA2_MLE,
    AlternativeHypothesis,
    ANOVA1_CI_linear_combs,
    ANOVA1_is_contrast,
    ANOVA1_is_orthogonal,
    ANOVA1_partition_TSS,
    ANOVA1_test_equality,
    ANOVA1_test_linear_combs,
    ANOVA2_partition_TSS,
    ANOVA2_test_equality,
    Bonferroni_correction,
    CorrectionMethod,
    LeveneResult,
    LoadedData,
    MeanConfidenceIntervalResult,
    Mult_LR_Least_squares,
    Mult_LR_partition_TSS,
    Mult_norm_LR_CR,
    Mult_norm_LR_is_in_CR,
    Mult_norm_LR_pred_CI,
    Mult_norm_LR_simul_CI,
    Mult_norm_LR_test_comp,
    Mult_norm_LR_test_general,
    Mult_norm_LR_test_linear_reg,
    PredictionMethod,
    RegressionDiagnosticsResult,
    RegressionSummaryResult,
    ShapiroWilkResult,
    Sidak_correction,
    TwoWayTestFactor,
    bayes_normal_mean_known_var,
    bayes_normal_mean_unknown_var,
    chi2_test_variance,
    f_test_variances,
    levene_test,
    load_data,
    mean_confidence_interval,
    plot_anova_groups,
    plot_posterior_normal,
    plot_qq,
    plot_regression,
    plot_residuals,
    regression_diagnostics,
    regression_summary,
    shapiro_wilk_test,
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
# DEMO 1: ANOVA1_partition_TSS
# =============================================================================
separator("1. ANOVA1_partition_TSS")
print("Data: Toxin experiment (4 groups)")
print(f"  Toxin 1: {toxin1}")
print(f"  Toxin 2: {toxin2}")
print(f"  Toxin 3: {toxin3}")
print(f"  Control: {control}")

result = ANOVA1_partition_TSS(anova_data)
print("\nResults:")
print(f"  Group means: {result.group_means}")
print(f"  Grand mean:  {result.grand_mean:.4f}")
print(f"  SS_total:    {result.SS_total:.4f}")
print(f"  SS_within:   {result.SS_within:.4f}")
print(f"  SS_between:  {result.SS_between:.4f}")
print(f"  Check: SS_w + SS_b = {result.SS_within + result.SS_between:.4f} (should equal SS_total)")


# =============================================================================
# DEMO 2: ANOVA1_test_equality
# =============================================================================
separator("2. ANOVA1_test_equality")
print("Testing H0: mu_1 = mu_2 = mu_3 = mu_4 at alpha = 0.05")

result = ANOVA1_test_equality(anova_data, alpha=0.05)
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
# DEMO 3: ANOVA1_is_contrast
# =============================================================================
separator("3. ANOVA1_is_contrast")
c1 = np.array([1, -1, 0, 0])
c2 = np.array([1, 1, -1, -1])
c3 = np.array([1, 1, 1, 0])

print(f"  c = {c1} -> is contrast? {ANOVA1_is_contrast(c1)}")
print(f"  c = {c2} -> is contrast? {ANOVA1_is_contrast(c2)}")
print(f"  c = {c3} -> is contrast? {ANOVA1_is_contrast(c3)}")


# =============================================================================
# DEMO 4: ANOVA1_is_orthogonal
# =============================================================================
separator("4. ANOVA1_is_orthogonal")
n = np.array([5, 5, 4, 4])
c1 = np.array([1, -1, 0, 0])
c2 = np.array([0, 0, 1, -1])
c3 = np.array([1, 0, -1, 0])

result12 = ANOVA1_is_orthogonal(n, c1, c2)
result13 = ANOVA1_is_orthogonal(n, c1, c3)

print(f"  Group sizes: {n}")
print(f"  c1 = {c1}, c2 = {c2}")
print(f"    Orthogonal? {result12.is_orthogonal}")
print(f"  c1 = {c1}, c3 = {c3}")
print(f"    Orthogonal? {result13.is_orthogonal}")

c_bad = np.array([1, 1, 0, 0])
result_bad = ANOVA1_is_orthogonal(n, c_bad, c2)
print(f"\n  c_bad = {c_bad}, c2 = {c2}")
print(f"    Warning: {result_bad.warning}")


# =============================================================================
# DEMO 5: Bonferroni_correction
# =============================================================================
separator("5. Bonferroni_correction")
alpha = 0.05
m = 4
corrected = Bonferroni_correction(alpha, m)
print(f"  FWER alpha = {alpha}, number of tests m = {m}")
print(f"  Individual significance level: alpha/m = {corrected:.6f}")


# =============================================================================
# DEMO 6: Sidak_correction
# =============================================================================
separator("6. Sidak_correction")
corrected = Sidak_correction(alpha, m)
print(f"  FWER alpha = {alpha}, number of tests m = {m}")
print(f"  Individual significance level: 1-(1-alpha)^(1/m) = {corrected:.6f}")
print(f"  (Compare Bonferroni: {Bonferroni_correction(alpha, m):.6f})")
print("  Sidak is less conservative (larger individual alpha).")


# =============================================================================
# DEMO 7: ANOVA1_CI_linear_combs
# =============================================================================
separator("7. ANOVA1_CI_linear_combs")
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

result = ANOVA1_CI_linear_combs(anova_data, alpha=0.05, C=C, method=CorrectionMethod.BEST)
print(f"\n  Method used: {result.method_used.value}")
for i, (lo, hi) in enumerate(result.intervals):
    print(f"  CI_{i + 1}: [{lo:.4f}, {hi:.4f}]  (point est: {result.point_estimates[i]:.4f})")


# =============================================================================
# DEMO 8: ANOVA1_test_linear_combs
# =============================================================================
separator("8. ANOVA1_test_linear_combs")
print("Testing contrasts at FWER = 0.05:")
d = np.array([0.0, 0.0, 0.0, 0.0])

result = ANOVA1_test_linear_combs(anova_data, alpha=0.05, C=C, d=d, method=CorrectionMethod.BEST)
print(f"  Method used: {result.method_used.value}")
print(f"\n  {'Hypothesis':<35} {'|T|':<8} {'Crit':<8} {'p-value':<10} {'Reject?'}")
labels = ["mu1 - mu4 = 0", "mu2 - mu4 = 0", "mu3 - mu4 = 0", "avg_toxin - mu4 = 0"]
for i in range(4):
    print(
        f"  {labels[i]:<35} {result.test_statistics[i]:<8.4f} {result.critical_values[i]:<8.4f} {result.p_values[i]:<10.6f} {result.reject[i]}"
    )


# =============================================================================
# DEMO 9: ANOVA2_partition_TSS
# =============================================================================
separator("9. ANOVA2_partition_TSS")
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

result = ANOVA2_partition_TSS(data_2way)
print(f"\n  SS_total: {result.SS_total:.4f}")
print(f"  SS_A:     {result.SS_A:.4f}")
print(f"  SS_B:     {result.SS_B:.4f}")
print(f"  SS_AB:    {result.SS_AB:.4f}")
print(f"  SS_E:     {result.SS_E:.4f}")
print(
    f"  Check: SS_A + SS_B + SS_AB + SS_E = {result.SS_A + result.SS_B + result.SS_AB + result.SS_E:.4f}"
)


# =============================================================================
# DEMO 10: ANOVA2_MLE
# =============================================================================
separator("10. ANOVA2_MLE")
mle = ANOVA2_MLE(data_2way)
print(f"  mu_hat = {mle.mu:.4f}")
print(f"  a_hat  = {mle.a}")
print(f"  b_hat  = {mle.b}")
print("  delta_hat =")
print(f"    {mle.delta}")


# =============================================================================
# DEMO 11: ANOVA2_test_equality
# =============================================================================
separator("11. ANOVA2_test_equality")
print("Testing at alpha = 0.05:\n")

for test_type in [TwoWayTestFactor.A, TwoWayTestFactor.B, TwoWayTestFactor.AB]:
    result = ANOVA2_test_equality(data_2way, alpha=0.05, test=test_type)
    print(f"  Test for factor {test_type.value}:")
    print(f"    df={result.df}, SS={result.SS:.4f}, MS={result.MS:.4f}")
    print(f"    F={result.F_statistic:.4f}, F_crit={result.F_critical:.4f}, p={result.p_value:.6f}")
    print(f"    Decision: {'Reject H0' if result.reject_H0 else 'Do not reject H0'}\n")

result = ANOVA2_test_equality(data_2way, alpha=0.05, test=TwoWayTestFactor.A)
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
# DEMO 12: Mult_LR_Least_squares
# =============================================================================
separator("12. Mult_LR_Least_squares")
print("Multiple linear regression: Grade = beta0 + beta1*Attend + beta2*Homework + e")

result = Mult_LR_Least_squares(X, y)
print(f"\n  beta_hat = {result.beta_hat}")
print(f"    beta_0 (intercept): {result.beta_hat[0]:.4f}")
print(f"    beta_1 (attend):    {result.beta_hat[1]:.4f}")
print(f"    beta_2 (homework):  {result.beta_hat[2]:.4f}")
print(f"  sigma^2 MLE:             {result.sigma2_mle:.4f}")
print(f"  sigma^2 unbiased (Se^2): {result.sigma2_unbiased:.4f}")
print(f"  Se = {np.sqrt(result.sigma2_unbiased):.4f}")


# =============================================================================
# DEMO 13: Mult_LR_partition_TSS
# =============================================================================
separator("13. Mult_LR_partition_TSS")
result = Mult_LR_partition_TSS(X, y)
print(f"  TSS   = {result.TSS:.4f}")
print(f"  RegSS = {result.RegSS:.4f}")
print(f"  RSS   = {result.RSS:.4f}")
print(f"  R^2   = {result.R_squared:.4f}")
print(f"  Check: RegSS + RSS = {result.RegSS + result.RSS:.4f} (should = TSS)")


# =============================================================================
# DEMO 14: Mult_norm_LR_simul_CI
# =============================================================================
separator("14. Mult_norm_LR_simul_CI")
result = Mult_norm_LR_simul_CI(X, y, alpha=0.1)
print("  Simultaneous 90% confidence intervals for beta:")
print(f"  Method: {result.method.value}")
labels = ["beta_0 (intercept)", "beta_1 (attend)", "beta_2 (homework)"]
for i, (lo, hi) in enumerate(result.intervals):
    print(f"    {labels[i]:<22}: [{lo:.4f}, {hi:.4f}]")


# =============================================================================
# DEMO 15: Mult_norm_LR_CR
# =============================================================================
separator("15. Mult_norm_LR_CR")
C_full = np.eye(3)
cr = Mult_norm_LR_CR(X, y, C_full, alpha=0.1)
print("  90% Confidence region for beta (full vector):")
print(f"  Center: {cr.center}")
print("  Shape matrix (C(X^TX)^-1 C^T):")
for row in cr.shape_matrix:
    print(f"    {row}")
print(f"  Radius^2: {cr.radius_squared:.4f}")
print(f"  F_critical: {cr.F_critical:.4f}")


# =============================================================================
# DEMO 16: Mult_norm_LR_is_in_CR
# =============================================================================
separator("16. Mult_norm_LR_is_in_CR")
c0_in = np.array([20, 40, 20])
c0_out = np.array([0, 0, 0])
print(
    f"  Testing if c0 = {c0_in} is in 90% CR: {Mult_norm_LR_is_in_CR(X, y, C_full, c0_in, alpha=0.1)}"
)
print(
    f"  Testing if c0 = {c0_out} is in 90% CR: {Mult_norm_LR_is_in_CR(X, y, C_full, c0_out, alpha=0.1)}"
)


# =============================================================================
# DEMO 17: Mult_norm_LR_test_general
# =============================================================================
separator("17. Mult_norm_LR_test_general")
print("Testing H0: beta_1 = beta_2 (attend has same effect as homework)")
C_eq = np.array([[0, 1, -1]])
c0_eq = np.array([0.0])
result = Mult_norm_LR_test_general(X, y, C_eq, c0_eq, alpha=0.1)
print(f"  C = {C_eq[0]}, c0 = {c0_eq}")
print(f"  F statistic: {result.test_statistic:.4f}")
print(f"  F critical:  {result.F_critical:.4f}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 18: Mult_norm_LR_test_comp
# =============================================================================
separator("18. Mult_norm_LR_test_comp")
print("Testing H0: beta_1 = 0 (attending lectures has no effect)")
result = Mult_norm_LR_test_comp(X, y, alpha=0.1, components=[1])
print(f"  F statistic: {result.test_statistic:.4f}")
print(f"  F critical:  {result.F_critical:.4f}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")

print("\nTesting H0: beta_2 = 0 (homework has no effect)")
result = Mult_norm_LR_test_comp(X, y, alpha=0.1, components=[2])
print(f"  F statistic: {result.test_statistic:.4f}")
print(f"  F critical:  {result.F_critical:.4f}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 19: Mult_norm_LR_test_linear_reg
# =============================================================================
separator("19. Mult_norm_LR_test_linear_reg")
print("Testing H0: beta_1 = beta_2 = 0 (no linear regression at all)")
result = Mult_norm_LR_test_linear_reg(X, y, alpha=0.1)
print(f"  F statistic: {result.test_statistic:.4f}")
print(f"  F critical:  {result.F_critical:.4f}")
print(f"  p-value:     {result.p_value:.6f}")
print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# DEMO 20: Mult_norm_LR_pred_CI
# =============================================================================
separator("20. Mult_norm_LR_pred_CI")
print("Prediction: A student with full attendance and no homework (d = (1, 1.0, 0.0))")
print("Also predict: half attendance, full homework completion (d = (1, 0.5, 1.0))")

D = np.array(
    [
        [1, 1.0, 0.0],
        [1, 0.5, 1.0],
    ]
)
result = Mult_norm_LR_pred_CI(X, y, D, alpha=0.1, method=PredictionMethod.BEST)
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
ols_simple = Mult_LR_Least_squares(np.column_stack([np.ones(n_obs), attend]), y)
fig = plot_regression(attend, y, ols_simple.beta_hat, x_label="Attendance", y_label="Grade")
print(f"  Figure type: {type(fig).__name__}  axes: {len(fig.axes)}")
fig.clf()


# =============================================================================
# DEMO 37: plot_residuals
# =============================================================================
separator("37. plot_residuals")
ols_full = Mult_LR_Least_squares(X, y)
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
separator("DEMO COMPLETE")
print("All statscore functions demonstrated successfully.")
print()
