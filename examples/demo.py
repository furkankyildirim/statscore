"""
Demonstration script for stats_toolbox.

Exercises all 20 public functions end-to-end with representative sample data.
"""

import numpy as np

from stats_toolbox import (
    ANOVA1_partition_TSS,
    ANOVA1_test_equality,
    ANOVA1_is_contrast,
    ANOVA1_is_orthogonal,
    Bonferroni_correction,
    Sidak_correction,
    ANOVA1_CI_linear_combs,
    ANOVA1_test_linear_combs,
    ANOVA2_partition_TSS,
    ANOVA2_MLE,
    ANOVA2_test_equality,
    Mult_LR_Least_squares,
    Mult_LR_partition_TSS,
    Mult_norm_LR_simul_CI,
    Mult_norm_LR_CR,
    Mult_norm_LR_is_in_CR,
    Mult_norm_LR_test_general,
    Mult_norm_LR_test_comp,
    Mult_norm_LR_test_linear_reg,
    Mult_norm_LR_pred_CI,
    CorrectionMethod,
    PredictionMethod,
    TwoWayTestFactor,
)


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


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
print(f"\nResults:")
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
print(f"\nANOVA Table:")
print(f"  {'Source':<12} {'df':<5} {'SS':<12} {'MS':<12} {'F':<10}")
print(f"  {'Between':<12} {result.df_between:<5} {result.SS_between:<12.4f} {result.MS_between:<12.4f} {result.F_statistic:<10.4f}")
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

# Test with non-contrast
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
print(f"  Sidak is less conservative (larger individual alpha).")


# =============================================================================
# DEMO 7: ANOVA1_CI_linear_combs
# =============================================================================
separator("7. ANOVA1_CI_linear_combs")
print("Simultaneous 95% confidence intervals for contrasts:")
print("  H01: mu_1 - mu_4 (Toxin 1 vs Control)")
print("  H02: mu_2 - mu_4 (Toxin 2 vs Control)")
print("  H03: mu_3 - mu_4 (Toxin 3 vs Control)")
print("  H04: (mu_1+mu_2+mu_3)/3 - mu_4 (Average toxin vs Control)")

C = np.array([
    [1, 0, 0, -1],
    [0, 1, 0, -1],
    [0, 0, 1, -1],
    [1/3, 1/3, 1/3, -1],
])

result = ANOVA1_CI_linear_combs(anova_data, alpha=0.05, C=C, method=CorrectionMethod.BEST)
print(f"\n  Method used: {result.method_used.value}")
for i, (lo, hi) in enumerate(result.intervals):
    print(f"  CI_{i+1}: [{lo:.4f}, {hi:.4f}]  (point est: {result.point_estimates[i]:.4f})")


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
    print(f"  {labels[i]:<35} {result.test_statistics[i]:<8.4f} {result.critical_values[i]:<8.4f} {result.p_values[i]:<10.6f} {result.reject[i]}")


# =============================================================================
# DEMO 9: ANOVA2_partition_TSS
# =============================================================================
separator("9. ANOVA2_partition_TSS")
print("Two-way ANOVA: Detergent x Temperature experiment")
print("  Factor A (detergent): Super, Best (I=2)")
print("  Factor B (temperature): Cold, Warm, Hot (J=3)")
print("  K=4 replicates per cell")

data_2way = np.array([
    [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
    [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
], dtype=float)

result = ANOVA2_partition_TSS(data_2way)
print(f"\n  SS_total: {result.SS_total:.4f}")
print(f"  SS_A:     {result.SS_A:.4f}")
print(f"  SS_B:     {result.SS_B:.4f}")
print(f"  SS_AB:    {result.SS_AB:.4f}")
print(f"  SS_E:     {result.SS_E:.4f}")
print(f"  Check: SS_A + SS_B + SS_AB + SS_E = {result.SS_A + result.SS_B + result.SS_AB + result.SS_E:.4f}")


# =============================================================================
# DEMO 10: ANOVA2_MLE
# =============================================================================
separator("10. ANOVA2_MLE")
mle = ANOVA2_MLE(data_2way)
print(f"  mu_hat = {mle.mu:.4f}")
print(f"  a_hat  = {mle.a}")
print(f"  b_hat  = {mle.b}")
print(f"  delta_hat =")
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

# Print full table
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
n = len(grade)
X = np.column_stack([np.ones(n), attend, homework])
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
print(f"  sigma^2 MLE:     {result.sigma2_mle:.4f}")
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
print(f"  Simultaneous 90% confidence intervals for beta:")
print(f"  Method: {result.method.value}")
labels = ["beta_0 (intercept)", "beta_1 (attend)", "beta_2 (homework)"]
for i, (lo, hi) in enumerate(result.intervals):
    print(f"    {labels[i]:<22}: [{lo:.4f}, {hi:.4f}]")


# =============================================================================
# DEMO 15: Mult_norm_LR_CR
# =============================================================================
separator("15. Mult_norm_LR_CR")
C = np.eye(3)
cr = Mult_norm_LR_CR(X, y, C, alpha=0.1)
print(f"  90% Confidence region for beta (full vector):")
print(f"  Center: {cr.center}")
print(f"  Shape matrix (C(X^TX)^-1 C^T):")
for row in cr.shape_matrix:
    print(f"    {row}")
print(f"  Radius^2: {cr.radius_squared:.4f}")
print(f"  F_critical: {cr.F_critical:.4f}")


# =============================================================================
# DEMO 16: Mult_norm_LR_is_in_CR
# =============================================================================
separator("16. Mult_norm_LR_is_in_CR")
C = np.eye(3)
c0_in = np.array([20, 40, 20])
c0_out = np.array([0, 0, 0])
print(f"  Testing if c0 = {c0_in} is in 90% CR: {Mult_norm_LR_is_in_CR(X, y, C, c0_in, alpha=0.1)}")
print(f"  Testing if c0 = {c0_out} is in 90% CR: {Mult_norm_LR_is_in_CR(X, y, C, c0_out, alpha=0.1)}")


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
print("Prediction: A student with full attendance and no homework.")
print("  d = (1, 1.0, 0.0)")
print("Also predict for: half attendance and full homework completion.")
print("  d = (1, 0.5, 1.0)")

D = np.array([
    [1, 1.0, 0.0],
    [1, 0.5, 1.0],
])
result = Mult_norm_LR_pred_CI(X, y, D, alpha=0.1, method=PredictionMethod.BEST)
print(f"\n  Method: {result.method_used.value}")
print(f"  Point estimates: {result.point_estimates}")
for i, (lo, hi) in enumerate(result.intervals):
    print(f"  CI_{i+1}: [{lo:.4f}, {hi:.4f}] (half-width: {result.half_widths[i]:.4f})")


# =============================================================================
separator("DEMO COMPLETE")
print("All 20 stats_toolbox functions demonstrated successfully.")
print()
