"""
CLI usage examples for statscore.

Demonstrates how to drive the same computations that the interactive CLI
performs (menu items 1–21), but without user prompts — useful for scripting,
automated testing, and understanding what each menu choice does under the hood.

Run:
    python examples/cli_example.py

Interactive CLI:
    statscore          # installed entry-point
    python -m statscore
"""

import numpy as np

from statscore import (
    AlternativeHypothesis,
    CorrectionMethod,
    PredictionMethod,
    TwoWayTestFactor,
    anova1_ci_linear_combs,
    anova1_test_equality,
    anova1_test_linear_combs,
    anova2_test_equality,
    bayes_beta_binomial,
    bayes_gamma_poisson,
    bayes_normal_mean_known_var,
    bayes_normal_mean_unknown_var,
    chi2_test_variance,
    f_test_variances,
    levene_test,
    mcmc_linear_regression,
    mcmc_normal_mean_unknown_var,
    mean_confidence_interval,
    mult_lr_least_squares,
    mult_norm_lr_pred_ci,
    mult_norm_lr_simul_ci,
    mult_norm_lr_test_general,
    plot_qq,
    plot_regression,
    plot_residuals,
    regression_diagnostics,
    regression_summary,
    run_mcmc,
    shapiro_wilk_test,
    t_test_mean,
    t_test_paired,
    t_test_two_sample,
    z_test_mean,
)


def sep(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  CLI Example: {title}")
    print(f"{'─' * 64}")


# =============================================================================
# Shared sample data (mirrors what a user would type into the CLI prompts)
# =============================================================================

# ANOVA groups — toxin experiment
toxin1 = np.array([28, 23, 14, 27, 31], dtype=float)
toxin2 = np.array([33, 36, 34, 29, 24], dtype=float)
toxin3 = np.array([18, 21, 20, 22], dtype=float)
control = np.array([11, 14, 11, 16], dtype=float)
anova_groups = [toxin1, toxin2, toxin3, control]

# Regression — student grades (attendance, homework, grade)
attend  = np.array([1, 0.5, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.6, 0.1, 0, 0, 0.7, 0.8, 1])
homework = np.array([0.25, 1, 0.5, 1, 1, 0.75, 1, 0.25, 0, 0, 1, 0.5, 0.25, 0.75, 1])
grade   = np.array([60, 65, 40, 70, 65, 70, 85, 70, 44, 20, 40, 30, 50, 77, 90], dtype=float)
n_obs   = len(grade)
X_simple = np.column_stack([np.ones(n_obs), attend])
X_full   = np.column_stack([np.ones(n_obs), attend, homework])

# Hypothesis tests — blood pressure
bp_treat   = np.array([138, 142, 130, 145, 137, 140, 133, 148, 135, 141], dtype=float)
bp_control = np.array([155, 162, 149, 158, 160, 153, 157, 151, 164, 156], dtype=float)
bp_before  = np.array([158, 163, 151, 162, 155, 160, 157, 165, 153, 159], dtype=float)

# Bayesian — sensor measurements
sensor = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4, 9.6, 10.1], dtype=float)


# =============================================================================
# [1] One-Way ANOVA
# =============================================================================
sep("[1] One-Way ANOVA — anova1_test_equality")
result = anova1_test_equality(anova_groups, alpha=0.05)
result.summary()
# Optional plot (CLI asks "Show box plot? y/n"):
fig = result.plot()
fig.savefig("/tmp/cli_anova_boxplot.png", dpi=150)
print("  → Plot saved to /tmp/cli_anova_boxplot.png")


# =============================================================================
# [2] Two-Way ANOVA
# =============================================================================
sep("[2] Two-Way ANOVA — anova2_test_equality")
# I=2 detergents × J=3 temps × K=4 replicates
data_2way = np.array(
    [
        [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
        [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
    ],
    dtype=float,
)
for factor in [TwoWayTestFactor.A, TwoWayTestFactor.B, TwoWayTestFactor.AB]:
    r = anova2_test_equality(data_2way, alpha=0.05, test=factor)
    print(f"  Factor {factor.value}: F={r.F_statistic:.4f}, p={r.p_value:.6f}, "
          f"reject={r.reject_H0}")


# =============================================================================
# [3] Multiple Comparisons — simultaneous CIs and hypothesis tests
# =============================================================================
sep("[3] Multiple Comparisons — anova1_ci_linear_combs / anova1_test_linear_combs")
# Contrast matrix: each toxin group vs control
C = np.array([
    [1,  0,  0, -1],
    [0,  1,  0, -1],
    [0,  0,  1, -1],
    [1/3, 1/3, 1/3, -1],
])
ci_result = anova1_ci_linear_combs(anova_groups, alpha=0.05, C=C, method=CorrectionMethod.BEST)
ci_result.summary()

d = np.zeros(4)
test_result = anova1_test_linear_combs(anova_groups, alpha=0.05, C=C, d=d, method=CorrectionMethod.BEST)
test_result.summary()

# Optional plot:
fig = ci_result.plot()
fig.savefig("/tmp/cli_simul_ci.png", dpi=150)
print("  → CI plot saved to /tmp/cli_simul_ci.png")


# =============================================================================
# [4] Z-test for Mean (sigma known)
# =============================================================================
sep("[4] Z-test for Mean — z_test_mean")
r = z_test_mean(bp_treat, mu0=150.0, sigma=10.0, alpha=0.05,
                alternative=AlternativeHypothesis.TWO_SIDED)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_z_test.png", dpi=150)
print("  → Plot saved to /tmp/cli_z_test.png")


# =============================================================================
# [5] One-Sample t-test
# =============================================================================
sep("[5] One-Sample t-test — t_test_mean")
r = t_test_mean(bp_treat, mu0=150.0, alpha=0.05,
                alternative=AlternativeHypothesis.TWO_SIDED)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_t_test_one.png", dpi=150)
print("  → Plot saved to /tmp/cli_t_test_one.png")


# =============================================================================
# [6] Two-Sample t-test (pooled and Welch)
# =============================================================================
sep("[6] Two-Sample t-test — t_test_two_sample")
# Pooled (equal_var=True)
r_pool = t_test_two_sample(bp_treat, bp_control, alpha=0.05,
                            alternative=AlternativeHypothesis.TWO_SIDED, equal_var=True)
print(f"  Pooled:  T={r_pool.t_statistic:.4f}, df={r_pool.df}, p={r_pool.p_value:.6f}")

# Welch (equal_var=False)
r_welch = t_test_two_sample(bp_treat, bp_control, alpha=0.05,
                             alternative=AlternativeHypothesis.TWO_SIDED, equal_var=False)
print(f"  Welch:   T={r_welch.t_statistic:.4f}, df={r_welch.df:.2f}, p={r_welch.p_value:.6f}")
fig = r_welch.plot()
fig.savefig("/tmp/cli_t_test_two.png", dpi=150)
print("  → Plot saved to /tmp/cli_t_test_two.png")


# =============================================================================
# [7] Paired t-test
# =============================================================================
sep("[7] Paired t-test — t_test_paired")
r = t_test_paired(bp_before, bp_treat, alpha=0.05,
                  alternative=AlternativeHypothesis.GREATER)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_t_test_paired.png", dpi=150)
print("  → Plot saved to /tmp/cli_t_test_paired.png")


# =============================================================================
# [8] Chi-squared Test for Variance
# =============================================================================
sep("[8] Chi-squared Test for Variance — chi2_test_variance")
r = chi2_test_variance(bp_treat, sigma0_sq=100.0, alpha=0.05,
                       alternative=AlternativeHypothesis.TWO_SIDED)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_chi2_test.png", dpi=150)
print("  → Plot saved to /tmp/cli_chi2_test.png")


# =============================================================================
# [9] F-test for Variances
# =============================================================================
sep("[9] F-test for Variances — f_test_variances")
r = f_test_variances(bp_treat, bp_control, alpha=0.05,
                     alternative=AlternativeHypothesis.TWO_SIDED)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_f_test.png", dpi=150)
print("  → Plot saved to /tmp/cli_f_test.png")


# =============================================================================
# [10] Simple Linear Regression
# =============================================================================
sep("[10] Simple Linear Regression — regression_summary (attend → grade)")
summary = regression_summary(X_simple, grade, alpha=0.05,
                              feature_names=["Attend"])
summary.summary(feature_names=["Attend"])

ols = mult_lr_least_squares(X_simple, grade)
fig = plot_regression(attend, grade, ols.beta_hat,
                      x_label="Attendance", y_label="Grade")
fig.savefig("/tmp/cli_regression_plot.png", dpi=150)
fig2 = plot_residuals(ols.fitted_values, ols.residuals)
fig2.savefig("/tmp/cli_residual_plot.png", dpi=150)
fig3 = plot_qq(ols.residuals)
fig3.savefig("/tmp/cli_qq_plot.png", dpi=150)
print("  → Plots saved to /tmp/cli_regression_plot.png, residual, qq")


# =============================================================================
# [11] Multiple Linear Regression (full inference suite)
# =============================================================================
sep("[11] Multiple Linear Regression — regression_summary + simul CI + general test + pred CI")
summary = regression_summary(X_full, grade, alpha=0.05,
                              feature_names=["Attend", "Homework"])
summary.summary(feature_names=["Attend", "Homework"])

# Simultaneous CIs for beta
sci = mult_norm_lr_simul_ci(X_full, grade, alpha=0.05)
sci.summary()

# General hypothesis test: H0: beta_attend = beta_homework
C_eq = np.array([[0, 1, -1]])
c0_eq = np.array([0.0])
test = mult_norm_lr_test_general(X_full, grade, C_eq, c0_eq, alpha=0.05)
test.summary()

# Prediction intervals for two new students
D = np.array([[1, 1.0, 0.5], [1, 0.3, 0.8]])
pred = mult_norm_lr_pred_ci(X_full, grade, D, alpha=0.05, method=PredictionMethod.BEST)
pred.summary()


# =============================================================================
# [12] Regression Diagnostics
# =============================================================================
sep("[12] Regression Diagnostics — regression_diagnostics")
diag = regression_diagnostics(X_full, grade)
diag.summary()
fig = diag.plot()
fig.savefig("/tmp/cli_diagnostics.png", dpi=150)
print("  → Plot saved to /tmp/cli_diagnostics.png")


# =============================================================================
# [13] Normality Check (Shapiro-Wilk + Q-Q)
# =============================================================================
sep("[13] Normality Check — shapiro_wilk_test")
sw = shapiro_wilk_test(bp_treat, alpha=0.05)
sw.summary()
fig = sw.plot(bp_treat)
fig.savefig("/tmp/cli_shapiro_qq.png", dpi=150)
print("  → Q-Q plot saved to /tmp/cli_shapiro_qq.png")


# =============================================================================
# [14] Variance Homogeneity (Levene)
# =============================================================================
sep("[14] Levene Test — levene_test")
lev = levene_test(anova_groups, alpha=0.05)
lev.summary()


# =============================================================================
# [15] Confidence Interval for Mean
# =============================================================================
sep("[15] Mean Confidence Interval — mean_confidence_interval")
# t-interval (sigma unknown)
ci_t = mean_confidence_interval(bp_treat, alpha=0.05)
ci_t.summary()

# z-interval (sigma known)
ci_z = mean_confidence_interval(bp_treat, alpha=0.05, sigma=10.0)
ci_z.summary()


# =============================================================================
# [16] Bayesian — Normal (known variance)
# =============================================================================
sep("[16] Bayesian Normal Known Variance — bayes_normal_mean_known_var")
r = bayes_normal_mean_known_var(sensor, sigma_sq=0.04, mu0=10.0, kappa0=2.0, alpha=0.05)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_posterior_known.png", dpi=150)
print("  → Plot saved to /tmp/cli_posterior_known.png")


# =============================================================================
# [17] Bayesian — Normal-Gamma (unknown variance)
# =============================================================================
sep("[17] Bayesian Normal-Gamma — bayes_normal_mean_unknown_var")
r = bayes_normal_mean_unknown_var(sensor, mu0=10.0, kappa0=1.0,
                                   alpha0=2.0, beta0=0.1, alpha=0.05)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_posterior_unknown.png", dpi=150)
print("  → Plot saved to /tmp/cli_posterior_unknown.png")


# =============================================================================
# [18] Bayesian — Beta-Binomial (success probability)
# =============================================================================
sep("[18] Beta-Binomial — bayes_beta_binomial")
# 23 successes in 30 trials, uniform prior Beta(1,1)
r = bayes_beta_binomial(n_trials=30, n_successes=23, alpha0=1.0, beta0=1.0, alpha=0.05)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_beta_binomial.png", dpi=150)
print("  → Plot saved to /tmp/cli_beta_binomial.png")


# =============================================================================
# [19] Bayesian — Gamma-Poisson (count rate)
# =============================================================================
sep("[19] Gamma-Poisson — bayes_gamma_poisson")
# Daily server error counts; weak prior Gamma(1,1)
counts = np.array([3, 5, 2, 4, 6, 3, 7, 2, 4, 5], dtype=float)
r = bayes_gamma_poisson(counts, alpha0=1.0, beta0=1.0, alpha=0.05)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_gamma_poisson.png", dpi=150)
print("  → Plot saved to /tmp/cli_gamma_poisson.png")


# =============================================================================
# [20] Bayesian MCMC — Normal mean & variance
# =============================================================================
sep("[20] MCMC Normal — mcmc_normal_mean_unknown_var")
r = mcmc_normal_mean_unknown_var(
    sensor,
    mu_prior_mean=10.0,
    mu_prior_std=5.0,
    sigma_prior_alpha=2.0,
    sigma_prior_beta=0.5,
    n_iter=12_000,
    n_burnin=2_000,
    alpha=0.05,
)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_mcmc_normal.png", dpi=150)
print("  → Plot saved to /tmp/cli_mcmc_normal.png")


# =============================================================================
# [21] Bayesian MCMC — Linear Regression
# =============================================================================
sep("[21] MCMC Linear Regression — mcmc_linear_regression")
r = mcmc_linear_regression(
    X_full,
    grade,
    beta_prior_std=20.0,
    sigma_prior_alpha=2.0,
    sigma_prior_beta=1.0,
    n_iter=15_000,
    n_burnin=3_000,
    alpha=0.05,
)
r.summary()
fig = r.plot()
fig.savefig("/tmp/cli_mcmc_regression.png", dpi=150)
print("  → Plot saved to /tmp/cli_mcmc_regression.png")


# =============================================================================
# Bonus: custom MCMC via run_mcmc (advanced usage)
# =============================================================================
sep("Bonus: Custom MCMC via run_mcmc — exponential rate estimation")
# Model: x_i ~ Exp(lambda), prior lambda ~ Gamma(2, 1)
# log-posterior (up to constant): (n + alpha0 - 1)*log(lambda) - lambda*(sum_x + beta0)
exp_data = np.array([1.2, 0.8, 2.1, 0.5, 1.7, 0.9, 1.4, 2.3, 0.6, 1.1], dtype=float)
n_exp = len(exp_data)
sum_x = float(exp_data.sum())
alpha0_gamma, beta0_gamma = 2.0, 1.0

def log_post_exponential(params: np.ndarray) -> float:
    log_lam = float(params[0])
    lam = np.exp(log_lam)
    ll = n_exp * log_lam - lam * sum_x
    lp = (alpha0_gamma - 1) * log_lam - beta0_gamma * lam
    return float(ll + lp + log_lam)  # Jacobian for log-lambda reparameterisation

mcmc_result = run_mcmc(
    log_posterior=log_post_exponential,
    init=np.array([np.log(1.0 / exp_data.mean())]),
    param_names=["log_lambda"],
    n_iter=10_000,
    n_burnin=2_000,
    proposal_std=np.array([0.3]),
    alpha=0.05,
    model_name="Exponential(lambda)",
)
mcmc_result.summary()
# Back-transform: lambda = exp(log_lambda)
lam_samples = np.exp(mcmc_result.posterior_samples[:, 0])
print(f"  Posterior mean lambda = {lam_samples.mean():.4f}  "
      f"(MLE = {1.0/exp_data.mean():.4f})")


print("\n" + "=" * 64)
print("  All CLI examples completed.")
print("  Plots written to /tmp/cli_*.png")
print("=" * 64)
