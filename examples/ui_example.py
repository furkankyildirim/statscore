"""
Streamlit UI usage examples for statscore.

This file serves two purposes:

1. **Quick reference** — shows which Python calls map to each UI page so you
   can reproduce any Streamlit result programmatically.

2. **Data preparation helpers** — shows how to build the CSV / DataFrame that
   the UI's "Data Input" page expects, and how to pass it to statscore
   functions directly (useful for automated pipelines or CI).

Running the Streamlit app
-------------------------
    streamlit run statscore/app.py

The app opens at http://localhost:8501 with a sidebar for page navigation:
    • Data Input          → upload or paste data
    • ANOVA               → one-way and two-way
    • Significance Tests  → Z, t, Chi², F tests
    • Regression          → OLS + diagnostics
    • Bayesian Inference  → conjugate priors
    • Multiple Comparisons→ simultaneous CIs / tests

Programmatic equivalents (this file)
--------------------------------------
Each section below mirrors a Streamlit page.
"""

import io
import textwrap

import numpy as np
import pandas as pd

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
    mult_lr_least_squares,
    mult_norm_lr_pred_ci,
    mult_norm_lr_simul_ci,
    regression_diagnostics,
    regression_summary,
    t_test_mean,
    t_test_paired,
    t_test_two_sample,
    z_test_mean,
)
from statscore.plots import (
    plot_anova_groups,
    plot_posterior_normal,
    plot_qq,
    plot_regression,
    plot_residuals,
    plot_simultaneous_ci,
)

# =============================================================================
# Helpers shared across all sections
# =============================================================================

def sep(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  UI Equivalent: {title}")
    print(f"{'─' * 64}")


def df_to_uploaded_file(df: pd.DataFrame, fmt: str = "csv") -> io.BytesIO:
    """Simulate what the Streamlit file-uploader would receive.

    In the actual UI the user drags a file onto the widget.  Here we serialise
    a DataFrame into bytes so the same `load_data` helper can be used.
    """
    buf = io.BytesIO()
    if fmt == "csv":
        df.to_csv(buf, index=False)
    elif fmt == "json":
        df.to_json(buf, orient="records")
    elif fmt == "xlsx":
        df.to_excel(buf, index=False)
    buf.seek(0)
    return buf


# =============================================================================
# PAGE: Data Input
# UI: user uploads a file or pastes numbers into the text area
# =============================================================================
sep("Data Input page — building and loading a DataFrame")

# The UI shows df.describe() after load; replicate it here:
raw_csv = textwrap.dedent("""\
    group,score
    A,72
    A,68
    A,75
    A,80
    B,55
    B,60
    B,58
    B,63
""")
df = pd.read_csv(io.StringIO(raw_csv))
print("  DataFrame preview:")
print(df.to_string(index=False))
print("\n  df.describe():")
print(df.describe())


# =============================================================================
# PAGE: ANOVA
# UI: "One-Way ANOVA" sub-tab — paste group data, choose alpha
# =============================================================================
sep("ANOVA page — One-Way ANOVA")

group_a = df[df["group"] == "A"]["score"].values.astype(float)
group_b = df[df["group"] == "B"]["score"].values.astype(float)
groups = [group_a, group_b]

result = anova1_test_equality(groups, alpha=0.05)
result.summary()

# The UI renders the ANOVA table then shows a box plot
fig = plot_anova_groups(groups, group_labels=["Group A", "Group B"],
                        title="One-Way ANOVA — Score by Group")
fig.savefig("/tmp/ui_anova_boxplot.png", dpi=150)
print("  → Box plot saved to /tmp/ui_anova_boxplot.png")


# =============================================================================
# PAGE: ANOVA
# UI: "Two-Way ANOVA" sub-tab — enter I, J, K and cell values
# =============================================================================
sep("ANOVA page — Two-Way ANOVA (I=2, J=3, K=4)")

data_2way = np.array([
    [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
    [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
], dtype=float)

for factor in [TwoWayTestFactor.A, TwoWayTestFactor.B, TwoWayTestFactor.AB]:
    r = anova2_test_equality(data_2way, alpha=0.05, test=factor)
    print(f"  Factor {factor.value}: F={r.F_statistic:.4f}, p={r.p_value:.6f}, reject={r.reject_H0}")


# =============================================================================
# PAGE: Significance Tests
# UI: Z-test widget — sample data, mu0, sigma, alpha, alternative
# =============================================================================
sep("Significance Tests page — Z-test for Mean")

bp_treat = np.array([138, 142, 130, 145, 137, 140, 133, 148, 135, 141], dtype=float)

r = z_test_mean(bp_treat, mu0=150.0, sigma=10.0, alpha=0.05,
                alternative=AlternativeHypothesis.TWO_SIDED)
r.summary()

# The UI calls result.plot() which is a t-distribution plot reused for Z
fig = r.plot()
fig.savefig("/tmp/ui_z_test.png", dpi=150)
print("  → Z-test distribution plot saved to /tmp/ui_z_test.png")


# =============================================================================
# PAGE: Significance Tests
# UI: one-sample t-test, two-sample t-test, paired t-test widgets
# =============================================================================
sep("Significance Tests page — t-tests")

bp_control = np.array([155, 162, 149, 158, 160, 153, 157, 151, 164, 156], dtype=float)
bp_before  = np.array([158, 163, 151, 162, 155, 160, 157, 165, 153, 159], dtype=float)

# One-sample
r1 = t_test_mean(bp_treat, mu0=150.0, alpha=0.05,
                 alternative=AlternativeHypothesis.LESS)
print(f"  One-sample t: T={r1.t_statistic:.4f}, p={r1.p_value:.6f}, reject={r1.reject_H0}")

# Two-sample Welch
r2 = t_test_two_sample(bp_treat, bp_control, alpha=0.05,
                        alternative=AlternativeHypothesis.TWO_SIDED, equal_var=False)
print(f"  Two-sample t: T={r2.t_statistic:.4f}, df={r2.df:.2f}, p={r2.p_value:.6f}")

# Paired
r3 = t_test_paired(bp_before, bp_treat, alpha=0.05,
                   alternative=AlternativeHypothesis.GREATER)
print(f"  Paired t:     T={r3.t_statistic:.4f}, p={r3.p_value:.6f}, reject={r3.reject_H0}")

# UI shows distribution plots for each — save one as example
fig = r3.plot()
fig.savefig("/tmp/ui_paired_t.png", dpi=150)
print("  → Paired t-test distribution plot saved to /tmp/ui_paired_t.png")


# =============================================================================
# PAGE: Significance Tests
# UI: Chi² and F-test widgets
# =============================================================================
sep("Significance Tests page — Chi² test for variance / F-test for variances")

r_chi2 = chi2_test_variance(bp_treat, sigma0_sq=100.0, alpha=0.05,
                             alternative=AlternativeHypothesis.TWO_SIDED)
print(f"  Chi²: χ²={r_chi2.chi2_statistic:.4f}, p={r_chi2.p_value:.6f}, reject={r_chi2.reject_H0}")

r_f = f_test_variances(bp_treat, bp_control, alpha=0.05,
                        alternative=AlternativeHypothesis.TWO_SIDED)
print(f"  F-var: F={r_f.f_statistic:.4f}, p={r_f.p_value:.6f}, reject={r_f.reject_H0}")


# =============================================================================
# PAGE: Regression
# UI: enter design matrix X and response y (or upload CSV); then tabs for
#     Summary / Simultaneous CI / Diagnostic Plots / Prediction
# =============================================================================
sep("Regression page — OLS summary + diagnostics + prediction")

attend  = np.array([1, 0.5, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.6, 0.1, 0, 0, 0.7, 0.8, 1])
homework = np.array([0.25, 1, 0.5, 1, 1, 0.75, 1, 0.25, 0, 0, 1, 0.5, 0.25, 0.75, 1])
grade   = np.array([60, 65, 40, 70, 65, 70, 85, 70, 44, 20, 40, 30, 50, 77, 90], dtype=float)
n_obs   = len(grade)
X       = np.column_stack([np.ones(n_obs), attend, homework])

# OLS summary tab
summary = regression_summary(X, grade, alpha=0.05, feature_names=["Attend", "Homework"])
summary.summary(feature_names=["Attend", "Homework"])

# Simultaneous CI tab
sci = mult_norm_lr_simul_ci(X, grade, alpha=0.05)
sci.summary()

# Diagnostic plots tab
ols  = mult_lr_least_squares(X, grade)
diag = regression_diagnostics(X, grade)
diag.summary()

fig_reg  = plot_regression(attend, grade, mult_lr_least_squares(
    np.column_stack([np.ones(n_obs), attend]), grade).beta_hat,
    x_label="Attendance", y_label="Grade")
fig_res  = plot_residuals(ols.fitted_values, ols.residuals)
fig_qq   = plot_qq(ols.residuals, title="Q-Q Residuals")
fig_diag = diag.plot()

fig_reg.savefig("/tmp/ui_regression.png",  dpi=150)
fig_res.savefig("/tmp/ui_residuals.png",   dpi=150)
fig_qq.savefig("/tmp/ui_qq.png",           dpi=150)
fig_diag.savefig("/tmp/ui_diagnostics.png", dpi=150)
print("  → Plots saved to /tmp/ui_regression.png, residuals, qq, diagnostics")

# Prediction tab
D = np.array([[1, 0.9, 0.8], [1, 0.4, 0.3]])
pred = mult_norm_lr_pred_ci(X, grade, D, alpha=0.05, method=PredictionMethod.BEST)
pred.summary()


# =============================================================================
# PAGE: Bayesian Inference
# UI: Normal (known σ²) / Normal-Gamma (unknown σ²) / Beta-Binomial / Gamma-Poisson tabs
# =============================================================================
sep("Bayesian Inference page")

sensor = np.array([9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4, 9.6, 10.1], dtype=float)

# Normal — known σ²
r_kv = bayes_normal_mean_known_var(sensor, sigma_sq=0.04, mu0=10.0, kappa0=2.0)
r_kv.summary()
fig = plot_posterior_normal(r_kv, title="Posterior — Sensor Mean (known σ²)")
fig.savefig("/tmp/ui_posterior_known.png", dpi=150)

# Normal-Gamma — unknown σ²
r_uv = bayes_normal_mean_unknown_var(sensor, mu0=10.0, kappa0=1.0, alpha0=2.0, beta0=0.1)
r_uv.summary()
fig = r_uv.plot()
fig.savefig("/tmp/ui_posterior_unknown.png", dpi=150)

# Beta-Binomial — drug trial: 47 / 60 responders
r_bb = bayes_beta_binomial(n_trials=60, n_successes=47, alpha0=5.0, beta0=5.0)
r_bb.summary()
fig = r_bb.plot()
fig.savefig("/tmp/ui_beta_binomial.png", dpi=150)

# Gamma-Poisson — call centre arrival rate
arrivals = np.array([8, 12, 7, 10, 9, 11, 13, 8, 10, 9], dtype=float)
r_gp = bayes_gamma_poisson(arrivals, alpha0=2.0, beta0=0.2)
r_gp.summary()
fig = r_gp.plot()
fig.savefig("/tmp/ui_gamma_poisson.png", dpi=150)
print("  → Bayesian plots saved to /tmp/ui_posterior_*.png, ui_beta_binomial.png, ui_gamma_poisson.png")


# =============================================================================
# PAGE: Multiple Comparisons
# UI: enter groups, contrast matrix C, alpha, method (Scheffé / Tukey / …)
# =============================================================================
sep("Multiple Comparisons page — simultaneous CIs and tests")

toxin1  = np.array([28, 23, 14, 27, 31], dtype=float)
toxin2  = np.array([33, 36, 34, 29, 24], dtype=float)
toxin3  = np.array([18, 21, 20, 22],     dtype=float)
ctrl    = np.array([11, 14, 11, 16],     dtype=float)
all_groups = [toxin1, toxin2, toxin3, ctrl]

C = np.array([
    [1,  0,  0, -1],
    [0,  1,  0, -1],
    [0,  0,  1, -1],
    [1/3, 1/3, 1/3, -1],
])
labels = ["Toxin1 − Ctrl", "Toxin2 − Ctrl", "Toxin3 − Ctrl", "Mean − Ctrl"]

ci_result = anova1_ci_linear_combs(all_groups, alpha=0.05, C=C, method=CorrectionMethod.BEST)
ci_result.summary()

# The UI renders a forest plot of the CIs
fig = plot_simultaneous_ci(
    ci_result.point_estimates,
    ci_result.intervals,
    method=ci_result.method_used.value,
    labels=labels,
    title="Simultaneous CIs — Toxin vs Control",
)
fig.savefig("/tmp/ui_simul_ci.png", dpi=150)
print("  → CI forest plot saved to /tmp/ui_simul_ci.png")

# Hypothesis tests
d = np.zeros(4)
test_result = anova1_test_linear_combs(all_groups, alpha=0.05, C=C, d=d,
                                        method=CorrectionMethod.BEST)
test_result.summary()


print("\n" + "=" * 64)
print("  All Streamlit UI equivalents demonstrated.")
print("  Plots written to /tmp/ui_*.png")
print("=" * 64)
