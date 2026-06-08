"""
Example analyses using the static fixture files in tests/fixtures/.

Each section loads a fixture file and runs a statscore analysis on it.
"""

from pathlib import Path

import numpy as np

from statscore import (
    AlternativeHypothesis,
    load_data,
    mean_confidence_interval,
    mult_lr_least_squares,
    mult_lr_partition_tss,
    mult_norm_lr_test_linear_reg,
    regression_summary,
    shapiro_wilk_test,
    t_test_mean,
    t_test_two_sample,
)

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# =============================================================================
# ANALYSIS 1: basic.csv — two-sample t-test (treatment A vs B)
# =============================================================================
separator("1. basic.csv — Two-sample t-test: treatment A vs B")

data = load_data(str(FIXTURES / "basic.csv"))
print(f"Loaded: {data.n_rows} rows, columns: {data.column_names}")

group_a = data.df[data.df["treatment"] == "A"]["score"].values
group_b = data.df[data.df["treatment"] == "B"]["score"].values

print(f"\n  Treatment A scores: {group_a}  (mean={group_a.mean():.2f})")
print(f"  Treatment B scores: {group_b}  (mean={group_b.mean():.2f})")

result = t_test_two_sample(
    group_a,
    group_b,
    alpha=0.05,
    alternative=AlternativeHypothesis.TWO_SIDED,
    equal_var=False,
)
print("\n  Welch t-test H0: mu_A = mu_B")
print(f"  T = {result.t_statistic:.4f},  df = {result.df:.2f},  p = {result.p_value:.4f}")
print(f"  Decision: {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# ANALYSIS 2: basic.csv — one-sample t-test on overall scores
# =============================================================================
separator("2. basic.csv — One-sample t-test: is mean score > 70?")

scores = data.df["score"].values
print(f"  All scores: {scores}")

result = t_test_mean(
    scores,
    mu0=70.0,
    alpha=0.05,
    alternative=AlternativeHypothesis.GREATER,
)
print("\n  H0: mu = 70  vs  H1: mu > 70")
print(f"  n = {result.n},  x_bar = {result.x_bar:.4f},  s = {result.s:.4f}")
print(
    f"  T = {result.t_statistic:.4f},  t_crit = {result.t_critical:.4f},  p = {result.p_value:.4f}"
)
print(f"  Decision: {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")

ci = mean_confidence_interval(scores, alpha=0.05)
print(f"\n  95% CI for mean score: [{ci.lower:.4f}, {ci.upper:.4f}]")


# =============================================================================
# ANALYSIS 3: basic.csv — normality check on scores
# =============================================================================
separator("3. basic.csv — Shapiro-Wilk normality test on scores")

result = shapiro_wilk_test(scores, alpha=0.05)
print(f"  W = {result.statistic:.4f},  p = {result.p_value:.4f}")
print(f"  Normal at alpha=0.05? {'Yes' if not result.reject_H0 else 'No'}")


# =============================================================================
# ANALYSIS 4: groups.tsv — reconstruct group data and run t-test (control vs high)
# =============================================================================
separator("4. groups.tsv — t-test with reconstructed group data (control vs high)")

grp = load_data(str(FIXTURES / "groups.tsv"))
print(f"Loaded: {grp.n_rows} rows, columns: {grp.column_names}\n")
print(grp.df.to_string(index=False))

# Reconstruct synthetic observations from summary stats using N(mean, sd)
rng = np.random.default_rng(seed=42)
row = grp.df.set_index("group")

control_obs = rng.normal(
    loc=row.loc["control", "mean"],
    scale=row.loc["control", "sd"],
    size=int(row.loc["control", "n"]),
)
high_obs = rng.normal(
    loc=row.loc["high", "mean"], scale=row.loc["high", "sd"], size=int(row.loc["high", "n"])
)

result = t_test_two_sample(
    control_obs,
    high_obs,
    alpha=0.05,
    alternative=AlternativeHypothesis.TWO_SIDED,
    equal_var=False,
)
print("\n  H0: mu_control = mu_high")
print(f"  control: n={len(control_obs)}, mean={control_obs.mean():.3f}")
print(f"  high:    n={len(high_obs)}, mean={high_obs.mean():.3f}")
print(f"  T = {result.t_statistic:.4f},  p = {result.p_value:.6f}")
print(f"  Decision: {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# ANALYSIS 5: records.json — OLS regression (y ~ x1 + x2)
# =============================================================================
separator("5. records.json — OLS regression: y ~ x1 + x2")

rec = load_data(str(FIXTURES / "records.json"))
print(f"Loaded: {rec.n_rows} observations, columns: {rec.column_names}\n")
print(rec.df.to_string(index=False))

y = rec.df["y"].values.astype(float)
x1 = rec.df["x1"].values.astype(float)
# x1 + x2 = 1 for all rows, so including both would cause perfect multicollinearity.
# Use only x1 as the single predictor.
X = np.column_stack([np.ones(len(y)), x1])

ols = mult_lr_least_squares(X, y)
tss = mult_lr_partition_tss(X, y)

print("\n  Model: y = beta0 + beta1*x1  (x2 dropped: x1+x2=1 => perfect collinearity)")
print(f"  beta_hat: intercept={ols.beta_hat[0]:.4f}, x1={ols.beta_hat[1]:.4f}")
print(f"  Se^2 (unbiased) = {ols.sigma2_unbiased:.4f}")
print(f"  R^2 = {tss.R_squared:.4f}")
print(f"  TSS={tss.TSS:.4f},  RegSS={tss.RegSS:.4f},  RSS={tss.RSS:.4f}")

result = mult_norm_lr_test_linear_reg(X, y, alpha=0.05)
print("\n  H0: beta_x1 = 0  (no linear regression)")
print(f"  F = {result.test_statistic:.4f},  p = {result.p_value:.6f}")
print(f"  Decision: {'Reject H0' if result.reject_H0 else 'Do not reject H0'}")


# =============================================================================
# ANALYSIS 6: records.json — full regression summary
# =============================================================================
separator("6. records.json — Full OLS regression summary")

regression_summary(X, y, alpha=0.05, feature_names=["intercept", "x1"]).summary()


# =============================================================================
# ANALYSIS 7: measurements.xlsx — CI for each measurement column
# =============================================================================
separator("7. measurements.xlsx — 95% confidence intervals for measurements")

meas = load_data(str(FIXTURES / "measurements.xlsx"))
print(f"Loaded: {meas.n_rows} samples, columns: {meas.column_names}\n")

for col in ["measurement_1", "measurement_2"]:
    vals = meas.df[col].values.astype(float)
    ci = mean_confidence_interval(vals, alpha=0.05)
    sw = shapiro_wilk_test(vals, alpha=0.05)
    print(f"  {col}:")
    print(f"    mean = {vals.mean():.4f},  sd = {vals.std(ddof=1):.4f}")
    print(f"    95% CI: [{ci.lower:.4f}, {ci.upper:.4f}]")
    print(
        f"    Shapiro-Wilk: W={sw.statistic:.4f}, p={sw.p_value:.4f}, normal={'Yes' if not sw.reject_H0 else 'No'}"
    )
    print()


# =============================================================================
separator("ALL ANALYSES COMPLETE")
print("Files used: basic.csv, groups.tsv, records.json, measurements.xlsx")
print()
