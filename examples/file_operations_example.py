"""
File operations examples for statscore.

Covers the complete lifecycle of data files used with statscore:
  1. Writing sample files (CSV, TSV, XLSX, JSON)
  2. Loading them with load_data()
  3. Extracting arrays and running analyses
  4. Exporting results (summary text, matplotlib figures)

Run:
    python examples/file_operations_example.py

The script writes all temporary files to /tmp/statscore_io/ and cleans up
at the end.  Set KEEP_FILES=True to inspect them after the run.
"""

import contextlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from statscore import (
    AlternativeHypothesis,
    anova1_test_equality,
    bayes_normal_mean_known_var,
    load_data,
    mean_confidence_interval,
    mult_lr_least_squares,
    mult_lr_partition_tss,
    regression_summary,
    shapiro_wilk_test,
    t_test_mean,
    t_test_two_sample,
)
from statscore.plots import plot_anova_groups, plot_qq, plot_residuals

KEEP_FILES = False
WORK_DIR = Path("/tmp/statscore_io")
WORK_DIR.mkdir(parents=True, exist_ok=True)


def sep(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


# =============================================================================
# 1. CSV — single numeric column
# =============================================================================
sep("1. CSV — write, load, one-sample t-test")

csv_path = WORK_DIR / "temperatures.csv"
temps = pd.DataFrame({
    "temperature": [36.6, 37.1, 36.8, 37.0, 36.9, 37.2, 36.7, 37.3, 36.5, 37.0],
})
temps.to_csv(csv_path, index=False)
print(f"  Written: {csv_path}")

data = load_data(str(csv_path), sep=",")
print(f"  Loaded:  {data.n_rows} rows × {data.n_cols} cols  columns={data.column_names}")
print(f"  Format:  {data.format}")

x = data.df["temperature"].values.astype(float)
result = t_test_mean(x, mu0=37.0, alpha=0.05, alternative=AlternativeHypothesis.TWO_SIDED)
result.summary()

fig = result.plot()
fig.savefig(str(WORK_DIR / "temperatures_t_test.png"), dpi=150)
print(f"  → Plot saved to {WORK_DIR / 'temperatures_t_test.png'}")


# =============================================================================
# 2. CSV — multiple columns: treatment groups
# =============================================================================
sep("2. CSV — two treatment groups, two-sample t-test from columns")

groups_csv = WORK_DIR / "treatment_groups.csv"
rng = np.random.default_rng(0)
df_groups = pd.DataFrame({
    "drug_A": rng.normal(loc=68.0, scale=8.0, size=20).round(1),
    "drug_B": rng.normal(loc=75.0, scale=9.0, size=20).round(1),
})
df_groups.to_csv(groups_csv, index=False)
print(f"  Written: {groups_csv}")

data = load_data(str(groups_csv), sep=",")
drug_a = data.df["drug_A"].values.astype(float)
drug_b = data.df["drug_B"].values.astype(float)

# Normality checks first
sw_a = shapiro_wilk_test(drug_a, alpha=0.05)
sw_b = shapiro_wilk_test(drug_b, alpha=0.05)
print(f"  Shapiro-Wilk drug_A: W={sw_a.statistic:.4f}, normal={'Yes' if not sw_a.reject_H0 else 'No'}")
print(f"  Shapiro-Wilk drug_B: W={sw_b.statistic:.4f}, normal={'Yes' if not sw_b.reject_H0 else 'No'}")

result = t_test_two_sample(drug_a, drug_b, alpha=0.05,
                            alternative=AlternativeHypothesis.TWO_SIDED, equal_var=False)
result.summary()


# =============================================================================
# 3. CSV with semicolon separator (European locale)
# =============================================================================
sep("3. CSV — semicolon separator, mean confidence interval")

euro_csv = WORK_DIR / "european_data.csv"
euro_df = pd.DataFrame({
    "measurement": [102.3, 98.7, 105.1, 99.8, 103.4, 101.2, 97.6, 104.8, 100.5, 102.9],
})
euro_df.to_csv(euro_csv, index=False, sep=";")
print(f"  Written: {euro_csv}  (semicolon-separated)")

data = load_data(str(euro_csv), sep=";")
vals = data.df["measurement"].values.astype(float)
ci = mean_confidence_interval(vals, alpha=0.05)
ci.summary()


# =============================================================================
# 4. TSV — ANOVA groups encoded as long-format table
# =============================================================================
sep("4. TSV — long-format ANOVA, one-way F-test")

anova_tsv = WORK_DIR / "anova_long.tsv"
df_anova = pd.DataFrame({
    "group": (["control"] * 5 + ["low_dose"] * 5 + ["high_dose"] * 5),
    "response": [8.2, 9.1, 7.8, 8.5, 8.9,
                 12.3, 11.8, 13.1, 12.7, 12.0,
                 15.6, 16.2, 14.9, 15.3, 16.0],
})
df_anova.to_csv(anova_tsv, index=False, sep="\t")
print(f"  Written: {anova_tsv}")

data = load_data(str(anova_tsv))
print(f"  Loaded:  {data.n_rows} rows × {data.n_cols} cols  format={data.format}")

groups = [
    data.df[data.df["group"] == g]["response"].values.astype(float)
    for g in ["control", "low_dose", "high_dose"]
]
result = anova1_test_equality(groups, alpha=0.05)
result.summary()

fig = plot_anova_groups(groups, group_labels=["Control", "Low dose", "High dose"],
                        y_label="Response", title="Treatment Effect (one-way ANOVA)")
fig.savefig(str(WORK_DIR / "anova_groups.png"), dpi=150)
print(f"  → Box plot saved to {WORK_DIR / 'anova_groups.png'}")


# =============================================================================
# 5. JSON — records format, OLS regression
# =============================================================================
sep("5. JSON — records format, OLS regression")

reg_json = WORK_DIR / "regression_data.json"
n = 30
rng2 = np.random.default_rng(1)
x1 = rng2.uniform(0, 10, n).round(2)
x2 = rng2.uniform(0, 5,  n).round(2)
y  = (2.5 + 3.1 * x1 - 1.8 * x2 + rng2.normal(0, 2, n)).round(3)
records = [{"x1": float(x1[i]), "x2": float(x2[i]), "y": float(y[i])} for i in range(n)]
reg_json.write_text(json.dumps(records, indent=2))
print(f"  Written: {reg_json}  ({n} records)")

data = load_data(str(reg_json))
print(f"  Loaded:  {data.n_rows} rows × {data.n_cols} cols  columns={data.column_names}")

y_arr = data.df["y"].values.astype(float)
X = np.column_stack([np.ones(n), data.df["x1"].values, data.df["x2"].values])

summary = regression_summary(X, y_arr, alpha=0.05, feature_names=["x1", "x2"])
summary.summary(feature_names=["x1", "x2"])

tss = mult_lr_partition_tss(X, y_arr)
print(f"\n  R² = {tss.R_squared:.4f},  RegSS = {tss.RegSS:.4f},  RSS = {tss.RSS:.4f}")

ols = mult_lr_least_squares(X, y_arr)
fig_res = plot_residuals(ols.fitted_values, ols.residuals)
fig_qq  = plot_qq(ols.residuals, title="Q-Q Residuals — JSON Regression")
fig_res.savefig(str(WORK_DIR / "json_residuals.png"), dpi=150)
fig_qq.savefig( str(WORK_DIR / "json_qq.png"),        dpi=150)
print(f"  → Diagnostic plots saved to {WORK_DIR / 'json_residuals.png'}, json_qq.png")


# =============================================================================
# 6. Excel (.xlsx) — multiple sheets, per-sheet analysis
# =============================================================================
sep("6. Excel — multiple sheets, per-sheet mean CI and normality check")

xlsx_path = WORK_DIR / "lab_measurements.xlsx"
rng3 = np.random.default_rng(2)
sheet_data = {
    "Lab_A": pd.DataFrame({"value": rng3.normal(100.0, 5.0, 15).round(2)}),
    "Lab_B": pd.DataFrame({"value": rng3.normal(102.5, 6.5, 15).round(2)}),
}
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    for sheet, df in sheet_data.items():
        df.to_excel(writer, sheet_name=sheet, index=False)
print(f"  Written: {xlsx_path}  (2 sheets: Lab_A, Lab_B)")

for sheet in ["Lab_A", "Lab_B"]:
    data = load_data(str(xlsx_path), sheet_name=sheet)
    vals = data.df["value"].values.astype(float)
    ci   = mean_confidence_interval(vals, alpha=0.05)
    sw   = shapiro_wilk_test(vals, alpha=0.05)
    print(f"\n  Sheet '{sheet}':")
    print(f"    n={len(vals)}, mean={vals.mean():.2f}, sd={vals.std(ddof=1):.2f}")
    print(f"    95% CI: [{ci.lower:.3f}, {ci.upper:.3f}]")
    print(f"    Shapiro-Wilk: W={sw.statistic:.4f}, normal={'Yes' if not sw.reject_H0 else 'No'}")


# =============================================================================
# 7. Bayesian analysis on loaded data
# =============================================================================
sep("7. Loaded CSV → Bayesian inference (Normal, known variance)")

bayes_csv = WORK_DIR / "factory_weights.csv"
rng4 = np.random.default_rng(3)
weights = rng4.normal(loc=500.0, scale=2.0, size=25).round(2)
pd.DataFrame({"weight_g": weights}).to_csv(bayes_csv, index=False)
print(f"  Written: {bayes_csv}")

data = load_data(str(bayes_csv), sep=",")
x_b = data.df["weight_g"].values.astype(float)

result_bayes = bayes_normal_mean_known_var(
    x_b, sigma_sq=4.0, mu0=500.0, kappa0=2.0, alpha=0.05
)
result_bayes.summary()

fig = result_bayes.plot()
fig.savefig(str(WORK_DIR / "bayes_factory.png"), dpi=150)
print(f"  → Posterior plot saved to {WORK_DIR / 'bayes_factory.png'}")


# =============================================================================
# 8. Exporting results to file (text summary → .txt, figure → .png / .pdf)
# =============================================================================
sep("8. Exporting results — text summary to .txt, figure to .png and .pdf")

export_dir = WORK_DIR / "exports"
export_dir.mkdir(exist_ok=True)

# Redirect summary() output to a text file
txt_path = export_dir / "anova_summary.txt"
with open(txt_path, "w") as fh, contextlib.redirect_stdout(fh):
    result.summary()           # result from section 4 (ANOVA)
print(f"  Summary text → {txt_path}")

# Save figure as both PNG and PDF
fig_export = plot_anova_groups(groups,
    group_labels=["Control", "Low dose", "High dose"],
    title="Treatment Effect — Export Demo")
fig_export.savefig(str(export_dir / "anova_groups.png"), dpi=150)
fig_export.savefig(str(export_dir / "anova_groups.pdf"))
print(f"  Figure PNG  → {export_dir / 'anova_groups.png'}")
print(f"  Figure PDF  → {export_dir / 'anova_groups.pdf'}")


# =============================================================================
# 9. Round-trip: analyse → save predictions as CSV → reload and verify
# =============================================================================
sep("9. Round-trip — OLS predictions saved to CSV, reloaded for verification")

preds_csv = WORK_DIR / "predictions.csv"
fitted    = ols.fitted_values      # from section 5 regression
residuals = ols.residuals

pd.DataFrame({
    "y_actual":  y_arr,
    "y_fitted":  fitted,
    "residual":  residuals,
}).to_csv(preds_csv, index=False)
print(f"  Predictions saved to {preds_csv}")

loaded_preds = load_data(str(preds_csv), sep=",")
df_preds = loaded_preds.df
print(f"  Reloaded: {loaded_preds.n_rows} rows, columns: {loaded_preds.column_names}")

resid_check = df_preds["residual"].values.astype(float)
sw_resid = shapiro_wilk_test(resid_check, alpha=0.05)
ci_resid = mean_confidence_interval(resid_check, alpha=0.05)
print("\n  Residuals after reload:")
print(f"    Shapiro-Wilk: W={sw_resid.statistic:.4f}, p={sw_resid.p_value:.4f}, "
      f"normal={'Yes' if not sw_resid.reject_H0 else 'No'}")
print(f"    95% CI for residual mean: [{ci_resid.lower:.4f}, {ci_resid.upper:.4f}]")
print("    (should be ≈ 0 for an unbiased model)")


# =============================================================================
# Cleanup
# =============================================================================
if not KEEP_FILES:
    shutil.rmtree(WORK_DIR)
    print(f"\n  Cleaned up {WORK_DIR}")
else:
    print(f"\n  Files kept in {WORK_DIR}")

print("\n" + "=" * 64)
print("  All file operations examples completed.")
print("=" * 64)
