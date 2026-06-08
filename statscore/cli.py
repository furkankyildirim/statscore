"""Interactive CLI for the statscore statistical toolbox."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _parse_data_input(prompt: str) -> np.ndarray:
    """Prompt user for data: file path or inline numbers."""
    raw = input(prompt).strip()
    if not raw:
        raise ValueError("No input provided.")

    if raw.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".json")):
        from statscore.io import load_data

        loaded = load_data(raw)
        print(f"  Loaded {loaded.n_rows} rows x {loaded.n_cols} cols from '{loaded.path}'")
        print(f"  Columns: {loaded.column_names}")
        col = input("  Enter column name: ").strip()
        if col not in loaded.column_names:
            raise ValueError(f"Column '{col}' not found.")
        return np.asarray(loaded.df[col].dropna(), dtype=float)

    raw = raw.replace(",", " ")
    return np.array([float(v) for v in raw.split()])


def _parse_groups_input(prompt: str) -> list[np.ndarray]:
    """Prompt user for multiple groups of data."""
    groups: list[np.ndarray] = []
    print(prompt)
    while True:
        raw = input(f"  Group {len(groups) + 1} (empty to finish): ").strip()
        if not raw:
            break
        if raw.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".json")):
            from statscore.io import load_data

            loaded = load_data(raw)
            print(f"    Loaded from '{loaded.path}'. Columns: {loaded.column_names}")
            col = input("    Enter column name: ").strip()
            if col not in loaded.column_names:
                raise ValueError(f"Column '{col}' not found.")
            groups.append(np.asarray(loaded.df[col].dropna(), dtype=float))
        else:
            raw = raw.replace(",", " ")
            groups.append(np.array([float(v) for v in raw.split()]))
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups.")
    return groups


def _run_one_way_anova() -> None:
    from statscore.anova import anova1_test_equality

    groups = _parse_groups_input("Enter group data (numbers or file path per group):")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = anova1_test_equality(groups, alpha=alpha)
    result.summary()

    show_plot = input("  Show box plot? (y/n) [n]: ").strip().lower()
    if show_plot == "y":
        from statscore.utils.plots import plot_anova_groups

        fig = plot_anova_groups(groups)
        fig.savefig("anova_plot.png", dpi=150)
        print("  Plot saved to anova_plot.png")


def _run_two_way_anova() -> None:
    from statscore.anova import anova2_test_equality
    from statscore.utils.enums import TwoWayTestFactor

    print("  Enter data as a 3-D array (I levels x J levels x K replicates).")
    print("  Provide data row by row. Format per cell group: space-separated numbers.")
    I = int(input("  Number of levels for Factor A (I): ").strip())
    J = int(input("  Number of levels for Factor B (J): ").strip())
    K = int(input("  Number of replicates per cell (K): ").strip())

    data = np.zeros((I, J, K))
    for i in range(I):
        for j in range(J):
            raw = input(f"  Cell ({i + 1},{j + 1}) [{K} values]: ").strip().replace(",", " ")
            vals = [float(v) for v in raw.split()]
            if len(vals) != K:
                raise ValueError(f"Expected {K} values, got {len(vals)}.")
            data[i, j, :] = vals

    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    factor_str = input("  Test factor (A/B/AB) [A]: ").strip().upper() or "A"
    factor = TwoWayTestFactor(factor_str)

    result = anova2_test_equality(data, alpha=alpha, test=factor)
    result.summary()


def _run_z_test() -> None:
    from statscore.testing import z_test_mean
    from statscore.utils.enums import AlternativeHypothesis

    x = _parse_data_input("  Enter sample data: ")
    mu0 = float(input("  Hypothesized mean (mu0): ").strip())
    sigma = float(input("  Known population std (sigma): ").strip())
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = z_test_mean(x, mu0=mu0, sigma=sigma, alpha=alpha, alternative=alt)
    print(f"\n  Z-statistic: {result.z_statistic:.4f}")
    print(f"  Z-critical:  {result.z_critical:.4f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_one_sample_t_test() -> None:
    from statscore.testing import t_test_mean
    from statscore.utils.enums import AlternativeHypothesis

    x = _parse_data_input("  Enter sample data: ")
    mu0 = float(input("  Hypothesized mean (mu0): ").strip())
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = t_test_mean(x, mu0, alpha=alpha, alternative=alt)
    print(f"\n  t-statistic: {result.t_statistic:.4f}")
    print(f"  t-critical:  {result.t_critical:.4f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_two_sample_t_test() -> None:
    from statscore.testing import t_test_two_sample
    from statscore.utils.enums import AlternativeHypothesis

    x1 = _parse_data_input("  Enter sample 1 data: ")
    x2 = _parse_data_input("  Enter sample 2 data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)
    eq_var = input("  Assume equal variances? (y/n) [y]: ").strip().lower() != "n"

    result = t_test_two_sample(x1, x2, alpha=alpha, alternative=alt, equal_var=eq_var)
    print(f"\n  t-statistic: {result.t_statistic:.4f}")
    print(f"  t-critical:  {result.t_critical:.4f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_paired_t_test() -> None:
    from statscore.testing import t_test_paired
    from statscore.utils.enums import AlternativeHypothesis

    x1 = _parse_data_input("  Enter sample 1 (before): ")
    x2 = _parse_data_input("  Enter sample 2 (after): ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = t_test_paired(x1, x2, alpha=alpha, alternative=alt)
    print(f"\n  t-statistic: {result.t_statistic:.4f}")
    print(f"  t-critical:  {result.t_critical:.4f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_f_test() -> None:
    from statscore.testing import f_test_variances
    from statscore.utils.enums import AlternativeHypothesis

    x1 = _parse_data_input("  Enter sample 1 data: ")
    x2 = _parse_data_input("  Enter sample 2 data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = f_test_variances(x1, x2, alpha=alpha, alternative=alt)
    print(f"\n  F-statistic: {result.f_statistic:.4f}")
    print(f"  F-critical:  [{result.f_critical_lower:.4f}, {result.f_critical_upper:.4f}]")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_simple_regression() -> None:
    from statscore.regression.least_squares import mult_lr_least_squares
    from statscore.regression.summary import regression_summary

    x = _parse_data_input("  Enter predictor (x) data: ")
    y = _parse_data_input("  Enter response (y) data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    X = np.column_stack([np.ones(len(x)), x])
    result = regression_summary(X, y, alpha=alpha)

    names_input = input("  Feature name for x [x]: ").strip() or "x"
    feature_names = ["(Intercept)", names_input]
    result.summary(feature_names=feature_names)

    show_reg = input("  Show regression plot? (y/n) [n]: ").strip().lower()
    if show_reg == "y":
        from statscore.utils.plots import plot_regression

        fig = plot_regression(x, y, result.beta_hat)
        fig.savefig("regression_plot.png", dpi=150)
        print("  Plot saved to regression_plot.png")

    show_resid = input("  Show residual plot? (y/n) [n]: ").strip().lower()
    if show_resid == "y":
        from statscore.utils.plots import plot_residuals

        ols = mult_lr_least_squares(X, y)
        fig = plot_residuals(ols.fitted_values, ols.residuals)
        fig.savefig("residual_plot.png", dpi=150)
        print("  Plot saved to residual_plot.png")

    show_qq = input("  Show Q-Q plot of residuals? (y/n) [n]: ").strip().lower()
    if show_qq == "y":
        from statscore.utils.plots import plot_qq

        ols = mult_lr_least_squares(X, y)
        fig = plot_qq(ols.residuals)
        fig.savefig("qq_plot.png", dpi=150)
        print("  Plot saved to qq_plot.png")


def _run_normality_check() -> None:
    from statscore.diagnostics import shapiro_wilk_test

    x = _parse_data_input("  Enter sample data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = shapiro_wilk_test(x, alpha=alpha)
    print(f"\n  Shapiro-Wilk statistic: {result.statistic:.4f}")
    print(f"  p-value:               {result.p_value:.4f}")
    print(
        f"  Decision:              {'Reject H0 (not normal)' if result.reject_H0 else 'Fail to reject H0 (consistent with normality)'}"
    )


def _run_levene_check() -> None:
    from statscore.diagnostics import levene_test

    groups = _parse_groups_input("Enter group data:")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = levene_test(groups, alpha=alpha)
    print(f"\n  Levene statistic: {result.statistic:.4f}")
    print(f"  p-value:          {result.p_value:.4f}")
    print(
        f"  Decision:         {'Reject H0 (variances differ)' if result.reject_H0 else 'Fail to reject H0 (variances are homogeneous)'}"
    )


def _run_chi2_test() -> None:
    from statscore.testing import chi2_test_variance
    from statscore.utils.enums import AlternativeHypothesis

    x = _parse_data_input("  Enter sample data: ")
    sigma0_sq = float(input("  Hypothesized variance (sigma0^2): ").strip())
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = chi2_test_variance(x, sigma0_sq=sigma0_sq, alpha=alpha, alternative=alt)
    print(f"\n  Chi2-statistic: {result.chi2_statistic:.4f}")
    print(f"  p-value:        {result.p_value:.4f}")
    print(f"  Decision:       {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_regression_diagnostics() -> None:
    from statscore.diagnostics import regression_diagnostics

    print("  Enter design matrix X row by row (each row: space-separated values).")
    print("  Type 'done' when finished.")
    rows: list[list[float]] = []
    while True:
        raw = input(f"  Row {len(rows) + 1}: ").strip()
        if raw.lower() == "done":
            break
        raw = raw.replace(",", " ")
        rows.append([float(v) for v in raw.split()])
    if not rows:
        raise ValueError("No rows entered.")
    X = np.array(rows, dtype=float)
    y = _parse_data_input("  Enter response (y) data: ")

    result = regression_diagnostics(X, y)
    result.summary()


def _run_mean_ci() -> None:
    from statscore.diagnostics import mean_confidence_interval

    x = _parse_data_input("  Enter sample data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    sigma_str = input("  Known population std (sigma) [leave blank for t-interval]: ").strip()
    sigma = float(sigma_str) if sigma_str else None

    result = mean_confidence_interval(x, alpha=alpha, sigma=sigma)
    result.summary()


def _run_bayes_known_var() -> None:
    from statscore.bayes import bayes_normal_mean_known_var

    x = _parse_data_input("  Enter sample data: ")
    sigma_sq = float(input("  Known variance (sigma^2): ").strip())
    mu0 = float(input("  Prior mean (mu0): ").strip())
    kappa0 = float(input("  Prior precision scaling (kappa0): ").strip())
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = bayes_normal_mean_known_var(x, sigma_sq=sigma_sq, mu0=mu0, kappa0=kappa0, alpha=alpha)
    pct = int((1 - alpha) * 100)
    print(f"\n  Posterior mean:   {result.posterior_mean:.6f}")
    print(f"  Posterior std:    {result.posterior_std:.6f}")
    print(
        f"  {pct}% Credible interval: ({result.credible_interval[0]:.6f}, {result.credible_interval[1]:.6f})"
    )

    show_plot = input("  Show posterior plot? (y/n) [n]: ").strip().lower()
    if show_plot == "y":
        from statscore.utils.plots import plot_posterior_normal

        fig = plot_posterior_normal(result)
        fig.savefig("posterior_plot.png", dpi=150)
        print("  Plot saved to posterior_plot.png")


def _run_bayes_unknown_var() -> None:
    from statscore.bayes import bayes_normal_mean_unknown_var

    x = _parse_data_input("  Enter sample data: ")
    mu0 = float(input("  Prior mean (mu0): ").strip())
    kappa0 = float(input("  Prior precision scaling (kappa0): ").strip())
    alpha0 = float(input("  Prior shape (alpha0): ").strip())
    beta0 = float(input("  Prior rate (beta0): ").strip())
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = bayes_normal_mean_unknown_var(
        x, mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0, alpha=alpha
    )
    result.summary()


_MENU_HANDLERS = {
    "1": _run_one_way_anova,
    "2": _run_two_way_anova,
    "3": _run_z_test,
    "4": _run_one_sample_t_test,
    "5": _run_two_sample_t_test,
    "6": _run_paired_t_test,
    "7": _run_chi2_test,
    "8": _run_f_test,
    "9": _run_simple_regression,
    "10": _run_regression_diagnostics,
    "11": _run_normality_check,
    "12": _run_levene_check,
    "13": _run_mean_ci,
    "14": _run_bayes_known_var,
    "15": _run_bayes_unknown_var,
}


def _print_menu() -> None:
    print("=" * 60)
    print("  statscore — Statistical Toolbox")
    print("  Type a number to select a method, or 'q' to quit.")
    print("=" * 60)
    print("  [1]  One-Way ANOVA")
    print("  [2]  Two-Way ANOVA")
    print("  [3]  Z-test for Mean (sigma known)")
    print("  [4]  One-Sample t-test")
    print("  [5]  Two-Sample t-test")
    print("  [6]  Paired t-test")
    print("  [7]  Chi-squared Test for Variance")
    print("  [8]  F-test for Variances")
    print("  [9]  Simple Linear Regression")
    print("  [10] Regression Diagnostics (Leverage / Cook's D)")
    print("  [11] Normality Check (Shapiro-Wilk)")
    print("  [12] Variance Homogeneity (Levene)")
    print("  [13] Confidence Interval for Mean")
    print("  [14] Bayesian Inference (Normal, known variance)")
    print("  [15] Bayesian Inference (Normal, unknown variance)")
    print("=" * 60)


def main() -> None:
    """Entry point for the statscore CLI."""
    parser = argparse.ArgumentParser(
        prog="statscore",
        description="statscore — Interactive Statistical Toolbox",
    )
    parser.parse_args()

    _print_menu()
    while True:
        try:
            choice = input("Select> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            sys.exit(0)

        if choice.lower() == "q":
            print("Goodbye.")
            sys.exit(0)

        handler = _MENU_HANDLERS.get(choice)
        if handler is None:
            print("  Invalid selection. Try again.")
            continue

        try:
            handler()
        except ValueError as e:
            print(f"  Error: {e}")
        except KeyboardInterrupt:
            print("\n  Interrupted. Returning to menu.")
        print()
