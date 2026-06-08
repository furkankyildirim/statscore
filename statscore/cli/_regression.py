"""CLI handlers for regression and Bayesian analyses."""

from __future__ import annotations

import numpy as np

from statscore.cli._io import _parse_data_input


def _run_simple_regression() -> None:
    from statscore.methods.regression import mult_lr_least_squares
    from statscore.methods.regression import regression_summary

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
        from statscore.plots import plot_regression

        fig = plot_regression(x, y, result.beta_hat)
        fig.savefig("regression_plot.png", dpi=150)
        print("  Plot saved to regression_plot.png")

    show_resid = input("  Show residual plot? (y/n) [n]: ").strip().lower()
    if show_resid == "y":
        from statscore.plots import plot_residuals

        ols = mult_lr_least_squares(X, y)
        fig = plot_residuals(ols.fitted_values, ols.residuals)
        fig.savefig("residual_plot.png", dpi=150)
        print("  Plot saved to residual_plot.png")

    show_qq = input("  Show Q-Q plot of residuals? (y/n) [n]: ").strip().lower()
    if show_qq == "y":
        from statscore.plots import plot_qq

        ols = mult_lr_least_squares(X, y)
        fig = plot_qq(ols.residuals)
        fig.savefig("qq_plot.png", dpi=150)
        print("  Plot saved to qq_plot.png")


def _run_regression_diagnostics() -> None:
    from statscore.methods.diagnostics import regression_diagnostics

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


def _run_bayes_known_var() -> None:
    from statscore.methods.bayes import bayes_normal_mean_known_var

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
        fig = result.plot()
        fig.savefig("posterior_plot.png", dpi=150)
        print("  Plot saved to posterior_plot.png")


def _run_bayes_unknown_var() -> None:
    from statscore.methods.bayes import bayes_normal_mean_unknown_var

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
