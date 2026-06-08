"""CLI handlers for regression and Bayesian analyses."""

from __future__ import annotations

import numpy as np

from statscore.cli._io import _parse_data_input


def _run_simple_regression() -> None:
    from statscore.methods.regression import mult_lr_least_squares, regression_summary

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


def _run_multiple_regression() -> None:
    """Multiple linear regression: user supplies full design matrix X and y.

    Covers:
    - OLS summary (coefficients, SE, t-stats, p-values, R², F-test)
    - Simultaneous CIs for beta
    - General hypothesis test H0: C*beta = c0
    - Prediction intervals for new observations
    """
    from statscore.cli._io import _parse_matrix_input
    from statscore.methods.regression import (
        mult_norm_lr_pred_ci,
        mult_norm_lr_simul_ci,
        mult_norm_lr_test_general,
        regression_summary,
    )
    from statscore.utils.enums import PredictionMethod

    print("  Multiple Linear Regression")
    print("  ---------------------------")
    print("  Enter design matrix X (include a column of 1s for the intercept).")
    X = _parse_matrix_input("  Design matrix X:")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = _parse_data_input("  Enter response (y) data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    n, p = X.shape
    # Ask for optional feature names
    names_raw = input(
        f"  Feature names (comma-separated, {p} names, or leave blank): "
    ).strip()
    feature_names = [s.strip() for s in names_raw.split(",")] if names_raw else None

    # OLS summary
    summary = regression_summary(X, y, alpha=alpha)
    print()
    summary.summary(feature_names=feature_names)

    # Simultaneous CIs for beta
    show_sci = input("\n  Show simultaneous CIs for beta? (y/n) [n]: ").strip().lower()
    if show_sci == "y":
        sci = mult_norm_lr_simul_ci(X, y, alpha=alpha)
        sci.summary()
        show_sci_plot = input("  Save CI plot? (y/n) [n]: ").strip().lower()
        if show_sci_plot == "y":
            fig = sci.plot()
            fig.savefig("simul_ci_beta.png", dpi=150)
            print("  Plot saved to simul_ci_beta.png")

    # Residual / QQ / regression plots
    show_plots = input("\n  Save diagnostic plots (residuals, Q-Q)? (y/n) [n]: ").strip().lower()
    if show_plots == "y":
        from statscore.methods.regression import mult_lr_least_squares
        from statscore.plots import plot_qq, plot_residuals

        ols = mult_lr_least_squares(X, y)
        plot_residuals(ols.fitted_values, ols.residuals).savefig("residuals.png", dpi=150)
        plot_qq(ols.residuals).savefig("qq_residuals.png", dpi=150)
        print("  Saved residuals.png, qq_residuals.png")

    # General hypothesis test
    run_test = input("\n  Run general hypothesis test H0: C*beta = c0? (y/n) [n]: ").strip().lower()
    if run_test == "y":
        print(f"  Enter C matrix ({p} columns). Use semicolons to separate rows.")
        C_raw = input("  C: ").strip()
        from statscore.cli._io import _parse_raw_string
        C = _parse_raw_string(C_raw)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        r = C.shape[0]
        c0_raw = input(f"  c0 vector ({r} values, space-separated): ").strip().replace(",", " ")
        c0 = np.array([float(v) for v in c0_raw.split()])
        test_result = mult_norm_lr_test_general(X, y, C, c0, alpha=alpha)
        test_result.summary()

    # Prediction intervals
    run_pred = input("\n  Compute prediction intervals for new observations? (y/n) [n]: ").strip().lower()
    if run_pred == "y":
        print(f"  Enter new observation rows (each row = {p} values). Use semicolons.")
        D_raw = input("  D: ").strip()
        from statscore.cli._io import _parse_raw_string
        D = _parse_raw_string(D_raw)
        if D.ndim == 1:
            D = D.reshape(1, -1)
        pred = mult_norm_lr_pred_ci(X, y, D, alpha=alpha, method=PredictionMethod.BEST)
        pred.summary()


def _run_regression_diagnostics() -> None:
    from statscore.cli._io import _parse_matrix_input
    from statscore.methods.diagnostics import regression_diagnostics

    print("  Regression Diagnostics — enter design matrix X, then response y.")
    print("  Include an intercept column of 1s in X if needed.")
    X = _parse_matrix_input("  Design matrix X:")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = _parse_data_input("  Enter response (y) data: ")

    result = regression_diagnostics(X, y)
    result.summary()

    show_plot = input("  Show diagnostics plots? (y/n) [n]: ").strip().lower()
    if show_plot == "y":
        fig = result.plot()
        fig.savefig("regression_diagnostics.png", dpi=150)
        print("  Plot saved to regression_diagnostics.png")


def _run_bayes_known_var() -> None:
    from statscore.methods.bayes import bayes_normal_mean_known_var

    x = _parse_data_input("  Enter sample data: ")
    sigma_sq = float(input("  Known variance (sigma^2): ").strip())
    mu0 = float(input("  Prior mean (mu0): ").strip())
    kappa0 = float(input("  Prior precision scaling (kappa0): ").strip())
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = bayes_normal_mean_known_var(x, sigma_sq=sigma_sq, mu0=mu0, kappa0=kappa0, alpha=alpha)
    result.summary()

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

    show_plot = input("  Show posterior plot? (y/n) [n]: ").strip().lower()
    if show_plot == "y":
        fig = result.plot()
        fig.savefig("posterior_unknown_var.png", dpi=150)
        print("  Plot saved to posterior_unknown_var.png")


def _run_mcmc_regression() -> None:
    """MCMC for linear regression — Bayesian alternative to OLS."""
    from statscore.cli._io import _parse_matrix_input
    from statscore.methods.bayes import mcmc_linear_regression

    print("  MCMC Linear Regression")
    print("  ----------------------")
    print("  Model: Y = X*beta + eps,  eps ~ N(0, sigma^2)")
    print("  Prior: beta_j ~ N(0, beta_prior_std^2), sigma ~ InvGamma(alpha, beta)")
    X = _parse_matrix_input("  Design matrix X (include intercept column of 1s):")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = _parse_data_input("  Enter response (y) data: ")

    beta_prior_std = float(input("  Prior std for beta [10.0]: ").strip() or "10.0")
    sig_alpha = float(input("  sigma prior alpha [2.0]: ").strip() or "2.0")
    sig_beta = float(input("  sigma prior beta [1.0]: ").strip() or "1.0")
    n_iter = int(input("  MCMC iterations [15000]: ").strip() or "15000")
    n_burnin = int(input("  Burn-in [3000]: ").strip() or "3000")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    print("  Running MCMC...")
    result = mcmc_linear_regression(
        X, y,
        beta_prior_std=beta_prior_std,
        sigma_prior_alpha=sig_alpha,
        sigma_prior_beta=sig_beta,
        n_iter=n_iter,
        n_burnin=n_burnin,
        alpha=alpha,
    )
    result.summary()

    show_plot = input("  Show MCMC trace/posterior plots? (y/n) [n]: ").strip().lower()
    if show_plot == "y":
        fig = result.plot()
        fig.savefig("mcmc_regression.png", dpi=150)
        print("  Plot saved to mcmc_regression.png")


def _run_mcmc_normal() -> None:
    from statscore.methods.bayes import mcmc_normal_mean_unknown_var

    x = _parse_data_input("  Enter sample data: ")
    print("  Prior for mu: Normal(mu_prior_mean, mu_prior_std^2)")
    mu_prior_mean = float(input("  mu_prior_mean [0.0]: ").strip() or "0.0")
    mu_prior_std = float(input("  mu_prior_std [10.0]: ").strip() or "10.0")
    print("  Prior for sigma^2: InvGamma(sigma_prior_alpha, sigma_prior_beta)")
    sig_alpha = float(input("  sigma_prior_alpha [2.0]: ").strip() or "2.0")
    sig_beta = float(input("  sigma_prior_beta [1.0]: ").strip() or "1.0")
    n_iter = int(input("  MCMC iterations [12000]: ").strip() or "12000")
    n_burnin = int(input("  Burn-in [2000]: ").strip() or "2000")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    print("  Running MCMC... (this may take a few seconds)")
    result = mcmc_normal_mean_unknown_var(
        x,
        mu_prior_mean=mu_prior_mean,
        mu_prior_std=mu_prior_std,
        sigma_prior_alpha=sig_alpha,
        sigma_prior_beta=sig_beta,
        n_iter=n_iter,
        n_burnin=n_burnin,
        alpha=alpha,
    )
    result.summary()

    show_plot = input("  Show MCMC trace/posterior plots? (y/n) [n]: ").strip().lower()
    if show_plot == "y":
        fig = result.plot()
        fig.savefig("mcmc_normal.png", dpi=150)
        print("  Plot saved to mcmc_normal.png")


def _run_bayes_beta_binomial() -> None:
    from statscore.methods.bayes import bayes_beta_binomial

    n_trials = int(input("  Number of trials (n): ").strip())
    n_successes = int(input("  Number of successes (k): ").strip())
    alpha0 = float(input("  Prior alpha0 (pseudo-successes) [1.0]: ").strip() or "1.0")
    beta0 = float(input("  Prior beta0 (pseudo-failures) [1.0]: ").strip() or "1.0")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = bayes_beta_binomial(
        n_trials=n_trials, n_successes=n_successes,
        alpha0=alpha0, beta0=beta0, alpha=alpha,
    )
    result.summary()

    show_plot = input("  Show prior vs posterior plot? (y/n) [n]: ").strip().lower()
    if show_plot == "y":
        fig = result.plot()
        fig.savefig("beta_binomial.png", dpi=150)
        print("  Plot saved to beta_binomial.png")


def _run_bayes_gamma_poisson() -> None:
    from statscore.methods.bayes import bayes_gamma_poisson

    x = _parse_data_input("  Enter count data (non-negative integers): ")
    alpha0 = float(input("  Prior alpha0 (shape) [1.0]: ").strip() or "1.0")
    beta0 = float(input("  Prior beta0 (rate) [1.0]: ").strip() or "1.0")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = bayes_gamma_poisson(x, alpha0=alpha0, beta0=beta0, alpha=alpha)
    result.summary()

    show_plot = input("  Show prior vs posterior plot? (y/n) [n]: ").strip().lower()
    if show_plot == "y":
        fig = result.plot()
        fig.savefig("gamma_poisson.png", dpi=150)
        print("  Plot saved to gamma_poisson.png")
