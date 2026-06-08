"""Visualization functions returning matplotlib Figure objects."""

from collections.abc import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import stats

from statscore.bayes.conjugate import NormalMeanKnownVarResult


def plot_regression(
    x: np.ndarray,
    y: np.ndarray,
    beta_hat: np.ndarray,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "Regression Fit",
) -> Figure:
    """Scatter plot with fitted regression line (simple regression).

    Parameters
    ----------
    x : array-like, 1-D
        Predictor values.
    y : array-like, 1-D
        Response values.
    beta_hat : array of length 2
        [intercept, slope].
    x_label, y_label, title : str
        Axis labels and title.

    Returns
    -------
    Figure
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    beta_hat = np.asarray(beta_hat, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = beta_hat[0] + beta_hat[1] * x_line
    ax.plot(x_line, y_line, color="crimson", linewidth=2, label="Fitted line")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(
    fitted: np.ndarray,
    residuals: np.ndarray,
    title: str = "Residuals vs Fitted",
) -> Figure:
    """Residuals versus fitted values plot.

    Parameters
    ----------
    fitted : array-like
        Fitted values.
    residuals : array-like
        Residuals.
    title : str
        Plot title.

    Returns
    -------
    Figure
    """
    fitted = np.asarray(fitted, dtype=float).ravel()
    residuals = np.asarray(residuals, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, residuals, color="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_qq(
    x: np.ndarray,
    title: str = "Normal Q-Q Plot",
) -> Figure:
    """Normal Q-Q plot.

    Parameters
    ----------
    x : array-like
        Sample data.
    title : str
        Plot title.

    Returns
    -------
    Figure
    """
    x = np.asarray(x, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    (osm, osr), (slope, intercept, _) = stats.probplot(x, dist="norm")
    ax.scatter(osm, osr, color="steelblue", alpha=0.7, edgecolors="k", linewidths=0.5)

    line_x = np.array([osm.min(), osm.max()])
    line_y = intercept + slope * line_x
    ax.plot(line_x, line_y, color="crimson", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_anova_groups(
    data: Sequence[np.ndarray],
    group_labels: list[str] | None = None,
    x_label: str = "Group",
    y_label: str = "Value",
    title: str = "Group Distributions",
) -> Figure:
    """Side-by-side box plots with jittered data points for ANOVA groups.

    Parameters
    ----------
    data : Sequence of array-like
        One array per group.
    group_labels : list[str] or None
        Labels for each group.
    x_label, y_label, title : str
        Axis labels and title.

    Returns
    -------
    Figure
    """
    groups: list[np.ndarray] = [np.asarray(g, dtype=float) for g in data]
    n_groups = len(groups)

    if group_labels is None:
        group_labels = [f"Group {i + 1}" for i in range(n_groups)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=group_labels, widths=0.5)

    for i, g in enumerate(groups):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(g))
        ax.scatter(
            np.full(len(g), i + 1) + jitter,
            g,
            alpha=0.5,
            s=25,
            color="steelblue",
            zorder=3,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_posterior_normal(
    result: NormalMeanKnownVarResult,
    x_range_sigma: float = 4.0,
    title: str = "Posterior Distribution",
) -> Figure:
    """Plot prior and posterior normal distributions for Normal-Normal conjugate model.

    Parameters
    ----------
    result : NormalMeanKnownVarResult
        Output from bayes_normal_mean_known_var.
    x_range_sigma : float
        Number of posterior standard deviations for x-axis range.
    title : str
        Plot title.

    Returns
    -------
    Figure
    """
    post_mean = result.mu_n
    post_var = result.posterior_variance
    post_std = result.posterior_std

    # Reconstruct prior: kappa0 = kappa_n - n
    kappa0 = result.kappa_n - result.n
    # prior_var = sigma_sq / kappa0 = posterior_variance * kappa_n / kappa0
    prior_var = post_var * result.kappa_n / kappa0 if kappa0 > 0 else post_var * 10
    prior_std = float(np.sqrt(prior_var))
    # prior mean: mu0 = (mu_n * kappa_n - n * x_bar) / kappa0
    prior_mean = (
        (post_mean * result.kappa_n - result.n * result.x_bar) / kappa0 if kappa0 > 0 else post_mean
    )

    x_min = post_mean - x_range_sigma * max(post_std, prior_std)
    x_max = post_mean + x_range_sigma * max(post_std, prior_std)
    x = np.linspace(x_min, x_max, 500)

    prior_pdf = stats.norm.pdf(x, loc=prior_mean, scale=prior_std)
    post_pdf = stats.norm.pdf(x, loc=post_mean, scale=post_std)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, prior_pdf, color="gray", linestyle="--", linewidth=1.5, label="Prior")
    ax.plot(x, post_pdf, color="steelblue", linewidth=2, label="Posterior")

    ci_lower, ci_upper = result.credible_interval
    ci_mask: list[bool] = ((x >= ci_lower) & (x <= ci_upper)).tolist()
    ax.fill_between(
        x, post_pdf, where=ci_mask, alpha=0.3, color="steelblue", label="Credible interval"
    )

    ax.axvline(post_mean, color="crimson", linestyle="-", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("μ")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
