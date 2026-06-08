"""Tests for the plots module."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from statscore.bayes.conjugate import bayes_normal_mean_known_var
from statscore.plots import (
    plot_anova_groups,
    plot_posterior_normal,
    plot_qq,
    plot_regression,
    plot_residuals,
)


class TestPlotRegression:
    def test_returns_figure(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])
        beta_hat = np.array([0.0, 2.0])
        fig = plot_regression(x, y, beta_hat)
        assert isinstance(fig, Figure)

    def test_custom_labels(self):
        x = np.arange(10, dtype=float)
        y = 2 * x + 1
        beta_hat = np.array([1.0, 2.0])
        fig = plot_regression(x, y, beta_hat, x_label="Hours", y_label="Score", title="Test")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Hours"
        assert ax.get_ylabel() == "Score"
        assert ax.get_title() == "Test"

    def test_axes_have_data(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        beta_hat = np.array([0.0, 2.0])
        fig = plot_regression(x, y, beta_hat)
        ax = fig.axes[0]
        assert len(ax.collections) >= 1  # scatter
        assert len(ax.lines) >= 1  # fitted line


class TestPlotResiduals:
    def test_returns_figure(self):
        fitted = np.array([2.0, 4.0, 6.0, 8.0])
        residuals = np.array([0.1, -0.2, 0.3, -0.1])
        fig = plot_residuals(fitted, residuals)
        assert isinstance(fig, Figure)

    def test_has_horizontal_line(self):
        fitted = np.arange(10, dtype=float)
        residuals = np.random.default_rng(0).normal(0, 1, 10)
        fig = plot_residuals(fitted, residuals)
        ax = fig.axes[0]
        hlines = [line for line in ax.lines if np.allclose(line.get_ydata(), [0, 0])]
        assert len(hlines) >= 1

    def test_custom_title(self):
        fig = plot_residuals(np.arange(5.0), np.zeros(5), title="Custom")
        ax = fig.axes[0]
        assert ax.get_title() == "Custom"


class TestPlotQQ:
    def test_returns_figure(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        fig = plot_qq(x)
        assert isinstance(fig, Figure)

    def test_has_reference_line(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 30)
        fig = plot_qq(x)
        ax = fig.axes[0]
        assert len(ax.lines) >= 1

    def test_custom_title(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fig = plot_qq(x, title="My QQ")
        ax = fig.axes[0]
        assert ax.get_title() == "My QQ"


class TestPlotAnovaGroups:
    def test_returns_figure(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        fig = plot_anova_groups([g1, g2])
        assert isinstance(fig, Figure)

    def test_with_labels(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        g3 = np.array([7.0, 8.0, 9.0])
        fig = plot_anova_groups([g1, g2, g3], group_labels=["A", "B", "C"])
        assert isinstance(fig, Figure)

    def test_custom_labels_and_title(self):
        g1 = np.array([1.0, 2.0])
        g2 = np.array([3.0, 4.0])
        fig = plot_anova_groups(
            [g1, g2], x_label="Treatment", y_label="Response", title="My Plot"
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Treatment"
        assert ax.get_ylabel() == "Response"
        assert ax.get_title() == "My Plot"

    def test_many_groups(self):
        groups = [np.random.default_rng(i).normal(i, 1, 10) for i in range(5)]
        fig = plot_anova_groups(groups)
        assert isinstance(fig, Figure)


class TestPlotPosteriorNormal:
    def test_returns_figure(self):
        x = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        result = bayes_normal_mean_known_var(x, sigma_sq=4.0, mu0=5.0, kappa0=2.0)
        fig = plot_posterior_normal(result)
        assert isinstance(fig, Figure)

    def test_has_legend(self):
        x = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        result = bayes_normal_mean_known_var(x, sigma_sq=1.0, mu0=5.0, kappa0=1.0)
        fig = plot_posterior_normal(result)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "Prior" in labels
        assert "Posterior" in labels

    def test_custom_title(self):
        x = np.array([1.0, 2.0, 3.0])
        result = bayes_normal_mean_known_var(x, sigma_sq=1.0, mu0=2.0, kappa0=1.0)
        fig = plot_posterior_normal(result, title="Custom Posterior")
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Posterior"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
