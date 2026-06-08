"""Tests for the regression summary module."""

import numpy as np
import pytest
from scipy import stats

from statscore.regression.summary import (
    RegressionSummaryResult,
    regression_summary,
)


class TestRegressionSummary:
    def setup_method(self):
        np.random.seed(42)
        n = 50
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        self.X = np.column_stack([np.ones(n), x1, x2])
        self.beta_true = np.array([2.0, 3.0, -1.5])
        self.y = self.X @ self.beta_true + np.random.normal(0, 1, n)

    def test_result_type(self):
        result = regression_summary(self.X, self.y)
        assert isinstance(result, RegressionSummaryResult)

    def test_dimensions(self):
        result = regression_summary(self.X, self.y)
        assert result.n == 50
        assert result.p == 3
        assert result.df_residual == 47
        assert len(result.beta_hat) == 3
        assert len(result.std_errors) == 3
        assert len(result.t_statistics) == 3
        assert len(result.p_values) == 3
        assert len(result.conf_intervals) == 3

    def test_t_statistics_computation(self):
        result = regression_summary(self.X, self.y)
        expected_t = result.beta_hat / result.std_errors
        np.testing.assert_allclose(result.t_statistics, expected_t, rtol=1e-10)

    def test_p_values_two_sided(self):
        result = regression_summary(self.X, self.y)
        for i in range(result.p):
            expected_p = 2 * (1 - stats.t.cdf(abs(result.t_statistics[i]), result.df_residual))
            np.testing.assert_allclose(result.p_values[i], expected_p, rtol=1e-6)

    def test_confidence_intervals_contain_true_beta(self):
        np.random.seed(7)
        n = 100
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        X = np.column_stack([np.ones(n), x1, x2])
        beta_true = np.array([2.0, 3.0, -1.5])
        y = X @ beta_true + np.random.normal(0, 0.5, n)
        result = regression_summary(X, y, alpha=0.05)
        for i in range(3):
            lo, hi = result.conf_intervals[i]
            assert lo < beta_true[i] < hi

    def test_confidence_interval_symmetry(self):
        result = regression_summary(self.X, self.y)
        for i in range(result.p):
            lo, hi = result.conf_intervals[i]
            center = result.beta_hat[i]
            np.testing.assert_allclose(center - lo, hi - center, rtol=1e-10)

    def test_r_squared_range(self):
        result = regression_summary(self.X, self.y)
        assert 0 <= result.R_squared <= 1
        assert result.adj_R_squared <= result.R_squared

    def test_f_statistic_positive(self):
        result = regression_summary(self.X, self.y)
        assert result.F_statistic > 0
        assert result.F_p_value >= 0

    def test_significant_regression(self):
        result = regression_summary(self.X, self.y, alpha=0.05)
        assert result.F_p_value < 0.05

    def test_se_positive(self):
        result = regression_summary(self.X, self.y)
        assert result.Se > 0

    def test_alpha_stored(self):
        result = regression_summary(self.X, self.y, alpha=0.01)
        assert result.alpha == 0.01

    def test_simple_regression(self):
        np.random.seed(7)
        n = 20
        x = np.arange(n, dtype=float)
        X = np.column_stack([np.ones(n), x])
        y = 5.0 + 2.0 * x + np.random.normal(0, 0.5, n)
        result = regression_summary(X, y)
        assert result.p == 2
        assert result.R_squared > 0.95

    def test_no_regression_low_r_squared(self):
        np.random.seed(42)
        n = 30
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        result = regression_summary(X, y)
        assert result.R_squared < 0.3


class TestRegressionSummaryPrint:
    def test_output_contains_key_sections(self, capsys):
        np.random.seed(42)
        n = 30
        x = np.random.uniform(0, 10, n)
        X = np.column_stack([np.ones(n), x])
        y = 2.0 + 3.0 * x + np.random.normal(0, 1, n)
        result = regression_summary(X, y)
        result.summary(feature_names=["(Intercept)", "x1"])
        captured = capsys.readouterr()
        assert "Regression Summary" in captured.out
        assert "Observations" in captured.out
        assert "R²" in captured.out
        assert "Coefficients" in captured.out
        assert "Overall F" in captured.out
        assert "Significance" in captured.out

    def test_output_with_default_names(self, capsys):
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        y = np.random.randn(n)
        result = regression_summary(X, y)
        result.summary()
        captured = capsys.readouterr()
        assert "(Intercept)" in captured.out
        assert "x1" in captured.out
        assert "x2" in captured.out

    def test_significance_stars(self, capsys):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        X = np.column_stack([np.ones(n), x])
        y = 2.0 + 5.0 * x + np.random.normal(0, 0.5, n)
        result = regression_summary(X, y)
        result.summary(feature_names=["(Intercept)", "x1"])
        captured = capsys.readouterr()
        assert "***" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
