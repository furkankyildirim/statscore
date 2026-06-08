"""Tests for the diagnostics module."""

import numpy as np
import pytest
from scipy import stats

from statscore.diagnostics import (
    LeveneResult,
    MeanConfidenceIntervalResult,
    RegressionDiagnosticsResult,
    ShapiroWilkResult,
    levene_test,
    mean_confidence_interval,
    regression_diagnostics,
    shapiro_wilk_test,
)


class TestShapiroWilkTest:
    def test_normal_data_not_rejected(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        result = shapiro_wilk_test(x, alpha=0.05)
        assert isinstance(result, ShapiroWilkResult)
        assert not result.reject_H0
        assert result.p_value > 0.05
        assert result.n == 50
        assert result.alpha == 0.05

    def test_uniform_data_rejected(self):
        np.random.seed(42)
        x = np.random.uniform(0, 1, 100)
        result = shapiro_wilk_test(x, alpha=0.05)
        assert result.reject_H0
        assert result.p_value < 0.05

    def test_agrees_with_scipy(self):
        np.random.seed(7)
        x = np.random.normal(5, 2, 30)
        result = shapiro_wilk_test(x)
        scipy_stat, scipy_p = stats.shapiro(x)
        np.testing.assert_allclose(result.statistic, scipy_stat, rtol=1e-6)
        np.testing.assert_allclose(result.p_value, scipy_p, rtol=1e-6)

    def test_minimum_sample_size(self):
        x = np.array([1.0, 2.0, 3.0])
        result = shapiro_wilk_test(x)
        assert result.n == 3

    def test_too_few_observations(self):
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="at least 3"):
            shapiro_wilk_test(x)

    def test_large_sample_warns(self):
        np.random.seed(0)
        x = np.random.normal(0, 1, 5001)
        with pytest.warns(UserWarning, match="not reliable for n > 5000"):
            shapiro_wilk_test(x)

    def test_custom_alpha(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 30)
        result = shapiro_wilk_test(x, alpha=0.01)
        assert result.alpha == 0.01


class TestLeveneTest:
    def test_equal_variances_not_rejected(self):
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 30)
        g2 = np.random.normal(0, 1, 30)
        g3 = np.random.normal(0, 1, 30)
        result = levene_test([g1, g2, g3])
        assert isinstance(result, LeveneResult)
        assert not result.reject_H0
        assert result.n_groups == 3

    def test_unequal_variances_rejected(self):
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(0, 5, 50)
        result = levene_test([g1, g2], alpha=0.05)
        assert result.reject_H0
        assert result.p_value < 0.05

    def test_agrees_with_scipy(self):
        np.random.seed(7)
        g1 = np.random.normal(0, 2, 20)
        g2 = np.random.normal(0, 3, 25)
        result = levene_test([g1, g2])
        scipy_stat, scipy_p = stats.levene(g1, g2, center="mean")
        np.testing.assert_allclose(result.statistic, scipy_stat, rtol=1e-6)
        np.testing.assert_allclose(result.p_value, scipy_p, rtol=1e-6)

    def test_too_few_groups(self):
        with pytest.raises(ValueError, match="at least 2 groups"):
            levene_test([np.array([1.0, 2.0])])

    def test_custom_alpha(self):
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 20)
        g2 = np.random.normal(0, 1, 20)
        result = levene_test([g1, g2], alpha=0.01)
        assert result.alpha == 0.01


class TestRegressionDiagnostics:
    def setup_method(self):
        np.random.seed(42)
        n = 50
        x = np.random.uniform(0, 10, n)
        self.X = np.column_stack([np.ones(n), x])
        self.y = 2.0 + 3.0 * x + np.random.normal(0, 1, n)

    def test_result_fields(self):
        result = regression_diagnostics(self.X, self.y)
        assert isinstance(result, RegressionDiagnosticsResult)
        assert result.n == 50
        assert result.p == 2
        assert result.leverage.shape == (50,)
        assert result.standardized_residuals.shape == (50,)
        assert result.cooks_distance.shape == (50,)

    def test_leverage_range(self):
        result = regression_diagnostics(self.X, self.y)
        assert np.all(result.leverage >= 0)
        assert np.all(result.leverage <= 1)

    def test_leverage_sum_equals_p(self):
        result = regression_diagnostics(self.X, self.y)
        np.testing.assert_allclose(result.leverage.sum(), 2.0, atol=1e-10)

    def test_cooks_distance_non_negative(self):
        result = regression_diagnostics(self.X, self.y)
        assert np.all(result.cooks_distance >= 0)

    def test_high_leverage_mask(self):
        result = regression_diagnostics(self.X, self.y)
        threshold = 2.0 * result.p / result.n
        expected = result.leverage > threshold
        np.testing.assert_array_equal(result.high_leverage_mask, expected)

    def test_influential_mask(self):
        result = regression_diagnostics(self.X, self.y)
        threshold = 4.0 / result.n
        expected = result.cooks_distance > threshold
        np.testing.assert_array_equal(result.influential_mask, expected)

    def test_with_influential_point(self):
        X = self.X.copy()
        y = self.y.copy()
        X[-1] = [1, 100]
        y[-1] = 1000
        result = regression_diagnostics(X, y)
        assert result.influential_mask[-1]


class TestMeanConfidenceInterval:
    def test_z_interval_known_sigma(self):
        x = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 11.0, 10.0, 12.0])
        result = mean_confidence_interval(x, alpha=0.05, sigma=2.0)
        assert isinstance(result, MeanConfidenceIntervalResult)
        assert result.method == "z"
        assert result.n == 8
        np.testing.assert_allclose(result.point_estimate, x.mean(), rtol=1e-10)
        assert result.lower < result.point_estimate < result.upper

    def test_t_interval_unknown_sigma(self):
        x = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 11.0, 10.0, 12.0])
        result = mean_confidence_interval(x, alpha=0.05)
        assert result.method == "t"
        assert result.lower < result.point_estimate < result.upper

    def test_z_interval_narrower_than_t(self):
        np.random.seed(42)
        x = np.random.normal(5, 2, 20)
        s = float(x.std(ddof=1))
        z_result = mean_confidence_interval(x, sigma=s)
        t_result = mean_confidence_interval(x)
        z_width = z_result.upper - z_result.lower
        t_width = t_result.upper - t_result.lower
        assert z_width < t_width

    def test_contains_true_mean(self):
        np.random.seed(42)
        true_mu = 100.0
        x = np.random.normal(true_mu, 5, 50)
        result = mean_confidence_interval(x, alpha=0.05)
        assert result.lower < true_mu < result.upper

    def test_margin_of_error(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_confidence_interval(x, alpha=0.05)
        np.testing.assert_allclose(
            result.upper - result.lower, 2 * result.margin_of_error, rtol=1e-10
        )

    def test_alpha_stored(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_confidence_interval(x, alpha=0.10)
        assert result.alpha == 0.10

    def test_wider_at_lower_alpha(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        r01 = mean_confidence_interval(x, alpha=0.01)
        r05 = mean_confidence_interval(x, alpha=0.05)
        assert (r01.upper - r01.lower) > (r05.upper - r05.lower)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
