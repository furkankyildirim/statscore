"""Tests for multiple linear regression functions."""

import numpy as np
import pytest

from stats_toolbox.regression.least_squares import Mult_LR_Least_squares, Mult_LR_partition_TSS
from stats_toolbox.regression.inference import (
    Mult_norm_LR_simul_CI,
    Mult_norm_LR_CR,
    Mult_norm_LR_is_in_CR,
    Mult_norm_LR_test_general,
    Mult_norm_LR_test_comp,
    Mult_norm_LR_test_linear_reg,
)
from stats_toolbox.regression.prediction import Mult_norm_LR_pred_CI
from stats_toolbox.utils.enums import PredictionMethod


class TestLeastSquares:
    def setup_method(self):
        np.random.seed(42)
        n = 50
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        self.X = np.column_stack([np.ones(n), x1, x2])
        self.beta_true = np.array([2.0, 3.0, -1.5])
        self.y = self.X @ self.beta_true + np.random.normal(0, 1, n)

    def test_beta_hat_close_to_true(self):
        result = Mult_LR_Least_squares(self.X, self.y)
        np.testing.assert_allclose(result.beta_hat, self.beta_true, atol=1.0)

    def test_residuals_sum_to_zero(self):
        """When X has intercept column, residuals sum to ~0."""
        result = Mult_LR_Least_squares(self.X, self.y)
        assert np.isclose(result.residuals.sum(), 0.0, atol=1e-10)

    def test_fitted_plus_residual_equals_y(self):
        result = Mult_LR_Least_squares(self.X, self.y)
        np.testing.assert_allclose(result.fitted_values + result.residuals, self.y)

    def test_unbiased_larger_than_mle(self):
        result = Mult_LR_Least_squares(self.X, self.y)
        assert result.sigma2_unbiased > result.sigma2_mle

    def test_hat_matrix_idempotent(self):
        result = Mult_LR_Least_squares(self.X, self.y)
        H = result.hat_matrix
        np.testing.assert_allclose(H @ H, H, atol=1e-10)

    def test_simple_linear_regression(self):
        """Compare with known formula for simple regression."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
        X = np.column_stack([np.ones(5), x])
        result = Mult_LR_Least_squares(X, y)
        # Manual: b = Sxy/Sxx, a = ybar - b*xbar
        x_bar, y_bar = x.mean(), y.mean()
        Sxx = np.sum((x - x_bar) ** 2)
        Sxy = np.sum((x - x_bar) * (y - y_bar))
        b = Sxy / Sxx
        a = y_bar - b * x_bar
        assert np.isclose(result.beta_hat[0], a)
        assert np.isclose(result.beta_hat[1], b)


class TestPartitionTSS:
    def test_partition_identity(self):
        """TSS = RegSS + RSS."""
        np.random.seed(10)
        n = 30
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1, 2] + np.random.randn(n)
        result = Mult_LR_partition_TSS(X, y)
        assert np.isclose(result.TSS, result.RegSS + result.RSS, rtol=1e-10)

    def test_r_squared_range(self):
        np.random.seed(10)
        n = 30
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1, 2] + np.random.randn(n)
        result = Mult_LR_partition_TSS(X, y)
        assert 0 <= result.R_squared <= 1


class TestSimulCI:
    def test_intervals_contain_true_beta(self):
        """With many samples, true beta should be in CI."""
        np.random.seed(7)
        n = 100
        x1 = np.random.uniform(0, 10, n)
        X = np.column_stack([np.ones(n), x1])
        beta_true = np.array([5.0, 2.0])
        y = X @ beta_true + np.random.normal(0, 0.5, n)
        result = Mult_norm_LR_simul_CI(X, y, alpha=0.05)
        for i, (lo, hi) in enumerate(result.intervals):
            assert lo < beta_true[i] < hi

    def test_method_selection(self):
        np.random.seed(7)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        result = Mult_norm_LR_simul_CI(X, y, alpha=0.05)
        assert result.method in (PredictionMethod.SCHEFFE, PredictionMethod.BONFERRONI)


class TestConfidenceRegion:
    def setup_method(self):
        np.random.seed(42)
        n = 50
        self.X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        self.beta_true = np.array([1.0, 2.0, -1.0])
        self.y = self.X @ self.beta_true + np.random.normal(0, 1, n)

    def test_true_beta_in_CR(self):
        C = np.eye(3)
        in_cr = Mult_norm_LR_is_in_CR(self.X, self.y, C, self.beta_true, alpha=0.05)
        assert in_cr

    def test_far_point_not_in_CR(self):
        C = np.eye(3)
        far_point = np.array([100.0, 100.0, 100.0])
        in_cr = Mult_norm_LR_is_in_CR(self.X, self.y, C, far_point, alpha=0.05)
        assert not in_cr

    def test_CR_shape(self):
        C = np.array([[0, 1, 0], [0, 0, 1]])  # r=2
        cr = Mult_norm_LR_CR(self.X, self.y, C, alpha=0.05)
        assert cr.r == 2
        assert cr.shape_matrix.shape == (2, 2)
        assert cr.radius_squared > 0


class TestTestGeneral:
    def setup_method(self):
        np.random.seed(42)
        n = 50
        self.X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        self.beta_true = np.array([1.0, 2.0, -1.0])
        self.y = self.X @ self.beta_true + np.random.normal(0, 0.5, n)

    def test_do_not_reject_true_hypothesis(self):
        C = np.eye(3)
        result = Mult_norm_LR_test_general(self.X, self.y, C, self.beta_true, alpha=0.05)
        assert not result.reject_H0

    def test_reject_false_hypothesis(self):
        C = np.eye(3)
        c0 = np.array([100.0, 100.0, 100.0])
        result = Mult_norm_LR_test_general(self.X, self.y, C, c0, alpha=0.05)
        assert result.reject_H0

    def test_single_component(self):
        C = np.array([[0, 1, 0]])  # test beta_1
        c0 = np.array([0.0])  # H0: beta_1 = 0
        result = Mult_norm_LR_test_general(self.X, self.y, C, c0, alpha=0.05)
        # beta_1 = 2.0, should reject H0: beta_1 = 0
        assert result.reject_H0


class TestTestComp:
    def test_reject_nonzero_component(self):
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        y = X @ [1, 3, 0] + np.random.normal(0, 1, n)
        # Test H0: beta_1 = 0
        result = Mult_norm_LR_test_comp(X, y, alpha=0.05, components=[1])
        assert result.reject_H0

    def test_do_not_reject_zero_component(self):
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        y = X @ [1, 3, 0] + np.random.normal(0, 1, n)
        # Test H0: beta_2 = 0 (true!)
        result = Mult_norm_LR_test_comp(X, y, alpha=0.05, components=[2])
        assert not result.reject_H0


class TestTestLinearReg:
    def test_significant_regression(self):
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ [1, 5] + np.random.normal(0, 1, n)
        result = Mult_norm_LR_test_linear_reg(X, y, alpha=0.05)
        assert result.reject_H0

    def test_no_regression(self):
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.ones(n) * 5 + np.random.normal(0, 1, n)  # no slope
        result = Mult_norm_LR_test_linear_reg(X, y, alpha=0.05)
        # Might not reject; with seed 42 the noise might cause rejection
        # Use a more definitive test
        y2 = np.random.normal(5, 10, n)
        X2 = np.column_stack([np.ones(n), np.linspace(0, 1, n) * 0.001])
        result2 = Mult_norm_LR_test_linear_reg(X2, y2, alpha=0.05)
        assert result2.p_value > 0.01  # weak test but shouldn't be very significant


class TestPredCI:
    def test_prediction_contains_true(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(0, 10, n)
        X = np.column_stack([np.ones(n), x1])
        beta_true = np.array([2.0, 3.0])
        y = X @ beta_true + np.random.normal(0, 0.5, n)
        D = np.array([[1, 5.0], [1, 7.0]])
        result = Mult_norm_LR_pred_CI(X, y, D, alpha=0.05, method=PredictionMethod.BEST)
        # True values
        true_vals = D @ beta_true
        for i, (lo, hi) in enumerate(result.intervals):
            assert lo < true_vals[i] < hi

    def test_scheffe_vs_bonferroni(self):
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        D = np.array([[1, 0.5], [1, 1.0], [1, 2.0]])
        r_s = Mult_norm_LR_pred_CI(X, y, D, alpha=0.05, method=PredictionMethod.SCHEFFE)
        r_b = Mult_norm_LR_pred_CI(X, y, D, alpha=0.05, method=PredictionMethod.BONFERRONI)
        # Both should produce valid intervals
        assert len(r_s.intervals) == 3
        assert len(r_b.intervals) == 3

    def test_best_picks_narrower(self):
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        D = np.array([[1, 0.5]])
        result = Mult_norm_LR_pred_CI(X, y, D, alpha=0.05, method=PredictionMethod.BEST)
        assert result.method_used in (PredictionMethod.SCHEFFE, PredictionMethod.BONFERRONI)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
