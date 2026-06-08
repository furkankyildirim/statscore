"""Tests for one-sample hypothesis testing functions."""

import numpy as np
import pytest
from scipy import stats

from statscore.methods.testing import chi2_test_variance, t_test_mean, z_test_mean
from statscore.utils.enums import AlternativeHypothesis

TWO_SIDED = AlternativeHypothesis.TWO_SIDED
GREATER = AlternativeHypothesis.GREATER
LESS = AlternativeHypothesis.LESS


class TestZTestMean:
    def test_two_sided_reject(self):
        """Known case: sample mean far from mu0 should reject."""
        x = np.array([12.1, 11.8, 12.3, 12.0, 11.9, 12.2, 12.4, 11.7, 12.1, 12.0])
        result = z_test_mean(x, mu0=11.0, sigma=0.5, alpha=0.05, alternative=TWO_SIDED)
        np.testing.assert_allclose(result.x_bar, x.mean(), rtol=1e-6)
        expected_z = (x.mean() - 11.0) / (0.5 / np.sqrt(10))
        np.testing.assert_allclose(result.z_statistic, expected_z, rtol=1e-6)
        assert result.reject_H0

    def test_two_sided_fail_to_reject(self):
        """Sample mean close to mu0 should not reject."""
        x = np.array([5.1, 4.9, 5.0, 5.2, 4.8])
        result = z_test_mean(x, mu0=5.0, sigma=1.0, alpha=0.05, alternative=TWO_SIDED)
        assert not result.reject_H0
        assert result.p_value > 0.05

    def test_one_sided_greater(self):
        """One-sided greater test."""
        x = np.array([10.5, 10.8, 10.3, 10.6, 10.9, 10.4])
        result = z_test_mean(x, mu0=10.0, sigma=0.5, alpha=0.05, alternative=GREATER)
        assert result.reject_H0
        assert result.alternative is GREATER
        expected_p = 1.0 - stats.norm.cdf(result.z_statistic)
        np.testing.assert_allclose(result.p_value, expected_p, rtol=1e-6)

    def test_one_sided_less(self):
        """One-sided less test."""
        x = np.array([9.5, 9.3, 9.7, 9.4, 9.6])
        result = z_test_mean(x, mu0=10.0, sigma=0.5, alpha=0.05, alternative=LESS)
        assert result.reject_H0
        assert result.alternative is LESS
        expected_p = stats.norm.cdf(result.z_statistic)
        np.testing.assert_allclose(result.p_value, expected_p, rtol=1e-6)


class TestTTestMean:
    def test_two_sided_known_values(self):
        """Verify against scipy.stats.ttest_1samp."""
        x = np.array([2.3, 2.5, 2.1, 2.8, 2.4, 2.6, 2.2, 2.7])
        result = t_test_mean(x, mu0=2.0, alpha=0.05, alternative=TWO_SIDED)
        scipy_t, scipy_p = stats.ttest_1samp(x, 2.0)
        np.testing.assert_allclose(result.t_statistic, scipy_t, rtol=1e-6)
        np.testing.assert_allclose(result.p_value, scipy_p, rtol=1e-6)
        assert result.df == 7

    def test_one_sided_greater(self):
        """One-sided greater t-test."""
        x = np.array([105, 108, 103, 107, 110, 106, 109, 104, 111, 102])
        result = t_test_mean(x, mu0=100, alpha=0.05, alternative=GREATER)
        assert result.reject_H0
        assert result.t_statistic > 0

    def test_n_equals_2(self):
        """Edge case: minimum sample size n=2."""
        x = np.array([10.0, 12.0])
        result = t_test_mean(x, mu0=5.0, alpha=0.05, alternative=TWO_SIDED)
        assert result.n == 2
        assert result.df == 1
        np.testing.assert_allclose(result.x_bar, 11.0, rtol=1e-6)

    def test_one_sided_less(self):
        """One-sided less t-test should reject when mean is clearly below mu0."""
        x = np.array([3.1, 2.9, 3.0, 2.8, 3.2, 2.7, 3.1, 2.9])
        result = t_test_mean(x, mu0=5.0, alpha=0.05, alternative=LESS)
        assert result.reject_H0
        assert result.t_statistic < 0
        assert result.alternative is LESS

    def test_fail_to_reject(self):
        """Sample close to mu0 should not reject."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 30)
        result = t_test_mean(x, mu0=0.0, alpha=0.05, alternative=TWO_SIDED)
        assert not result.reject_H0


class TestChi2TestVariance:
    def test_two_sided_known_values(self):
        """Hand-calculated chi-squared test."""
        x = np.array([10.2, 9.8, 10.1, 10.3, 9.9, 10.0, 10.4, 9.7])
        result = chi2_test_variance(x, sigma0_sq=0.04, alpha=0.05, alternative=TWO_SIDED)
        s2 = float(x.var(ddof=1))
        expected_chi2 = 7 * s2 / 0.04
        np.testing.assert_allclose(result.chi2_statistic, expected_chi2, rtol=1e-6)
        assert result.df == 7
        np.testing.assert_allclose(result.s2, s2, rtol=1e-6)

    def test_greater_reject(self):
        """Variance clearly larger than hypothesized should reject (greater)."""
        np.random.seed(99)
        x = np.random.normal(0, 5, 50)
        result = chi2_test_variance(x, sigma0_sq=1.0, alpha=0.05, alternative=GREATER)
        assert result.reject_H0

    def test_less_fail_to_reject(self):
        """Variance larger than hypothesized should not reject (less)."""
        np.random.seed(99)
        x = np.random.normal(0, 5, 50)
        result = chi2_test_variance(x, sigma0_sq=1.0, alpha=0.05, alternative=LESS)
        assert not result.reject_H0

    def test_invalid_alternative(self):
        """Non-enum alternative raises TypeError."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="AlternativeHypothesis"):
            chi2_test_variance(x, sigma0_sq=1.0, alternative="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
