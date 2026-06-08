"""Tests for two-sample hypothesis testing functions."""

import numpy as np
import pytest
from scipy import stats

from statscore.methods.testing import f_test_variances, t_test_paired, t_test_two_sample
from statscore.utils.enums import AlternativeHypothesis

TWO_SIDED = AlternativeHypothesis.TWO_SIDED
GREATER = AlternativeHypothesis.GREATER
LESS = AlternativeHypothesis.LESS


class TestTTestTwoSample:
    def test_equal_var_known_values(self):
        """Verify pooled t-test against scipy."""
        x1 = np.array([23, 25, 18, 29, 21])
        x2 = np.array([31, 33, 35, 30, 28])
        result = t_test_two_sample(x1, x2, alpha=0.05, equal_var=True)
        scipy_t, scipy_p = stats.ttest_ind(x1, x2, equal_var=True)
        np.testing.assert_allclose(result.t_statistic, scipy_t, rtol=1e-6)
        np.testing.assert_allclose(result.p_value, scipy_p, rtol=1e-6)
        assert result.df == 8
        assert result.equal_var is True
        assert result.pooled_var is not None

    def test_welch_known_values(self):
        """Verify Welch's t-test against scipy (approximately, due to df flooring)."""
        x1 = np.array([14, 15, 15, 15, 16, 18, 22, 23, 24, 25, 25])
        x2 = np.array([10, 12, 14, 15, 18, 22, 24, 27, 31, 33, 34, 34, 34])
        result = t_test_two_sample(x1, x2, alpha=0.05, equal_var=False)
        assert result.equal_var is False
        assert result.pooled_var is None
        scipy_t, _ = stats.ttest_ind(x1, x2, equal_var=False)
        np.testing.assert_allclose(result.t_statistic, scipy_t, rtol=1e-6)

    def test_one_sided_greater(self):
        """Two-sample one-sided greater test."""
        x1 = np.array([50, 52, 48, 55, 51, 53])
        x2 = np.array([40, 42, 38, 45, 41, 43])
        result = t_test_two_sample(x1, x2, alpha=0.05, alternative=GREATER)
        assert result.reject_H0
        assert result.t_statistic > 0

    def test_one_sided_less(self):
        """Two-sample one-sided less test."""
        x1 = np.array([40, 42, 38, 45, 41, 43])
        x2 = np.array([50, 52, 48, 55, 51, 53])
        result = t_test_two_sample(x1, x2, alpha=0.05, alternative=LESS)
        assert result.reject_H0
        assert result.t_statistic < 0

    def test_fail_to_reject(self):
        """Samples from same distribution should not reject."""
        np.random.seed(42)
        x1 = np.random.normal(10, 2, 20)
        x2 = np.random.normal(10, 2, 20)
        result = t_test_two_sample(x1, x2, alpha=0.05)
        assert not result.reject_H0


class TestTTestPaired:
    def test_known_values(self):
        """Verify paired t-test against scipy."""
        x1 = np.array([85, 90, 78, 92, 88, 76, 95, 80])
        x2 = np.array([82, 86, 75, 90, 85, 70, 91, 78])
        result = t_test_paired(x1, x2, alpha=0.05)
        scipy_t, scipy_p = stats.ttest_rel(x1, x2)
        np.testing.assert_allclose(result.t_statistic, scipy_t, rtol=1e-6)
        np.testing.assert_allclose(result.p_value, scipy_p, rtol=1e-6)
        assert result.n == 8
        assert result.df == 7

    def test_unequal_lengths_raises(self):
        """Mismatched lengths should raise ValueError."""
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            t_test_paired(x1, x2)

    def test_one_sided_less(self):
        """Paired one-sided less test."""
        x1 = np.array([10, 11, 9, 12, 10, 8])
        x2 = np.array([15, 16, 14, 17, 16, 14])
        result = t_test_paired(x1, x2, alpha=0.05, alternative=LESS)
        assert result.reject_H0
        assert result.d_bar < 0

    def test_one_sided_greater(self):
        """Paired one-sided greater test."""
        x1 = np.array([15, 16, 14, 17, 16, 14])
        x2 = np.array([10, 11, 9, 12, 10, 8])
        result = t_test_paired(x1, x2, alpha=0.05, alternative=GREATER)
        assert result.reject_H0
        assert result.d_bar > 0


class TestFTestVariances:
    def test_two_sided_equal_variances(self):
        """Samples with equal variances should not reject."""
        np.random.seed(42)
        x1 = np.random.normal(0, 2, 30)
        x2 = np.random.normal(0, 2, 30)
        result = f_test_variances(x1, x2, alpha=0.05)
        assert not result.reject_H0

    def test_two_sided_unequal_variances(self):
        """Samples with very different variances should reject."""
        np.random.seed(42)
        x1 = np.random.normal(0, 10, 30)
        x2 = np.random.normal(0, 1, 30)
        result = f_test_variances(x1, x2, alpha=0.05)
        assert result.reject_H0
        expected_f = float(x1.var(ddof=1) / x2.var(ddof=1))
        np.testing.assert_allclose(result.f_statistic, expected_f, rtol=1e-6)
        assert result.df1 == 29
        assert result.df2 == 29

    def test_one_sided_greater(self):
        """F-test one-sided greater."""
        np.random.seed(55)
        x1 = np.random.normal(0, 5, 20)
        x2 = np.random.normal(0, 1, 20)
        result = f_test_variances(x1, x2, alpha=0.05, alternative=GREATER)
        assert result.reject_H0

    def test_one_sided_less(self):
        """F-test one-sided less: x1 has smaller variance than x2."""
        np.random.seed(55)
        x1 = np.random.normal(0, 1, 20)
        x2 = np.random.normal(0, 5, 20)
        result = f_test_variances(x1, x2, alpha=0.05, alternative=LESS)
        assert result.reject_H0
        assert result.f_statistic < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
