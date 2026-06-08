"""Tests for multiple testing ANOVA functions."""

import numpy as np
import pytest

from statscore.anova.multiple_tests import (
    anova1_ci_linear_combs,
    anova1_is_contrast,
    anova1_is_orthogonal,
    anova1_test_linear_combs,
    bonferroni_correction,
    sidak_correction,
)
from statscore.utils.enums import CorrectionMethod


class TestIsContrast:
    def test_valid_contrast(self):
        assert anova1_is_contrast([1, -1, 0]) is True
        assert anova1_is_contrast([1, 0, -1]) is True
        assert anova1_is_contrast([1, -0.5, -0.5]) is True

    def test_not_contrast(self):
        assert anova1_is_contrast([1, 1, 0]) is False
        assert anova1_is_contrast([1, 2, 3]) is False

    def test_pairwise_difference(self):
        assert anova1_is_contrast([1, -1, 0, 0]) is True


class TestIsOrthogonal:
    def test_orthogonal_equal_sizes(self):
        n = np.array([10, 10, 10])
        c1 = np.array([1, -1, 0])
        c2 = np.array([1, 1, -2])
        result = anova1_is_orthogonal(n, c1, c2)
        assert result.is_orthogonal
        assert result.c1_is_contrast
        assert result.c2_is_contrast
        assert result.warning is None

    def test_not_orthogonal(self):
        n = np.array([10, 10, 10])
        c1 = np.array([1, -1, 0])
        c2 = np.array([1, 0, -1])
        result = anova1_is_orthogonal(n, c1, c2)
        assert not result.is_orthogonal

    def test_warning_if_not_contrast(self):
        n = np.array([5, 5, 5])
        c1 = np.array([1, 1, 0])  # not a contrast
        c2 = np.array([1, -1, 0])
        result = anova1_is_orthogonal(n, c1, c2)
        assert result.warning is not None
        assert not result.c1_is_contrast
        assert result.c2_is_contrast

    def test_unequal_sizes(self):
        n = np.array([4, 6, 5, 5])
        c1 = np.array([1, -1, 0, 0])
        c2 = np.array([0, 0, 1, -1])
        result = anova1_is_orthogonal(n, c1, c2)
        assert result.is_orthogonal


class TestCorrections:
    def test_bonferroni(self):
        assert np.isclose(bonferroni_correction(0.05, 10), 0.005)
        assert np.isclose(bonferroni_correction(0.1, 4), 0.025)

    def test_sidak(self):
        alpha_corr = sidak_correction(0.05, 10)
        # 1 - (1-0.05)^(1/10) ≈ 0.005116
        assert np.isclose(alpha_corr, 1 - 0.95**0.1, rtol=1e-10)

    def test_sidak_less_conservative_than_bonferroni(self):
        alpha = 0.05
        m = 5
        assert sidak_correction(alpha, m) > bonferroni_correction(alpha, m)

    def test_single_test(self):
        assert np.isclose(bonferroni_correction(0.05, 1), 0.05)
        assert np.isclose(sidak_correction(0.05, 1), 0.05)


class TestCILinearCombs:
    def setup_method(self):
        np.random.seed(42)
        self.data = [
            np.array([28, 23, 14, 27, 31]),
            np.array([33, 36, 34, 29, 24]),
            np.array([18, 21, 20, 22]),
            np.array([11, 14, 11, 16]),
        ]

    def test_scheffe_contrasts(self):
        C = np.array([[1, -1, 0, 0], [0, 0, 1, -1]])
        result = anova1_ci_linear_combs(self.data, 0.05, C, method=CorrectionMethod.SCHEFFE)
        assert len(result.intervals) == 2
        for lo, hi in result.intervals:
            assert lo < hi

    def test_bonferroni(self):
        C = np.array([[1, -1, 0, 0], [1, 0, -1, 0]])
        result = anova1_ci_linear_combs(self.data, 0.05, C, method=CorrectionMethod.BONFERRONI)
        assert len(result.intervals) == 2
        assert result.method_used == CorrectionMethod.BONFERRONI

    def test_best_method(self):
        C = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
        result = anova1_ci_linear_combs(self.data, 0.05, C, method=CorrectionMethod.BEST)
        assert len(result.intervals) == 3

    def test_tukey_invalid_raises(self):
        # non-pairwise — Tukey requires pairwise comparisons
        C2 = np.array([[1, 1, -1, -1]])
        with pytest.raises(ValueError):
            anova1_ci_linear_combs(self.data, 0.05, C2, method=CorrectionMethod.TUKEY)


class TestTestLinearCombs:
    def setup_method(self):
        self.data = [
            np.array([28, 23, 14, 27, 31]),
            np.array([33, 36, 34, 29, 24]),
            np.array([18, 21, 20, 22]),
            np.array([11, 14, 11, 16]),
        ]

    def test_reject_clear_difference(self):
        C = np.array([[1, 0, 0, -1]])  # mu1 - mu4
        d = np.array([0.0])
        result = anova1_test_linear_combs(self.data, 0.05, C, d, method=CorrectionMethod.BONFERRONI)
        assert result.reject[0]

    def test_do_not_reject_similar_groups(self):
        data = [np.array([5, 5, 5, 5]), np.array([5, 5, 5, 5])]
        C = np.array([[1, -1]])
        d = np.array([0.0])
        result = anova1_test_linear_combs(data, 0.05, C, d, method=CorrectionMethod.BONFERRONI)
        assert not result.reject[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
