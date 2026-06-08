"""Tests for one-way ANOVA functions."""

import numpy as np
import pytest
from scipy import stats

from stats_toolbox.anova.one_way import ANOVA1_partition_TSS, ANOVA1_test_equality


class TestANOVA1PartitionTSS:
    def test_basic_partition_identity(self):
        """SS_total = SS_within + SS_between."""
        data = [[4, 5, 6, 5], [6, 6, 4, 4], [7, 9, 8, 12]]
        result = ANOVA1_partition_TSS(data)
        assert np.isclose(result.SS_total, result.SS_within + result.SS_between)

    def test_known_values(self):
        """Verify against hand-calculated values."""
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = ANOVA1_partition_TSS(data)
        # grand mean = 5, group means = [2, 5, 8]
        assert np.isclose(result.grand_mean, 5.0)
        np.testing.assert_array_almost_equal(result.group_means, [2, 5, 8])
        # SS_between = 3*(2-5)^2 + 3*(5-5)^2 + 3*(8-5)^2 = 27+0+27 = 54
        assert np.isclose(result.SS_between, 54.0)
        # SS_within = 2+2+2 = 6
        assert np.isclose(result.SS_within, 6.0)
        assert np.isclose(result.SS_total, 60.0)

    def test_unequal_group_sizes(self):
        """Works with unequal group sizes."""
        data = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        result = ANOVA1_partition_TSS(data)
        assert np.isclose(result.SS_total, result.SS_within + result.SS_between)
        np.testing.assert_array_equal(result.group_sizes, [2, 3, 4])

    def test_single_obs_per_group(self):
        """Groups with single observations."""
        data = [[5], [10], [15]]
        result = ANOVA1_partition_TSS(data)
        assert np.isclose(result.SS_within, 0.0)
        assert np.isclose(result.SS_total, result.SS_between)


class TestANOVA1TestEquality:
    def test_clear_difference(self):
        """Groups with very different means should reject H0."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 30)
        g2 = np.random.normal(5, 1, 30)
        g3 = np.random.normal(10, 1, 30)
        result = ANOVA1_test_equality([g1, g2, g3], alpha=0.05)
        assert result.reject_H0
        assert result.p_value < 0.05
        assert result.df_between == 2
        assert result.df_within == 87

    def test_no_difference(self):
        """Groups from same distribution should not reject (usually)."""
        np.random.seed(123)
        g1 = np.random.normal(5, 1, 50)
        g2 = np.random.normal(5, 1, 50)
        g3 = np.random.normal(5, 1, 50)
        result = ANOVA1_test_equality([g1, g2, g3], alpha=0.05)
        assert not result.reject_H0
        assert result.p_value > 0.05

    def test_matches_scipy(self):
        """Result should match scipy.stats.f_oneway."""
        data = [[23, 25, 18, 29], [31, 33, 35, 30, 28], [20, 22, 24]]
        result = ANOVA1_test_equality(data, alpha=0.05)
        f_scipy, p_scipy = stats.f_oneway(*[np.array(g) for g in data])
        assert np.isclose(result.F_statistic, f_scipy, rtol=1e-10)
        assert np.isclose(result.p_value, p_scipy, rtol=1e-10)

    def test_anova_table_consistency(self):
        """MS = SS / df, F = MS_b / MS_w."""
        data = [[4, 5, 6, 5], [6, 6, 4, 4], [10, 12, 11, 9]]
        result = ANOVA1_test_equality(data, alpha=0.05)
        assert np.isclose(result.MS_between, result.SS_between / result.df_between)
        assert np.isclose(result.MS_within, result.SS_within / result.df_within)
        assert np.isclose(result.F_statistic, result.MS_between / result.MS_within)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
