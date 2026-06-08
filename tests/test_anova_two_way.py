"""Tests for two-way ANOVA functions."""

import numpy as np
import pytest

from statscore.methods.anova.two_way import anova2_mle, anova2_partition_tss, anova2_test_equality
from statscore.utils.enums import TwoWayTestFactor


class TestANOVA2PartitionTSS:
    def setup_method(self):
        # Detergent x temperature experiment data
        # Factor A: detergent (Super, Best) -> I=2
        # Factor B: temperature (Cold, Warm, Hot) -> J=3
        # K=4 replicates
        self.data = np.array(
            [
                [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],  # Super
                [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],  # Best
            ],
            dtype=float,
        )

    def test_partition_identity(self):
        result = anova2_partition_tss(self.data)
        total = result.SS_A + result.SS_B + result.SS_AB + result.SS_E
        assert np.isclose(result.SS_total, total, rtol=1e-10)

    def test_known_values(self):
        # Simple 2x2x2 dataset for hand verification
        data = np.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
            dtype=float,
        )
        result = anova2_partition_tss(data)
        # grand mean = 4.5
        # X_bar_i: [2.5, 6.5], X_bar_j: [3.5, 5.5]
        I, J, K = 2, 2, 2
        assert np.isclose(result.SS_A, J * K * ((2.5 - 4.5) ** 2 + (6.5 - 4.5) ** 2))
        assert np.isclose(result.SS_B, I * K * ((3.5 - 4.5) ** 2 + (5.5 - 4.5) ** 2))

    def test_nonnegative(self):
        result = anova2_partition_tss(self.data)
        assert result.SS_A >= 0
        assert result.SS_B >= 0
        assert result.SS_AB >= 0
        assert result.SS_E >= 0


class TestANOVA2MLE:
    def test_constraints(self):
        """MLE estimates must satisfy sum constraints."""
        data = np.array(
            [
                [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
                [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
            ],
            dtype=float,
        )
        result = anova2_mle(data)
        I, J, K = data.shape
        # sum(a_i) = 0
        assert np.isclose(result.a.sum(), 0.0, atol=1e-10)
        # sum(b_j) = 0
        assert np.isclose(result.b.sum(), 0.0, atol=1e-10)
        # sum_j(delta_ij) = 0 for each i
        for i in range(I):
            assert np.isclose(result.delta[i].sum(), 0.0, atol=1e-10)
        # sum_i(delta_ij) = 0 for each j
        for j in range(J):
            assert np.isclose(result.delta[:, j].sum(), 0.0, atol=1e-10)

    def test_reconstruction(self):
        """mu + a_i + b_j + delta_ij = X_bar_ij."""
        data = np.array(
            [
                [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
                [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
            ],
            dtype=float,
        )
        result = anova2_mle(data)
        X_bar_ij = data.mean(axis=2)
        I, J, _ = data.shape
        for i in range(I):
            for j in range(J):
                reconstructed = result.mu + result.a[i] + result.b[j] + result.delta[i, j]
                assert np.isclose(reconstructed, X_bar_ij[i, j])


class TestANOVA2TestEquality:
    def setup_method(self):
        self.data = np.array(
            [
                [[4, 5, 6, 5], [7, 9, 8, 12], [10, 12, 11, 19]],
                [[6, 6, 4, 4], [13, 15, 12, 12], [12, 13, 10, 13]],
            ],
            dtype=float,
        )

    def test_factor_A(self):
        result = anova2_test_equality(self.data, alpha=0.05, test=TwoWayTestFactor.A)
        assert result.source == TwoWayTestFactor.A
        assert result.df == 1  # I-1 = 2-1

    def test_factor_B(self):
        result = anova2_test_equality(self.data, alpha=0.05, test=TwoWayTestFactor.B)
        assert result.source == TwoWayTestFactor.B
        assert result.df == 2  # J-1 = 3-1
        # Temperature should have significant effect
        assert result.reject_H0

    def test_interaction(self):
        result = anova2_test_equality(self.data, alpha=0.05, test=TwoWayTestFactor.AB)
        assert result.source == TwoWayTestFactor.AB
        assert result.df == 2  # (I-1)(J-1) = 1*2

    def test_invalid_test(self):
        with pytest.raises(ValueError):
            anova2_test_equality(self.data, test=TwoWayTestFactor("C"))

    def test_k1_raises(self):
        # K=1 makes df_E=0; validation must reject this before division by zero.
        data_k1 = np.ones((2, 3, 1))
        with pytest.raises(ValueError, match="K >= 2"):
            anova2_test_equality(data_k1, alpha=0.05, test=TwoWayTestFactor.A)

    def test_full_table_structure(self):
        result = anova2_test_equality(self.data, alpha=0.1, test=TwoWayTestFactor.A)
        table = result.full_table
        assert "A" in table
        assert "B" in table
        assert "AB" in table
        assert "within" in table
        assert "total" in table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
