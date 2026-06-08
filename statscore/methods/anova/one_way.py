"""One-way ANOVA: partitioning TSS and testing equality of means."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from statscore.methods.anova._results import ANOVA1PartitionResult, ANOVA1TestResult
from statscore.utils.distributions import f_critical, f_pvalue
from statscore.utils.validation import validate_data_groups


def anova1_partition_tss(data: Sequence[np.ndarray]) -> ANOVA1PartitionResult:
    """Partition total sum of squares in a one-way ANOVA layout.

    Parameters
    ----------
    data : Sequence of array-like
        Data X_{i,j} for j=1,...,n_i and i=1,...,I.
        Each element is an array of observations for one group.

    Returns
    -------
    ANOVA1PartitionResult with SS_total, SS_within, SS_between.
    """
    validate_data_groups(data)

    groups: list[np.ndarray] = [np.asarray(g, dtype=float) for g in data]
    group_sizes: np.ndarray = np.array([len(g) for g in groups])
    n: int = int(group_sizes.sum())
    group_means: np.ndarray = np.array([g.mean() for g in groups])
    grand_mean: float = float(sum(g.sum() for g in groups) / n)

    SS_within: float = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    SS_between: float = float(
        sum(ni * (xi_bar - grand_mean) ** 2 for ni, xi_bar in zip(group_sizes, group_means))
    )
    SS_total: float = SS_within + SS_between

    return ANOVA1PartitionResult(
        SS_total=SS_total,
        SS_within=SS_within,
        SS_between=SS_between,
        group_means=group_means,
        grand_mean=grand_mean,
        group_sizes=group_sizes,
    )


def anova1_test_equality(data: Sequence[np.ndarray], alpha: float = 0.05) -> ANOVA1TestResult:
    """Test equality of means in a one-way ANOVA layout.

    H0: mu_1 = mu_2 = ... = mu_I
    H1: not all means are equal

    Parameters
    ----------
    data : Sequence of array-like
        Data for each group.
    alpha : float
        Significance level.

    Returns
    -------
    ANOVA1TestResult with full ANOVA table quantities, critical value,
    p-value, and decision.
    """
    partition: ANOVA1PartitionResult = anova1_partition_tss(data)

    I: int = len(data)
    n: int = int(partition.group_sizes.sum())
    df_between: int = I - 1
    df_within: int = n - I
    df_total: int = n - 1

    MS_between: float = partition.SS_between / df_between
    MS_within: float = partition.SS_within / df_within

    F_stat: float = MS_between / MS_within
    F_crit: float = f_critical(alpha, df_between, df_within)
    p_val: float = f_pvalue(F_stat, df_between, df_within)

    return ANOVA1TestResult(
        SS_total=partition.SS_total,
        SS_within=partition.SS_within,
        SS_between=partition.SS_between,
        df_between=df_between,
        df_within=df_within,
        df_total=df_total,
        MS_between=MS_between,
        MS_within=MS_within,
        F_statistic=F_stat,
        F_critical=F_crit,
        p_value=p_val,
        reject_H0=(F_stat > F_crit),
        alpha=alpha,
    )
