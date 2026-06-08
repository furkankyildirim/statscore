"""One-way ANOVA: partitioning TSS and testing equality of means."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from stats_toolbox.utils.distributions import f_critical, f_pvalue
from stats_toolbox.utils.validation import validate_data_groups


@dataclass
class ANOVA1PartitionResult:
    """Result of one-way ANOVA sum-of-squares partitioning."""

    SS_total: float
    SS_within: float
    SS_between: float
    group_means: np.ndarray
    grand_mean: float
    group_sizes: np.ndarray


@dataclass
class ANOVA1TestResult:
    """Result of one-way ANOVA F-test for equality of means."""

    SS_total: float
    SS_within: float
    SS_between: float
    df_between: int
    df_within: int
    df_total: int
    MS_between: float
    MS_within: float
    F_statistic: float
    F_critical: float
    p_value: float
    reject_H0: bool
    alpha: float


def ANOVA1_partition_TSS(data: Sequence[np.ndarray]) -> ANOVA1PartitionResult:
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
    SS_between: float = float(sum(
        ni * (xi_bar - grand_mean) ** 2
        for ni, xi_bar in zip(group_sizes, group_means)
    ))
    SS_total: float = SS_within + SS_between

    return ANOVA1PartitionResult(
        SS_total=SS_total,
        SS_within=SS_within,
        SS_between=SS_between,
        group_means=group_means,
        grand_mean=grand_mean,
        group_sizes=group_sizes,
    )


def ANOVA1_test_equality(data: Sequence[np.ndarray], alpha: float = 0.05) -> ANOVA1TestResult:
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
    partition: ANOVA1PartitionResult = ANOVA1_partition_TSS(data)

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
