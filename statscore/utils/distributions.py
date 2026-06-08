"""Statistical distribution utilities: critical values and p-values."""

from scipy import stats


def f_critical(alpha: float, df1: int, df2: int) -> float:
    """Upper alpha critical value of the F distribution."""
    return float(stats.f.ppf(1 - alpha, df1, df2))


def t_critical(alpha: float, df: int) -> float:
    """Upper alpha critical value of the t distribution (two-tailed: use alpha/2)."""
    return float(stats.t.ppf(1 - alpha, df))


def chi2_critical(alpha: float, df: int) -> float:
    """Upper alpha critical value of the chi-squared distribution."""
    return float(stats.chi2.ppf(1 - alpha, df))


def studentized_range_critical(alpha: float, k: int, df: int) -> float:
    """Upper alpha critical value of the Studentized range distribution q_{k,df}."""
    return float(stats.studentized_range.ppf(1 - alpha, k, df))


def f_pvalue(F: float, df1: int, df2: int) -> float:
    """P-value for an observed F statistic (upper tail)."""
    return float(1.0 - stats.f.cdf(F, df1, df2))


def t_pvalue(t_stat: float, df: int) -> float:
    """Two-sided p-value for an observed t statistic."""
    return float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)))
