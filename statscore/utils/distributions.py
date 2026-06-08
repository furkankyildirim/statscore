"""Statistical distribution utilities: critical values and p-values."""

from scipy import stats

from statscore.utils.enums import AlternativeHypothesis


def f_critical(alpha: float, df1: int, df2: int) -> float:
    """Upper alpha critical value of the F distribution."""
    return float(stats.f.ppf(1 - alpha, df1, df2))


def f_critical_lower(alpha: float, df1: int, df2: int) -> float:
    """Lower alpha critical value of the F distribution."""
    return float(stats.f.ppf(alpha, df1, df2))


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


def f_pvalue_lower(F: float, df1: int, df2: int) -> float:
    """P-value for an observed F statistic (lower tail)."""
    return float(stats.f.cdf(F, df1, df2))


def t_cdf(t_stat: float, df: float) -> float:
    """CDF of the t distribution at t_stat with df degrees of freedom."""
    return float(stats.t.cdf(t_stat, df))


def t_pvalue(t_stat: float, df: int) -> float:
    """Two-sided p-value for an observed t statistic."""
    return float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)))


def t_pvalue_one_sided(t_stat: float, df: int, alternative: AlternativeHypothesis) -> float:
    """One-sided p-value for an observed t statistic."""
    if alternative is AlternativeHypothesis.GREATER:
        return float(1.0 - stats.t.cdf(t_stat, df))
    return float(stats.t.cdf(t_stat, df))


def z_pvalue(z: float, alternative: AlternativeHypothesis) -> float:
    """P-value for an observed Z statistic under N(0,1)."""
    if alternative is AlternativeHypothesis.TWO_SIDED:
        return float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    elif alternative is AlternativeHypothesis.GREATER:
        return float(1.0 - stats.norm.cdf(z))
    return float(stats.norm.cdf(z))


def chi2_pvalue(chi2_stat: float, df: int, alternative: AlternativeHypothesis) -> float:
    """P-value for an observed chi-squared statistic."""
    if alternative is AlternativeHypothesis.TWO_SIDED:
        p_lower = float(stats.chi2.cdf(chi2_stat, df))
        p_upper = float(1.0 - stats.chi2.cdf(chi2_stat, df))
        return float(2.0 * min(p_lower, p_upper))
    elif alternative is AlternativeHypothesis.GREATER:
        return float(1.0 - stats.chi2.cdf(chi2_stat, df))
    return float(stats.chi2.cdf(chi2_stat, df))


def norm_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF)."""
    return float(stats.norm.ppf(p))


def t_ppf(p: float, df: float) -> float:
    """Student's t quantile (inverse CDF)."""
    return float(stats.t.ppf(p, df))


def chi2_ppf(p: float, df: float) -> float:
    """Chi-squared quantile (inverse CDF, lower tail)."""
    return float(stats.chi2.ppf(p, df))
