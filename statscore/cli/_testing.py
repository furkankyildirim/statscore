"""CLI handlers for hypothesis tests and diagnostics."""

from __future__ import annotations

from statscore.cli._io import _parse_data_input, _parse_groups_input


def _run_z_test() -> None:
    from statscore.methods.testing import z_test_mean
    from statscore.utils.enums import AlternativeHypothesis

    x = _parse_data_input("  Enter sample data: ")
    mu0 = float(input("  Hypothesized mean (mu0): ").strip())
    sigma = float(input("  Known population std (sigma): ").strip())
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = z_test_mean(x, mu0=mu0, sigma=sigma, alpha=alpha, alternative=alt)
    print(f"\n  Z-statistic: {result.z_statistic:.4f}")
    print(f"  Z-critical:  {result.z_critical:.4f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_one_sample_t_test() -> None:
    from statscore.methods.testing import t_test_mean
    from statscore.utils.enums import AlternativeHypothesis

    x = _parse_data_input("  Enter sample data: ")
    mu0 = float(input("  Hypothesized mean (mu0): ").strip())
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = t_test_mean(x, mu0, alpha=alpha, alternative=alt)
    print(f"\n  t-statistic: {result.t_statistic:.4f}")
    print(f"  t-critical:  {result.t_critical:.4f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_two_sample_t_test() -> None:
    from statscore.methods.testing import t_test_two_sample
    from statscore.utils.enums import AlternativeHypothesis

    x1 = _parse_data_input("  Enter sample 1 data: ")
    x2 = _parse_data_input("  Enter sample 2 data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)
    eq_var = input("  Assume equal variances? (y/n) [y]: ").strip().lower() != "n"

    result = t_test_two_sample(x1, x2, alpha=alpha, alternative=alt, equal_var=eq_var)
    print(f"\n  t-statistic: {result.t_statistic:.4f}")
    print(f"  t-critical:  {result.t_critical:.4f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_paired_t_test() -> None:
    from statscore.methods.testing import t_test_paired
    from statscore.utils.enums import AlternativeHypothesis

    x1 = _parse_data_input("  Enter sample 1 (before): ")
    x2 = _parse_data_input("  Enter sample 2 (after): ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = t_test_paired(x1, x2, alpha=alpha, alternative=alt)
    print(f"\n  t-statistic: {result.t_statistic:.4f}")
    print(f"  t-critical:  {result.t_critical:.4f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_f_test() -> None:
    from statscore.methods.testing import f_test_variances
    from statscore.utils.enums import AlternativeHypothesis

    x1 = _parse_data_input("  Enter sample 1 data: ")
    x2 = _parse_data_input("  Enter sample 2 data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = f_test_variances(x1, x2, alpha=alpha, alternative=alt)
    print(f"\n  F-statistic: {result.f_statistic:.4f}")
    print(f"  F-critical:  [{result.f_critical_lower:.4f}, {result.f_critical_upper:.4f}]")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  Decision:    {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_chi2_test() -> None:
    from statscore.methods.testing import chi2_test_variance
    from statscore.utils.enums import AlternativeHypothesis

    x = _parse_data_input("  Enter sample data: ")
    sigma0_sq = float(input("  Hypothesized variance (sigma0^2): ").strip())
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    alt_str = input("  Alternative (two-sided/less/greater) [two-sided]: ").strip() or "two-sided"
    alt = AlternativeHypothesis(alt_str)

    result = chi2_test_variance(x, sigma0_sq=sigma0_sq, alpha=alpha, alternative=alt)
    print(f"\n  Chi2-statistic: {result.chi2_statistic:.4f}")
    print(f"  p-value:        {result.p_value:.4f}")
    print(f"  Decision:       {'Reject H0' if result.reject_H0 else 'Fail to reject H0'}")


def _run_normality_check() -> None:
    from statscore.methods.diagnostics import shapiro_wilk_test

    x = _parse_data_input("  Enter sample data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = shapiro_wilk_test(x, alpha=alpha)
    print(f"\n  Shapiro-Wilk statistic: {result.statistic:.4f}")
    print(f"  p-value:               {result.p_value:.4f}")
    print(
        f"  Decision:              {'Reject H0 (not normal)' if result.reject_H0 else 'Fail to reject H0 (consistent with normality)'}"
    )


def _run_levene_check() -> None:
    from statscore.methods.diagnostics import levene_test

    groups = _parse_groups_input("Enter group data:")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = levene_test(groups, alpha=alpha)
    print(f"\n  Levene statistic: {result.statistic:.4f}")
    print(f"  p-value:          {result.p_value:.4f}")
    print(
        f"  Decision:         {'Reject H0 (variances differ)' if result.reject_H0 else 'Fail to reject H0 (variances are homogeneous)'}"
    )


def _run_mean_ci() -> None:
    from statscore.methods.diagnostics import mean_confidence_interval

    x = _parse_data_input("  Enter sample data: ")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    sigma_str = input("  Known population std (sigma) [leave blank for t-interval]: ").strip()
    sigma = float(sigma_str) if sigma_str else None

    result = mean_confidence_interval(x, alpha=alpha, sigma=sigma)
    result.summary()
