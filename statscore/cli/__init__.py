"""Interactive CLI for the statscore statistical toolbox."""

from __future__ import annotations

import argparse
import sys

from statscore.cli._anova import (
    _run_anova_multiple_comparisons,
    _run_one_way_anova,
    _run_two_way_anova,
)
from statscore.cli._regression import (
    _run_bayes_beta_binomial,
    _run_bayes_gamma_poisson,
    _run_bayes_known_var,
    _run_bayes_unknown_var,
    _run_mcmc_normal,
    _run_mcmc_regression,
    _run_multiple_regression,
    _run_regression_diagnostics,
    _run_simple_regression,
)
from statscore.cli._testing import (
    _run_chi2_test,
    _run_f_test,
    _run_levene_check,
    _run_mean_ci,
    _run_normality_check,
    _run_one_sample_t_test,
    _run_paired_t_test,
    _run_two_sample_t_test,
    _run_z_test,
)

_MENU_HANDLERS = {
    # ── ANOVA ──────────────────────────────────────────────────────────────
    "1":  _run_one_way_anova,
    "2":  _run_two_way_anova,
    "3":  _run_anova_multiple_comparisons,
    # ── Significance Tests ─────────────────────────────────────────────────
    "4":  _run_z_test,
    "5":  _run_one_sample_t_test,
    "6":  _run_two_sample_t_test,
    "7":  _run_paired_t_test,
    "8":  _run_chi2_test,
    "9":  _run_f_test,
    # ── Regression ────────────────────────────────────────────────────────
    "10": _run_simple_regression,
    "11": _run_multiple_regression,
    "12": _run_regression_diagnostics,
    # ── Diagnostics ───────────────────────────────────────────────────────
    "13": _run_normality_check,
    "14": _run_levene_check,
    "15": _run_mean_ci,
    # ── Bayesian (Conjugate) ───────────────────────────────────────────────
    "16": _run_bayes_known_var,
    "17": _run_bayes_unknown_var,
    "18": _run_bayes_beta_binomial,
    "19": _run_bayes_gamma_poisson,
    # ── Bayesian (MCMC) ────────────────────────────────────────────────────
    "20": _run_mcmc_normal,
    "21": _run_mcmc_regression,
}


def _print_menu() -> None:
    print("=" * 64)
    print("  statscore — Statistical Toolbox  (v0.0.3)")
    print("  Type a number to select a method, or 'q' to quit.")
    print("=" * 64)
    print("  ── ANOVA ──────────────────────────────────────────────")
    print("  [1]  One-Way ANOVA (F-test, ANOVA table)")
    print("  [2]  Two-Way ANOVA (Factor A / B / Interaction)")
    print("  [3]  Multiple Comparisons (Bonferroni/Scheffe/Tukey/Sidak)")
    print("  ── Significance Tests ─────────────────────────────────")
    print("  [4]  Z-test for Mean (sigma known)")
    print("  [5]  One-Sample t-test")
    print("  [6]  Two-Sample t-test (pooled or Welch)")
    print("  [7]  Paired t-test")
    print("  [8]  Chi-squared Test for Variance")
    print("  [9]  F-test for Variances")
    print("  ── Regression ─────────────────────────────────────────")
    print("  [10] Simple Linear Regression")
    print("  [11] Multiple Linear Regression (full inference suite)")
    print("  [12] Regression Diagnostics (Leverage / Cook's D)")
    print("  ── Diagnostics ────────────────────────────────────────")
    print("  [13] Normality Check (Shapiro-Wilk + Q-Q plot)")
    print("  [14] Variance Homogeneity (Levene)")
    print("  [15] Confidence Interval for Mean")
    print("  ── Bayesian Inference — Conjugate Priors ──────────────")
    print("  [16] Bayesian Inference — Normal (known variance)")
    print("  [17] Bayesian Inference — Normal-Gamma (unknown variance)")
    print("  [18] Bayesian Inference — Beta-Binomial (success prob.)")
    print("  [19] Bayesian Inference — Gamma-Poisson (count rate)")
    print("  ── Bayesian Inference — MCMC ──────────────────────────")
    print("  [20] Bayesian MCMC — Normal mean & variance")
    print("  [21] Bayesian MCMC — Linear Regression")
    print("=" * 64)


def main() -> None:
    """Entry point for the statscore CLI."""
    parser = argparse.ArgumentParser(
        prog="statscore",
        description="statscore — Interactive Statistical Toolbox",
    )
    parser.parse_args()

    _print_menu()
    while True:
        try:
            choice = input("Select> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            sys.exit(0)

        if choice.lower() == "q":
            print("Goodbye.")
            sys.exit(0)

        handler = _MENU_HANDLERS.get(choice)
        if handler is None:
            print("  Invalid selection. Try again.")
            continue

        try:
            handler()
        except ValueError as e:
            print(f"  Error: {e}")
        except KeyboardInterrupt:
            print("\n  Interrupted. Returning to menu.")
        print()
