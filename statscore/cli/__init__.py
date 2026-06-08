"""Interactive CLI for the statscore statistical toolbox."""

from __future__ import annotations

import argparse
import sys

from statscore.cli._anova import _run_one_way_anova, _run_two_way_anova
from statscore.cli._regression import (
    _run_bayes_known_var,
    _run_bayes_unknown_var,
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
    "1": _run_one_way_anova,
    "2": _run_two_way_anova,
    "3": _run_z_test,
    "4": _run_one_sample_t_test,
    "5": _run_two_sample_t_test,
    "6": _run_paired_t_test,
    "7": _run_chi2_test,
    "8": _run_f_test,
    "9": _run_simple_regression,
    "10": _run_regression_diagnostics,
    "11": _run_normality_check,
    "12": _run_levene_check,
    "13": _run_mean_ci,
    "14": _run_bayes_known_var,
    "15": _run_bayes_unknown_var,
}


def _print_menu() -> None:
    print("=" * 60)
    print("  statscore — Statistical Toolbox")
    print("  Type a number to select a method, or 'q' to quit.")
    print("=" * 60)
    print("  [1]  One-Way ANOVA")
    print("  [2]  Two-Way ANOVA")
    print("  [3]  Z-test for Mean (sigma known)")
    print("  [4]  One-Sample t-test")
    print("  [5]  Two-Sample t-test")
    print("  [6]  Paired t-test")
    print("  [7]  Chi-squared Test for Variance")
    print("  [8]  F-test for Variances")
    print("  [9]  Simple Linear Regression")
    print("  [10] Regression Diagnostics (Leverage / Cook's D)")
    print("  [11] Normality Check (Shapiro-Wilk)")
    print("  [12] Variance Homogeneity (Levene)")
    print("  [13] Confidence Interval for Mean")
    print("  [14] Bayesian Inference (Normal, known variance)")
    print("  [15] Bayesian Inference (Normal, unknown variance)")
    print("=" * 60)


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
