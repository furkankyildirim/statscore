"""CLI handlers for ANOVA analyses."""

from __future__ import annotations

import numpy as np

from statscore.cli._io import _parse_groups_input


def _run_one_way_anova() -> None:
    from statscore.methods.anova import anova1_test_equality

    groups = _parse_groups_input("Enter group data (numbers or file path per group):")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")

    result = anova1_test_equality(groups, alpha=alpha)
    result.summary()

    show_plot = input("  Show box plot? (y/n) [n]: ").strip().lower()
    if show_plot == "y":
        from statscore.plots import plot_anova_groups

        fig = plot_anova_groups(groups)
        fig.savefig("anova_plot.png", dpi=150)
        print("  Plot saved to anova_plot.png")

    # Offer multiple comparisons as a follow-up
    run_mc = input("  Run multiple comparisons (simultaneous CIs/tests)? (y/n) [n]: ").strip().lower()
    if run_mc == "y":
        _run_anova_multiple_comparisons_with(groups, alpha)


def _run_anova_multiple_comparisons_with(groups: list[np.ndarray], alpha: float) -> None:
    """Run simultaneous CIs and/or tests for linear combinations of group means.

    Called both as a standalone menu option and as a follow-up after one-way ANOVA.
    """
    from statscore.methods.anova import anova1_ci_linear_combs, anova1_test_linear_combs
    from statscore.utils.enums import CorrectionMethod

    I = len(groups)
    print(f"\n  {I} groups detected.")
    print("  Enter the contrast matrix C (m × I).")
    print(f"  Each row defines one linear combination of the {I} group means.")
    print("  Use semicolons to separate rows, commas or spaces within a row.")
    print("  Example pairwise: '1,-1,0; 0,1,-1'  (for I=3)")

    from statscore.cli._io import _parse_raw_string

    C_raw = input("  C matrix: ").strip()
    C = _parse_raw_string(C_raw)
    if C.ndim == 1:
        C = C.reshape(1, -1)

    method_str = input(
        "  Method (scheffe/tukey/bonferroni/sidak/best) [best]: "
    ).strip().lower() or "best"
    method_map = {
        "scheffe": CorrectionMethod.SCHEFFE,
        "tukey": CorrectionMethod.TUKEY,
        "bonferroni": CorrectionMethod.BONFERRONI,
        "sidak": CorrectionMethod.SIDAK,
        "best": CorrectionMethod.BEST,
    }
    if method_str not in method_map:
        print(f"  Unknown method '{method_str}', using 'best'.")
        method_str = "best"
    method = method_map[method_str]

    # Simultaneous CIs
    ci_result = anova1_ci_linear_combs(groups, alpha=alpha, C=C, method=method)
    ci_result.summary()

    show_ci_plot = input("  Save CI forest plot? (y/n) [n]: ").strip().lower()
    if show_ci_plot == "y":
        fig = ci_result.plot()
        fig.savefig("simul_ci_anova.png", dpi=150)
        print("  Plot saved to simul_ci_anova.png")

    # Simultaneous tests
    run_tests = input("\n  Also run simultaneous hypothesis tests? (y/n) [n]: ").strip().lower()
    if run_tests == "y":
        m = C.shape[0]
        d_raw = input(
            f"  Hypothesized values d (space-separated, {m} values) [all zeros]: "
        ).strip()
        d = np.array([float(v) for v in d_raw.replace(",", " ").split()]) if d_raw else np.zeros(m)
        test_result = anova1_test_linear_combs(groups, alpha=alpha, C=C, d=d, method=method)
        test_result.summary()

        show_test_plot = input("  Save test statistics plot? (y/n) [n]: ").strip().lower()
        if show_test_plot == "y":
            fig = test_result.plot()
            fig.savefig("simul_tests_anova.png", dpi=150)
            print("  Plot saved to simul_tests_anova.png")


def _run_anova_multiple_comparisons() -> None:
    """Standalone menu entry for simultaneous CIs/tests after one-way ANOVA."""
    groups = _parse_groups_input("Enter group data (numbers or file path per group):")
    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    _run_anova_multiple_comparisons_with(groups, alpha)


def _run_two_way_anova() -> None:
    from statscore.methods.anova import anova2_test_equality
    from statscore.utils.enums import TwoWayTestFactor

    print("  Enter data as a 3-D array (I levels x J levels x K replicates).")
    print("  Provide data row by row. Format per cell group: space-separated numbers.")
    I = int(input("  Number of levels for Factor A (I): ").strip())
    J = int(input("  Number of levels for Factor B (J): ").strip())
    K = int(input("  Number of replicates per cell (K): ").strip())

    data = np.zeros((I, J, K))
    for i in range(I):
        for j in range(J):
            raw = input(f"  Cell ({i + 1},{j + 1}) [{K} values]: ").strip().replace(",", " ")
            vals = [float(v) for v in raw.split()]
            if len(vals) != K:
                raise ValueError(f"Expected {K} values, got {len(vals)}.")
            data[i, j, :] = vals

    alpha = float(input("  Alpha [0.05]: ").strip() or "0.05")
    factor_str = input("  Test factor (A/B/AB) [A]: ").strip().upper() or "A"
    factor = TwoWayTestFactor(factor_str)

    result = anova2_test_equality(data, alpha=alpha, test=factor)
    result.summary()
