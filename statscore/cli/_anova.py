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
