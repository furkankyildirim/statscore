"""CLI input helpers — shared data parsing utilities."""

from __future__ import annotations

import numpy as np


def _parse_data_input(prompt: str) -> np.ndarray:
    """Prompt user for data: file path or inline numbers."""
    raw = input(prompt).strip()
    if not raw:
        raise ValueError("No input provided.")

    if raw.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".json")):
        from statscore.io import load_data

        loaded = load_data(raw)
        print(f"  Loaded {loaded.n_rows} rows x {loaded.n_cols} cols from '{loaded.path}'")
        print(f"  Columns: {loaded.column_names}")
        col = input("  Enter column name: ").strip()
        if col not in loaded.column_names:
            raise ValueError(f"Column '{col}' not found.")
        return np.asarray(loaded.df[col].dropna(), dtype=float)

    raw = raw.replace(",", " ")
    return np.array([float(v) for v in raw.split()])


def _parse_groups_input(prompt: str) -> list[np.ndarray]:
    """Prompt user for multiple groups of data."""
    groups: list[np.ndarray] = []
    print(prompt)
    while True:
        raw = input(f"  Group {len(groups) + 1} (empty to finish): ").strip()
        if not raw:
            break
        if raw.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".json")):
            from statscore.io import load_data

            loaded = load_data(raw)
            print(f"    Loaded from '{loaded.path}'. Columns: {loaded.column_names}")
            col = input("    Enter column name: ").strip()
            if col not in loaded.column_names:
                raise ValueError(f"Column '{col}' not found.")
            groups.append(np.asarray(loaded.df[col].dropna(), dtype=float))
        else:
            raw = raw.replace(",", " ")
            groups.append(np.array([float(v) for v in raw.split()]))
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups.")
    return groups
