"""CLI input helpers — shared data parsing utilities."""

from __future__ import annotations

import numpy as np


def _parse_data_input(prompt: str) -> np.ndarray:
    """Prompt user for data: file path, inline numbers, or semicolon-separated matrix rows.

    Accepted formats:
    - Space/comma separated numbers: "1.2 3.4 5.6"  or  "1.2, 3.4, 5.6"
    - File path ending in .csv / .tsv / .xlsx / .xls / .json
    - Semicolon-separated rows for a 2-D matrix:
        "1,0.5,0.25; 1,0.7,0.5; 1,1.0,1.0"
      Returns shape (n_rows, n_cols).  Use this when the method expects a
      design matrix X rather than a 1-D vector.
    """
    raw = input(prompt).strip()
    if not raw:
        raise ValueError("No input provided.")

    if raw.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".json")):
        from statscore.io import load_data

        loaded = load_data(raw)
        print(f"  Loaded {loaded.n_rows} rows x {loaded.n_cols} cols from '{loaded.path}'")
        print(f"  Columns: {loaded.column_names}")
        if loaded.n_cols == 1:
            col = loaded.column_names[0]
            print(f"  Using single column: '{col}'")
        else:
            col = input("  Enter column name (or 'ALL' for full matrix): ").strip()
            if col.upper() == "ALL":
                return np.asarray(loaded.df.dropna().values, dtype=float)
            if col not in loaded.column_names:
                raise ValueError(f"Column '{col}' not found.")
        return np.asarray(loaded.df[col].dropna(), dtype=float)

    # Semicolon-separated rows → 2-D matrix
    if ";" in raw:
        rows: list[list[float]] = []
        for row_str in raw.split(";"):
            row_str = row_str.strip()
            if not row_str:
                continue
            row_str = row_str.replace(",", " ")
            rows.append([float(v) for v in row_str.split()])
        if not rows:
            raise ValueError("No rows parsed from semicolon input.")
        lengths = [len(r) for r in rows]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All rows must have the same number of columns. Got lengths: {lengths}"
            )
        return np.array(rows, dtype=float)

    raw = raw.replace(",", " ")
    return np.array([float(v) for v in raw.split()])


def _parse_matrix_input(prompt: str) -> np.ndarray:
    """Prompt user for a 2-D design matrix.

    Accepts the same formats as _parse_data_input, but always returns a 2-D
    array.  If the user provides 1-D data (n,), it is treated as a (n, 1) column.

    Interactive row-by-row entry is also offered when the user types 'interactive'.
    """
    print(prompt)
    print("  Options:")
    print("    (a) Inline rows separated by semicolons, e.g.: '1,0.5; 1,0.7; 1,1.0'")
    print("    (b) File path (.csv / .tsv / .xlsx / .json)  — type 'ALL' for full matrix")
    print("    (c) Type 'interactive' to enter row by row")
    raw = input("  Input: ").strip()

    if raw.lower() == "interactive":
        rows2: list[list[float]] = []
        print("  Enter each row as space/comma-separated values. Empty line to finish.")
        while True:
            row_raw = input(f"  Row {len(rows2) + 1}: ").strip()
            if not row_raw:
                break
            row_raw = row_raw.replace(",", " ")
            rows2.append([float(v) for v in row_raw.split()])
        if not rows2:
            raise ValueError("No rows entered.")
        lengths2 = [len(r) for r in rows2]
        if len(set(lengths2)) != 1:
            raise ValueError(f"All rows must have the same length. Got: {lengths2}")
        return np.array(rows2, dtype=float)

    result = _parse_raw_string(raw)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    return result


def _parse_raw_string(raw: str) -> np.ndarray:
    """Parse a raw string (already stripped) into a numpy array.

    Handles: file paths, semicolon-separated rows, and flat number lists.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("No input provided.")

    if raw.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".json")):
        from statscore.io import load_data

        loaded = load_data(raw)
        col_prompt = input(
            f"  Columns: {loaded.column_names}. Enter column name or 'ALL': "
        ).strip()
        if col_prompt.upper() == "ALL":
            return np.asarray(loaded.df.dropna().values, dtype=float)
        if col_prompt not in loaded.column_names:
            raise ValueError(f"Column '{col_prompt}' not found.")
        return np.asarray(loaded.df[col_prompt].dropna(), dtype=float)

    if ";" in raw:
        rows: list[list[float]] = []
        for row_str in raw.split(";"):
            row_str = row_str.strip().replace(",", " ")
            if row_str:
                rows.append([float(v) for v in row_str.split()])
        if not rows:
            raise ValueError("No rows parsed.")
        return np.array(rows, dtype=float)

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
        arr = _parse_raw_string(raw)
        if arr.ndim != 1:
            raise ValueError("Each group must be a 1-D array of numbers.")
        groups.append(arr)
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups.")
    return groups
