"""Data I/O: load tabular data from various file formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class LoadedData:
    """Result of loading tabular data from a file."""

    df: pd.DataFrame
    path: str
    format: str
    n_rows: int
    n_cols: int
    column_names: list[str]


def load_data(path: str, **kwargs: Any) -> LoadedData:
    """Load tabular data from a file and return a LoadedData result.

    Supported formats: .csv, .tsv, .xlsx, .xls, .json.
    Additional keyword arguments are passed to the underlying pandas reader.

    Parameters
    ----------
    path : str
        Path to the data file.
    **kwargs
        Passed through to the pandas reader function.

    Returns
    -------
    LoadedData
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".csv":
        fmt = "csv"
        kwargs.setdefault("sep", None)
        kwargs.setdefault("engine", "python")
        df = pd.read_csv(path, **kwargs)
    elif ext == ".tsv":
        fmt = "tsv"
        kwargs.setdefault("sep", "\t")
        df = pd.read_csv(path, **kwargs)
    elif ext in (".xlsx", ".xls"):
        fmt = "xlsx"
        df = pd.read_excel(path, **kwargs)
    elif ext == ".json":
        fmt = "json"
        df = pd.read_json(path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats: .csv, .tsv, .xlsx, .xls, .json"
        )

    return LoadedData(
        df=df,
        path=str(p),
        format=fmt,
        n_rows=len(df),
        n_cols=len(df.columns),
        column_names=list(df.columns),
    )
