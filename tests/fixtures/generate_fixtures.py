"""Generate static fixture files used by test_io_fixtures.py."""

import json
from pathlib import Path

import pandas as pd

FIXTURES = Path(__file__).parent


def make_csv_basic():
    df = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "treatment": ["A", "B", "A", "B", "A"],
            "score": [72.5, 88.0, 65.3, 91.2, 78.9],
            "passed": [True, True, False, True, True],
        }
    )
    df.to_csv(FIXTURES / "basic.csv", index=False)


def make_csv_semicolon():
    lines = [
        "name;value;unit",
        "alpha;1.23;kg",
        "beta;4.56;kg",
        "gamma;7.89;kg",
    ]
    (FIXTURES / "semicolon.csv").write_text("\n".join(lines))


def make_tsv():
    df = pd.DataFrame(
        {
            "group": ["control", "low", "high"],
            "n": [20, 21, 19],
            "mean": [10.1, 12.4, 15.7],
            "sd": [1.2, 1.5, 1.8],
        }
    )
    df.to_csv(FIXTURES / "groups.tsv", sep="\t", index=False)


def make_json_records():
    records = [
        {"obs": 1, "x1": 0.1, "x2": 0.9, "y": 55.2},
        {"obs": 2, "x1": 0.4, "x2": 0.6, "y": 62.8},
        {"obs": 3, "x1": 0.7, "x2": 0.3, "y": 71.4},
        {"obs": 4, "x1": 1.0, "x2": 0.0, "y": 80.0},
    ]
    (FIXTURES / "records.json").write_text(json.dumps(records, indent=2))


def make_xlsx():
    df = pd.DataFrame(
        {
            "sample": [f"S{i:02d}" for i in range(1, 9)],
            "measurement_1": [9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4],
            "measurement_2": [5.1, 5.3, 4.9, 5.2, 5.0, 5.4, 5.1, 4.8],
            "flag": [True, False, False, True, True, False, True, False],
        }
    )
    df.to_excel(FIXTURES / "measurements.xlsx", index=False)


if __name__ == "__main__":
    FIXTURES.mkdir(exist_ok=True)
    make_csv_basic()
    make_csv_semicolon()
    make_tsv()
    make_json_records()
    make_xlsx()
    print("Fixtures generated in", FIXTURES)
