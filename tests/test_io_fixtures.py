"""Tests for load_data using static fixture files in tests/fixtures/."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statscore.io import LoadedData, load_data

FIXTURES = Path(__file__).parent / "fixtures"


class TestCSVBasic:
    def test_shape(self):
        result = load_data(str(FIXTURES / "basic.csv"))
        assert result.n_rows == 5
        assert result.n_cols == 4

    def test_format(self):
        result = load_data(str(FIXTURES / "basic.csv"))
        assert result.format == "csv"

    def test_column_names(self):
        result = load_data(str(FIXTURES / "basic.csv"))
        assert result.column_names == ["subject_id", "treatment", "score", "passed"]

    def test_score_values(self):
        result = load_data(str(FIXTURES / "basic.csv"))
        expected = [72.5, 88.0, 65.3, 91.2, 78.9]
        np.testing.assert_allclose(result.df["score"].values, expected)

    def test_boolean_column(self):
        result = load_data(str(FIXTURES / "basic.csv"))
        assert result.df["passed"].dtype == bool or result.df["passed"].dtype == object

    def test_string_column(self):
        result = load_data(str(FIXTURES / "basic.csv"))
        assert list(result.df["treatment"]) == ["A", "B", "A", "B", "A"]

    def test_path_preserved(self):
        path = str(FIXTURES / "basic.csv")
        result = load_data(path)
        assert result.path == path

    def test_returns_loaded_data(self):
        result = load_data(str(FIXTURES / "basic.csv"))
        assert isinstance(result, LoadedData)
        assert isinstance(result.df, pd.DataFrame)


class TestCSVSemicolon:
    def test_shape(self):
        result = load_data(str(FIXTURES / "semicolon.csv"))
        assert result.n_rows == 3
        assert result.n_cols == 3

    def test_column_names(self):
        result = load_data(str(FIXTURES / "semicolon.csv"))
        assert result.column_names == ["name", "value", "unit"]

    def test_values(self):
        result = load_data(str(FIXTURES / "semicolon.csv"))
        np.testing.assert_allclose(result.df["value"].values, [1.23, 4.56, 7.89])

    def test_string_column(self):
        result = load_data(str(FIXTURES / "semicolon.csv"))
        assert list(result.df["name"]) == ["alpha", "beta", "gamma"]


class TestTSV:
    def test_format(self):
        result = load_data(str(FIXTURES / "groups.tsv"))
        assert result.format == "tsv"

    def test_shape(self):
        result = load_data(str(FIXTURES / "groups.tsv"))
        assert result.n_rows == 3
        assert result.n_cols == 4

    def test_column_names(self):
        result = load_data(str(FIXTURES / "groups.tsv"))
        assert result.column_names == ["group", "n", "mean", "sd"]

    def test_group_labels(self):
        result = load_data(str(FIXTURES / "groups.tsv"))
        assert list(result.df["group"]) == ["control", "low", "high"]

    def test_numeric_columns(self):
        result = load_data(str(FIXTURES / "groups.tsv"))
        np.testing.assert_allclose(result.df["mean"].values, [10.1, 12.4, 15.7])
        np.testing.assert_allclose(result.df["sd"].values, [1.2, 1.5, 1.8])


class TestJSON:
    def test_format(self):
        result = load_data(str(FIXTURES / "records.json"))
        assert result.format == "json"

    def test_shape(self):
        result = load_data(str(FIXTURES / "records.json"))
        assert result.n_rows == 4
        assert result.n_cols == 4

    def test_column_names(self):
        result = load_data(str(FIXTURES / "records.json"))
        assert set(result.column_names) == {"obs", "x1", "x2", "y"}

    def test_y_values(self):
        result = load_data(str(FIXTURES / "records.json"))
        np.testing.assert_allclose(result.df["y"].values, [55.2, 62.8, 71.4, 80.0])

    def test_x_columns_sum_to_one(self):
        result = load_data(str(FIXTURES / "records.json"))
        x_sums = result.df["x1"].values + result.df["x2"].values
        np.testing.assert_allclose(x_sums, np.ones(4))


class TestXLSX:
    def test_format(self):
        result = load_data(str(FIXTURES / "measurements.xlsx"))
        assert result.format == "xlsx"

    def test_shape(self):
        result = load_data(str(FIXTURES / "measurements.xlsx"))
        assert result.n_rows == 8
        assert result.n_cols == 4

    def test_column_names(self):
        result = load_data(str(FIXTURES / "measurements.xlsx"))
        assert result.column_names == ["sample", "measurement_1", "measurement_2", "flag"]

    def test_sample_labels(self):
        result = load_data(str(FIXTURES / "measurements.xlsx"))
        assert list(result.df["sample"]) == [f"S{i:02d}" for i in range(1, 9)]

    def test_measurement_1_values(self):
        result = load_data(str(FIXTURES / "measurements.xlsx"))
        expected = [9.8, 10.2, 10.1, 9.9, 10.3, 9.7, 10.0, 10.4]
        np.testing.assert_allclose(result.df["measurement_1"].values, expected)

    def test_measurement_2_range(self):
        result = load_data(str(FIXTURES / "measurements.xlsx"))
        vals = result.df["measurement_2"].values
        assert vals.min() >= 4.5
        assert vals.max() <= 5.5


class TestFixturesExist:
    @pytest.mark.parametrize(
        "filename",
        ["basic.csv", "semicolon.csv", "groups.tsv", "records.json", "measurements.xlsx"],
    )
    def test_fixture_file_exists(self, filename):
        assert (FIXTURES / filename).exists(), f"Missing fixture: {filename}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
