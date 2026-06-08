"""Tests for the data I/O module."""

import json

import numpy as np
import pandas as pd
import pytest

from statscore.io import LoadedData, load_data


class TestLoadDataCSV:
    def test_comma_separated(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
        result = load_data(str(csv_file))
        assert isinstance(result, LoadedData)
        assert result.format == "csv"
        assert result.n_rows == 3
        assert result.n_cols == 3
        assert result.column_names == ["a", "b", "c"]
        assert result.path == str(csv_file)

    def test_semicolon_separated(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x;y\n1.1;2.2\n3.3;4.4\n")
        result = load_data(str(csv_file))
        assert result.n_rows == 2
        assert result.column_names == ["x", "y"]

    def test_kwargs_passthrough(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("skip\na,b\n1,2\n3,4\n")
        result = load_data(str(csv_file), skiprows=1, sep=",")
        assert result.n_rows == 2
        assert result.column_names == ["a", "b"]


class TestLoadDataTSV:
    def test_tab_separated(self, tmp_path):
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("col1\tcol2\n10\t20\n30\t40\n")
        result = load_data(str(tsv_file))
        assert result.format == "tsv"
        assert result.n_rows == 2
        assert result.n_cols == 2
        assert result.column_names == ["col1", "col2"]


class TestLoadDataJSON:
    def test_records_orientation(self, tmp_path):
        json_file = tmp_path / "data.json"
        data = [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}]
        json_file.write_text(json.dumps(data))
        result = load_data(str(json_file))
        assert result.format == "json"
        assert result.n_rows == 3
        assert result.n_cols == 2

    def test_columns_orientation(self, tmp_path):
        json_file = tmp_path / "data.json"
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        json_file.write_text(json.dumps(data))
        result = load_data(str(json_file))
        assert result.n_rows == 3
        assert result.column_names == ["a", "b"]


class TestLoadDataXLSX:
    def test_xlsx_file(self, tmp_path):
        xlsx_file = tmp_path / "data.xlsx"
        df = pd.DataFrame({"alpha": [1.0, 2.0], "beta": [3.0, 4.0]})
        df.to_excel(str(xlsx_file), index=False)
        result = load_data(str(xlsx_file))
        assert result.format == "xlsx"
        assert result.n_rows == 2
        assert result.column_names == ["alpha", "beta"]


class TestLoadDataErrors:
    def test_unsupported_extension(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_data(str(txt_file))

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/data.csv")


class TestLoadedDataFields:
    def test_dataframe_preserved(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n1.5,2.5\n3.5,4.5\n")
        result = load_data(str(csv_file))
        assert isinstance(result.df, pd.DataFrame)
        np.testing.assert_allclose(result.df["x"].values, [1.5, 3.5])
        np.testing.assert_allclose(result.df["y"].values, [2.5, 4.5])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
