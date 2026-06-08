"""Tests for the CLI module."""

import subprocess
import sys

import pytest


class TestCLIHelp:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "statscore", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Statistical Toolbox" in result.stdout

    def test_quit_immediately(self):
        result = subprocess.run(
            [sys.executable, "-m", "statscore"],
            input="q\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "statscore" in result.stdout
        assert "Goodbye" in result.stdout

    def test_invalid_selection(self):
        result = subprocess.run(
            [sys.executable, "-m", "statscore"],
            input="99\nq\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Invalid selection" in result.stdout

    def test_menu_shows_all_options(self):
        result = subprocess.run(
            [sys.executable, "-m", "statscore"],
            input="q\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "[1]" in result.stdout
        assert "[11]" in result.stdout
        assert "One-Way ANOVA" in result.stdout
        assert "Bayesian Inference" in result.stdout


class TestCLIImport:
    def test_main_importable(self):
        from statscore.cli import main
        assert callable(main)

    def test_module_runnable(self):
        result = subprocess.run(
            [sys.executable, "-c", "from statscore.__main__ import main"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
