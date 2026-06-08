"""Console-script entry point — safe to import without a Streamlit runtime."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_STREAMLIT_FLAGS = [
    "--server.headless=true",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=true",
    "--server.maxUploadSize=50",
    "--runner.magicEnabled=false",
    "--runner.fastReruns=true",
    "--browser.gatherUsageStats=false",
    "--client.showErrorDetails=false",
    "--client.toolbarMode=minimal",
    "--theme.base=light",
    "--theme.primaryColor=#1f77b4",
    "--theme.backgroundColor=#ffffff",
    "--theme.secondaryBackgroundColor=#f0f4f8",
    "--theme.textColor=#1a1a2e",
    "--theme.font=sans serif",
]


def run() -> None:
    """Launch the Streamlit UI (used by the ``statscore-ui`` console script)."""
    app_file = Path(__file__).parent / "__init__.py"
    sys.exit(subprocess.call(["streamlit", "run", str(app_file), *_STREAMLIT_FLAGS, *sys.argv[1:]]))
