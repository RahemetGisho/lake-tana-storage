"""Headless test: every dashboard page renders on synthetic data without raising."""
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["Overview", "Reservoir lab", "Rainfall forecasts", "Solar & wind backup", "Battery lab", "Method & limits", "About me"]


def test_all_pages_render():
    os.environ["TANA_SYNTHETIC"] = "1"
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py"), default_timeout=600).run()
    assert not at.exception, [e.value for e in at.exception]
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
