"""Smoke tests for the Streamlit dashboard.

``AppTest`` executes ``app.py`` the same way the server does, so a broken
widget call or a bad DataFrame column shows up here instead of as a
traceback in the browser during a demo.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception(app):
    assert not app.exception


def test_header_metrics_are_rendered(app):
    labels = [m.label for m in app.metric]
    assert "Tonnage above cut-off" in labels
    assert {"Cu", "Pb", "Zn"} <= set(labels)


def test_default_cutoff_reports_the_published_tonnage(app):
    tonnage = next(m for m in app.metric if m.label == "Tonnage above cut-off")
    assert tonnage.value.endswith("Mt")
    assert float(tonnage.value.removesuffix(" Mt")) == pytest.approx(1.52, abs=0.08)


def test_raising_the_cutoff_shrinks_the_reported_tonnage():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()

    def tonnage(a):
        m = next(x for x in a.metric if x.label == "Tonnage above cut-off")
        return float(m.value.removesuffix(" Mt"))

    before = tonnage(at)
    # Sidebar slider 0 is the TMC cut-off.
    at.sidebar.slider[0].set_value(10.0).run()
    after = tonnage(at)

    assert after < before, "cut-off slider did not reduce tonnage"


def test_all_four_tabs_exist(app):
    assert len(app.tabs) == 4
