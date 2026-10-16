from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_sector_60m_macd import classify_latest


def _frame(latest: dict) -> pd.DataFrame:
    previous = {
        "above_ma5": True,
        "above_ma10": True,
        "above_ma20": True,
        "close": 101.0,
        "ma5": 100.0,
        "ma10": 99.0,
        "ma20": 98.0,
        "dif": 2.1,
        "dea": 1.5,
        "macd": 1.2,
        "macd_up": False,
        "dif_up": False,
        "dif_gt_dea": True,
        "green_shrink_3": False,
        "red_shrink_3": False,
    }
    return pd.DataFrame([previous, {**previous, **latest}])


def test_escape_warning_precedes_strong_slowdown() -> None:
    frame = _frame(
        {
            "close": 102.0,
            "dif": 2.0,
            "macd": 0.8,
            "red_shrink_3": True,
        }
    )
    assert classify_latest(frame) == "逃顶预警"


def test_strong_slowdown_requires_no_escape_trigger() -> None:
    frame = _frame(
        {
            "close": 102.0,
            "dif": 2.2,
            "dif_up": True,
            "macd": 0.8,
            "red_shrink_3": True,
        }
    )
    assert classify_latest(frame) == "强势放缓"


def test_bottom_watch_is_observation_state() -> None:
    frame = _frame(
        {
            "close": 100.5,
            "above_ma5": True,
            "above_ma10": True,
            "above_ma20": False,
            "dif": -0.3,
            "dea": -0.2,
            "macd": -0.2,
            "dif_up": True,
            "dif_gt_dea": False,
            "green_shrink_3": True,
        }
    )
    assert classify_latest(frame) == "抄底观察"
