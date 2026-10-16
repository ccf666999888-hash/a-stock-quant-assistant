import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_index_60m_macd import (
    date_window as index_date_window,
    merge_minute_frames,
    minute_frame_covers_date,
)
from plot_sector_60m_macd import date_window as sector_date_window


def test_index_default_window_is_60_calendar_days_including_end_date() -> None:
    start, end = index_date_window("000001.SH", None, None, "20260828")

    assert (start, end) == ("20260630", "20260828")


def test_sector_default_window_is_60_calendar_days_including_end_date() -> None:
    start, end = sector_date_window("20260828")

    assert (start, end) == ("20260630", "20260828")


def test_explicit_start_date_overrides_index_window() -> None:
    start, end = index_date_window(
        "000001.SH",
        None,
        "20260701",
        "20260828",
        days=30,
    )

    assert (start, end) == ("20260701", "20260828")


def test_stale_primary_index_bars_require_and_merge_fallback() -> None:
    primary = pd.DataFrame(
        {
            "ts_code": ["000001.SH"],
            "trade_time": pd.to_datetime(["2026-09-02 15:00:00"]),
            "trade_date": ["20260902"],
            "freq": ["60min"],
        }
    )
    fallback = pd.DataFrame(
        {
            "ts_code": ["000001.SH", "000001.SH"],
            "trade_time": pd.to_datetime(
                ["2026-09-02 15:00:00", "2026-09-03 15:00:00"]
            ),
            "trade_date": ["20260902", "20260903"],
            "freq": ["60min", "60min"],
        }
    )

    assert not minute_frame_covers_date(primary, "20260903")
    merged = merge_minute_frames(primary, fallback)
    assert minute_frame_covers_date(merged, "20260903")
    assert len(merged) == 2
