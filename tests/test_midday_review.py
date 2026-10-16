from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts")
)

import generate_midday_html as midday  # noqa: E402
from generate_midday_html import (  # noqa: E402
    market_summary,
    normalize_index_minutes,
    provider_time_delta_minutes,
    quality_assessment,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _index_raw() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2026-08-20", "2026-09-02"):
        if day.weekday() >= 5:
            continue
        for hour, minute in [(10, 30), (11, 30), (14, 0), (15, 0)]:
            rows.append(
                {
                    "day": day.replace(hour=hour, minute=minute),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1_000,
                    "amount": 2_000,
                }
            )
    for hour, minute in [(10, 30), (11, 30), (14, 0)]:
        rows.append(
            {
                "day": datetime(2026, 9, 3, hour, minute),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000,
                "amount": 2_000,
            }
        )
    return pd.DataFrame(rows)


def test_midday_bar_filter_drops_future_labeled_bar() -> None:
    result, dropped = normalize_index_minutes(
        _index_raw(), "20260903"
    )

    today = result[result["trade_date"] == "20260903"]
    assert dropped == 1
    assert today["trade_time"].dt.strftime("%H:%M").tolist() == [
        "10:30",
        "11:30",
    ]


def test_provider_time_delta_uses_shanghai_clock() -> None:
    captured = datetime(
        2026, 9, 3, 11, 42, tzinfo=SHANGHAI
    )

    assert provider_time_delta_minutes("11:37:00", captured) == 5
    assert provider_time_delta_minutes("13:37:00", captured) == 115


def test_quality_blocks_selection_when_provider_time_is_ahead() -> None:
    codes = [f"{index:06d}.SZ" for index in range(5_100)]
    pct = pd.Series(range(5_100), dtype=float) / 1_000
    spot = pd.DataFrame({"ts_code": codes, "pct_chg": pct})
    flow = pd.DataFrame({"ts_code": codes, "pct_chg": pct})
    anchor = pd.DataFrame(
        {"ts_code": codes, "anchor_pct_chg": pct}
    )
    bars, _ = normalize_index_minutes(_index_raw(), "20260903")
    live = {
        "spot": spot,
        "anchor": anchor,
        "provider_time": "13:37:00",
        "stock_flow": flow,
        "index_bars": bars,
        "future_rows": 1,
        "errors": [],
    }
    captured = datetime(
        2026, 9, 3, 11, 42, tzinfo=SHANGHAI
    )

    result = quality_assessment(live, captured, "20260903")

    assert result["grade"] == "C"
    assert result["can_select"] is False
    assert any(
        "供应商时间与本地相差" in reason
        for reason in result["reasons"]
    )


def test_market_breadth_below_35_is_midday_retreat() -> None:
    stocks = pd.DataFrame(
        {
            "pct_chg": [-1.0] * 70 + [1.0] * 30,
            "amount": [100_000_000.0] * 100,
        }
    )
    flow = pd.DataFrame(
        {
            "net_inflow_amount": [-1_000_000.0] * 100,
            "turnover_amount": [100_000_000.0] * 100,
        }
    )
    events = pd.DataFrame(
        {"event_type": ["limit_up"] * 20 + ["broken_board"] * 5}
    )

    result = market_summary(
        stocks, flow, events, pd.DataFrame()
    )

    assert result["breadth"] == 30
    assert result["regime"] == "上午退潮"


def test_snapshot_persistence_uses_separate_midday_db(
    tmp_path: Path,
    monkeypatch,
) -> None:
    midday_db = tmp_path / "midday.duckdb"
    monkeypatch.setattr(midday, "MIDDAY_DB", midday_db)
    market = {
        "equal_return": 0.2,
        "breadth": 55.0,
        "amount_yi": 5000.0,
        "flow_yi": 12.0,
        "flow_ratio": 1.0,
        "limit_up": 30,
        "broken": 10,
        "limit_down": 2,
        "broken_rate": 25.0,
        "regime": "震荡偏强",
        "technical_state": "强势放缓",
    }
    stocks = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "last": 12.0,
                "pct_chg": 1.0,
                "amount": 100_000_000.0,
                "volume": 1_000.0,
                "amplitude": 2.0,
                "open": 11.9,
                "high": 12.1,
                "low": 11.8,
                "pre_close": 11.88,
            }
        ]
    )
    sectors = pd.DataFrame(
        [
            {
                "sector": "银行",
                "stocks": 42,
                "return_pct": 1.0,
                "breadth_pct": 60.0,
                "amount_yi": 300.0,
                "relative_market": 0.8,
                "rel3": 1.0,
                "rel5": 1.5,
                "flow_yi": 3.0,
                "flow_ratio": 1.2,
                "limit_up": 0,
                "broken_board": 0,
                "limit_down": 0,
                "score": 70.0,
                "state": "上午强势延续",
            }
        ]
    )
    bars, _ = normalize_index_minutes(_index_raw(), "20260903")
    quality = {
        "grade": "A",
        "can_select": True,
        "reasons": [],
    }
    captured = datetime(
        2026, 9, 3, 11, 42, tzinfo=SHANGHAI
    )

    midday.persist_snapshot(
        "20260903_114200",
        "20260903",
        captured,
        "11:40:00",
        quality,
        market,
        stocks,
        sectors,
        pd.DataFrame(),
        bars,
        tmp_path / "report.html",
    )

    import duckdb

    with duckdb.connect(str(midday_db), read_only=True) as con:
        assert con.execute(
            "select count(*) from midday_run"
        ).fetchone()[0] == 1
        assert con.execute(
            "select count(*) from midday_stock_snapshot"
        ).fetchone()[0] == 1
