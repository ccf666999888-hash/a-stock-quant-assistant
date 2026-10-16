from __future__ import annotations

import pandas as pd

from factors.risk import calculate_risk_factors
from strategies.breakout import breakout_signal


def test_breakout_model_runs():
    frame = pd.DataFrame([{
        "close": 12, "ma5": 11, "ma10": 10.8, "ma20": 10.5,
        "macd_hist": 0.2, "macd_hist_prev": 0.1, "kdj_k": 60, "kdj_d": 50,
        "volume_ratio": 1.6, "turnover_rate": 6, "close_position": 0.9,
        "new_high_5d": True, "new_high_10d": True, "stage_overheated": False,
    }])
    assert bool(breakout_signal(frame).iloc[0])


def test_risk_model_identifies_long_upper_shadow():
    frame = pd.DataFrame([{
        "ts_code": "000001.SZ", "trade_date": "20260612",
        "open": 10, "close": 10.1, "low": 9.9, "high": 12,
        "close_position": 0.1,
    }])
    risk = calculate_risk_factors(frame)
    assert "long_upper_shadow" in risk["risk_reasons"].iloc[0]

