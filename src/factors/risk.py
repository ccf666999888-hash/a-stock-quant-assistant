from __future__ import annotations

import numpy as np
import pandas as pd


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce").fillna(default)


def calculate_risk_factors(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    close = _num(result, "close")
    high = _num(result, "high", close)
    low = _num(result, "low", close)
    open_ = _num(result, "open", close)
    spread = (high - low).replace(0, pd.NA)
    upper_shadow_ratio = ((high - pd.concat([open_, close], axis=1).max(axis=1)) / spread).fillna(0)

    conditions = {
        "long_upper_shadow": upper_shadow_ratio > 0.45,
        "huge_volume_stall": (_num(result, "volume_ratio") > 3) & (_num(result, "pct_chg").abs() < 3),
        "weak_close": _num(result, "close_position", 0.5) < 0.5,
        "kdj_high_dead_cross": (_num(result, "kdj_k") < _num(result, "kdj_d")) & (_num(result, "kdj_d") > 80),
        "rsi_overbought": _num(result, "rsi6") > 80,
        "failed_breakout": result.get("is_platform", pd.Series(False, index=result.index)).fillna(False) & (close < _num(result, "ma20", close)),
        "main_outflow": _num(result, "main_net_inflow") < 0,
        "five_day_overheated": _num(result, "return_5d") > 20,
        "volume_break_ma20": (_num(result, "volume_ratio") > 1.3) & (close < _num(result, "ma20", close)),
    }
    deductions = {
        "long_upper_shadow": 10,
        "huge_volume_stall": 15,
        "weak_close": 10,
        "kdj_high_dead_cross": 8,
        "rsi_overbought": 8,
        "failed_breakout": 15,
        "main_outflow": 12,
        "five_day_overheated": 8,
        "volume_break_ma20": 20,
    }
    risk_scores = pd.Series(0.0, index=result.index)
    reason_rows = pd.Series("", index=result.index, dtype="string")
    for name, condition in conditions.items():
        active = condition.fillna(False).astype(bool)
        risk_scores = risk_scores + active.astype(float) * deductions[name]
        separator = np.where(active & reason_rows.ne(""), ";", "")
        addition = np.where(active, name, "")
        reason_rows = reason_rows + separator + addition
    severe_rows = (
        conditions["huge_volume_stall"]
        | conditions["failed_breakout"]
        | conditions["volume_break_ma20"]
    ).fillna(False)
    return pd.DataFrame(
        {
            "ts_code": result.get("ts_code", pd.Series("", index=result.index)),
            "trade_date": result.get("trade_date", pd.Series("", index=result.index)),
            "risk_triggered": risk_scores > 0,
            "severe_risk": severe_rows.astype(bool),
            "risk_score": risk_scores.astype(float),
            "risk_reasons": reason_rows.astype(str),
        }
    )
