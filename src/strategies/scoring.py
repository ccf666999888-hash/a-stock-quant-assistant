from __future__ import annotations

import pandas as pd

from factors.risk import calculate_risk_factors
from strategies.breakout import breakout_signal
from strategies.oversold_rebound import oversold_rebound_signal
from strategies.platform_breakout import platform_breakout_signal
from strategies.pullback import pullback_signal


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.get(column, pd.Series(False, index=frame.index)).fillna(False).astype(bool)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce").fillna(default)


def _join_true(labels: dict[str, pd.Series], index) -> list[str]:
    return [
        ";".join(name for name, values in labels.items() if bool(values.loc[row_index]))
        for row_index in index
    ]


def score_universe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result["macd_hist_prev"] = result.groupby("ts_code")["macd_hist"].shift(1) if "macd_hist" in result else 0

    technical = (
        (_num(result, "close") > _num(result, "ma5")) * 20
        + (_num(result, "close") > _num(result, "ma10")) * 10
        + (_num(result, "close") > _num(result, "ma20")) * 10
        + (_num(result, "ma5") >= _num(result, "ma10")) * 10
        + ((_num(result, "macd_hist") > 0) | _bool(result, "technical_strengthening")) * 20
        + ((_num(result, "kdj_k") >= _num(result, "kdj_d")) & (_num(result, "kdj_k") < 80)) * 15
        + _num(result, "rsi6").between(35, 70) * 10
        + (_num(result, "close") >= _num(result, "boll_mid")) * 5
    ).astype(float).clip(0, 100)

    market = (
        _num(result, "volume_ratio").between(1.3, 2.5) * 20
        + _num(result, "turnover_rate").between(3, 12) * 15
        + (_num(result, "amount") > _num(result, "amount_ma5")) * 10
        + ((_num(result, "main_net_inflow") > 0) | (_num(result, "large_net_inflow") > 0)) * 20
        + (_num(result, "close_position", 0.5) > 0.7) * 20
        + _bool(result, "stronger_than_market") * 10
        + _bool(result, "stronger_than_sector") * 5
    ).astype(float).clip(0, 100)

    stage = (
        _bool(result, "platform_breakout") * 30
        + (_bool(result, "new_high_5d") | _bool(result, "new_high_10d") | _bool(result, "new_high_20d")) * 25
        + (~_bool(result, "stage_overheated")) * 20
        + (_num(result, "turnover_5d") >= 10) * 15
        + _bool(result, "stage_volume_expansion") * 10
    ).astype(float).clip(0, 100)

    risk = calculate_risk_factors(result).set_axis(result.index)
    model_flags = {
        "breakout": breakout_signal(result),
        "pullback": pullback_signal(result),
        "oversold_rebound": oversold_rebound_signal(result),
        "platform_breakout": platform_breakout_signal(result),
    }
    result["technical_score"] = technical
    result["market_score"] = market
    result["stage_score"] = stage
    result["risk_deduction"] = risk["risk_score"]
    result["total_score"] = (technical * 0.4 + market * 0.4 + stage * 0.2 - risk["risk_score"]).clip(0, 100)
    result["models"] = _join_true(model_flags, result.index)
    result["risk_reasons"] = risk["risk_reasons"]
    result["reasons"] = _join_true(
        {
            "技术面转强": technical >= 60,
            "量价配合": market >= 60,
            "阶段形态积极": stage >= 60,
        },
        result.index,
    )
    positive = pd.Series(False, index=result.index)
    for values in model_flags.values():
        positive = positive | values
    result["classification"] = "回避"
    cautious = (result["total_score"] >= 60) & ~risk["severe_risk"]
    priority = (result["total_score"] >= 75) & positive & ~risk["severe_risk"]
    result.loc[cautious, "classification"] = "谨慎观察"
    result.loc[priority, "classification"] = "优先观察"
    columns = [
        "ts_code", "trade_date", "name", "models", "technical_score", "market_score",
        "stage_score", "risk_deduction", "total_score", "classification",
        "reasons", "risk_reasons",
    ]
    return result[[column for column in columns if column in result]].sort_values(
        "total_score", ascending=False
    ).reset_index(drop=True)

