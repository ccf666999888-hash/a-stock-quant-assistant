from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.compare_groups import compare_success_control
from backtest.metrics import describe_numeric, success_rate


DEFAULT_METRICS = [
    "macd_hist", "kdj_k", "rsi6", "volume_ratio", "turnover_rate",
    "close_position", "main_net_inflow", "return_5d", "return_20d",
    "technical_score", "market_score", "stage_score", "total_score",
]


@dataclass
class EventStudyResult:
    samples: pd.DataFrame
    success_samples: pd.DataFrame
    control_samples: pd.DataFrame
    comparison: pd.DataFrame
    summary: pd.DataFrame
    success_rate: float


def run_event_study(
    features: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    threshold: float = 5.0,
    control_threshold: float = 2.0,
    ts_codes: list[str] | None = None,
    metrics: list[str] | None = None,
) -> EventStudyResult:
    if features.empty or not {"ts_code", "trade_date", "pct_chg"}.issubset(features.columns):
        empty = pd.DataFrame()
        return EventStudyResult(empty, empty, empty, empty, empty, 0.0)
    frame = features.sort_values(["ts_code", "trade_date"]).copy()
    frame["next_pct_chg"] = frame.groupby("ts_code")["pct_chg"].shift(-1)
    frame = frame[frame["trade_date"].astype(str).between(start_date, end_date)]
    if ts_codes:
        frame = frame[frame["ts_code"].isin(ts_codes)]
    frame["group_name"] = "other"
    frame.loc[frame["next_pct_chg"] >= threshold, "group_name"] = "success"
    frame.loc[frame["next_pct_chg"] < control_threshold, "group_name"] = "control"
    success = frame[frame["group_name"] == "success"].copy()
    control = frame[frame["group_name"] == "control"].copy()
    selected_metrics = metrics or DEFAULT_METRICS
    comparison = compare_success_control(success, control, selected_metrics)
    summary = describe_numeric(success, ["next_pct_chg", *selected_metrics])
    return EventStudyResult(
        samples=frame.reset_index(drop=True),
        success_samples=success.reset_index(drop=True),
        control_samples=control.reset_index(drop=True),
        comparison=comparison,
        summary=summary,
        success_rate=success_rate(frame, threshold),
    )

