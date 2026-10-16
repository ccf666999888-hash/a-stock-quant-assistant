from __future__ import annotations

import pandas as pd


def describe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return pd.DataFrame()
    return frame[available].agg(["count", "mean", "median", "min", "max"]).T.reset_index(
        names="metric"
    )


def success_rate(frame: pd.DataFrame, threshold: float = 5.0) -> float:
    if frame.empty or "next_pct_chg" not in frame:
        return 0.0
    return float((pd.to_numeric(frame["next_pct_chg"], errors="coerce") >= threshold).mean())

