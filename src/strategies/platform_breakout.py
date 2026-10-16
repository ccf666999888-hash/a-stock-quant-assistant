from __future__ import annotations

import pandas as pd


def platform_breakout_signal(frame: pd.DataFrame) -> pd.Series:
    def n(column: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)), errors="coerce").fillna(default)

    return (
        frame.get("platform_breakout", pd.Series(False, index=frame.index)).fillna(False)
        & (n("volume_ratio") >= 1.3)
        & (n("close_position", 0.5) > 0.7)
        & n("turnover_rate").between(2, 15)
    )

