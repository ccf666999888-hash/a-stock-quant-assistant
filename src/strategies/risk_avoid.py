from __future__ import annotations

import pandas as pd

from factors.risk import calculate_risk_factors


def risk_avoid_signal(frame: pd.DataFrame) -> pd.Series:
    risk = calculate_risk_factors(frame)
    if risk.empty:
        return pd.Series(False, index=frame.index)
    return risk["risk_triggered"].set_axis(frame.index)

