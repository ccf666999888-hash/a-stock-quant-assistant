from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    count = 100
    close = np.linspace(10, 15, count) + np.sin(np.arange(count) / 4) * 0.3
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * count,
            "trade_date": pd.date_range("2025-01-01", periods=count).strftime("%Y%m%d"),
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "pre_close": np.r_[close[0], close[:-1]],
            "pct_chg": pd.Series(close).pct_change().fillna(0) * 100,
            "vol": np.linspace(1000, 1800, count),
            "amount": np.linspace(10000, 25000, count),
        }
    )

