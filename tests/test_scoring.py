from __future__ import annotations

import pandas as pd

from strategies.scoring import score_universe


def test_total_score_is_bounded():
    frame = pd.DataFrame([{
        "ts_code": "000001.SZ", "trade_date": "20260612", "name": "示例",
        "open": 10, "high": 10.8, "low": 9.9, "close": 10.7,
        "ma5": 10.2, "ma10": 10.0, "ma20": 9.8, "boll_mid": 10,
        "boll_upper": 10.9, "macd_hist": 0.2, "kdj_k": 60, "kdj_d": 50,
        "rsi6": 60, "volume_ratio": 1.5, "turnover_rate": 6,
        "amount": 200, "amount_ma5": 100, "main_net_inflow": 10,
        "large_net_inflow": 5, "close_position": 0.9, "stronger_than_market": True,
        "new_high_5d": True, "new_high_10d": True, "return_5d": 8,
        "return_20d": 15, "turnover_5d": 20, "stage_volume_expansion": True,
        "stage_overheated": False, "platform_breakout": False, "new_low_5d": False,
    }])
    scored = score_universe(frame)
    assert scored["total_score"].between(0, 100).all()
    assert scored["classification"].iloc[0] in {"优先观察", "谨慎观察", "回避"}

