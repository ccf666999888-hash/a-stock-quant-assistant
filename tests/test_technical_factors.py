from __future__ import annotations

import pandas as pd

from factors.technical import calculate_technical_factors


def test_ma_calculation_is_correct(sample_prices):
    factors = calculate_technical_factors(sample_prices)
    assert factors["ma5"].iloc[4] == sample_prices["close"].iloc[:5].mean()


def test_macd_is_not_empty_and_rsi_is_bounded(sample_prices):
    factors = calculate_technical_factors(sample_prices)
    assert factors["macd_hist"].notna().all()
    assert factors["rsi6"].between(0, 100).all()


def test_empty_price_frame_does_not_crash():
    assert calculate_technical_factors(pd.DataFrame()).empty


def test_multiple_stocks_keep_their_own_factor_identity(sample_prices):
    first = sample_prices.copy()
    second = sample_prices.copy()
    second["ts_code"] = "600000.SH"
    second["close"] = second["close"] * 10
    second["high"] = second["high"] * 10
    second["low"] = second["low"] * 10
    factors = calculate_technical_factors(pd.concat([first, second], ignore_index=True))
    latest = factors.groupby("ts_code").tail(1).set_index("ts_code")
    assert latest.loc["600000.SH", "ma20"] > latest.loc["000001.SZ", "ma20"] * 9
