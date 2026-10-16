from __future__ import annotations

import pandas as pd


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(0, index=frame.index)), errors="coerce").fillna(0)


def _calculate_group(group: pd.DataFrame) -> pd.DataFrame:
    result = group.sort_values("trade_date").copy()
    close = _series(result, "close")
    high = _series(result, "high")
    low = _series(result, "low")
    vol = _series(result, "vol")
    turnover = _series(result, "turnover_rate")
    pre_close = _series(result, "pre_close").replace(0, pd.NA)
    daily_amplitude = ((high - low) / pre_close * 100).fillna(0)

    for period in (5, 10, 20, 60):
        result[f"return_{period}d"] = (close / close.shift(period) - 1).fillna(0) * 100
    result["turnover_5d"] = turnover.rolling(5, min_periods=1).sum()
    result["turnover_20d"] = turnover.rolling(20, min_periods=1).sum()
    result["amplitude_5d"] = daily_amplitude.rolling(5, min_periods=1).mean()
    result["amplitude_20d"] = daily_amplitude.rolling(20, min_periods=1).mean()
    for period in (5, 10, 20):
        result[f"new_high_{period}d"] = close >= close.rolling(period, min_periods=1).max()
    for period in (5, 10):
        result[f"new_low_{period}d"] = close <= close.rolling(period, min_periods=1).min()

    platform_high = high.shift(1).rolling(10, min_periods=5).max()
    platform_low = low.shift(1).rolling(10, min_periods=5).min()
    platform_range = (platform_high - platform_low) / platform_low.replace(0, pd.NA)
    recent_volatility = close.pct_change().rolling(5, min_periods=3).std()
    result["is_platform"] = (platform_range <= 0.12) & (recent_volatility <= 0.035)
    result["platform_breakout"] = result["is_platform"] & (close > platform_high)
    vol_ma5 = vol.rolling(5, min_periods=1).mean()
    vol_ma20 = vol.rolling(20, min_periods=1).mean()
    result["stage_volume_expansion"] = vol_ma5 > vol_ma20 * 1.2
    result["stage_volume_contraction"] = vol_ma5 < vol_ma20 * 0.8
    result["stage_overheated"] = (result["return_5d"] > 20) | (result["return_20d"] > 40)
    return result


def calculate_stage_factors(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "high", "low", "close", "vol"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame()
    result = pd.concat(
        [_calculate_group(group) for _, group in prices.groupby("ts_code", sort=False)],
        ignore_index=True,
    )
    columns = [
        "ts_code", "trade_date", "return_5d", "return_10d", "return_20d",
        "return_60d", "turnover_5d", "turnover_20d", "amplitude_5d",
        "amplitude_20d", "new_high_5d", "new_high_10d", "new_high_20d",
        "new_low_5d", "new_low_10d", "is_platform", "platform_breakout",
        "stage_volume_expansion", "stage_volume_contraction", "stage_overheated",
    ]
    return result[[column for column in columns if column in result]].reset_index(drop=True)
