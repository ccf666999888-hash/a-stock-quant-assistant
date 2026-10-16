from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50).clip(0, 100)


def _calculate_group(group: pd.DataFrame) -> pd.DataFrame:
    result = group.sort_values("trade_date").copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")

    for period in (5, 10, 20, 60):
        result[f"ma{period}"] = close.rolling(period, min_periods=1).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["macd_dif"] = ema12 - ema26
    result["macd_dea"] = result["macd_dif"].ewm(span=9, adjust=False).mean()
    result["macd_hist"] = 2 * (result["macd_dif"] - result["macd_dea"])

    lowest9 = low.rolling(9, min_periods=1).min()
    highest9 = high.rolling(9, min_periods=1).max()
    rsv = ((close - lowest9) / (highest9 - lowest9).replace(0, np.nan) * 100).fillna(50)
    result["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    result["kdj_d"] = result["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    result["kdj_j"] = 3 * result["kdj_k"] - 2 * result["kdj_d"]

    result["rsi6"] = _rsi(close, 6)
    result["rsi12"] = _rsi(close, 12)
    result["boll_mid"] = close.rolling(20, min_periods=1).mean()
    boll_std = close.rolling(20, min_periods=2).std(ddof=0).fillna(0)
    result["boll_upper"] = result["boll_mid"] + 2 * boll_std
    result["boll_lower"] = result["boll_mid"] - 2 * boll_std

    typical = (high + low + close) / 3
    typical_ma = typical.rolling(14, min_periods=1).mean()
    mean_dev = typical.rolling(14, min_periods=1).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))), raw=True
    )
    result["cci"] = ((typical - typical_ma) / (0.015 * mean_dev.replace(0, np.nan))).fillna(0)

    for period in (5, 10, 20):
        result[f"bias{period}"] = (
            (close - result[f"ma{period}"]) / result[f"ma{period}"].replace(0, np.nan) * 100
        ).fillna(0)

    highest14 = high.rolling(14, min_periods=1).max()
    lowest14 = low.rolling(14, min_periods=1).min()
    result["wr"] = (
        -100 * (highest14 - close) / (highest14 - lowest14).replace(0, np.nan)
    ).fillna(-50)
    result["mtm"] = close - close.shift(10)

    result["above_ma5"] = close > result["ma5"]
    result["above_ma10"] = close > result["ma10"]
    result["above_ma20"] = close > result["ma20"]
    result["ma5_cross_ma10"] = (result["ma5"] >= result["ma10"]) & (
        result["ma5"].shift(1) < result["ma10"].shift(1)
    )
    result["macd_golden_cross"] = (result["macd_dif"] >= result["macd_dea"]) & (
        result["macd_dif"].shift(1) < result["macd_dea"].shift(1)
    )
    result["kdj_golden_cross"] = (result["kdj_k"] >= result["kdj_d"]) & (
        result["kdj_k"].shift(1) < result["kdj_d"].shift(1)
    )
    result["technical_strengthening"] = (
        result["above_ma5"]
        & (result["macd_hist"] > result["macd_hist"].shift(1))
        & (result["kdj_k"] > result["kdj_k"].shift(1))
    )
    return result


def calculate_technical_factors(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "high", "low", "close"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame()
    calculated = pd.concat(
        [_calculate_group(group) for _, group in prices.groupby("ts_code", sort=False)],
        ignore_index=True,
    )
    columns = [
        "ts_code", "trade_date", "close", "ma5", "ma10", "ma20", "ma60",
        "macd_dif", "macd_dea", "macd_hist", "kdj_k", "kdj_d", "kdj_j",
        "rsi6", "rsi12", "boll_upper", "boll_mid", "boll_lower", "cci",
        "bias5", "bias10", "bias20", "wr", "mtm", "above_ma5", "above_ma10",
        "above_ma20", "ma5_cross_ma10", "macd_golden_cross", "kdj_golden_cross",
        "technical_strengthening",
    ]
    return calculated[[column for column in columns if column in calculated]].reset_index(drop=True)
