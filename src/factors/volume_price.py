from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _calculate_group(group: pd.DataFrame) -> pd.DataFrame:
    result = group.sort_values("trade_date").copy()
    high = _numeric(result, "high")
    low = _numeric(result, "low")
    close = _numeric(result, "close")
    pre_close = _numeric(result, "pre_close").replace(0, np.nan)
    vol = _numeric(result, "vol")
    amount = _numeric(result, "amount")

    if "pct_chg" not in result:
        result["pct_chg"] = (close / close.shift(1) - 1) * 100
    result["amplitude"] = ((high - low) / pre_close * 100).fillna(0)
    result["vol_ma5"] = vol.rolling(5, min_periods=1).mean()
    result["amount_ma5"] = amount.rolling(5, min_periods=1).mean()
    result["volume_ratio"] = (vol / result["vol_ma5"].replace(0, np.nan)).fillna(0)
    spread = high - low
    result["close_position"] = ((close - low) / spread.replace(0, np.nan)).fillna(0.5).clip(0, 1)
    result["is_volume_expansion"] = result["volume_ratio"] >= 1.3
    result["is_volume_contraction"] = result["volume_ratio"] <= 0.8
    return result


def calculate_volume_price_factors(
    prices: pd.DataFrame,
    daily_basic: pd.DataFrame | None = None,
    moneyflow: pd.DataFrame | None = None,
    market_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "high", "low", "close", "vol"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame()
    result = pd.concat(
        [_calculate_group(group) for _, group in prices.groupby("ts_code", sort=False)],
        ignore_index=True,
    )

    for extra in (daily_basic, moneyflow):
        if extra is not None and not extra.empty and {"ts_code", "trade_date"}.issubset(extra.columns):
            incoming = extra.drop_duplicates(["ts_code", "trade_date"], keep="last")
            overlapping = [
                column for column in incoming.columns
                if column in result.columns and column not in {"ts_code", "trade_date"}
            ]
            incoming = incoming.drop(columns=overlapping)
            result = result.merge(incoming, on=["ts_code", "trade_date"], how="left")

    result["turnover_rate"] = _numeric(result, "turnover_rate")
    result["main_net_inflow"] = _numeric(result, "net_mf_amount")
    result["large_net_inflow"] = _numeric(result, "buy_lg_amount") - _numeric(result, "sell_lg_amount")
    result["small_net_inflow"] = _numeric(result, "buy_sm_amount") - _numeric(result, "sell_sm_amount")

    market_map: dict[str, float] = {}
    if market_returns is not None and not market_returns.empty and "trade_date" in market_returns:
        market_column = "market_pct_chg" if "market_pct_chg" in market_returns else "pct_chg"
        if market_column in market_returns:
            market_map = market_returns.groupby("trade_date")[market_column].mean().to_dict()
    market_benchmark = result["trade_date"].map(market_map)
    result["stronger_than_market"] = market_benchmark.notna() & (
        result["pct_chg"].fillna(0) > market_benchmark
    )
    result["stronger_than_sector"] = False

    columns = [
        "ts_code", "trade_date", "pct_chg", "amplitude", "turnover_rate", "vol",
        "amount", "vol_ma5", "amount_ma5", "volume_ratio", "close_position",
        "main_net_inflow", "large_net_inflow", "small_net_inflow",
        "is_volume_expansion", "is_volume_contraction", "stronger_than_market",
        "stronger_than_sector",
    ]
    return result[[column for column in columns if column in result]].reset_index(drop=True)
