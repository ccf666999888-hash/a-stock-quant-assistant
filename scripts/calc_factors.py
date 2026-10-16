from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from database.reader import DatabaseReader
from database.writer import DatabaseWriter
from factors.risk import calculate_risk_factors
from factors.stage import calculate_stage_factors
from factors.technical import calculate_technical_factors
from factors.volume_price import calculate_volume_price_factors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate and persist factor tables")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    return parser.parse_args()


def merge_factors(*frames):
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame()
    result = usable[0]
    for frame in usable[1:]:
        overlapping = [
            column
            for column in frame.columns
            if column in result.columns and column not in {"ts_code", "trade_date"}
        ]
        result = result.merge(
            frame.drop(columns=overlapping),
            on=["ts_code", "trade_date"],
            how="outer",
        )
    return result


def main() -> None:
    args = parse_args()
    reader, writer = DatabaseReader(), DatabaseWriter()
    prices = reader.table("daily_price", start_date=args.start_date, end_date=args.end_date)
    basic = reader.table("daily_basic", start_date=args.start_date, end_date=args.end_date)
    moneyflow = reader.table("moneyflow", start_date=args.start_date, end_date=args.end_date)
    indices = reader.table("index_daily", start_date=args.start_date, end_date=args.end_date)
    market_returns = (
        indices.groupby("trade_date", as_index=False)["pct_chg"].mean().rename(columns={"pct_chg": "market_pct_chg"})
        if not indices.empty else None
    )
    technical = calculate_technical_factors(prices)
    volume_price = calculate_volume_price_factors(prices, basic, moneyflow, market_returns)
    prices_with_turnover = prices.merge(
        basic[["ts_code", "trade_date", "turnover_rate"]]
        if not basic.empty and "turnover_rate" in basic else basic,
        on=["ts_code", "trade_date"],
        how="left",
    ) if not basic.empty else prices
    stage = calculate_stage_factors(prices_with_turnover)
    combined = merge_factors(prices, technical, volume_price, stage)
    risk = calculate_risk_factors(combined)
    for table, frame in [
        ("factor_technical", technical),
        ("factor_volume_price", volume_price),
        ("factor_stage", stage),
        ("factor_risk", risk),
    ]:
        writer.upsert_dataframe(table, frame)
        print(f"{table}: {len(frame)} rows")


if __name__ == "__main__":
    main()
