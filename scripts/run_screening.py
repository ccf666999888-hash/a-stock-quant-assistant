from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from database.reader import DatabaseReader
from database.writer import DatabaseWriter
from strategies.scoring import score_universe
from utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short-term stock screening")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--pool", default="all")
    return parser.parse_args()


def merge_date_tables(reader: DatabaseReader, trade_date: str) -> pd.DataFrame:
    tables = ["daily_price", "stock_basic", "factor_technical", "factor_volume_price", "factor_stage"]
    frames = []
    for table in tables:
        frame = reader.table(table, start_date=trade_date, end_date=trade_date) if table != "stock_basic" else reader.table(table)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = frames[0]
    for frame in frames[1:]:
        keys = ["ts_code"] if "trade_date" not in frame else ["ts_code", "trade_date"]
        overlapping = [column for column in frame if column in result and column not in keys]
        result = result.merge(frame.drop(columns=overlapping), on=keys, how="left")
    return result


def main() -> None:
    args = parse_args()
    reader, writer = DatabaseReader(), DatabaseWriter()
    universe = merge_date_tables(reader, args.trade_date)
    pools = load_yaml("configs/stock_pool.yaml").get("pools", {})
    ts_codes = pools.get(args.pool, {}).get("ts_codes", [])
    if ts_codes and not universe.empty:
        universe = universe[universe["ts_code"].isin(ts_codes)]
    scored = score_universe(universe)
    writer.upsert_dataframe("screening_result", scored)
    output = ROOT / "outputs" / "stock_pools" / f"watchlist_{args.trade_date}.csv"
    scored.to_csv(output, index=False, encoding="utf-8-sig")
    table = Table(title=f"Watchlist {args.trade_date} / {args.pool}")
    for column in ["ts_code", "name", "models", "total_score", "classification", "risk_reasons"]:
        table.add_column(column)
    for _, row in scored.head(30).iterrows():
        table.add_row(*(str(row.get(column, "")) for column in ["ts_code", "name", "models", "total_score", "classification", "risk_reasons"]))
    Console().print(table)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

