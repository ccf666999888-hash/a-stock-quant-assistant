from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest.event_study import run_event_study
from database.reader import DatabaseReader
from reports.markdown_writer import write_markdown
from utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T-1 factor event study")
    parser.add_argument("--pool", default="all")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--threshold", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = DatabaseReader()
    frames = [
        reader.table(table, start_date=args.start_date, end_date=args.end_date)
        for table in ["factor_volume_price", "factor_technical", "factor_stage", "screening_result"]
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        features = pd.DataFrame()
    else:
        features = frames[0]
        for frame in frames[1:]:
            overlap = [column for column in frame if column in features and column not in {"ts_code", "trade_date"}]
            features = features.merge(frame.drop(columns=overlap), on=["ts_code", "trade_date"], how="left")
    pools = load_yaml("configs/stock_pool.yaml").get("pools", {})
    result = run_event_study(
        features,
        start_date=args.start_date,
        end_date=args.end_date,
        threshold=args.threshold,
        ts_codes=pools.get(args.pool, {}).get("ts_codes") or None,
    )
    prefix = ROOT / "outputs" / "backtests" / f"event_study_{args.pool}_{args.end_date}"
    result.samples.to_csv(f"{prefix}_samples.csv", index=False, encoding="utf-8-sig")
    result.comparison.to_csv(f"{prefix}_comparison.csv", index=False, encoding="utf-8-sig")
    report = f"""# Event Study: {args.pool}

- Period: {args.start_date} to {args.end_date}
- Success threshold: next-day return >= {args.threshold:.2f}%
- Samples: {len(result.samples)}
- Success samples: {len(result.success_samples)}
- Control samples: {len(result.control_samples)}
- Success rate: {result.success_rate:.2%}

## Success vs Control

{result.comparison.to_markdown(index=False) if not result.comparison.empty else "No comparable samples."}

> 仅供复盘研究和交易计划参考，不构成投资建议。
"""
    write_markdown(f"{prefix}.md", report)
    print(f"Saved event study: {prefix}.md")


if __name__ == "__main__":
    main()

