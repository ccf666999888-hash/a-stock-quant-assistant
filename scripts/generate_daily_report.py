from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from database.reader import DatabaseReader
from database.writer import DatabaseWriter
from reports.daily_review import build_daily_review
from reports.markdown_writer import write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily A-share review")
    parser.add_argument("--trade-date", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader, writer = DatabaseReader(), DatabaseWriter()
    screening = reader.table("screening_result", start_date=args.trade_date, end_date=args.trade_date)
    technical = reader.table("factor_technical", start_date=args.trade_date, end_date=args.trade_date)
    prices = reader.table("daily_price", start_date=args.trade_date, end_date=args.trade_date)
    for frame in [technical, prices]:
        if not frame.empty and not screening.empty:
            overlap = [column for column in frame if column in screening and column not in {"ts_code", "trade_date"}]
            screening = screening.merge(frame.drop(columns=overlap), on=["ts_code", "trade_date"], how="left")
    indices = reader.table("index_daily", start_date=args.trade_date, end_date=args.trade_date)
    content = build_daily_review(trade_date=args.trade_date, screening=screening, indices=indices)
    report_path = ROOT / "outputs" / "reports" / f"daily_review_{args.trade_date}.md"
    watchlist_path = ROOT / "outputs" / "stock_pools" / f"watchlist_{args.trade_date}.csv"
    write_markdown(report_path, content)
    if not screening.empty:
        screening.to_csv(watchlist_path, index=False, encoding="utf-8-sig")
    writer.append_dataframe(
        "daily_report_log",
        __import__("pandas").DataFrame(
            [{
                "trade_date": args.trade_date,
                "generated_at": datetime.now(),
                "report_path": str(report_path),
                "watchlist_path": str(watchlist_path),
                "status": "ok",
            }]
        ),
    )
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()

