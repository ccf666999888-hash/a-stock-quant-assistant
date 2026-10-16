from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.downloader import TushareDownloader
from data.trade_calendar import fetch_trade_calendar, open_dates
from database.reader import DatabaseReader
from database.writer import DatabaseWriter
from utils.config import load_yaml

DAILY_ENDPOINTS = {
    "daily",
    "daily_basic",
    "moneyflow",
    "stk_limit",
    "top_list",
    "top_inst",
}
RANGE_PAGINATED_ENDPOINTS = {"daily", "daily_basic", "moneyflow", "stk_limit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Tushare data into DuckDB")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--endpoints", nargs="*")
    parser.add_argument(
        "--index-codes",
        nargs="*",
        help="Optional index-code override; defaults to configs/tushare.yaml.",
    )
    parser.add_argument(
        "--top-list-days",
        type=int,
        default=20,
        help="Only sync top_list/top_inst for the most recent N open days; 0 means all.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0,
        help="Seconds to wait between per-trade-date requests for low-frequency accounts.",
    )
    parser.add_argument(
        "--paginate-range",
        action="store_true",
        help="Paginate date-range endpoints with limit/offset instead of requesting each trade date.",
    )
    parser.add_argument("--page-size", type=int, default=6000)
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=14,
        help="Calendar-day chunk size used with --paginate-range to avoid offset caps.",
    )
    return parser.parse_args()


def date_chunks(start_date: str, end_date: str, chunk_days: int) -> list[tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    chunks: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def main() -> None:
    args = parse_args()
    config = load_yaml("configs/tushare.yaml")
    endpoints = args.endpoints or config.get("endpoints", [])
    downloader = TushareDownloader()
    results = []

    reader = DatabaseReader()
    needs_calendar = any(
        endpoint in DAILY_ENDPOINTS
        and not (args.paginate_range and endpoint in RANGE_PAGINATED_ENDPOINTS)
        for endpoint in endpoints
    )
    calendar = pd.DataFrame()
    trade_dates: list[str] = []
    if needs_calendar:
        cached_calendar = reader.query(
            "SELECT * FROM trade_calendar WHERE cal_date >= ? AND cal_date <= ?",
            [args.start_date, args.end_date],
        )
        expected_dates = set(
            pd.date_range(args.start_date, args.end_date).strftime("%Y%m%d")
        )
        cached_dates = set(cached_calendar["cal_date"].astype(str))
        calendar = (
            cached_calendar
            if expected_dates.issubset(cached_dates)
            else fetch_trade_calendar(args.start_date, args.end_date)
        )
        if not expected_dates.issubset(set(calendar["cal_date"].astype(str))):
            raise RuntimeError("trade_cal did not cover the requested date range")
        trade_dates = open_dates(calendar)
        if not calendar.empty:
            DatabaseWriter().upsert_dataframe(
                "trade_calendar",
                calendar,
                keys=("exchange", "cal_date"),
            )
        print(f"Open trade dates: {len(trade_dates)}")

    for endpoint in endpoints:
        if endpoint == "stock_basic":
            results.append(downloader.sync(endpoint, list_status="L"))
        elif endpoint == "trade_cal":
            continue
        elif endpoint == "index_daily":
            for ts_code in args.index_codes or config.get("index_codes", []):
                results.append(
                    downloader.sync(
                        endpoint,
                        ts_code=ts_code,
                        start_date=args.start_date,
                        end_date=args.end_date,
                    )
                )
        elif args.paginate_range and endpoint in RANGE_PAGINATED_ENDPOINTS:
            chunks = date_chunks(args.start_date, args.end_date, args.chunk_days)
            for chunk_position, (chunk_start, chunk_end) in enumerate(chunks, start=1):
                offset = 0
                page = 1
                while True:
                    result = downloader.sync(
                        endpoint,
                        start_date=chunk_start,
                        end_date=chunk_end,
                        limit=args.page_size,
                        offset=offset,
                    )
                    results.append(result)
                    print(
                        f"{endpoint:15} chunk={chunk_position:>2}/{len(chunks):<2} "
                        f"{chunk_start}-{chunk_end} page={page:<2} "
                        f"offset={offset:<7} {result.status:22} rows={result.rows}"
                    )
                    if args.request_interval > 0:
                        time.sleep(args.request_interval)
                    if result.rows < args.page_size:
                        break
                    offset += args.page_size
                    page += 1
        elif endpoint in DAILY_ENDPOINTS:
            endpoint_dates = trade_dates
            if endpoint in {"top_list", "top_inst"} and args.top_list_days > 0:
                endpoint_dates = trade_dates[-args.top_list_days :]
            for position, trade_date in enumerate(endpoint_dates, start=1):
                result = downloader.sync(endpoint, trade_date=trade_date)
                results.append(result)
                print(
                    f"{endpoint:15} {position:>3}/{len(endpoint_dates):<3} "
                    f"{trade_date} {result.status:22} rows={result.rows}"
                )
                if args.request_interval > 0 and position < len(endpoint_dates):
                    time.sleep(args.request_interval)
        else:
            results.append(
                downloader.sync(endpoint, start_date=args.start_date, end_date=args.end_date)
            )
    print("\nSummary")
    for endpoint in endpoints:
        endpoint_results = [result for result in results if result.endpoint == endpoint]
        if endpoint_results:
            print(
                f"{endpoint:15} calls={len(endpoint_results):<4} "
                f"rows={sum(result.rows for result in endpoint_results)}"
            )


if __name__ == "__main__":
    main()
