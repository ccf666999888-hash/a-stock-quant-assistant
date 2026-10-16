from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "a_stock_quant.duckdb"
sys.path.insert(0, str(ROOT / "src"))

from data.akshare_client import AkshareClient, aggregate_market_from_stock  # noqa: E402
from database.writer import DatabaseWriter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync AKShare fund-flow and limit-event supplements")
    parser.add_argument("--trade-date", help="YYYYMMDD; defaults to latest local daily_price date")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["market", "sector", "stock", "limits"],
        default=["market", "sector", "stock", "limits"],
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.5)
    return parser.parse_args()


def resolve_trade_date(requested: str | None) -> str:
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        if requested:
            value = con.execute(
                "select max(trade_date) from daily_price where trade_date <= ?", [requested]
            ).fetchone()[0]
        else:
            value = con.execute("select max(trade_date) from daily_price").fetchone()[0]
    if not value:
        raise RuntimeError("daily_price has no usable trade date")
    return str(value)


def local_daily_snapshot(trade_date: str) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(
            "select ts_code, close, pct_chg from daily_price where trade_date = ?",
            [trade_date],
        ).fetchdf()


def validate_stock_snapshot(
    frame: pd.DataFrame,
    local: pd.DataFrame,
) -> tuple[bool, str]:
    if frame.empty or local.empty:
        return False, "stock snapshot or local daily data is empty"
    merged = frame.merge(local, on="ts_code", suffixes=("_ak", "_local"))
    valid = merged.dropna(subset=["pct_chg_ak", "pct_chg_local"])
    if len(valid) < min(1000, max(100, int(len(local) * 0.25))):
        return False, f"only {len(valid)} codes overlap local daily data"
    differences = valid["pct_chg_ak"].sub(valid["pct_chg_local"]).abs()
    correlation = valid[["pct_chg_ak", "pct_chg_local"]].corr().iloc[0, 1]
    median_difference = differences.median()
    within_one_point = (differences <= 1.0).mean()
    if correlation < 0.97 or median_difference > 0.25 or within_one_point < 0.85:
        return (
            False,
            "cross-section date check failed: "
            f"corr={correlation:.3f}, median_diff={median_difference:.3f}, "
            f"within_1pp={within_one_point:.1%}",
        )
    return (
        True,
        f"{len(valid)} overlapping codes; corr={correlation:.3f}, "
        f"median_diff={median_difference:.3f}, within_1pp={within_one_point:.1%}",
    )


def write_frame(
    writer: DatabaseWriter,
    table: str,
    frame: pd.DataFrame,
    keys: tuple[str, ...],
) -> int:
    if frame.empty:
        return 0
    return writer.upsert_dataframe(table, frame, keys=keys)


def main() -> None:
    args = parse_args()
    trade_date = resolve_trade_date(args.trade_date)
    requested = set(args.only)
    writer = DatabaseWriter(DB_PATH)
    writer.initialize()
    client = AkshareClient(retries=args.retries, delay_seconds=args.delay)
    local = local_daily_snapshot(trade_date)
    status: list[str] = []

    market_current = False
    if "market" in requested:
        try:
            market = client.market_fund_flow()
            market = market[market["trade_date"] == trade_date].copy()
            market_current = not market.empty
            rows = write_frame(writer, "ak_market_fund_flow", market, ("trade_date",))
            status.append(f"market=ok rows={rows}" if rows else "market=unavailable_for_date")
        except Exception as exc:
            status.append(f"market=error {type(exc).__name__}: {exc}")

    stock_verified = False
    stock = pd.DataFrame()
    if "stock" in requested or "sector" in requested:
        try:
            stock = client.stock_fund_flow(trade_date)
            stock_verified, message = validate_stock_snapshot(stock, local)
            if stock_verified and "stock" in requested:
                coverage = len(stock) / max(len(local), 1)
                stock["data_status"] = "complete" if coverage >= 0.8 else "partial"
                rows = write_frame(
                    writer,
                    "ak_stock_fund_flow",
                    stock,
                    ("trade_date", "ts_code"),
                )
                source = str(stock["source"].iloc[0]) if "source" in stock and not stock.empty else "unknown"
                status.append(
                    f"stock=ok rows={rows} coverage={coverage:.1%} source={source}; {message}"
                )
            elif not stock_verified:
                status.append(f"stock=rejected {message}")
        except Exception as exc:
            status.append(f"stock=error {type(exc).__name__}: {exc}")

    if "market" in requested and not market_current and stock_verified:
        aggregate = aggregate_market_from_stock(stock, trade_date)
        rows = write_frame(writer, "ak_market_fund_flow", aggregate, ("trade_date",))
        market_current = rows > 0
        status.append(f"market=fallback_stock_aggregate rows={rows}")

    if "sector" in requested:
        if not stock_verified and not market_current:
            status.append("sector=skipped because snapshot date could not be verified")
        else:
            try:
                sector = client.sector_fund_flow(trade_date)
                sector["data_status"] = "complete" if len(sector) >= 30 else "partial"
                rows = write_frame(
                    writer,
                    "ak_sector_fund_flow",
                    sector,
                    ("trade_date", "provider_sector"),
                )
                source = str(sector["source"].iloc[0]) if "source" in sector and not sector.empty else "unknown"
                status.append(f"sector=ok rows={rows} source={source}")
            except Exception as exc:
                status.append(f"sector=error {type(exc).__name__}: {exc}")

    if "limits" in requested:
        event_frames: list[pd.DataFrame] = []
        for event_type in ["limit_up", "broken_board", "limit_down"]:
            try:
                event = client.limit_pool(trade_date, event_type)
                event_frames.append(event)
                status.append(f"{event_type}=ok rows={len(event)}")
            except Exception as exc:
                status.append(f"{event_type}=error {type(exc).__name__}: {exc}")
        events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
        rows = write_frame(
            writer,
            "ak_limit_event",
            events,
            ("trade_date", "ts_code", "event_type"),
        )
        status.append(f"limit_events_written={rows}")

    print(f"trade_date={trade_date}")
    for line in status:
        print(line)


if __name__ == "__main__":
    main()
