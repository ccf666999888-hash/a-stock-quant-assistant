from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MARKET_DB = ROOT / "data" / "db" / "a_stock_quant.duckdb"
STATE_DB = ROOT / "data" / "db" / "paper_trading.duckdb"
CONFIG_PATH = ROOT / "configs" / "paper_trading.yaml"
OUTPUT_ROOT = ROOT / "outputs" / "paper_trading"

sys.path.insert(0, str(ROOT / "src"))
from paper_trading.engine import PaperTradingEngine  # noqa: E402
from paper_trading.report import write_paper_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily A-share paper-trading account")
    parser.add_argument("--trade-date", help="Completed market date in YYYYMMDD")
    parser.add_argument("--start-wall-date", help="Account inception date in YYYYMMDD")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--sync", action="store_true", help="Refresh after-close inputs first")
    parser.add_argument("--skip-minute-fetch", action="store_true")
    return parser.parse_args()


def latest_market_date() -> str:
    with duckdb.connect(str(MARKET_DB), read_only=True) as con:
        value = con.execute("select max(trade_date) from daily_price").fetchone()[0]
    if not value:
        raise RuntimeError("daily_price has no completed trade date")
    return str(value)


def run_after_close(target_date: str, skip_minute_fetch: bool) -> str:
    latest_before = latest_market_date()
    if latest_before < target_date:
        start_date = (pd.to_datetime(latest_before) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        sync_command = [
            sys.executable,
            str(ROOT / "scripts" / "sync_tushare_data.py"),
            "--start-date",
            start_date,
            "--end-date",
            target_date,
            "--endpoints",
            "daily",
            "daily_basic",
            "index_daily",
            "--index-codes",
            "000001.SH",
        ]
        subprocess.run(sync_command, cwd=ROOT, check=True)
    completed_date = latest_market_date()
    supplement_current = False
    with duckdb.connect(str(MARKET_DB), read_only=True) as con:
        supplement_current = bool(
            con.execute(
                "select count(1) from ak_limit_event where trade_date=?",
                [completed_date],
            ).fetchone()[0]
        )
    if completed_date > latest_before or not supplement_current:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "sync_akshare_supplement.py"),
                "--trade-date",
                completed_date,
            ],
            cwd=ROOT,
            check=True,
        )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_after_close_html.py"),
        "--trade-date",
        completed_date,
    ]
    if skip_minute_fetch:
        command.append("--skip-minute-fetch")
    subprocess.run(command, cwd=ROOT, check=True)
    return completed_date


def export_snapshot(snapshot: dict, output_dir: Path, suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["positions", "pending", "trades", "signals", "equity"]:
        frame = snapshot[name]
        frame.to_csv(output_dir / f"{name}_{suffix}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    now = datetime.now()
    wall_date = args.start_wall_date or now.strftime("%Y%m%d")
    engine = PaperTradingEngine(MARKET_DB, STATE_DB, CONFIG_PATH, OUTPUT_ROOT)
    engine.initialize_account(
        wall_date,
        "用户授权建立一万元日线模拟盘；启动前历史开盘不得回填为成交。",
    )
    if args.sync and not args.initialize_only:
        synced_date = run_after_close(
            args.trade_date or now.strftime("%Y%m%d"), args.skip_minute_fetch
        )
    else:
        synced_date = None
    data_date = args.trade_date or synced_date or engine.latest_market_date()
    if data_date > engine.latest_market_date():
        data_date = engine.latest_market_date()
    context = engine.market_context(data_date)
    if args.initialize_only or data_date < str(engine.account()["start_wall_date"]):
        report_date = now.strftime("%Y%m%d")
        status = (
            "模拟账户已经启动，目前空仓。今天收盘后才会产生第一批有效信号，"
            "最早在下一交易日开盘模拟成交。"
        )
        filename = f"paper_trading_inception_{report_date}.html"
    else:
        result = engine.process_trade_date(data_date, pd.DataFrame())
        context = result["context"]
        report_date = data_date
        if result["idempotent"]:
            status = "本交易日已处理，重复运行不会重复成交。"
        elif any("错过下一交易日" in message for message in result["fills"]):
            status = (
                "检测到交易日断档：逾期订单已取消，未补录历史开盘成交；"
                "按今日收盘重新生成下一交易日计划。"
            )
        else:
            status = "已按开盘价处理上一交易日订单，并按今日收盘生成下一交易日计划。"
        filename = f"paper_trading_{report_date}.html"
    snapshot = engine.snapshot()
    report_dir = OUTPUT_ROOT / report_date
    market_chart = (
        ROOT / "outputs" / "after_close" / data_date / "assets" / f"market_60m_kline_{data_date}.png"
    )
    output = write_paper_report(
        report_dir / filename,
        snapshot,
        data_date,
        report_date,
        context,
        status,
        market_chart,
    )
    export_snapshot(snapshot, report_dir, report_date)
    print(f"account_id={engine.account_id}")
    print(f"data_date={data_date}")
    print(f"cash={float(snapshot['account']['cash']):.2f}")
    print(f"positions={len(snapshot['positions'])}")
    print(f"pending_orders={len(snapshot['pending'])}")
    print(f"html={output}")


if __name__ == "__main__":
    main()
