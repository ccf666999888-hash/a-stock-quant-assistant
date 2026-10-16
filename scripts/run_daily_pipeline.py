from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily end-to-end pipeline")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--pool", default="all")
    parser.add_argument("--skip-sync", action="store_true")
    args = parser.parse_args()
    trade_date = datetime.strptime(args.trade_date, "%Y%m%d")
    start_date = (trade_date - timedelta(days=args.lookback_days)).strftime("%Y%m%d")
    run("setup_database.py")
    if not args.skip_sync:
        run("sync_tushare_data.py", "--start-date", start_date, "--end-date", args.trade_date)
    run("calc_factors.py", "--start-date", start_date, "--end-date", args.trade_date)
    run("run_screening.py", "--trade-date", args.trade_date, "--pool", args.pool)
    run("generate_daily_report.py", "--trade-date", args.trade_date)


if __name__ == "__main__":
    main()

