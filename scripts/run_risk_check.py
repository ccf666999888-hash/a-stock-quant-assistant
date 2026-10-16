from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from database.reader import DatabaseReader


def main() -> None:
    parser = argparse.ArgumentParser(description="List triggered short-term risks")
    parser.add_argument("--trade-date", required=True)
    args = parser.parse_args()
    risks = DatabaseReader().table("factor_risk", start_date=args.trade_date, end_date=args.trade_date)
    triggered = risks[risks["risk_triggered"]] if not risks.empty else risks
    output = ROOT / "outputs" / "stock_pools" / f"risk_check_{args.trade_date}.csv"
    triggered.to_csv(output, index=False, encoding="utf-8-sig")
    print(triggered.to_string(index=False) if not triggered.empty else "No triggered risks or no data.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

