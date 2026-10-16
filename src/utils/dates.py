from __future__ import annotations

from datetime import date, datetime


def normalize_trade_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    cleaned = str(value).replace("-", "").replace("/", "")
    if len(cleaned) != 8 or not cleaned.isdigit():
        raise ValueError(f"Invalid trade date: {value!r}; expected YYYYMMDD")
    return cleaned

