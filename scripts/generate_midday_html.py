from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MARKET_DB = ROOT / "data" / "db" / "a_stock_quant.duckdb"
MIDDAY_DB = ROOT / "data" / "db" / "a_stock_midday.duckdb"
OUTPUT_ROOT = ROOT / "outputs" / "midday"
SHANGHAI = ZoneInfo("Asia/Shanghai")
WINDOW_START = clock_time(11, 35)
WINDOW_END = clock_time(12, 55)
CURRENT_BAR_CUTOFF = clock_time(11, 30)
LOOKBACK_DAYS = 60

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data.akshare_client import (  # noqa: E402
    canonical_sector_name,
    normalize_limit_pool,
    normalize_ths_sector_fund_flow,
    normalize_ths_stock_fund_flow,
    standardize_stock_code,
)
from plot_sector_60m_macd import add_macd, build_sector_bar, classify_latest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an A-share midday review")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Check connectivity and freshness without writing snapshots",
    )
    parser.add_argument("--max-sector-charts", type=int, default=4)
    parser.add_argument("--members-per-sector", type=int, default=6)
    return parser.parse_args()


def shanghai_now() -> datetime:
    return datetime.now(SHANGHAI)


def in_midday_window(value: datetime) -> bool:
    local_time = value.timetz().replace(tzinfo=None)
    return value.weekday() < 5 and WINDOW_START <= local_time <= WINDOW_END


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_code(value: object) -> str:
    text = str(value or "").strip().lower()
    digits = "".join(char for char in text if char.isdigit())[-6:]
    return standardize_stock_code(digits) if len(digits) == 6 else ""


def normalize_tencent_spot(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    frame["ts_code"] = frame["code"].map(_clean_code)
    frame["name"] = frame["name"].astype(str)
    frame["last"] = _number(frame["zxj"])
    frame["pct_chg"] = _number(frame["zdf"])
    frame["amount"] = _number(frame["turnover"]) * 10_000.0
    frame["volume"] = _number(frame["volume"])
    frame["amplitude"] = _number(frame["zf"])
    columns = [
        "ts_code",
        "name",
        "last",
        "pct_chg",
        "amount",
        "volume",
        "amplitude",
    ]
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["ts_code", "last", "pct_chg"])
    )


def normalize_sina_spot(raw: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if raw is None or raw.empty:
        return pd.DataFrame(), None
    frame = raw.copy()
    frame["ts_code"] = frame["代码"].map(_clean_code)
    frame["anchor_last"] = _number(frame["最新价"])
    frame["anchor_pct_chg"] = _number(frame["涨跌幅"])
    frame["open"] = _number(frame["今开"])
    frame["high"] = _number(frame["最高"])
    frame["low"] = _number(frame["最低"])
    frame["pre_close"] = _number(frame["昨收"])
    frame["anchor_amount"] = _number(frame["成交额"])
    times = frame.get("时间戳", pd.Series(dtype=str)).dropna().astype(str)
    provider_time = times.mode().iloc[0] if not times.empty else None
    columns = [
        "ts_code",
        "anchor_last",
        "anchor_pct_chg",
        "open",
        "high",
        "low",
        "pre_close",
        "anchor_amount",
    ]
    return frame[columns].dropna(subset=["ts_code"]), provider_time


def provider_time_delta_minutes(
    provider_time: str | None, captured_at: datetime
) -> float | None:
    if not provider_time:
        return None
    try:
        parsed = datetime.strptime(provider_time, "%H:%M:%S").time()
    except ValueError:
        return None
    provider_dt = datetime.combine(captured_at.date(), parsed, tzinfo=SHANGHAI)
    return abs((provider_dt - captured_at).total_seconds()) / 60.0


def normalize_index_minutes(
    raw: pd.DataFrame, trade_date: str
) -> tuple[pd.DataFrame, int]:
    if raw is None or raw.empty:
        return pd.DataFrame(), 0
    frame = raw.rename(
        columns={"day": "trade_time", "volume": "vol"}
    ).copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    for column in ["open", "high", "low", "close", "vol", "amount"]:
        frame[column] = _number(frame[column]) if column in frame else np.nan
    frame = frame.dropna(
        subset=["trade_time", "open", "high", "low", "close"]
    )
    frame["trade_date"] = frame["trade_time"].dt.strftime("%Y%m%d")
    current = frame["trade_date"] == trade_date
    future = current & (frame["trade_time"].dt.time > CURRENT_BAR_CUTOFF)
    future_rows = int(future.sum())
    frame = frame[~future].copy()
    start = pd.to_datetime(trade_date) - pd.Timedelta(days=LOOKBACK_DAYS - 1)
    frame = frame[frame["trade_time"] >= start].sort_values("trade_time")
    frame["instrument"] = "000001.SH"
    frame["freq"] = "60min"
    frame["source"] = "sina_via_akshare"
    columns = [
        "instrument",
        "trade_time",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "freq",
        "source",
    ]
    return frame[columns], future_rows


def normalize_stock_minutes(
    raw: pd.DataFrame, ts_code: str, trade_date: str
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.rename(
        columns={"day": "trade_time", "volume": "vol"}
    ).copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    for column in ["open", "high", "low", "close", "vol", "amount"]:
        frame[column] = _number(frame[column]) if column in frame else np.nan
    frame = frame.dropna(
        subset=["trade_time", "open", "high", "low", "close"]
    )
    frame["trade_date"] = frame["trade_time"].dt.strftime("%Y%m%d")
    future = (frame["trade_date"] == trade_date) & (
        frame["trade_time"].dt.time > CURRENT_BAR_CUTOFF
    )
    frame = frame[~future]
    start = pd.to_datetime(trade_date) - pd.Timedelta(days=LOOKBACK_DAYS - 1)
    frame = frame[frame["trade_time"] >= start].copy()
    frame["ts_code"] = ts_code
    frame["freq"] = "60min"
    frame["source"] = "sina_via_akshare"
    columns = [
        "ts_code",
        "trade_time",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "freq",
        "source",
    ]
    return frame[columns]


def cross_validate(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    empty = {
        "overlap": 0,
        "correlation": np.nan,
        "median_diff": np.nan,
        "within_1pp": 0.0,
    }
    if left.empty or right.empty:
        return empty
    merged = left[["ts_code", "pct_chg"]].merge(
        right[["ts_code", "pct_chg"]].rename(
            columns={"pct_chg": "right_pct"}
        ),
        on="ts_code",
    ).dropna()
    if merged.empty:
        return empty
    diff = (merged["pct_chg"] - merged["right_pct"]).abs()
    return {
        "overlap": int(len(merged)),
        "correlation": float(merged["pct_chg"].corr(merged["right_pct"])),
        "median_diff": float(diff.median()),
        "within_1pp": float((diff <= 1.0).mean() * 100),
    }


def safe_call(label: str, function, retries: int = 2):
    error: Exception | None = None
    for attempt in range(retries):
        try:
            return function(), ""
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame(), f"{label}: {type(error).__name__}: {error}"


def collect_live(trade_date: str) -> dict[str, object]:
    import akshare as ak

    errors: list[str] = []
    tx_raw, error = safe_call("腾讯全A快照", ak.stock_zh_a_spot_tx)
    if error:
        errors.append(error)
    sina_raw, error = safe_call("新浪全A快照", ak.stock_zh_a_spot)
    if error:
        errors.append(error)
    stock_flow_raw, error = safe_call(
        "同花顺个股资金",
        lambda: ak.stock_fund_flow_individual(symbol="即时"),
    )
    if error:
        errors.append(error)
    sector_flow_raw, error = safe_call(
        "同花顺行业资金",
        lambda: ak.stock_fund_flow_industry(symbol="即时"),
    )
    if error:
        errors.append(error)
    index_raw, error = safe_call(
        "新浪指数60分钟",
        lambda: ak.stock_zh_a_minute(
            symbol="sh000001", period="60", adjust=""
        ),
    )
    if error:
        errors.append(error)

    events: list[pd.DataFrame] = []
    for event_type, function in [
        ("limit_up", ak.stock_zt_pool_em),
        ("broken_board", ak.stock_zt_pool_zbgc_em),
        ("limit_down", ak.stock_zt_pool_dtgc_em),
    ]:
        raw, error = safe_call(
            event_type, lambda target=function: target(date=trade_date)
        )
        if error:
            errors.append(error)
        elif not raw.empty:
            events.append(normalize_limit_pool(raw, trade_date, event_type))

    spot = normalize_tencent_spot(tx_raw)
    anchor, provider_time = normalize_sina_spot(sina_raw)
    stock_flow = (
        normalize_ths_stock_fund_flow(stock_flow_raw, trade_date)
        if not stock_flow_raw.empty
        else pd.DataFrame()
    )
    sector_flow = (
        normalize_ths_sector_fund_flow(sector_flow_raw, trade_date)
        if not sector_flow_raw.empty
        else pd.DataFrame()
    )
    index_bars, future_rows = normalize_index_minutes(index_raw, trade_date)
    return {
        "spot": spot,
        "anchor": anchor,
        "provider_time": provider_time,
        "stock_flow": stock_flow,
        "sector_flow": sector_flow,
        "events": (
            pd.concat(events, ignore_index=True)
            if events
            else pd.DataFrame()
        ),
        "index_bars": index_bars,
        "future_rows": future_rows,
        "errors": errors,
    }


def current_trade_day_is_open(trade_date: str) -> bool:
    with duckdb.connect(str(MARKET_DB), read_only=True) as con:
        row = con.execute(
            """
            select is_open from trade_calendar
            where cal_date = ?
            limit 1
            """,
            [trade_date],
        ).fetchone()
    if row is not None:
        return bool(row[0])
    if pd.to_datetime(trade_date).weekday() >= 5:
        return False

    # The local calendar is incrementally maintained and may stop at the
    # previous session. Query only the missing date without mutating it.
    try:
        env_path = ROOT / ".env"
        token = ""
        if env_path.exists():
            for line in env_path.read_text(
                encoding="utf-8"
            ).splitlines():
                if line.strip().startswith("TUSHARE_TOKEN="):
                    token = (
                        line.split("=", 1)[1]
                        .strip()
                        .strip('"')
                        .strip("'")
                    )
                    break
        if token:
            import tushare as ts

            frame = ts.pro_api(token).trade_cal(
                exchange="SSE",
                start_date=trade_date,
                end_date=trade_date,
            )
            if frame is not None and not frame.empty:
                return bool(int(frame.iloc[0]["is_open"]))
    except Exception:
        pass

    # A weekday with a missing calendar can still proceed to live data
    # validation. Grade C will suppress selections if the market is closed
    # or providers are stale.
    return True


def latest_completed_date(trade_date: str) -> str:
    with duckdb.connect(str(MARKET_DB), read_only=True) as con:
        value = con.execute(
            "select max(trade_date) from daily_price where trade_date < ?",
            [trade_date],
        ).fetchone()[0]
    if not value:
        raise RuntimeError(
            "No completed daily baseline exists before the midday session"
        )
    return str(value)


def quality_assessment(
    live: dict[str, object], captured_at: datetime, trade_date: str
) -> dict[str, object]:
    spot = live["spot"]
    anchor = live["anchor"]
    flow = live["stock_flow"]
    bars = live["index_bars"]
    assert isinstance(spot, pd.DataFrame)
    assert isinstance(anchor, pd.DataFrame)
    assert isinstance(flow, pd.DataFrame)
    assert isinstance(bars, pd.DataFrame)

    flow_validation = (
        cross_validate(spot, flow)
        if not flow.empty
        else cross_validate(pd.DataFrame(), pd.DataFrame())
    )
    anchor_for_check = (
        anchor[["ts_code", "anchor_pct_chg"]].rename(
            columns={"anchor_pct_chg": "pct_chg"}
        )
        if not anchor.empty
        else pd.DataFrame()
    )
    anchor_validation = cross_validate(spot, anchor_for_check)
    delta = provider_time_delta_minutes(
        live.get("provider_time"), captured_at
    )
    time_ok = delta is not None and delta <= 15
    current_bars = (
        bars[bars["trade_date"] == trade_date]
        if not bars.empty
        else pd.DataFrame()
    )
    bar_times = set(
        pd.to_datetime(
            current_bars.get(
                "trade_time", pd.Series(dtype="datetime64[ns]")
            ).dropna()
        ).dt.time
    )
    bars_ok = {
        clock_time(10, 30),
        clock_time(11, 30),
    }.issubset(bar_times)
    cross_ok = (
        flow_validation["overlap"] >= 1_000
        and flow_validation["correlation"] >= 0.97
        and flow_validation["median_diff"] <= 0.25
        and flow_validation["within_1pp"] >= 85
    )
    anchor_ok = (
        anchor_validation["overlap"] >= 1_000
        and anchor_validation["correlation"] >= 0.97
        and anchor_validation["median_diff"] <= 0.35
    )
    core_ok = len(spot) >= 5_000 and cross_ok and bars_ok
    if core_ok and anchor_ok and time_ok and not live["errors"]:
        grade = "A"
    elif core_ok and anchor_ok and time_ok:
        grade = "B"
    else:
        grade = "C"

    reasons: list[str] = []
    if len(spot) < 5_000:
        reasons.append(f"全A快照仅{len(spot)}只")
    if not cross_ok:
        reasons.append("行情与资金快照交叉校验未通过")
    if not anchor_ok:
        reasons.append("带时间戳行情与腾讯行情不一致")
    if not time_ok:
        reasons.append(
            f"供应商时间与本地相差{delta:.1f}分钟"
            if delta is not None
            else "供应商时间缺失"
        )
    if not bars_ok:
        reasons.append("10:30或11:30完整指数K线缺失")
    reasons.extend(live["errors"])
    return {
        "grade": grade,
        "can_select": grade in {"A", "B"},
        "provider_time_delta": delta,
        "flow_validation": flow_validation,
        "anchor_validation": anchor_validation,
        "current_index_bars": len(current_bars),
        "future_rows_dropped": int(live["future_rows"]),
        "reasons": reasons,
    }


def load_basic() -> pd.DataFrame:
    with duckdb.connect(str(MARKET_DB), read_only=True) as con:
        return con.execute(
            "select ts_code, name, industry, list_date from stock_basic"
        ).fetchdf()


def previous_relative_returns(previous_date: str) -> pd.DataFrame:
    with duckdb.connect(str(MARKET_DB), read_only=True) as con:
        dates = [
            row[0]
            for row in con.execute(
                """
                select distinct trade_date from daily_price
                where trade_date <= ?
                order by trade_date desc limit 5
                """,
                [previous_date],
            ).fetchall()
        ]
        frame = con.execute(
            """
            select p.trade_date, p.ts_code, p.pct_chg, b.industry
            from daily_price p join stock_basic b using(ts_code)
            where p.trade_date in (select * from unnest(?))
            """,
            [dates],
        ).fetchdf()
    if frame.empty:
        return pd.DataFrame(columns=["sector", "rel3", "rel5"])
    frame["sector"] = (
        frame["industry"].astype(str).map(canonical_sector_name)
    )
    market = (
        frame.groupby("trade_date")["pct_chg"]
        .mean()
        .rename("market_pct")
    )
    grouped = (
        frame.groupby(["trade_date", "sector"])["pct_chg"]
        .mean()
        .rename("sector_pct")
        .reset_index()
        .merge(market, on="trade_date")
    )
    grouped["relative"] = (
        grouped["sector_pct"] - grouped["market_pct"]
    )
    ordered = {
        date: index for index, date in enumerate(reversed(dates))
    }
    grouped["order"] = grouped["trade_date"].map(ordered)
    rows = []
    for sector, group in grouped.groupby("sector"):
        values = group.sort_values("order")["relative"]
        rows.append(
            {
                "sector": sector,
                "rel3": float(values.tail(3).sum()),
                "rel5": float(values.tail(5).sum()),
            }
        )
    return pd.DataFrame(rows)


def market_summary(
    stocks: pd.DataFrame,
    flow: pd.DataFrame,
    events: pd.DataFrame,
    bars: pd.DataFrame,
) -> dict[str, object]:
    equal_return = (
        float(stocks["pct_chg"].mean())
        if not stocks.empty
        else np.nan
    )
    breadth = (
        float((stocks["pct_chg"] > 0).mean() * 100)
        if not stocks.empty
        else np.nan
    )
    amount_yi = (
        float(stocks["amount"].sum() / 100_000_000.0)
        if not stocks.empty
        else np.nan
    )
    flow_yi = (
        float(flow["net_inflow_amount"].sum() / 100_000_000.0)
        if not flow.empty
        else np.nan
    )
    turnover = (
        float(flow["turnover_amount"].sum())
        if not flow.empty
        else np.nan
    )
    flow_ratio = (
        flow_yi * 100_000_000.0 / turnover * 100
        if turnover and turnover > 0
        else np.nan
    )
    counts = (
        events["event_type"].value_counts()
        if not events.empty
        else pd.Series(dtype=int)
    )
    limit_up = int(counts.get("limit_up", 0))
    broken = int(counts.get("broken_board", 0))
    limit_down = int(counts.get("limit_down", 0))
    attempts = limit_up + broken
    broken_rate = broken / attempts * 100 if attempts else 0.0
    if (
        breadth < 35
        or limit_down > limit_up
        or (broken_rate >= 40 and limit_down >= 10)
    ):
        regime = "上午退潮"
    elif equal_return >= 0.5 and breadth >= 55 and broken_rate < 25:
        regime = "上午强势"
    elif equal_return >= 0 and breadth >= 48:
        regime = "震荡偏强"
    else:
        regime = "震荡偏弱"
    state = "数据不足"
    if len(bars) >= 30:
        state = classify_latest(add_macd(bars))
    return {
        "equal_return": equal_return,
        "breadth": breadth,
        "amount_yi": amount_yi,
        "flow_yi": flow_yi,
        "flow_ratio": flow_ratio,
        "limit_up": limit_up,
        "broken": broken,
        "limit_down": limit_down,
        "broken_rate": broken_rate,
        "regime": regime,
        "technical_state": state,
    }


def build_sector_table(
    stocks: pd.DataFrame,
    sector_flow: pd.DataFrame,
    events: pd.DataFrame,
    basics: pd.DataFrame,
    previous_date: str,
    market: dict[str, object],
) -> pd.DataFrame:
    universe = stocks.merge(
        basics[["ts_code", "industry"]], on="ts_code", how="left"
    )
    universe["sector"] = (
        universe["industry"]
        .fillna("")
        .astype(str)
        .map(canonical_sector_name)
    )
    grouped = (
        universe.groupby("sector", dropna=False)
        .agg(
            stocks=("ts_code", "nunique"),
            return_pct=("pct_chg", "mean"),
            breadth_pct=(
                "pct_chg",
                lambda value: (value > 0).mean() * 100,
            ),
            amount_yi=(
                "amount",
                lambda value: value.sum() / 100_000_000.0,
            ),
        )
        .reset_index()
    )
    grouped = grouped[grouped["sector"].astype(bool)].copy()
    grouped["relative_market"] = (
        grouped["return_pct"] - float(market["equal_return"])
    )
    grouped = grouped.merge(
        previous_relative_returns(previous_date),
        on="sector",
        how="left",
    )
    if not sector_flow.empty:
        normalized_flow = sector_flow[
            [
                "canonical_sector",
                "net_inflow_amount",
                "net_inflow_ratio",
                "source",
            ]
        ].rename(
            columns={
                "canonical_sector": "sector",
                "net_inflow_amount": "flow_amount",
                "net_inflow_ratio": "flow_ratio",
            }
        )
        grouped = grouped.merge(
            normalized_flow.drop_duplicates("sector"),
            on="sector",
            how="left",
        )
    else:
        grouped["flow_amount"] = np.nan
        grouped["flow_ratio"] = np.nan
        grouped["source"] = ""

    if not events.empty:
        event_rows = events.merge(
            basics[["ts_code", "industry"]],
            on="ts_code",
            how="left",
            suffixes=("_provider", "_local"),
        )
        event_rows["sector"] = (
            event_rows["industry_local"]
            .fillna(event_rows["industry_provider"])
            .fillna("")
            .astype(str)
            .map(canonical_sector_name)
        )
        event_counts = (
            event_rows.pivot_table(
                index="sector",
                columns="event_type",
                values="ts_code",
                aggfunc="nunique",
                fill_value=0,
            )
            .reset_index()
        )
        grouped = grouped.merge(
            event_counts, on="sector", how="left"
        )
    for column in ["limit_up", "broken_board", "limit_down"]:
        if column not in grouped:
            grouped[column] = 0
        grouped[column] = grouped[column].fillna(0).astype(int)
    grouped[["rel3", "rel5"]] = grouped[
        ["rel3", "rel5"]
    ].fillna(0)
    grouped["flow_yi"] = (
        grouped["flow_amount"] / 100_000_000.0
    )
    amount_anchor = grouped["amount_yi"].quantile(0.8)
    if not amount_anchor or not np.isfinite(amount_anchor):
        amount_anchor = 1.0
    grouped["score"] = (
        np.clip((grouped["relative_market"] + 1) / 3, 0, 1) * 25
        + np.clip((grouped["breadth_pct"] - 30) / 35, 0, 1) * 20
        + np.clip(
            (grouped["flow_ratio"].fillna(0) + 5) / 10, 0, 1
        )
        * 20
        + np.clip((grouped["rel3"] + 2) / 6, 0, 1) * 10
        + np.clip((grouped["rel5"] + 3) / 8, 0, 1) * 10
        + np.clip(
            grouped["limit_up"]
            / np.maximum(grouped["stocks"] * 0.03, 1),
            0,
            1,
        )
        * 10
        + np.clip(grouped["amount_yi"] / amount_anchor, 0, 1) * 5
    ).round(1)

    def state(row: pd.Series) -> str:
        flow_ratio = (
            float(row.flow_ratio)
            if pd.notna(row.flow_ratio)
            else 0.0
        )
        if row.relative_market <= -0.5 or row.breadth_pct < 35:
            return "退潮回避"
        if (
            row.relative_market >= 0.8
            and row.breadth_pct >= 55
            and flow_ratio > 0
        ):
            return "上午强势延续"
        if (
            row.relative_market >= 0.3
            and row.breadth_pct >= 50
            and row.rel3 < 0
            and flow_ratio > 0
        ):
            return "午后回流观察"
        if row.return_pct > 0 and (
            row.breadth_pct < 45 or flow_ratio <= -3
        ):
            return "资金背离"
        if row.relative_market > 0 and row.breadth_pct >= 48:
            return "相对强势"
        if row.return_pct > 0:
            return "冲高放缓"
        return "震荡观察"

    grouped["state"] = grouped.apply(state, axis=1)
    return grouped.sort_values(
        ["score", "relative_market"], ascending=False
    ).reset_index(drop=True)


def fetch_sector_minutes(
    sectors: pd.DataFrame,
    stocks: pd.DataFrame,
    basics: pd.DataFrame,
    trade_date: str,
    max_sectors: int,
    members_per_sector: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    if sectors.empty or max_sectors <= 0:
        return {}, {}
    selected = sectors[
        sectors["state"].isin(
            {"上午强势延续", "午后回流观察", "相对强势"}
        )
    ].head(max_sectors)
    universe = stocks.merge(
        basics[["ts_code", "industry"]], on="ts_code", how="left"
    )
    universe["sector"] = (
        universe["industry"]
        .fillna("")
        .astype(str)
        .map(canonical_sector_name)
    )
    memberships: dict[str, pd.DataFrame] = {}
    codes: set[str] = set()
    for sector in selected["sector"]:
        members = universe[universe["sector"] == sector].nlargest(
            members_per_sector, "amount"
        )
        members = members[["ts_code", "amount"]].rename(
            columns={"amount": "latest_amount"}
        )
        memberships[str(sector)] = members
        codes.update(members["ts_code"].astype(str))
    if not codes:
        return {}, {}

    def fetch_one(ts_code: str) -> tuple[str, pd.DataFrame]:
        import akshare as ak

        if ts_code.endswith(".SH"):
            prefix = "sh"
        elif ts_code.endswith(".SZ"):
            prefix = "sz"
        else:
            prefix = "bj"
        raw = ak.stock_zh_a_minute(
            symbol=prefix + ts_code[:6],
            period="60",
            adjust="qfq",
        )
        return ts_code, normalize_stock_minutes(
            raw, ts_code, trade_date
        )

    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(fetch_one, code): code for code in codes
        }
        for future in as_completed(futures):
            try:
                code, frame = future.result()
                if not frame.empty:
                    frames[code] = frame
            except Exception:
                continue
    all_minutes = (
        pd.concat(frames.values(), ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    stock_states: dict[str, str] = {}
    for code, frame in frames.items():
        if len(frame) >= 30:
            stock_states[code] = classify_latest(add_macd(frame))
    sector_bars: dict[str, pd.DataFrame] = {}
    for sector, members in memberships.items():
        if all_minutes.empty:
            continue
        bar = build_sector_bar(
            sector,
            members,
            all_minutes,
            min_members=max(2, min(4, len(members))),
        )
        if not bar.empty:
            sector_bars[sector] = bar
    return sector_bars, stock_states


def load_previous_factors(previous_date: str) -> pd.DataFrame:
    with duckdb.connect(str(MARKET_DB), read_only=True) as con:
        return con.execute(
            """
            select p.ts_code, b.name as basic_name, b.list_date,
                   t.ma5, t.ma10, t.ma20, t.rsi6, t.kdj_j,
                   t.bias5, t.bias20, s.return_5d,
                   r.severe_risk, r.risk_reasons
            from daily_price p join stock_basic b using(ts_code)
            left join factor_technical t using(ts_code, trade_date)
            left join factor_stage s using(ts_code, trade_date)
            left join factor_risk r using(ts_code, trade_date)
            where p.trade_date = ?
            """,
            [previous_date],
        ).fetchdf()


def build_candidates(
    stocks: pd.DataFrame,
    flow: pd.DataFrame,
    events: pd.DataFrame,
    sectors: pd.DataFrame,
    previous_date: str,
    market: dict[str, object],
    quality: dict[str, object],
    stock_states: dict[str, str],
    trade_date: str,
) -> pd.DataFrame:
    columns = [
        "classification",
        "ts_code",
        "name",
        "sector",
        "sector_state",
        "score",
        "last",
        "pct_chg",
        "relative_sector",
        "close_position",
        "amount_yi",
        "flow_yi",
        "flow_ratio",
        "rsi6",
        "kdj_j",
        "minute_state",
        "support",
        "trigger",
        "plan",
    ]
    if not quality["can_select"]:
        return pd.DataFrame(columns=columns)
    universe = stocks.merge(
        load_previous_factors(previous_date),
        on="ts_code",
        how="left",
    )
    basics = load_basic()[["ts_code", "industry"]]
    universe = universe.merge(basics, on="ts_code", how="left")
    universe["sector"] = (
        universe["industry"]
        .fillna("")
        .astype(str)
        .map(canonical_sector_name)
    )
    sector_fields = sectors[
        ["sector", "state", "score", "return_pct"]
    ].rename(
        columns={
            "state": "sector_state",
            "score": "sector_score",
            "return_pct": "sector_return",
        }
    )
    universe = universe.merge(
        sector_fields, on="sector", how="left"
    )
    if not flow.empty:
        universe = universe.merge(
            flow[
                [
                    "ts_code",
                    "net_inflow_amount",
                    "net_inflow_ratio",
                    "flow_metric",
                ]
            ],
            on="ts_code",
            how="left",
        )
    else:
        universe["net_inflow_amount"] = np.nan
        universe["net_inflow_ratio"] = np.nan
        universe["flow_metric"] = ""
    event_map = (
        events[["ts_code", "event_type"]].drop_duplicates("ts_code")
        if not events.empty
        else pd.DataFrame(columns=["ts_code", "event_type"])
    )
    universe = universe.merge(event_map, on="ts_code", how="left")
    spread = universe["high"] - universe["low"]
    universe["close_position"] = np.where(
        spread > 0,
        (universe["last"] - universe["low"]) / spread,
        0.5,
    )
    universe["relative_sector"] = (
        universe["pct_chg"] - universe["sector_return"]
    )
    universe["amount_yi"] = universe["amount"] / 100_000_000.0
    universe["flow_yi"] = (
        universe["net_inflow_amount"] / 100_000_000.0
    )
    universe["flow_ratio"] = universe["net_inflow_ratio"]
    universe["minute_state"] = (
        universe["ts_code"].map(stock_states).fillna("分钟数据不足")
    )
    list_cutoff = (
        pd.to_datetime(trade_date) - pd.Timedelta(days=90)
    ).strftime("%Y%m%d")
    names = universe["name"].fillna(universe["basic_name"]).astype(str)
    universe = universe[~names.str.contains("ST|退", regex=True)]
    accepted_states = {
        "上午强势延续",
        "午后回流观察",
        "相对强势",
    }
    universe = universe[
        (universe["list_date"] <= list_cutoff)
        & universe["pct_chg"].between(-1, 5)
        & (universe["amount_yi"] >= 0.5)
        & (universe["close_position"] >= 0.7)
        & (universe["sector_score"] >= 55)
        & universe["sector_state"].isin(accepted_states)
        & (universe["relative_sector"] > 0)
        & (universe["last"] >= universe["ma20"] * 0.98)
        & (
            universe["last"]
            >= universe[["ma5", "ma10"]].min(axis=1)
        )
        & universe["rsi6"].between(35, 72)
        & (universe["kdj_j"] < 90)
        & (universe["return_5d"] <= 12)
        & (universe["net_inflow_amount"] > 0)
        & ~universe["severe_risk"].fillna(False)
        & ~universe["event_type"]
        .fillna("")
        .isin(["limit_up", "broken_board", "limit_down"])
    ].copy()
    if universe.empty:
        return pd.DataFrame(columns=columns)
    minute_points = np.where(
        universe["minute_state"].isin(["强势延续", "反弹确认"]),
        12,
        np.where(universe["minute_state"] == "抄底观察", 8, 3),
    )
    universe["score"] = (
        universe["sector_score"] * 0.3
        + np.clip(
            (universe["relative_sector"] + 0.5) / 3, 0, 1
        )
        * 18
        + np.clip(universe["close_position"], 0, 1) * 12
        + np.clip((universe["flow_ratio"] + 2) / 8, 0, 1) * 15
        + minute_points
        + np.where(universe["last"] >= universe["ma5"], 7, 3)
        + np.where(universe["kdj_j"] < 80, 6, 2)
    ).round(1)
    universe = universe[universe["score"] >= 65].copy()
    if universe.empty:
        return pd.DataFrame(columns=columns)
    universe["classification"] = np.where(
        market["regime"] == "上午退潮",
        "逆势观察",
        "午后条件观察",
    )
    universe["name"] = universe["name"].fillna(
        universe["basic_name"]
    )
    universe["support"] = (
        universe[["ma5", "ma10", "ma20"]]
        .where(
            universe[["ma5", "ma10", "ma20"]].le(
                universe["last"], axis=0
            )
        )
        .max(axis=1)
        .round(2)
    )
    universe["trigger"] = universe["high"].round(2)
    universe["plan"] = universe.apply(
        lambda row: (
            f"午后不追高；站稳{row.trigger:.2f}且板块广度不回落时观察，"
            f"跌破{row.support:.2f}取消"
        ),
        axis=1,
    )
    return universe.nlargest(5, "score")[columns]


def ensure_midday_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        create table if not exists midday_run (
            snapshot_id varchar, trade_date varchar, captured_at timestamp,
            provider_time varchar, grade varchar, can_select boolean,
            market_regime varchar, market_state varchar,
            report_path varchar, quality_json varchar
        )
        """
    )
    con.execute(
        """
        create table if not exists midday_market_snapshot (
            snapshot_id varchar, trade_date varchar,
            equal_return double, breadth double, amount_yi double,
            flow_yi double, flow_ratio double, limit_up integer,
            broken integer, limit_down integer, broken_rate double,
            regime varchar, technical_state varchar
        )
        """
    )
    con.execute(
        """
        create table if not exists midday_stock_snapshot (
            snapshot_id varchar, trade_date varchar, ts_code varchar,
            name varchar, last double, pct_chg double, amount double,
            volume double, amplitude double, open double, high double,
            low double, pre_close double, provider_time varchar,
            source varchar
        )
        """
    )
    con.execute(
        """
        create table if not exists midday_sector_snapshot (
            snapshot_id varchar, trade_date varchar, sector varchar,
            stocks integer, return_pct double, breadth_pct double,
            amount_yi double, relative_market double, rel3 double,
            rel5 double, flow_yi double, flow_ratio double,
            limit_up integer, broken_board integer, limit_down integer,
            score double, state varchar
        )
        """
    )
    con.execute(
        """
        create table if not exists midday_limit_snapshot (
            snapshot_id varchar, trade_date varchar, ts_code varchar,
            name varchar, event_type varchar, pct_chg double,
            close double, open_count integer, consecutive_boards integer,
            industry varchar, source varchar
        )
        """
    )
    con.execute(
        """
        create table if not exists midday_bar_60min (
            snapshot_id varchar, instrument varchar,
            trade_time timestamp, trade_date varchar,
            open double, high double, low double, close double,
            vol double, amount double, freq varchar, source varchar
        )
        """
    )


def _insert_frame(
    con: duckdb.DuckDBPyConnection,
    table: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return
    con.register("incoming_frame", frame)
    con.execute(
        f"insert into {table} by name select * from incoming_frame"
    )
    con.unregister("incoming_frame")


def persist_snapshot(
    snapshot_id: str,
    trade_date: str,
    captured_at: datetime,
    provider_time: str | None,
    quality: dict[str, object],
    market: dict[str, object],
    stocks: pd.DataFrame,
    sectors: pd.DataFrame,
    events: pd.DataFrame,
    bars: pd.DataFrame,
    report_path: Path,
) -> None:
    with duckdb.connect(str(MIDDAY_DB)) as con:
        ensure_midday_db(con)
        con.execute("begin transaction")
        try:
            run = pd.DataFrame(
                [
                    {
                        "snapshot_id": snapshot_id,
                        "trade_date": trade_date,
                        "captured_at": captured_at.replace(tzinfo=None),
                        "provider_time": provider_time or "",
                        "grade": quality["grade"],
                        "can_select": quality["can_select"],
                        "market_regime": market["regime"],
                        "market_state": market["technical_state"],
                        "report_path": str(report_path),
                        "quality_json": json.dumps(
                            quality,
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                ]
            )
            market_frame = pd.DataFrame(
                [
                    {
                        "snapshot_id": snapshot_id,
                        "trade_date": trade_date,
                        **market,
                    }
                ]
            )
            stock_columns = [
                "ts_code",
                "name",
                "last",
                "pct_chg",
                "amount",
                "volume",
                "amplitude",
                "open",
                "high",
                "low",
                "pre_close",
            ]
            stock_frame = stocks[stock_columns].copy()
            stock_frame.insert(0, "trade_date", trade_date)
            stock_frame.insert(0, "snapshot_id", snapshot_id)
            stock_frame["provider_time"] = provider_time or ""
            stock_frame["source"] = "tencent+sina_via_akshare"
            sector_columns = [
                "sector",
                "stocks",
                "return_pct",
                "breadth_pct",
                "amount_yi",
                "relative_market",
                "rel3",
                "rel5",
                "flow_yi",
                "flow_ratio",
                "limit_up",
                "broken_board",
                "limit_down",
                "score",
                "state",
            ]
            sector_frame = sectors[sector_columns].copy()
            sector_frame.insert(0, "trade_date", trade_date)
            sector_frame.insert(0, "snapshot_id", snapshot_id)
            event_columns = [
                "ts_code",
                "name",
                "event_type",
                "pct_chg",
                "close",
                "open_count",
                "consecutive_boards",
                "industry",
                "source",
            ]
            event_frame = (
                events[event_columns].copy()
                if not events.empty
                else pd.DataFrame(columns=event_columns)
            )
            event_frame.insert(0, "trade_date", trade_date)
            event_frame.insert(0, "snapshot_id", snapshot_id)
            bar_frame = bars.copy()
            bar_frame.insert(0, "snapshot_id", snapshot_id)
            _insert_frame(con, "midday_run", run)
            _insert_frame(
                con, "midday_market_snapshot", market_frame
            )
            _insert_frame(
                con, "midday_stock_snapshot", stock_frame
            )
            _insert_frame(
                con, "midday_sector_snapshot", sector_frame
            )
            _insert_frame(
                con, "midday_limit_snapshot", event_frame
            )
            _insert_frame(con, "midday_bar_60min", bar_frame)
            con.execute("commit")
        except Exception:
            con.execute("rollback")
            raise


def chart_data_url(frame: pd.DataFrame, title: str) -> str:
    if frame.empty or len(frame) < 10:
        return ""
    data = add_macd(frame).tail(240).reset_index(drop=True)
    x = np.arange(len(data))
    fig, (ax_price, ax_macd) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for index, row in data.iterrows():
        color = "#d84a3a" if row.close >= row.open else "#1f9d74"
        ax_price.vlines(
            index, row.low, row.high, color=color, linewidth=0.8
        )
        lower = min(row.open, row.close)
        height = max(
            abs(row.close - row.open),
            max(abs(row.close) * 0.0003, 0.001),
        )
        ax_price.add_patch(
            plt.Rectangle(
                (index - 0.3, lower),
                0.6,
                height,
                color=color,
                alpha=0.85,
            )
        )
    ax_price.plot(
        x, data["ma5"], label="MA5", color="#e5a13a", linewidth=1
    )
    ax_price.plot(
        x, data["ma10"], label="MA10", color="#4b82d0", linewidth=1
    )
    ax_price.plot(
        x, data["ma20"], label="MA20", color="#8b5fbf", linewidth=1
    )
    ax_price.set_title(title)
    ax_price.legend(loc="upper left", ncol=3)
    ax_price.grid(alpha=0.18)
    colors = np.where(
        data["macd"] >= 0, "#d84a3a", "#1f9d74"
    )
    ax_macd.bar(x, data["macd"], color=colors, width=0.65)
    ax_macd.plot(
        x, data["dif"], color="#e5a13a", linewidth=1, label="DIF"
    )
    ax_macd.plot(
        x, data["dea"], color="#4b82d0", linewidth=1, label="DEA"
    )
    ax_macd.axhline(0, color="#777", linewidth=0.7)
    ax_macd.legend(loc="upper left", ncol=2)
    tick_indices = np.linspace(
        0, len(data) - 1, min(8, len(data)), dtype=int
    )
    ax_macd.set_xticks(tick_indices)
    ax_macd.set_xticklabels(
        pd.to_datetime(
            data.loc[tick_indices, "trade_time"]
        ).dt.strftime("%m-%d\n%H:%M")
    )
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(
        buffer, format="png", dpi=135, bbox_inches="tight"
    )
    plt.close(fig)
    return (
        "data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode("ascii")
    )


def _fmt(
    value: object, digits: int = 2, suffix: str = ""
) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "无数据"
    if not np.isfinite(number):
        return "无数据"
    return f"{number:.{digits}f}{suffix}"


def html_table(
    frame: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
) -> str:
    if frame.empty:
        return '<div class="empty">当前没有满足条件的数据。</div>'
    return (
        frame[columns]
        .rename(columns=labels)
        .to_html(
            index=False,
            escape=True,
            classes="data-table",
            border=0,
        )
    )


def render_html(
    trade_date: str,
    previous_date: str,
    captured_at: datetime,
    provider_time: str | None,
    quality: dict[str, object],
    market: dict[str, object],
    sectors: pd.DataFrame,
    candidates: pd.DataFrame,
    index_bars: pd.DataFrame,
    sector_bars: dict[str, pd.DataFrame],
) -> str:
    quality_reasons = (
        "；".join(str(item) for item in quality["reasons"])
        or "核心数据、跨源校验和时间校验均通过"
    )
    index_chart = chart_data_url(
        index_bars,
        "上证指数 60分钟（午间仅含10:30、11:30完整K线）",
    )
    sector_sections = []
    for sector, bars in sector_bars.items():
        url = chart_data_url(
            bars, f"{sector} 合成60分钟K线"
        )
        if url:
            sector_sections.append(
                '<article class="chart-card">'
                f"<h3>{html.escape(sector)}</h3>"
                f'<img src="{url}" alt="{html.escape(sector)} 60分钟K线">'
                "</article>"
            )
    top_sectors = sectors.head(12).copy()
    for column in [
        "return_pct",
        "breadth_pct",
        "relative_market",
        "rel3",
        "rel5",
        "flow_yi",
        "flow_ratio",
        "score",
    ]:
        top_sectors[column] = top_sectors[column].map(_fmt)
    candidate_display = candidates.copy()
    for column in [
        "score",
        "last",
        "pct_chg",
        "relative_sector",
        "close_position",
        "amount_yi",
        "flow_yi",
        "flow_ratio",
        "rsi6",
        "kdj_j",
        "support",
        "trigger",
    ]:
        if column in candidate_display:
            candidate_display[column] = (
                candidate_display[column].map(_fmt)
            )
    if market["regime"] == "上午退潮":
        afternoon_plan = (
            "上午处于退潮：下午关闭追涨和首板接力，只观察指数"
            "是否收回上午均价、上涨家数占比是否回到40%以上。"
        )
    elif market["regime"] == "上午强势":
        afternoon_plan = (
            "上午广度较好：下午只观察强板块回踩确认，"
            "不追高开或已经大幅加速的个股。"
        )
    else:
        afternoon_plan = (
            "上午分化：等待13:30后板块广度和资金同向，"
            "无法确认时保持观望。"
        )
    selection_note = (
        "数据质量允许生成条件观察池；午间结果不改变日线模拟盘订单。"
        if quality["can_select"]
        else (
            "数据质量为C级，观察池已强制清空；"
            "本页仅用于数据诊断和盘面描述。"
        )
    )
    chart_html = (
        f'<img class="main-chart" src="{index_chart}" '
        'alt="上证指数60分钟K线">'
        if index_chart
        else '<div class="empty">指数60分钟数据不足。</div>'
    )
    sector_chart_html = (
        "".join(sector_sections)
        or (
            '<div class="empty">'
            "当前没有通过质量门槛的重点板块60分钟图。</div>"
        )
    )
    sector_table = html_table(
        top_sectors,
        [
            "sector",
            "state",
            "score",
            "return_pct",
            "relative_market",
            "breadth_pct",
            "flow_yi",
            "flow_ratio",
            "rel3",
            "rel5",
            "limit_up",
            "broken_board",
        ],
        {
            "sector": "板块",
            "state": "上午状态",
            "score": "评分",
            "return_pct": "上午涨幅%",
            "relative_market": "相对全A%",
            "breadth_pct": "上涨家数%",
            "flow_yi": "资金净额亿",
            "flow_ratio": "资金净占比%",
            "rel3": "昨日基线近3日相对%",
            "rel5": "昨日基线近5日相对%",
            "limit_up": "涨停",
            "broken_board": "炸板",
        },
    )
    candidate_table = html_table(
        candidate_display,
        [
            "classification",
            "ts_code",
            "name",
            "sector",
            "sector_state",
            "score",
            "last",
            "pct_chg",
            "relative_sector",
            "close_position",
            "amount_yi",
            "flow_yi",
            "flow_ratio",
            "rsi6",
            "kdj_j",
            "minute_state",
            "support",
            "trigger",
            "plan",
        ],
        {
            "classification": "分类",
            "ts_code": "代码",
            "name": "名称",
            "sector": "板块",
            "sector_state": "板块状态",
            "score": "评分",
            "last": "午间价",
            "pct_chg": "上午涨幅%",
            "relative_sector": "相对板块%",
            "close_position": "位置系数",
            "amount_yi": "成交额亿",
            "flow_yi": "资金净额亿",
            "flow_ratio": "资金净占比%",
            "rsi6": "昨日RSI6",
            "kdj_j": "昨日J值",
            "minute_state": "60分钟状态",
            "support": "支撑",
            "trigger": "确认位",
            "plan": "下午预案",
        },
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股午间复盘 {trade_date}</title><style>
body{{margin:0;background:#f4f6f8;color:#1f2937;font-family:"Microsoft YaHei",Arial,sans-serif}}
.wrap{{max-width:1450px;margin:auto;padding:24px}}
h1{{margin:0 0 8px}} h2{{margin-top:0}} .sub{{color:#64748b}}
.banner{{background:#172033;color:white;padding:24px;border-radius:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:18px}}
.metric,.section,.chart-card{{background:white;border-radius:12px;padding:16px;box-shadow:0 2px 10px #0f172a12}}
.metric strong{{font-size:24px;display:block;margin-top:6px}}
.section{{margin-top:18px;overflow:auto}}
.quality-A{{color:#4ade80}} .quality-B{{color:#facc15}} .quality-C{{color:#f87171}}
.warning{{background:#fff7ed;border-left:4px solid #f97316;padding:12px}}
.empty{{padding:24px;color:#64748b;text-align:center}}
.data-table{{width:100%;border-collapse:collapse;font-size:14px}}
.data-table th{{background:#f0b429;color:#111827}}
.data-table th,.data-table td{{padding:9px;border:1px solid #d6dbe1;text-align:center;white-space:nowrap}}
.main-chart,.chart-card img{{width:100%;height:auto}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}}
</style></head><body><main class="wrap">
<section class="banner"><h1>A股午间复盘</h1>
<div>数据日期 {trade_date} ｜ 抓取时间 {captured_at:%Y-%m-%d %H:%M:%S} ｜ 昨日基线 {previous_date}</div>
<div class="grid">
<div class="metric">上午状态<strong>{market['regime']}</strong></div>
<div class="metric">60分钟状态<strong>{market['technical_state']}</strong></div>
<div class="metric">全A等权<strong>{_fmt(market['equal_return'],2,'%')}</strong></div>
<div class="metric">上涨家数占比<strong>{_fmt(market['breadth'],2,'%')}</strong></div>
<div class="metric">上午成交额<strong>{_fmt(market['amount_yi'],0,'亿')}</strong></div>
<div class="metric">质量等级<strong class="quality-{quality['grade']}">{quality['grade']}</strong></div>
</div></section>
<section class="section"><h2>一、数据质量</h2>
<p>{html.escape(quality_reasons)}</p>
<p>供应商时间：{html.escape(provider_time or '无')}；
删除未来或未完成60分钟K线 {quality['future_rows_dropped']} 条。</p>
<p>行情/资金重合 {quality['flow_validation']['overlap']} 只，
相关系数 {_fmt(quality['flow_validation']['correlation'],4)}，
中位误差 {_fmt(quality['flow_validation']['median_diff'],2,'个百分点')}，
1个百分点内占比 {_fmt(quality['flow_validation']['within_1pp'],1,'%')}。</p>
<p class="warning">{selection_note}</p></section>
<section class="section"><h2>二、上午大盘与情绪</h2>
<p>涨停 {market['limit_up']} 只，炸板 {market['broken']} 只，
跌停 {market['limit_down']} 只，炸板率 {_fmt(market['broken_rate'],1,'%')}；
AKShare同花顺总资金净额 {_fmt(market['flow_yi'],2,'亿')}。
该口径不等同于可验证的机构主力资金。</p>{chart_html}</section>
<section class="section"><h2>三、板块排序</h2>{sector_table}</section>
<section class="section"><h2>四、重点板块60分钟图</h2>
<p>板块K线由当前成交额靠前成分股合成，不是官方行业指数。</p>
<div class="charts">{sector_chart_html}</div></section>
<section class="section"><h2>五、下午条件观察池</h2>
{candidate_table}</section>
<section class="section"><h2>六、下午预案</h2>
<p>{afternoon_plan}</p>
<p>午间报告仅是未收盘快照，不构成投资建议，不执行真实交易，
也不修改日线模拟盘账本。</p></section>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    captured_at = shanghai_now()
    trade_date = captured_at.strftime("%Y%m%d")
    if not args.probe and not in_midday_window(captured_at):
        raise SystemExit(
            "Midday report only runs from 11:35 to 12:55 "
            f"Asia/Shanghai; now={captured_at:%H:%M:%S}"
        )
    if not current_trade_day_is_open(trade_date):
        raise SystemExit(
            f"{trade_date} is not an open A-share trading day"
        )
    previous_date = latest_completed_date(trade_date)
    live = collect_live(trade_date)
    quality = quality_assessment(live, captured_at, trade_date)
    spot = live["spot"]
    anchor = live["anchor"]
    stock_flow = live["stock_flow"]
    sector_flow = live["sector_flow"]
    events = live["events"]
    index_bars = live["index_bars"]
    assert isinstance(spot, pd.DataFrame)
    assert isinstance(anchor, pd.DataFrame)
    assert isinstance(stock_flow, pd.DataFrame)
    assert isinstance(sector_flow, pd.DataFrame)
    assert isinstance(events, pd.DataFrame)
    assert isinstance(index_bars, pd.DataFrame)
    print(f"trade_date={trade_date}")
    print(f"captured_at={captured_at.isoformat()}")
    print(
        f"spot_rows={len(spot)} flow_rows={len(stock_flow)} "
        f"sector_flow_rows={len(sector_flow)}"
    )
    print(
        f"quality={quality['grade']} "
        f"can_select={quality['can_select']}"
    )
    print(
        f"provider_time={live['provider_time']} "
        f"delta_minutes={quality['provider_time_delta']}"
    )
    print(
        "flow_validation="
        + json.dumps(
            quality["flow_validation"], ensure_ascii=False
        )
    )
    if args.probe:
        for error in live["errors"]:
            print(f"warning={error}")
        return

    stocks = spot.merge(anchor, on="ts_code", how="left")
    basics = load_basic()
    market = market_summary(
        stocks, stock_flow, events, index_bars
    )
    sectors = build_sector_table(
        stocks,
        sector_flow,
        events,
        basics,
        previous_date,
        market,
    )
    sector_bars: dict[str, pd.DataFrame] = {}
    stock_states: dict[str, str] = {}
    if quality["can_select"]:
        sector_bars, stock_states = fetch_sector_minutes(
            sectors,
            stocks,
            basics,
            trade_date,
            args.max_sector_charts,
            args.members_per_sector,
        )
    candidates = build_candidates(
        stocks,
        stock_flow,
        events,
        sectors,
        previous_date,
        market,
        quality,
        stock_states,
        trade_date,
    )
    output_dir = OUTPUT_ROOT / trade_date
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_id = captured_at.strftime("%Y%m%d_%H%M%S")
    report_path = (
        output_dir
        / f"a_share_midday_review_{snapshot_id}.html"
    )
    report_path.write_text(
        render_html(
            trade_date,
            previous_date,
            captured_at,
            live["provider_time"],
            quality,
            market,
            sectors,
            candidates,
            index_bars,
            sector_bars,
        ),
        encoding="utf-8",
    )
    bars_to_store = index_bars.copy()
    for _, frame in sector_bars.items():
        item = frame.rename(columns={"sector": "instrument"})
        bars_to_store = pd.concat(
            [bars_to_store, item[bars_to_store.columns]],
            ignore_index=True,
        )
    persist_snapshot(
        snapshot_id,
        trade_date,
        captured_at,
        live["provider_time"],
        quality,
        market,
        stocks,
        sectors,
        events,
        bars_to_store,
        report_path,
    )
    sectors.to_csv(
        output_dir / f"sector_snapshot_{snapshot_id}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidates.to_csv(
        output_dir / f"stock_watchlist_{snapshot_id}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"market_regime={market['regime']}")
    print(f"market_state={market['technical_state']}")
    print(
        f"sectors={len(sectors)} candidates={len(candidates)}"
    )
    print(f"html={report_path}")
    print(f"db={MIDDAY_DB}")


if __name__ == "__main__":
    main()
