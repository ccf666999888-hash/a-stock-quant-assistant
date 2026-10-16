from __future__ import annotations

import argparse
import base64
import html
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "a_stock_quant.duckdb"
OUTPUT_ROOT = ROOT / "outputs" / "after_close"
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")
CHART_LOOKBACK_DAYS = 60

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data.akshare_client import canonical_sector_name  # noqa: E402
from plot_sector_60m_macd import (  # noqa: E402
    SECTOR_ALIASES,
    add_macd,
    classify_latest,
)


POSITIVE_SECTOR_STATES = {"强势延续", "反弹确认", "抄底观察"}
CONDITIONAL_SECTOR_STATES = POSITIVE_SECTOR_STATES | {"强势放缓", "震荡观察"}
POSITIVE_STOCK_STATES = {"强势延续", "反弹确认", "抄底观察", "强势放缓"}
RISK_STATES = {"逃顶预警", "弱势回避", "下跌延续"}

STATE_POINTS = {
    "强势延续": 20,
    "反弹确认": 17,
    "抄底观察": 14,
    "强势放缓": 12,
    "震荡观察": 7,
    "逃顶预警": 3,
    "下跌延续": 1,
    "弱势回避": 0,
}

STATE_OUTLOOK = {
    "强势延续": ("延续偏强", "等待回踩MA5/MA10承接，不在加速高开时追价"),
    "强势放缓": ("分歧风险上升", "观察MACD红柱能否重新放大；跌破MA10则降级"),
    "抄底观察": ("筑底尚未确认", "只观察；需DIF上拐、站回MA10并保持板块广度"),
    "反弹确认": ("反弹可能延续", "回踩MA10/MA20不破且量能正常时继续观察"),
    "逃顶预警": ("转弱风险上升", "停止追涨；反抽不能收回MA5时继续降级"),
    "弱势回避": ("弱势未改", "不新增观察，等待重新站回MA20"),
    "下跌延续": ("下行趋势延续", "回避左侧抢反弹，等待绿柱收缩和DIF上拐"),
    "震荡观察": ("方向不清", "等待放量突破或回踩确认，减少主观预测"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate A-share after-close HTML review")
    parser.add_argument("--trade-date", help="YYYYMMDD; defaults to latest local trade date")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync permitted Tushare core data and AKShare supplements before analysis",
    )
    parser.add_argument("--skip-sector-refresh", action="store_true")
    parser.add_argument("--skip-minute-fetch", action="store_true")
    parser.add_argument("--top-n", type=int, default=10, help="Members per synthetic sector index")
    return parser.parse_args()


def run_script(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    subprocess.run(command, cwd=ROOT, check=True)


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            "select count(*) from information_schema.tables where table_name = ?",
            [table],
        ).fetchone()[0]
    )


def table_max_date(con: duckdb.DuckDBPyConnection, table: str) -> str | None:
    if not table_exists(con, table):
        return None
    columns = {row[0] for row in con.execute(f"describe {table}").fetchall()}
    if "trade_date" not in columns:
        return None
    value = con.execute(f"select max(trade_date) from {table}").fetchone()[0]
    return str(value) if value else None


def resolve_trade_date(con: duckdb.DuckDBPyConnection, requested: str | None) -> str:
    if requested:
        value = con.execute(
            "select max(trade_date) from daily_price where trade_date <= ?",
            [requested],
        ).fetchone()[0]
    else:
        value = con.execute("select max(trade_date) from daily_price").fetchone()[0]
    if not value:
        raise RuntimeError("daily_price has no usable trade date")
    return str(value)


def ensure_factors(trade_date: str) -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        stale = any(
            table_max_date(con, table) != trade_date
            for table in ["factor_technical", "factor_volume_price", "factor_stage", "factor_risk"]
        )
    if stale:
        start_date = (pd.to_datetime(trade_date) - pd.DateOffset(months=7)).strftime("%Y%m%d")
        run_script("calc_factors.py", "--start-date", start_date, "--end-date", trade_date)


def refresh_market_minutes(trade_date: str, skip_fetch: bool) -> None:
    command = [
        "--index-code",
        "000001.SH",
        "--end-date",
        trade_date,
        "--days",
        str(CHART_LOOKBACK_DAYS),
    ]
    if skip_fetch:
        command.append("--skip-sync")
    run_script("plot_index_60m_macd.py", *command)


def refresh_sector_outputs(trade_date: str, top_n: int, skip_fetch: bool) -> Path:
    tracking_path = ROOT / "outputs" / "sector_tracking" / f"sector_tracking_{trade_date}.csv"
    run_script("generate_sector_tracking_table.py", "--trade-date", trade_date)
    command = [
        "--sector-file",
        str(tracking_path),
        "--end-date",
        trade_date,
        "--days",
        str(CHART_LOOKBACK_DAYS),
        "--top-n",
        str(top_n),
        "--extra-sectors",
        "电力",
        "电力设备",
        "旅游",
    ]
    if skip_fetch:
        command.append("--skip-fetch")
    run_script("plot_sector_60m_macd.py", *command)
    return ROOT / "outputs" / "sector_60m" / f"sector_60m_signal_summary_{trade_date}.csv"


def data_quality(con: duckdb.DuckDBPyConnection, trade_date: str) -> tuple[str, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    core_ok = True
    optional_missing = False
    for table, required in [
        ("daily_price", True),
        ("daily_basic", False),
        ("factor_technical", True),
        ("factor_volume_price", True),
        ("factor_stage", True),
        ("index_daily", True),
        ("index_bar_60min", True),
        ("sector_bar_60min", True),
        ("stock_bar_60min", False),
        ("ak_market_fund_flow", False),
        ("ak_sector_fund_flow", False),
        ("ak_stock_fund_flow", False),
        ("ak_limit_event", False),
    ]:
        max_date = table_max_date(con, table)
        count = 0
        if table_exists(con, table):
            if max_date:
                count = con.execute(
                    f"select count(*) from {table} where trade_date = ?", [trade_date]
                ).fetchone()[0]
            else:
                count = con.execute(f"select count(*) from {table}").fetchone()[0]
        current = max_date == trade_date if max_date else count > 0
        if required and not current:
            core_ok = False
        if not required and not current:
            optional_missing = True
        rows.append(
            {
                "数据表": table,
                "最新日期": max_date or "无",
                "目标日记录数": f"{count:,}",
                "状态": "可用" if current else ("缺失/滞后" if required else "可选数据不可用"),
            }
        )
    grade = "A" if core_ok and not optional_missing else "B" if core_ok else "C"
    return grade, rows


def fund_flow_status(
    main_ratio: object,
    super_large_amount: object,
    large_amount: object,
) -> str:
    ratio = pd.to_numeric(pd.Series([main_ratio]), errors="coerce").iloc[0]
    super_large = pd.to_numeric(pd.Series([super_large_amount]), errors="coerce").iloc[0]
    large = pd.to_numeric(pd.Series([large_amount]), errors="coerce").iloc[0]
    if pd.isna(ratio):
        return "数据不足"
    large_orders = (0.0 if pd.isna(super_large) else float(super_large)) + (
        0.0 if pd.isna(large) else float(large)
    )
    if ratio >= 3 and large_orders > 0:
        return "强净流入"
    if ratio > 1:
        return "温和净流入"
    if ratio >= -1:
        return "中性"
    if ratio > -3:
        return "温和净流出"
    return "强净流出"


def market_supplement(
    con: duckdb.DuckDBPyConnection,
    trade_date: str,
) -> dict[str, float | str | bool]:
    result: dict[str, float | str | bool] = {
        "ak_fund_available": False,
        "limit_data_available": False,
        "market_net_inflow_amount": np.nan,
        "market_net_inflow_ratio": np.nan,
        "market_super_large_net_amount": np.nan,
        "market_large_net_amount": np.nan,
        "market_fund_status": "数据不足",
        "market_flow_source": "无数据",
        "market_flow_metric": "无数据",
        "limit_up_count": np.nan,
        "first_board_count": np.nan,
        "multi_board_count": np.nan,
        "broken_board_count": np.nan,
        "limit_down_count": np.nan,
        "max_board_height": np.nan,
        "one_word_count": np.nan,
        "broken_board_rate": np.nan,
        "seal_rate": np.nan,
    }
    if table_exists(con, "ak_market_fund_flow"):
        flow = con.execute(
            "select * from ak_market_fund_flow where trade_date = ?", [trade_date]
        ).fetchdf()
        if not flow.empty:
            row = flow.iloc[-1]
            net_amount = row.net_inflow_amount if pd.notna(row.net_inflow_amount) else row.main_net_amount
            net_ratio = row.net_inflow_ratio if pd.notna(row.net_inflow_ratio) else row.main_net_ratio
            result.update(
                {
                    "ak_fund_available": True,
                    "market_net_inflow_amount": float(net_amount),
                    "market_net_inflow_ratio": float(net_ratio),
                    "market_super_large_net_amount": float(row.super_large_net_amount) if pd.notna(row.super_large_net_amount) else np.nan,
                    "market_large_net_amount": float(row.large_net_amount) if pd.notna(row.large_net_amount) else np.nan,
                    "market_fund_status": fund_flow_status(
                        net_ratio,
                        row.super_large_net_amount,
                        row.large_net_amount,
                    ),
                    "market_flow_source": str(row.source),
                    "market_flow_metric": str(row.flow_metric),
                }
            )
    if table_exists(con, "ak_limit_event"):
        events = con.execute(
            "select * from ak_limit_event where trade_date = ?", [trade_date]
        ).fetchdf()
        if not events.empty:
            limit_up = events[events["event_type"] == "limit_up"]
            broken = events[events["event_type"] == "broken_board"]
            limit_down = events[events["event_type"] == "limit_down"]
            attempts = len(limit_up) + len(broken)
            up_codes = limit_up["ts_code"].astype(str).tolist()
            one_word = 0
            if up_codes:
                one_word = con.execute(
                    """
                    select count(*) from daily_price
                    where trade_date = ? and ts_code in (select * from unnest(?))
                      and abs(high - low) < 0.000001
                    """,
                    [trade_date, up_codes],
                ).fetchone()[0]
            result.update(
                {
                    "limit_data_available": True,
                    "limit_up_count": len(limit_up),
                    "first_board_count": int((limit_up["consecutive_boards"] == 1).sum()),
                    "multi_board_count": int((limit_up["consecutive_boards"] >= 2).sum()),
                    "broken_board_count": len(broken),
                    "limit_down_count": len(limit_down),
                    "max_board_height": int(limit_up["consecutive_boards"].max()) if len(limit_up) else 0,
                    "one_word_count": int(one_word),
                    "broken_board_rate": len(broken) / attempts * 100 if attempts else 0.0,
                    "seal_rate": len(limit_up) / attempts * 100 if attempts else 0.0,
                }
            )
    return result


def market_emotion_history(
    con: duckdb.DuckDBPyConnection,
    trade_date: str,
) -> pd.DataFrame:
    if not table_exists(con, "ak_limit_event"):
        return pd.DataFrame()
    frame = con.execute(
        """
        select trade_date,
               sum(case when event_type = 'limit_up' then 1 else 0 end) as "涨停家数",
               sum(case when event_type = 'limit_up' and consecutive_boards = 1 then 1 else 0 end) as "首板家数",
               sum(case when event_type = 'broken_board' then 1 else 0 end) as "炸板家数",
               sum(case when event_type = 'limit_down' then 1 else 0 end) as "跌停家数",
               max(case when event_type = 'limit_up' then consecutive_boards else 0 end) as "最高连板"
        from ak_limit_event where trade_date <= ?
        group by trade_date order by trade_date desc limit 10
        """,
        [trade_date],
    ).fetchdf()
    if frame.empty:
        return frame
    frame["封板率%"] = frame["涨停家数"] / (frame["涨停家数"] + frame["炸板家数"]).replace(0, np.nan) * 100
    frame = frame.rename(columns={"trade_date": "交易日"}).sort_values("交易日", ascending=False)
    return frame


def market_snapshot(con: duckdb.DuckDBPyConnection, trade_date: str) -> dict[str, float | str]:
    dates = con.execute(
        "select distinct trade_date from daily_price where trade_date <= ? order by trade_date desc limit 2",
        [trade_date],
    ).fetchdf()["trade_date"].astype(str).tolist()
    current = con.execute(
        """
        select avg(pct_chg) as equal_pct,
               median(pct_chg) as median_pct,
               avg(case when pct_chg > 0 then 1.0 else 0.0 end) * 100 as up_ratio,
               sum(amount) / 100000.0 as amount_yi,
               count(*) as stocks
        from daily_price where trade_date = ?
        """,
        [trade_date],
    ).fetchdf().iloc[0]
    previous_amount = np.nan
    if len(dates) > 1:
        previous_amount = con.execute(
            "select sum(amount) / 100000.0 from daily_price where trade_date = ?",
            [dates[1]],
        ).fetchone()[0]
    index_row = con.execute(
        """
        select pct_chg, close, amount / 100000.0 as amount_yi
        from index_daily where ts_code = '000001.SH' and trade_date = ?
        """,
        [trade_date],
    ).fetchdf()
    index_pct = float(index_row.iloc[0]["pct_chg"]) if not index_row.empty else np.nan
    index_close = float(index_row.iloc[0]["close"]) if not index_row.empty else np.nan
    amount_change = (
        (float(current["amount_yi"]) / float(previous_amount) - 1) * 100
        if pd.notna(previous_amount) and previous_amount
        else np.nan
    )
    equal_pct = float(current["equal_pct"])
    up_ratio = float(current["up_ratio"])
    supplement = market_supplement(con, trade_date)
    emotion_retreat = bool(
        supplement["limit_data_available"]
        and (
            (
                float(supplement["broken_board_rate"]) >= 40
                and float(supplement["limit_down_count"]) >= 10
            )
            or float(supplement["limit_down_count"]) > float(supplement["limit_up_count"])
        )
    )
    if equal_pct <= -1.0 or up_ratio < 35 or (pd.notna(index_pct) and index_pct <= -1.2) or emotion_retreat:
        regime = "退潮"
    elif equal_pct < -0.3 or up_ratio < 45:
        regime = "弱势"
    elif equal_pct >= 0.5 and up_ratio >= 60 and (pd.isna(index_pct) or index_pct >= 0):
        regime = "强势"
    else:
        regime = "震荡"
    result = {
        "trade_date": trade_date,
        "equal_pct": equal_pct,
        "median_pct": float(current["median_pct"]),
        "up_ratio": up_ratio,
        "amount_yi": float(current["amount_yi"]),
        "amount_change_pct": amount_change,
        "stocks": int(current["stocks"]),
        "index_pct": index_pct,
        "index_close": index_close,
        "regime": regime,
    }
    result.update(supplement)
    return result


def add_index_state(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    work = frame.copy().sort_values("trade_time")
    work["trade_time"] = pd.to_datetime(work["trade_time"])
    factor = add_macd(work)
    return factor, classify_latest(factor)


def draw_candles(ax: plt.Axes, frame: pd.DataFrame) -> None:
    for i, row in frame.reset_index(drop=True).iterrows():
        color = "#D84A3A" if row.close >= row.open else "#2E9F6E"
        ax.vlines(i, row.low, row.high, color=color, linewidth=0.8)
        lower = min(row.open, row.close)
        height = abs(row.close - row.open)
        if height < 0.01:
            ax.hlines(row.close, i - 0.28, i + 0.28, color=color, linewidth=1.0)
        else:
            ax.add_patch(
                Rectangle((i - 0.28, lower), 0.56, height, facecolor=color, edgecolor=color)
            )


def render_market_chart(
    con: duckdb.DuckDBPyConnection,
    trade_date: str,
    output_path: Path,
) -> tuple[Path, str]:
    start_date = (
        pd.to_datetime(trade_date) - pd.Timedelta(days=CHART_LOOKBACK_DAYS - 1)
    ).strftime("%Y%m%d")
    frame = con.execute(
        """
        select ts_code, trade_time, trade_date, open, high, low, close, vol, amount
        from index_bar_60min
        where ts_code = '000001.SH'
          and freq = '60min'
          and trade_date between ? and ?
        order by trade_time
        """,
        [start_date, trade_date],
    ).fetchdf()
    if len(frame) < 40:
        raise RuntimeError("index_bar_60min does not contain enough Shanghai Composite history")
    factor, state = add_index_state(frame)
    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    bold = FontProperties(fname=str(FONT_BOLD_PATH)) if FONT_BOLD_PATH.exists() else font
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    x = np.arange(len(factor))
    fig = plt.figure(figsize=(13.5, 8), facecolor="#FAFBFD")
    grid = fig.add_gridspec(3, 1, height_ratios=[3.2, 0.9, 1.25], hspace=0.07)
    price_ax = fig.add_subplot(grid[0, 0])
    vol_ax = fig.add_subplot(grid[1, 0], sharex=price_ax)
    macd_ax = fig.add_subplot(grid[2, 0], sharex=price_ax)
    for ax in [price_ax, vol_ax, macd_ax]:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, color="#E8EBF2", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    draw_candles(price_ax, factor)
    price_ax.plot(x, factor.ma5, color="#4E75C4", linewidth=1.2, label="MA5")
    price_ax.plot(x, factor.ma10, color="#C8A83E", linewidth=1.2, label="MA10")
    price_ax.plot(x, factor.ma20, color="#8A4C32", linewidth=1.2, label="MA20")
    price_ax.legend(loc="upper left", frameon=False, ncol=3, prop=font)
    price_ax.set_ylabel("上证指数", fontproperties=font)
    candle_colors = np.where(factor.close >= factor.open, "#D84A3A", "#2E9F6E")
    vol_ax.bar(x, factor.vol, width=0.62, color=candle_colors, alpha=0.65)
    vol_ax.set_ylabel("成交量", fontproperties=font)
    macd_colors = np.where(factor.macd >= 0, "#D84A3A", "#2E9F6E")
    macd_ax.bar(x, factor.macd, width=0.62, color=macd_colors, alpha=0.68)
    macd_ax.plot(x, factor.dif, color="#F0A202", linewidth=1.25, label="DIF")
    macd_ax.plot(x, factor.dea, color="#3B6FB6", linewidth=1.25, label="DEA")
    macd_ax.axhline(0, color="#9AA1B3", linewidth=0.7)
    macd_ax.legend(loc="upper left", frameon=False, ncol=2, prop=font)
    labels = pd.to_datetime(factor.trade_time).dt.strftime("%m-%d\n%H:%M").tolist()
    step = max(1, len(factor) // 10)
    ticks = x[::step]
    macd_ax.set_xticks(ticks)
    macd_ax.set_xticklabels([labels[i] for i in ticks])
    plt.setp(price_ax.get_xticklabels(), visible=False)
    plt.setp(vol_ax.get_xticklabels(), visible=False)
    latest = factor.iloc[-1]
    fig.suptitle(
        f"上证指数60分钟K线与MACD（最近{CHART_LOOKBACK_DAYS}天，截至 {trade_date}）",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=17,
        fontproperties=bold,
    )
    fig.text(
        0.07,
        0.935,
        f"收盘 {latest.close:.2f} | DIF {latest.dif:.2f} | DEA {latest.dea:.2f} | MACD柱 {latest.macd:.2f} | 状态：{state}",
        fontsize=10.5,
        fontproperties=font,
        color="#667085",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return output_path, state


def latest_sector_daily_data(
    con: duckdb.DuckDBPyConnection,
    trade_date: str,
) -> pd.DataFrame:
    dates = con.execute(
        "select distinct trade_date from daily_price where trade_date <= ? order by trade_date desc limit 6",
        [trade_date],
    ).fetchdf()["trade_date"].astype(str).tolist()
    return con.execute(
        """
        select p.trade_date, p.ts_code, b.name, b.industry, p.pct_chg,
               p.amount / 100000.0 as amount_yi
        from daily_price p join stock_basic b using(ts_code)
        where p.trade_date in (select * from unnest(?))
          and b.name not like '%ST%' and b.name not like '%退%'
        """,
        [dates],
    ).fetchdf()


def sector_supplements(
    con: duckdb.DuckDBPyConnection,
    trade_date: str,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], bool]:
    flow_map: dict[str, dict[str, object]] = {}
    event_map: dict[str, dict[str, object]] = {}
    if table_exists(con, "ak_sector_fund_flow"):
        flow = con.execute(
            "select * from ak_sector_fund_flow where trade_date = ?", [trade_date]
        ).fetchdf()
        if not flow.empty:
            flow["canonical_sector"] = flow["canonical_sector"].map(canonical_sector_name)
            grouped = flow.groupby("canonical_sector", dropna=False).agg(
                main_net_amount=("main_net_amount", "sum"),
                main_net_ratio=("main_net_ratio", "mean"),
                net_inflow_amount=("net_inflow_amount", "sum"),
                net_inflow_ratio=("net_inflow_ratio", "mean"),
                super_large_net_amount=("super_large_net_amount", "sum"),
                large_net_amount=("large_net_amount", "sum"),
                source=("source", "first"),
                flow_metric=("flow_metric", "first"),
                largest_inflow_stock=("largest_inflow_stock", lambda values: "、".join([str(v) for v in values if str(v) and str(v) != "nan"])[:80]),
            )
            for name, row in grouped.iterrows():
                item = row.to_dict()
                effective_ratio = (
                    item["net_inflow_ratio"]
                    if pd.notna(item["net_inflow_ratio"])
                    else item["main_net_ratio"]
                )
                item["fund_status"] = fund_flow_status(
                    effective_ratio,
                    item["super_large_net_amount"],
                    item["large_net_amount"],
                )
                flow_map[str(name)] = item
    events_available = False
    if table_exists(con, "ak_limit_event"):
        events = con.execute(
            "select * from ak_limit_event where trade_date = ?", [trade_date]
        ).fetchdf()
        if not events.empty:
            events_available = True
            events["canonical_sector"] = events["industry"].map(canonical_sector_name)
            for name, group in events.groupby("canonical_sector", dropna=False):
                limit_up = group[group["event_type"] == "limit_up"]
                broken = group[group["event_type"] == "broken_board"]
                limit_down = group[group["event_type"] == "limit_down"]
                attempts = len(limit_up) + len(broken)
                event_map[str(name)] = {
                    "limit_up_count": len(limit_up),
                    "first_board_count": int((limit_up["consecutive_boards"] == 1).sum()),
                    "broken_board_count": len(broken),
                    "limit_down_count": len(limit_down),
                    "max_board_height": int(limit_up["consecutive_boards"].max()) if len(limit_up) else 0,
                    "broken_board_rate": len(broken) / attempts * 100 if attempts else 0.0,
                    "seal_rate": len(limit_up) / attempts * 100 if attempts else 0.0,
                }
    return flow_map, event_map, events_available


def sector_score(
    state: str,
    relative_pct: float,
    up_ratio: float,
    amount_ratio: float,
    rel3: float,
    rel5: float,
    main_net_ratio: float = np.nan,
    limit_density: float = np.nan,
    seal_rate: float = np.nan,
) -> float:
    components: list[tuple[float, float]] = []
    relative_points = float(np.clip((relative_pct + 1.0) / 3.0 * 20.0, 0, 20))
    breadth_points = float(np.clip((up_ratio - 35.0) / 35.0 * 15.0, 0, 15))
    if 0.85 <= amount_ratio <= 1.8:
        volume_points = 10.0
    elif 0.65 <= amount_ratio <= 2.2:
        volume_points = 6.0
    else:
        volume_points = 2.0
    continuity_points = float(np.clip((rel3 + rel5 + 2.0) / 8.0 * 10.0, 0, 10))
    components.extend(
        [
            (relative_points, 20),
            (breadth_points, 15),
            (volume_points, 10),
            (continuity_points, 10),
            (float(STATE_POINTS.get(state, 0)), 20),
        ]
    )
    if pd.notna(main_net_ratio):
        fund_points = float(np.clip((float(main_net_ratio) + 3.0) / 6.0 * 15.0, 0, 15))
        components.append((fund_points, 15))
    if pd.notna(limit_density) and pd.notna(seal_rate):
        ecology_points = float(np.clip(float(limit_density) / 5.0 * 7.0, 0, 7))
        ecology_points += float(np.clip(float(seal_rate) / 100.0 * 3.0, 0, 3))
        components.append((ecology_points, 10))
    raw = sum(points for points, _ in components)
    maximum = sum(weight for _, weight in components)
    return round(raw / maximum * 100.0, 1) if maximum else 0.0


def build_sector_table(
    con: duckdb.DuckDBPyConnection,
    trade_date: str,
    market: dict[str, float | str],
    summary_path: Path,
) -> pd.DataFrame:
    if not summary_path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary = summary[summary["end_time"].astype(str).str.startswith(pd.to_datetime(trade_date).strftime("%Y-%m-%d"))]
    daily = latest_sector_daily_data(con, trade_date)
    all_daily = daily.groupby("trade_date")["pct_chg"].mean().sort_index()
    flow_map, event_map, events_available = sector_supplements(con, trade_date)
    rows: list[dict[str, object]] = []
    for _, source in summary.iterrows():
        sector = str(source["sector"])
        industries = SECTOR_ALIASES.get(sector, [sector])
        sample = daily[daily["industry"].isin(industries)]
        grouped = sample.groupby("trade_date").agg(
            pct_chg=("pct_chg", "mean"),
            up_ratio=("pct_chg", lambda values: float((values > 0).mean() * 100)),
            amount_yi=("amount_yi", "sum"),
            stocks=("ts_code", "nunique"),
        ).sort_index()
        if grouped.empty or trade_date not in grouped.index:
            continue
        latest = grouped.loc[trade_date]
        previous_amount = grouped[grouped.index < trade_date].tail(5)["amount_yi"].mean()
        amount_ratio = float(latest.amount_yi / previous_amount) if previous_amount else np.nan
        sector3 = float(grouped.tail(3)["pct_chg"].sum())
        sector5 = float(grouped.tail(5)["pct_chg"].sum())
        market3 = float(all_daily.tail(3).sum())
        market5 = float(all_daily.tail(5).sum())
        rel3 = sector3 - market3
        rel5 = sector5 - market5
        relative_pct = float(latest.pct_chg) - float(market["equal_pct"])
        state = str(source["latest_signal"])
        outlook, plan = STATE_OUTLOOK.get(state, ("待确认", "等待更多数据"))
        flow = flow_map.get(sector, {})
        event = event_map.get(sector, {})
        net_inflow_amount = float(flow.get("net_inflow_amount", np.nan))
        net_inflow_ratio = float(flow.get("net_inflow_ratio", np.nan))
        if pd.isna(net_inflow_amount):
            net_inflow_amount = float(flow.get("main_net_amount", np.nan))
        if pd.isna(net_inflow_ratio):
            net_inflow_ratio = float(flow.get("main_net_ratio", np.nan))
        fund_status = str(flow.get("fund_status", "数据不足"))
        flow_source = str(flow.get("source", "无数据"))
        limit_up_count = int(event.get("limit_up_count", 0)) if events_available else np.nan
        first_board_count = int(event.get("first_board_count", 0)) if events_available else np.nan
        broken_board_count = int(event.get("broken_board_count", 0)) if events_available else np.nan
        broken_rate = float(event.get("broken_board_rate", 0.0)) if events_available else np.nan
        seal_rate = float(event.get("seal_rate", 0.0)) if events_available else np.nan
        limit_density = limit_up_count / float(latest.stocks) * 100 if events_available and latest.stocks else np.nan
        if fund_status in {"温和净流出", "强净流出"} and amount_ratio >= 1.2:
            outlook = "放量但资金承接偏弱"
            plan = "等待资金转正；不因单日价格反弹追价"
        if pd.notna(broken_rate) and broken_rate >= 40:
            outlook = "板块炸板分歧较大"
            plan = "停止追高；等待封板率和60分钟结构共同修复"
        reliability = "样本不足"
        if state == "抄底观察":
            n = int(source.get("bottom_watch_n", 0) or 0)
            rate = float(source.get("bottom_watch_up_rate_pct", np.nan))
            avg = float(source.get("bottom_watch_fwd4_avg_pct", np.nan))
            if n >= 20:
                reliability = f"近60天同类{n}次，后4根上涨率{rate:.1f}%，均值{avg:+.2f}%"
        elif state in {"强势延续", "反弹确认"}:
            n = int(source.get("bottom_confirm_n", 0) or 0)
            rate = float(source.get("bottom_confirm_up_rate_pct", np.nan))
            avg = float(source.get("bottom_confirm_fwd4_avg_pct", np.nan))
            if n >= 20:
                reliability = f"近60天确认信号{n}次，后4根上涨率{rate:.1f}%，均值{avg:+.2f}%"
        score = sector_score(
            state,
            relative_pct,
            float(latest.up_ratio),
            amount_ratio,
            rel3,
            rel5,
            net_inflow_ratio,
            limit_density,
            seal_rate,
        )
        rows.append(
            {
                "板块": sector,
                "状态": state,
                "板块评分": score,
                "当日涨幅%": round(float(latest.pct_chg), 2),
                "相对全A%": round(relative_pct, 2),
                "上涨家数占比%": round(float(latest.up_ratio), 1),
                "成交额亿元": round(float(latest.amount_yi), 2),
                "成交额/5日均额": round(amount_ratio, 2) if pd.notna(amount_ratio) else np.nan,
                "近3日相对收益%": round(rel3, 2),
                "近5日相对收益%": round(rel5, 2),
                "资金状态（AKShare）": fund_status,
                "资金来源": flow_source,
                "资金净额亿元": round(net_inflow_amount / 100_000_000.0, 2) if pd.notna(net_inflow_amount) else np.nan,
                "资金净占比%": round(net_inflow_ratio, 2) if pd.notna(net_inflow_ratio) else np.nan,
                "涨停家数": limit_up_count,
                "首板家数": first_board_count,
                "炸板家数": broken_board_count,
                "板块炸板率%": round(broken_rate, 1) if pd.notna(broken_rate) else np.nan,
                "涨停密度%": round(limit_density, 2) if pd.notna(limit_density) else np.nan,
                "60分钟MA5乖离%": round(float(source["bias5"]), 2),
                "60分钟MA20乖离%": round(float(source["bias20"]), 2),
                "MACD柱": round(float(source["macd"]), 2),
                "MACD柱变化": round(float(source["macd_change"]), 2),
                "次日倾向": outlook,
                "次日预案": plan,
                "历史验证": reliability,
                "图表路径": str(source.get("chart_path", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["板块评分", "当日涨幅%"], ascending=False).reset_index(drop=True)


def load_stock_minute_states(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    if not table_exists(con, "stock_bar_60min"):
        return {}
    bars = con.execute(
        "select * from stock_bar_60min where freq = '60min' order by ts_code, trade_time"
    ).fetchdf()
    states: dict[str, str] = {}
    for code, group in bars.groupby("ts_code", sort=False):
        if len(group) >= 30:
            states[str(code)] = classify_latest(add_macd(group))
    return states


def sector_for_industry(industry: str, sector_names: set[str]) -> str | None:
    if industry in sector_names:
        return industry
    for alias, industries in SECTOR_ALIASES.items():
        if industry in industries and alias in sector_names:
            return alias
    return None


def stock_reasons(row: pd.Series) -> str:
    reasons = []
    if row.get("above_ma5") and row.get("above_ma10"):
        reasons.append("站上MA5/MA10")
    if row.get("close_position", 0) >= 0.8:
        reasons.append("强收盘")
    if row.get("relative_sector", 0) >= 1:
        reasons.append("强于板块")
    if row.get("minute_state") in POSITIVE_STOCK_STATES:
        reasons.append(f"60分钟{row.minute_state}")
    if 0.8 <= row.get("volume_ratio", 0) <= 1.8:
        reasons.append("量能适中")
    if row.get("fund_confirmed", False):
        reasons.append("AKShare大单资金确认")
    return "、".join(reasons) or "条件尚少"


def stock_risks(row: pd.Series) -> tuple[bool, str, float]:
    risks: list[str] = []
    deduction = 0.0
    spread = float(row.high - row.low)
    upper_shadow = (float(row.high) - max(float(row.open), float(row.close))) / spread if spread else 0
    if upper_shadow > 0.45:
        risks.append("长上影")
        deduction += 10
    if float(row.close_position or 0) < 0.5:
        risks.append("弱收盘")
        deduction += 10
    if float(row.volume_ratio or 0) > 2.5 and abs(float(row.pct_chg or 0)) < 3:
        risks.append("放量滞涨")
        deduction += 15
    if float(row.rsi6 or 0) > 75 or float(row.kdj_j or 0) > 90:
        risks.append("指标过热")
        deduction += 10
    if float(row.return_5d or 0) > 15:
        risks.append("近5日涨幅过大")
        deduction += 10
    if float(row.bias5 or 0) > 6 or float(row.bias20 or 0) > 15:
        risks.append("均线乖离过大")
        deduction += 10
    if row.minute_state in RISK_STATES:
        risks.append(f"60分钟{row.minute_state}")
        deduction += 20
    flow_ratio = float(row.get("flow_ratio", np.nan))
    if pd.notna(flow_ratio) and float(row.pct_chg) > 0 and flow_ratio < 0:
        risks.append("价涨但资金净流出")
        deduction += 8
    if pd.notna(flow_ratio) and flow_ratio <= -3 and float(row.volume_ratio) >= 1.2:
        risks.append("放量强净流出")
        deduction += 18
    if row.get("limit_event", "") == "broken_board":
        risks.append("当日炸板")
        deduction += 25
    if row.get("limit_event", "") == "limit_down":
        risks.append("当日跌停")
        deduction += 30
    severe = (
        "放量滞涨" in risks
        or "放量强净流出" in risks
        or row.get("limit_event", "") in {"broken_board", "limit_down"}
        or row.minute_state in {"弱势回避", "下跌延续"}
    )
    return severe, "、".join(risks) or "无严重量价风险", deduction


def stock_score(row: pd.Series) -> float:
    sector_points = min(float(row.sector_score) / 100 * 20, 20)
    relative_points = float(np.clip((float(row.relative_sector) + 0.5) / 3 * 15, 0, 15))
    technical = 0.0
    technical += 3 if row.above_ma5 else 0
    technical += 3 if row.above_ma10 else 0
    technical += 3 if row.above_ma20 else 0
    technical += 6 if float(row.macd_dif) >= float(row.macd_dea) else 0
    trading = 0.0
    trading += 6 if float(row.close_position) >= 0.8 else 3 if float(row.close_position) >= 0.65 else 0
    trading += 5 if 0.8 <= float(row.volume_ratio) <= 1.8 else 2 if 0.6 <= float(row.volume_ratio) <= 2.2 else 0
    trading += 4 if float(row.amount_yi) >= 3 else 2
    stage = 0.0
    stage += 4 if 35 <= float(row.rsi6) <= 70 else 1 if float(row.rsi6) < 75 else 0
    stage += 3 if float(row.kdj_j) < 85 else 0
    stage += 3 if -2 <= float(row.bias5) <= 5 and float(row.bias20) <= 12 else 0
    minute = {"强势延续": 10, "反弹确认": 8, "抄底观察": 7, "强势放缓": 5}.get(
        str(row.minute_state), 0
    )
    funds = 0.0
    main_ratio = float(row.get("main_net_ratio", np.nan))
    super_large = row.get("super_large_net_amount", np.nan)
    large = row.get("large_net_amount", np.nan)
    large_orders = (0.0 if pd.isna(super_large) else float(super_large)) + (
        0.0 if pd.isna(large) else float(large)
    )
    if pd.notna(main_ratio) and row.get("flow_metric", "") == "main_order_net":
        funds += 5 if main_ratio > 0 else 0
        funds += 5 if large_orders > 0 else 0
        funds += 5 if main_ratio >= 3 else 2 if main_ratio >= 1 else 0
    return round(sector_points + relative_points + technical + trading + stage + minute + funds, 1)


def build_stock_watchlist(
    con: duckdb.DuckDBPyConnection,
    trade_date: str,
    market: dict[str, float | str],
    sectors: pd.DataFrame,
) -> pd.DataFrame:
    if sectors.empty:
        return pd.DataFrame()
    universe = con.execute(
        """
        select p.ts_code, b.name, b.industry, b.list_date,
               p.open, p.high, p.low, p.close, p.pct_chg,
               p.amount / 100000.0 as amount_yi,
               t.ma5, t.ma10, t.ma20, t.macd_dif, t.macd_dea, t.macd_hist,
               t.rsi6, t.kdj_j, t.bias5, t.bias10, t.bias20,
               t.above_ma5, t.above_ma10, t.above_ma20,
               v.volume_ratio, v.close_position,
               s.return_5d, s.return_10d, s.return_20d
        from daily_price p join stock_basic b using(ts_code)
        left join factor_technical t using(ts_code, trade_date)
        left join factor_volume_price v using(ts_code, trade_date)
        left join factor_stage s using(ts_code, trade_date)
        where p.trade_date = ?
          and b.name not like '%ST%' and b.name not like '%退%'
          and p.amount >= 100000
          and b.list_date <= ?
        """,
        [trade_date, (pd.to_datetime(trade_date) - pd.Timedelta(days=90)).strftime("%Y%m%d")],
    ).fetchdf()
    flow_columns = [
        "ts_code", "main_net_amount", "main_net_ratio",
        "super_large_net_amount", "large_net_amount", "net_inflow_amount",
        "net_inflow_ratio", "flow_metric", "source",
    ]
    if table_exists(con, "ak_stock_fund_flow"):
        flow = con.execute(
            "select * from ak_stock_fund_flow where trade_date = ?", [trade_date]
        ).fetchdf()
        flow = flow[flow_columns] if not flow.empty else pd.DataFrame(columns=flow_columns)
    else:
        flow = pd.DataFrame(columns=flow_columns)
    universe = universe.merge(flow, on="ts_code", how="left")
    universe["flow_amount"] = universe["net_inflow_amount"].where(
        universe["net_inflow_amount"].notna(), universe["main_net_amount"]
    )
    universe["flow_ratio"] = universe["net_inflow_ratio"].where(
        universe["net_inflow_ratio"].notna(), universe["main_net_ratio"]
    )
    if table_exists(con, "ak_limit_event"):
        events = con.execute(
            "select ts_code, event_type from ak_limit_event where trade_date = ?", [trade_date]
        ).fetchdf().drop_duplicates("ts_code")
    else:
        events = pd.DataFrame(columns=["ts_code", "event_type"])
    universe = universe.merge(
        events.rename(columns={"event_type": "limit_event"}), on="ts_code", how="left"
    )
    universe["limit_event"] = universe["limit_event"].fillna("")
    sector_names = set(sectors["板块"].astype(str))
    sector_map = sectors.set_index("板块").to_dict("index")
    universe["sector"] = universe["industry"].map(lambda value: sector_for_industry(str(value), sector_names))
    universe = universe[universe["sector"].notna()].copy()
    universe["sector_state"] = universe["sector"].map(lambda value: sector_map[value]["状态"])
    universe["sector_score"] = universe["sector"].map(lambda value: sector_map[value]["板块评分"])
    universe["sector_pct"] = universe["sector"].map(lambda value: sector_map[value]["当日涨幅%"])
    universe["relative_sector"] = universe["pct_chg"] - universe["sector_pct"]
    minute_states = load_stock_minute_states(con)
    universe["minute_state"] = universe["ts_code"].map(minute_states).fillna("分钟数据不足")
    numeric = [
        "volume_ratio", "close_position", "rsi6", "kdj_j", "bias5", "bias10", "bias20",
        "return_5d", "return_10d", "return_20d", "macd_dif", "macd_dea",
        "main_net_amount", "main_net_ratio", "super_large_net_amount", "large_net_amount",
        "flow_amount", "flow_ratio",
    ]
    for column in numeric:
        universe[column] = pd.to_numeric(universe[column], errors="coerce")
    universe = universe.dropna(subset=["volume_ratio", "close_position", "rsi6", "kdj_j", "ma5", "ma10", "ma20"])
    universe = universe[
        universe["sector_state"].isin(CONDITIONAL_SECTOR_STATES)
        & (universe["sector_score"] >= 50)
        & universe["volume_ratio"].between(0.6, 2.2)
        & (universe["close_position"] >= 0.6)
        & (universe["rsi6"] < 78)
        & (universe["kdj_j"] < 95)
        & (universe["bias5"] <= 6)
        & (universe["bias20"] <= 15)
        & (universe["return_5d"] <= 15)
        & (universe["relative_sector"] > 0)
        & (universe["limit_event"] != "limit_up")
    ].copy()
    if universe.empty:
        return universe
    risk_values = universe.apply(stock_risks, axis=1, result_type="expand")
    risk_values.columns = ["severe_risk", "risk_reasons", "risk_deduction"]
    universe = pd.concat([universe, risk_values], axis=1)
    universe["fund_status"] = universe.apply(
        lambda row: fund_flow_status(
            row.flow_ratio, row.super_large_net_amount, row.large_net_amount
        ),
        axis=1,
    )
    universe["fund_confirmed"] = (
        (universe["flow_metric"] == "main_order_net")
        &
        (universe["main_net_ratio"] > 0)
        & ((universe["super_large_net_amount"].fillna(0) + universe["large_net_amount"].fillna(0)) > 0)
    )
    universe["raw_score"] = universe.apply(stock_score, axis=1)
    universe["score"] = (universe["raw_score"] - universe["risk_deduction"]).clip(0, 100).round(1)
    universe["classification"] = "不入选"
    conditional = (
        (universe["score"] >= 62)
        & (universe["sector_score"] >= 55)
        & ~universe["severe_risk"]
    )
    priority = (
        (universe["score"] >= 72)
        & (universe["sector_score"] >= 65)
        & ~universe["severe_risk"]
        & universe["sector_state"].isin(POSITIVE_SECTOR_STATES)
        & universe["minute_state"].isin(POSITIVE_STOCK_STATES)
        & universe["fund_confirmed"]
    )
    if market["regime"] == "退潮":
        priority[:] = False
    universe.loc[conditional, "classification"] = "条件观察"
    universe.loc[priority, "classification"] = "优先观察"
    universe["reasons"] = universe.apply(stock_reasons, axis=1)
    universe["support"] = universe[["ma5", "ma10", "ma20"]].where(
        universe[["ma5", "ma10", "ma20"]].le(universe["close"], axis=0)
    ).max(axis=1).fillna(universe[["ma5", "ma10", "ma20"]].min(axis=1)).round(2)
    universe["resistance"] = (universe["close"] * 1.05).round(2)
    universe["next_plan"] = universe.apply(
        lambda row: (
            f"高开不追；平开守{row.support:.2f}且板块不转弱时观察；"
            f"低开跌破{row.support:.2f}取消观察"
        ),
        axis=1,
    )
    selected = pd.concat(
        [
            universe[universe["classification"] == "优先观察"].nlargest(5, "score"),
            universe[universe["classification"] == "条件观察"].nlargest(5, "score"),
        ],
        ignore_index=True,
    )
    columns = [
        "classification", "ts_code", "name", "sector", "sector_state", "minute_state",
        "score", "close", "pct_chg", "volume_ratio", "close_position", "rsi6", "kdj_j",
        "bias5", "relative_sector", "flow_amount", "flow_ratio", "fund_status", "source",
        "support", "resistance", "reasons", "risk_reasons", "next_plan",
    ]
    return selected[columns]


def _clock_minutes(value: object) -> int | None:
    text = str(value or "").strip()
    if not text or ":" not in text:
        return None
    try:
        hour, minute, *_ = [int(part) for part in text.split(":")]
        return hour * 60 + minute
    except ValueError:
        return None


def build_first_board_watchlist(
    con: duckdb.DuckDBPyConnection,
    trade_date: str,
    market: dict[str, float | str],
    sectors: pd.DataFrame,
) -> pd.DataFrame:
    if not table_exists(con, "ak_limit_event") or sectors.empty:
        return pd.DataFrame()
    events = con.execute(
        """
        select e.*, p.open, p.high, p.low, p.close as daily_close, p.amount as daily_amount,
               b.name as basic_name, b.industry as local_industry,
               f.main_net_amount, f.main_net_ratio,
               f.super_large_net_amount, f.large_net_amount,
               f.net_inflow_amount, f.net_inflow_ratio, f.flow_metric, f.source
        from ak_limit_event e
        left join daily_price p using(ts_code, trade_date)
        left join stock_basic b using(ts_code)
        left join ak_stock_fund_flow f using(ts_code, trade_date)
        where e.trade_date = ? and e.event_type = 'limit_up'
          and e.consecutive_boards = 1
          and coalesce(b.name, e.name) not like '%ST%'
          and coalesce(b.name, e.name) not like '%退%'
        """,
        [trade_date],
    ).fetchdf()
    if events.empty:
        return events
    sector_names = set(sectors["板块"].astype(str))
    sector_map = sectors.set_index("板块").to_dict("index")
    events["sector"] = events["local_industry"].map(
        lambda value: sector_for_industry(str(value), sector_names)
    )
    missing_sector = events["sector"].isna()
    events.loc[missing_sector, "sector"] = events.loc[missing_sector, "industry"].map(
        canonical_sector_name
    )
    events["sector_state"] = events["sector"].map(
        lambda value: sector_map.get(value, {}).get("状态", "未跟踪")
    )
    events["sector_score"] = events["sector"].map(
        lambda value: float(sector_map.get(value, {}).get("板块评分", 0))
    )
    events["is_one_word"] = (events["high"] - events["low"]).abs() < 0.000001
    events["seal_ratio_pct"] = events["seal_amount"] / events["float_mv"].replace(0, np.nan) * 100
    events["fund_status"] = events.apply(
        lambda row: fund_flow_status(
            row.net_inflow_ratio if pd.notna(row.net_inflow_ratio) else row.main_net_ratio,
            row.super_large_net_amount,
            row.large_net_amount,
        ),
        axis=1,
    )
    large_orders = events["super_large_net_amount"].fillna(0) + events["large_net_amount"].fillna(0)
    events["fund_confirmed"] = (
        (events["flow_metric"] == "main_order_net")
        & (events["main_net_ratio"] > 0)
        & (large_orders > 0)
    )

    def quality_score(row: pd.Series) -> float:
        score = min(float(row.sector_score) / 100 * 20, 20)
        seal_time = _clock_minutes(row.first_seal_time)
        score += 25 if seal_time is not None and seal_time <= 600 else 20 if seal_time is not None and seal_time <= 630 else 12 if seal_time is not None and seal_time <= 810 else 5
        score += 20 if int(row.open_count) == 0 else 12 if int(row.open_count) <= 2 else 5 if int(row.open_count) <= 4 else 0
        seal_ratio = float(row.seal_ratio_pct) if pd.notna(row.seal_ratio_pct) else 0.0
        score += 20 if seal_ratio >= 1 else 15 if seal_ratio >= 0.5 else 8 if seal_ratio >= 0.2 else 3
        turnover = float(row.turnover_rate) if pd.notna(row.turnover_rate) else 0.0
        score += 15 if 3 <= turnover <= 20 else 8 if 1 <= turnover <= 30 else 2
        return round(score, 1)

    events["score"] = events.apply(quality_score, axis=1)
    events["risks"] = events.apply(
        lambda row: "、".join(
            item
            for item, triggered in [
                ("一字板不可正常参与", bool(row.is_one_word)),
                ("封板偏晚", (_clock_minutes(row.first_seal_time) or 999) > 810),
                ("反复开板", int(row.open_count) > 2),
                ("资金未确认", not bool(row.fund_confirmed)),
                ("板块非正向状态", row.sector_state not in POSITIVE_SECTOR_STATES),
            ]
            if triggered
        ) or "无明显封板质量风险",
        axis=1,
    )
    qualified = (
        (events["score"] >= 70)
        & ~events["is_one_word"]
        & events["fund_confirmed"]
        & events["sector_state"].isin(POSITIVE_SECTOR_STATES)
    )
    if market["regime"] == "退潮":
        qualified[:] = False
    events = events[qualified].nlargest(8, "score").copy()
    if events.empty:
        return events
    events["next_plan"] = "只做次日竞价与开盘确认；高开加速不追，开板转弱或板块退潮即取消观察"
    events["name"] = events["basic_name"].fillna(events["name"])
    columns = [
        "ts_code", "name", "sector", "sector_state", "score", "first_seal_time",
        "last_seal_time", "open_count", "seal_ratio_pct", "turnover_rate",
        "net_inflow_ratio", "fund_status", "source", "risks", "next_plan",
    ]
    return events[columns]


def fmt(value: object, decimals: int = 2, sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "无数据"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    return html.escape(str(value))


def image_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def state_badge(state: str) -> str:
    css = {
        "强势延续": "good", "反弹确认": "good", "抄底观察": "watch",
        "强势放缓": "caution", "震荡观察": "neutral", "逃顶预警": "risk",
        "弱势回避": "risk", "下跌延续": "risk",
    }.get(state, "neutral")
    return f'<span class="badge {css}">{html.escape(state)}</span>'


def html_table(frame: pd.DataFrame, columns: list[str], percent_columns: set[str] | None = None) -> str:
    if frame.empty:
        return '<div class="empty">当前没有符合条件的数据。</div>'
    percent_columns = percent_columns or set()
    heads = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, "")
            if column == "状态":
                content = state_badge(str(value))
            elif column in percent_columns and pd.notna(value):
                content = f"{float(value):+.2f}%"
            elif isinstance(value, (float, np.floating)):
                content = fmt(value)
            else:
                content = html.escape(str(value))
            cells.append(f"<td>{content}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{heads}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def market_scenarios(market: dict[str, float | str]) -> list[tuple[str, str]]:
    regime = str(market["regime"])
    if regime == "强势":
        flat = "平开后量能不低于昨日且上涨家数保持55%以上，可继续观察强势延续板块。"
    elif regime == "退潮":
        flat = "即使平开也先等待情绪修复；没有板块广度确认时不新增优先观察。"
    else:
        flat = "平开后观察全A广度能否保持50%以上，再决定是否保留优先观察。"
    return [
        ("高开情景", "指数高开超过0.5%时不追价，等待首小时回踩与板块广度确认。"),
        ("平开情景", flat),
        ("低开情景", "指数低开超过0.5%时先看昨日低点与MA10；板块跌破MA20则取消对应观察池。"),
    ]


def build_html(
    trade_date: str,
    grade: str,
    quality_rows: list[dict[str, str]],
    market: dict[str, float | str],
    market_state: str,
    market_chart: Path,
    emotion_history: pd.DataFrame,
    sectors: pd.DataFrame,
    stocks: pd.DataFrame,
    first_boards: pd.DataFrame,
    akshare_fund_available: bool,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    market_conclusion = (
        f"市场环境为{market['regime']}，上证技术状态为{market_state}。"
        f"全A等权{float(market['equal_pct']):+.2f}%，上涨家数占比{float(market['up_ratio']):.1f}%。"
    )
    cards = [
        ("市场环境", str(market["regime"])),
        ("上证状态", market_state),
        ("全A等权", f"{float(market['equal_pct']):+.2f}%"),
        ("上涨家数", f"{float(market['up_ratio']):.1f}%"),
        ("两市成交额", f"{float(market['amount_yi']):,.0f}亿"),
        ("成交额环比", f"{float(market['amount_change_pct']):+.1f}%" if pd.notna(market["amount_change_pct"]) else "无数据"),
        ("资金状态（AKShare）", str(market["market_fund_status"])),
        ("资金净额", f"{float(market['market_net_inflow_amount']) / 100_000_000:+.1f}亿" if pd.notna(market["market_net_inflow_amount"]) else "无数据"),
        ("涨停/首板", f"{int(market['limit_up_count'])}/{int(market['first_board_count'])}" if pd.notna(market["limit_up_count"]) else "无数据"),
        ("炸板率", f"{float(market['broken_board_rate']):.1f}%" if pd.notna(market["broken_board_rate"]) else "无数据"),
        ("跌停家数", f"{int(market['limit_down_count'])}" if pd.notna(market["limit_down_count"]) else "无数据"),
        ("最高连板", f"{int(market['max_board_height'])}板" if pd.notna(market["max_board_height"]) else "无数据"),
    ]
    card_html = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    quality = pd.DataFrame(quality_rows)
    sector_columns = [
        "板块", "状态", "板块评分", "当日涨幅%", "相对全A%", "上涨家数占比%",
        "成交额/5日均额", "近3日相对收益%", "资金状态（AKShare）", "资金来源",
        "资金净额亿元", "资金净占比%", "涨停家数", "首板家数", "炸板家数", "板块炸板率%",
        "MACD柱变化", "次日倾向", "次日预案",
    ]
    bottom = sectors[sectors["状态"] == "抄底观察"] if not sectors.empty else pd.DataFrame()
    bottom_sections = []
    for _, row in bottom.iterrows():
        chart_path = Path(str(row["图表路径"]))
        image = (
            f'<img src="{image_data_uri(chart_path)}" alt="{html.escape(str(row["板块"]))}60分钟K线">'
            if chart_path.exists()
            else '<div class="empty">图表文件不可用。</div>'
        )
        bottom_sections.append(
            f"""
            <article class="chart-card">
              <div class="chart-title"><h3>{html.escape(str(row['板块']))}</h3>{state_badge(str(row['状态']))}</div>
              <p>{html.escape(str(row['历史验证']))}</p>
              <p><strong>次日预案：</strong>{html.escape(str(row['次日预案']))}</p>
              {image}
            </article>
            """
        )
    if not bottom_sections:
        bottom_sections.append('<div class="empty">当前没有严格命中“抄底观察”的板块，因此本节不展示板块K线。</div>')
    priority = stocks[stocks["classification"] == "优先观察"].copy() if not stocks.empty else pd.DataFrame()
    conditional = stocks[stocks["classification"] == "条件观察"].copy() if not stocks.empty else pd.DataFrame()
    for frame in [priority, conditional]:
        if "flow_amount" in frame.columns:
            frame.loc[:, "flow_amount"] = frame["flow_amount"] / 100_000_000.0
    stock_rename = {
        "ts_code": "股票代码", "name": "股票名称", "sector": "所属板块",
        "sector_state": "板块状态", "minute_state": "个股60分钟状态", "score": "评分",
        "close": "收盘价", "pct_chg": "涨跌幅%", "volume_ratio": "量比",
        "close_position": "收盘位置", "rsi6": "RSI6", "kdj_j": "KDJ-J",
        "flow_amount": "资金净额亿元", "flow_ratio": "资金净占比%",
        "fund_status": "资金状态（AKShare）", "source": "资金来源",
        "support": "支撑位", "reasons": "入选理由", "risk_reasons": "风险",
        "next_plan": "次日预案",
    }
    priority = priority.rename(columns=stock_rename)
    conditional = conditional.rename(columns=stock_rename)
    stock_columns = [
        "股票代码", "股票名称", "所属板块", "板块状态", "个股60分钟状态", "评分", "收盘价",
        "涨跌幅%", "量比", "收盘位置", "RSI6", "KDJ-J", "支撑位",
        "资金净额亿元", "资金净占比%", "资金状态（AKShare）", "资金来源", "入选理由", "风险", "次日预案",
    ]
    first_board_rename = {
        "ts_code": "股票代码", "name": "股票名称", "sector": "所属板块",
        "sector_state": "板块状态", "score": "封板质量分",
        "first_seal_time": "首次封板", "last_seal_time": "最后封板",
        "open_count": "炸板次数", "seal_ratio_pct": "封单/流通市值%",
        "turnover_rate": "换手率%", "net_inflow_ratio": "资金净占比%",
        "fund_status": "资金状态（AKShare）", "source": "资金来源", "risks": "风险",
        "next_plan": "次日预案",
    }
    first_boards = first_boards.rename(columns=first_board_rename)
    first_board_columns = list(first_board_rename.values())
    scenario_html = "".join(
        f'<div class="scenario"><h3>{html.escape(title)}</h3><p>{html.escape(text)}</p></div>'
        for title, text in market_scenarios(market)
    )
    fund_note = (
        f"资金流通过AKShare获取，当前市场来源为{market['market_flow_source']}，"
        f"指标口径为{market['market_flow_metric']}。"
        "total_flow_net_aggregate为个股总资金净额汇总，不能与东方财富主力净额直接比较。"
        "不同供应商字段不混算，也不等同于已确认的机构账户行为。"
        if akshare_fund_available
        else "AKShare资金流当前不可用；报告未编造资金净流入，相关股票最多进入条件观察。"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A股盘后决策复盘 {trade_date}</title>
<style>
:root{{--ink:#19202c;--muted:#667085;--line:#e5e9f0;--bg:#f4f6fa;--card:#fff;--red:#cf4439;--green:#228b62;--amber:#b7791f;--blue:#315a9b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC",sans-serif;line-height:1.6}}
.page{{max-width:1480px;margin:0 auto;padding:28px}} .hero{{background:linear-gradient(135deg,#17243b,#315a9b);color:#fff;padding:32px;border-radius:18px;box-shadow:0 12px 32px rgba(30,50,80,.16)}}
.hero h1{{margin:0 0 8px;font-size:30px}} .hero p{{margin:4px 0;color:#e9eef8}} .section{{background:var(--card);margin-top:22px;padding:24px;border-radius:16px;box-shadow:0 5px 18px rgba(28,39,58,.06)}}
.section h2{{margin:0 0 14px;font-size:22px}} .section-note{{color:var(--muted);margin-top:-8px}} .metrics{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:12px;margin:18px 0}}
.metric{{background:#f7f9fc;border:1px solid var(--line);border-radius:12px;padding:14px}} .metric span{{display:block;color:var(--muted);font-size:13px}} .metric strong{{display:block;font-size:21px;margin-top:4px}}
.chart{{width:100%;border:1px solid var(--line);border-radius:12px;background:#fff}} .table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}} table{{border-collapse:collapse;width:100%;font-size:13px;min-width:980px}}
th{{background:#f4f7fb;color:#344054;text-align:left;position:sticky;top:0}} th,td{{padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top}} tr:last-child td{{border-bottom:none}}
.badge{{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;white-space:nowrap}} .good{{background:#e7f6ef;color:#15704d}} .watch{{background:#eaf1ff;color:#315a9b}} .caution{{background:#fff4dc;color:#986515}} .risk{{background:#fdebea;color:#a63831}} .neutral{{background:#eef1f5;color:#596273}}
.scenario-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}} .scenario{{border:1px solid var(--line);border-radius:12px;padding:16px;background:#fafbfc}} .scenario h3{{margin:0 0 6px}}
.chart-card{{border:1px solid var(--line);border-radius:14px;padding:18px;margin-top:16px}} .chart-card img{{width:100%;border-radius:10px;border:1px solid var(--line)}} .chart-title{{display:flex;gap:12px;align-items:center}} .chart-title h3{{margin:0}}
.empty{{border:1px dashed #cbd2dc;background:#fafbfc;color:var(--muted);padding:18px;border-radius:12px}} .warning{{background:#fff8e8;border-left:4px solid #d69e2e;padding:13px 15px;border-radius:6px}}
.footer{{color:var(--muted);font-size:13px;text-align:center;padding:26px}} @media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.scenario-grid{{grid-template-columns:1fr}}.page{{padding:12px}}}}
</style>
</head>
<body><main class="page">
<header class="hero"><h1>A股盘后决策复盘</h1><p>交易日：{trade_date} | 生成时间：{generated_at} | 数据质量：{grade}</p><p>{html.escape(market_conclusion)}</p></header>

<section class="section"><h2>一、大盘环境</h2><div class="metrics">{card_html}</div><p class="section-note">图表统一使用最近{CHART_LOOKBACK_DAYS}个自然日的60分钟K线。资金流来源和计算口径在表格及数据质量区单独标明。</p><img class="chart" src="{image_data_uri(market_chart)}" alt="上证指数最近60天60分钟K线"><h3>近期涨停情绪</h3>{html_table(emotion_history, ['交易日','涨停家数','首板家数','炸板家数','跌停家数','最高连板','封板率%'])}</section>

<section class="section"><h2>二、板块状态与次日倾向</h2><p class="section-note">板块评分加入AKShare资金流和涨停生态；数据缺失时按可观察部分重标，不把缺失当流出。</p>{html_table(sectors, sector_columns, {'当日涨幅%','相对全A%','近3日相对收益%','资金净占比%','板块炸板率%'})}</section>

<section class="section"><h2>三、抄底观察板块60分钟K线</h2><p class="warning">本节图表统一使用最近{CHART_LOOKBACK_DAYS}个自然日数据。“抄底观察”只代表绿柱收缩和DIF修复，必须等待站回均线与板块广度确认，不等同于买入信号。</p>{''.join(bottom_sections)}</section>

<section class="section"><h2>四、明日市场情景</h2><div class="scenario-grid">{scenario_html}</div></section>

<section class="section"><h2>五、优先观察个股</h2><p class="section-note">优先观察仍要求东方财富主力净占比为正且超大单与大单合计净流入；同花顺总资金净额只能支持条件观察。</p>{html_table(priority, stock_columns, {'涨跌幅%','资金净占比%'})}<h2 style="margin-top:24px">条件观察个股</h2>{html_table(conditional, stock_columns, {'涨跌幅%','资金净占比%'})}</section>

<section class="section"><h2>六、首板接力观察</h2><p class="warning">本区只保留非一字、板块正向、东方财富大单资金确认且封板质量达到阈值的首板；只用于次日竞价与开盘确认，不等同于买入信号。</p>{html_table(first_boards, first_board_columns, {'封单/流通市值%','换手率%','资金净占比%'})}</section>

<section class="section"><h2>七、数据质量与限制</h2><p>{html.escape(fund_note)}</p>{html_table(quality, ['数据表','最新日期','目标日记录数','状态'])}<p class="warning">大盘及所展示板块均使用最近{CHART_LOOKBACK_DAYS}个自然日的60分钟K线。行业K线为最新成交额前10只成分股的成交额加权合成，不是官方行业指数。涨停池只保存AKShare/东方财富仍可返回的近期数据；缺失日期不会用当前快照冒充。报告仅用于量化复盘和次日观察计划，不构成投资建议。</p></section>

<footer class="footer">本报告不使用“必涨”“稳赚”“一定涨停”等确定性表述。所有观察条件需在下一交易日重新验证。</footer>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    run_script("setup_database.py")
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        current = resolve_trade_date(con, args.trade_date)
    if args.sync:
        sync_start = (pd.to_datetime(current) - pd.DateOffset(months=7)).strftime("%Y%m%d")
        run_script("sync_tushare_data.py", "--start-date", sync_start, "--end-date", args.trade_date or current)
        with duckdb.connect(str(DB_PATH), read_only=True) as con:
            current = resolve_trade_date(con, args.trade_date)
        run_script("sync_akshare_supplement.py", "--trade-date", current)
    ensure_factors(current)
    refresh_market_minutes(current, args.skip_minute_fetch)
    summary_path = ROOT / "outputs" / "sector_60m" / f"sector_60m_signal_summary_{current}.csv"
    if not args.skip_sector_refresh:
        summary_path = refresh_sector_outputs(current, args.top_n, args.skip_minute_fetch)
    report_dir = OUTPUT_ROOT / current
    asset_dir = report_dir / "assets"
    report_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        grade, quality_rows = data_quality(con, current)
        market = market_snapshot(con, current)
        emotion_history = market_emotion_history(con, current)
        market_chart, market_state = render_market_chart(
            con, current, asset_dir / f"market_60m_kline_{current}.png"
        )
        sectors = build_sector_table(con, current, market, summary_path)
        stocks = build_stock_watchlist(con, current, market, sectors)
        first_boards = build_first_board_watchlist(con, current, market, sectors)
        akshare_fund_available = bool(
            table_exists(con, "ak_stock_fund_flow")
            and con.execute(
                "select count(*) from ak_stock_fund_flow where trade_date = ?", [current]
            ).fetchone()[0]
        )
    sector_csv = report_dir / f"sector_summary_{current}.csv"
    stock_csv = report_dir / f"stock_watchlist_{current}.csv"
    first_board_csv = report_dir / f"first_board_watchlist_{current}.csv"
    sectors.to_csv(sector_csv, index=False, encoding="utf-8-sig")
    stocks.to_csv(stock_csv, index=False, encoding="utf-8-sig")
    first_boards.to_csv(first_board_csv, index=False, encoding="utf-8-sig")
    content = build_html(
        current,
        grade,
        quality_rows,
        market,
        market_state,
        market_chart,
        emotion_history,
        sectors,
        stocks,
        first_boards,
        akshare_fund_available,
    )
    output = report_dir / f"a_share_after_close_review_{current}.html"
    output.write_text(content, encoding="utf-8")
    print(f"trade_date={current}")
    print(f"market_regime={market['regime']}")
    print(f"market_state={market_state}")
    print(f"sectors={len(sectors)}")
    print(f"bottom_watch_sectors={(sectors['状态'] == '抄底观察').sum() if not sectors.empty else 0}")
    print(f"stocks={len(stocks)}")
    print(f"first_boards={len(first_boards)}")
    print(f"html={output}")
    print(f"sector_csv={sector_csv}")
    print(f"stock_csv={stock_csv}")
    print(f"first_board_csv={first_board_csv}")


if __name__ == "__main__":
    main()
