from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import load_env_file

DB_PATH = ROOT / "data" / "db" / "a_stock_quant.duckdb"
OUTPUT_DIR = ROOT / "outputs" / "charts"
MINUTE_TABLE = "index_bar_60min"

INDEX_NAMES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
}

FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 60-minute index MACD chart")
    parser.add_argument("--index-code", default="000001.SH")
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--days", type=int, help="Calendar-day window including end date; default 60")
    window.add_argument("--months", type=int, help="Legacy month-based window")
    parser.add_argument("--start-date", help="YYYYMMDD; overrides --days/--months")
    parser.add_argument("--end-date", help="YYYYMMDD; defaults to latest local index_daily date")
    parser.add_argument("--skip-sync", action="store_true", help="Use cached 60-minute data only")
    parser.add_argument("--retry-wait", type=float, default=65.0)
    return parser.parse_args()


def get_latest_index_date(index_code: str) -> str:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        latest = con.execute(
            "select max(trade_date) from index_daily where ts_code = ?",
            [index_code],
        ).fetchone()[0]
    if not latest:
        return pd.Timestamp.today().strftime("%Y%m%d")
    return str(latest)


def date_window(
    index_code: str,
    months: int | None,
    start_date: str | None,
    end_date: str | None,
    days: int | None = None,
) -> tuple[str, str]:
    end = end_date or get_latest_index_date(index_code)
    if start_date:
        start = start_date
    elif days is not None:
        if days < 1:
            raise ValueError("days must be >= 1")
        start = (pd.to_datetime(end) - pd.Timedelta(days=days - 1)).strftime("%Y%m%d")
    elif months is not None:
        if months < 1:
            raise ValueError("months must be >= 1")
        start = (pd.to_datetime(end) - pd.DateOffset(months=months)).strftime("%Y%m%d")
    else:
        start = (pd.to_datetime(end) - pd.Timedelta(days=59)).strftime("%Y%m%d")
    if start > end:
        raise ValueError(f"start_date must be <= end_date, got {start} > {end}")
    return start, end


def create_minute_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        create table if not exists {MINUTE_TABLE} (
            ts_code varchar,
            trade_time timestamp,
            trade_date varchar,
            open double,
            high double,
            low double,
            close double,
            vol double,
            amount double,
            freq varchar
        )
        """
    )


def fetch_index_60m(index_code: str, start_date: str, end_date: str, retry_wait: float) -> pd.DataFrame:
    load_env_file()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")

    import tushare as ts

    ts.set_token(token)
    start = f"{pd.to_datetime(start_date).strftime('%Y-%m-%d')} 09:00:00"
    end = f"{pd.to_datetime(end_date).strftime('%Y-%m-%d')} 15:30:00"
    primary = pd.DataFrame()
    try:
        frame = ts.pro_bar(
            ts_code=index_code,
            asset="I",
            freq="60min",
            start_date=start,
            end_date=end,
        )
        primary = normalize_pro_bar(frame)
    except Exception as exc:
        print(f"pro_bar 60min unavailable, falling back to legacy get_k_data: {exc}")

    if minute_frame_covers_date(primary, end_date):
        return primary

    latest = "empty" if primary.empty else str(primary["trade_date"].max())
    print(
        "pro_bar 60min is stale "
        f"(latest={latest}, required={end_date}); supplementing with legacy get_k_data"
    )
    time.sleep(min(retry_wait, 3.0))
    try:
        fallback = fetch_legacy_index_60m(index_code, start_date, end_date)
    except Exception as exc:
        if not primary.empty:
            print(f"legacy get_k_data unavailable; retaining stale primary bars: {exc}")
            return primary
        raise
    return merge_minute_frames(primary, fallback)


def minute_frame_covers_date(frame: pd.DataFrame, end_date: str) -> bool:
    if frame.empty or "trade_date" not in frame.columns:
        return False
    latest = frame["trade_date"].dropna().astype(str).max()
    return bool(latest and latest >= end_date)


def merge_minute_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    available = [frame for frame in frames if frame is not None and not frame.empty]
    if not available:
        return pd.DataFrame()
    return (
        pd.concat(available, ignore_index=True)
        .sort_values("trade_time")
        .drop_duplicates(["ts_code", "trade_time", "freq"], keep="last")
        .reset_index(drop=True)
    )


def normalize_pro_bar(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"])
    frame["trade_date"] = frame["trade_time"].dt.strftime("%Y%m%d")
    frame["freq"] = "60min"
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
    ]
    return frame[columns].sort_values("trade_time").drop_duplicates(
        ["ts_code", "trade_time", "freq"],
        keep="last",
    )


def fetch_legacy_index_60m(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    import tushare as ts

    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, ignore_index=False, **kwargs: pd.concat(  # type: ignore[attr-defined]
            [self, other],
            ignore_index=ignore_index,
        )
    code = index_code.split(".")[0]
    frame = ts.get_k_data(
        code,
        start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        end=pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        ktype="60",
        index=True,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()

    frame = frame.copy()
    frame["trade_time"] = pd.to_datetime(frame["date"])
    start_ts = pd.to_datetime(f"{start_date} 00:00:00")
    end_ts = pd.to_datetime(f"{end_date} 23:59:59")
    frame = frame[(frame["trade_time"] >= start_ts) & (frame["trade_time"] <= end_ts)]
    frame["ts_code"] = index_code
    frame["trade_date"] = frame["trade_time"].dt.strftime("%Y%m%d")
    frame["vol"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame["amount"] = np.nan
    frame["freq"] = "60min"
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
    ]
    return frame[columns].sort_values("trade_time").drop_duplicates(
        ["ts_code", "trade_time", "freq"],
        keep="last",
    )


def upsert_minute_data(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    with duckdb.connect(str(DB_PATH)) as con:
        create_minute_table(con)
        keys = frame[["ts_code", "trade_time", "freq"]].drop_duplicates()
        con.register("incoming_keys", keys)
        con.execute(
            f"""
            delete from {MINUTE_TABLE} as target
            using incoming_keys
            where target.ts_code = incoming_keys.ts_code
              and target.trade_time = incoming_keys.trade_time
              and target.freq = incoming_keys.freq
            """
        )
        con.register("incoming_frame", frame)
        con.execute(f"insert into {MINUTE_TABLE} by name select * from incoming_frame")
    return len(frame)


def load_cached_60m(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH)) as con:
        create_minute_table(con)
        frame = con.execute(
            f"""
            select ts_code, trade_time, trade_date, open, high, low, close, vol, amount
            from {MINUTE_TABLE}
            where ts_code = ?
              and freq = '60min'
              and trade_date between ? and ?
            order by trade_time
            """,
            [index_code, start_date, end_date],
        ).fetchdf()
    if frame.empty:
        return frame
    frame["trade_time"] = pd.to_datetime(frame["trade_time"])
    return frame.sort_values("trade_time").reset_index(drop=True)


def add_macd(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    close = frame["close"].astype(float)
    frame["ema12"] = close.ewm(span=12, adjust=False).mean()
    frame["ema26"] = close.ewm(span=26, adjust=False).mean()
    frame["dif"] = frame["ema12"] - frame["ema26"]
    frame["dea"] = frame["dif"].ewm(span=9, adjust=False).mean()
    frame["macd"] = (frame["dif"] - frame["dea"]) * 2
    for window in (5, 10, 20):
        frame[f"ma{window}"] = close.rolling(window).mean()
    return frame


def draw_candles(ax: plt.Axes, frame: pd.DataFrame) -> None:
    width = 0.58
    up_color = "#D84A3A"
    down_color = "#2E9F6E"
    flat_color = "#7A828F"
    for i, row in frame.iterrows():
        open_price = float(row["open"])
        close_price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        color = up_color if close_price > open_price else down_color if close_price < open_price else flat_color
        ax.vlines(i, low, high, color=color, linewidth=1.0, zorder=2)
        lower = min(open_price, close_price)
        height = abs(close_price - open_price)
        if height < 0.03:
            ax.hlines(close_price, i - width / 2, i + width / 2, color=color, linewidth=1.2)
        else:
            ax.add_patch(
                Rectangle(
                    (i - width / 2, lower),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.6,
                    zorder=3,
                )
            )
    ax.set_xlim(-1, len(frame))


def format_x_axis(ax: plt.Axes, frame: pd.DataFrame) -> None:
    labels = frame["trade_time"].dt.strftime("%m-%d\n%H:%M").tolist()
    step = max(1, len(frame) // 10)
    ticks = np.arange(len(frame))[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=0)


def render_chart(frame: pd.DataFrame, index_code: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = INDEX_NAMES.get(index_code, index_code)
    latest_date = frame["trade_date"].iloc[-1]
    output_png = OUTPUT_DIR / f"{index_code.replace('.', '_')}_60m_macd_{latest_date}.png"
    output_csv = OUTPUT_DIR / f"{index_code.replace('.', '_')}_60m_macd_{latest_date}.csv"

    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    bold_font = FontProperties(fname=str(FONT_BOLD_PATH)) if FONT_BOLD_PATH.exists() else font
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    x = np.arange(len(frame))
    fig = plt.figure(figsize=(15, 8.8), facecolor="#FCFCFD")
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.4], hspace=0.08)
    ax_price = fig.add_subplot(grid[0, 0])
    ax_macd = fig.add_subplot(grid[1, 0], sharex=ax_price)

    for ax in (ax_price, ax_macd):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, color="#E7EAF2", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D7DBE7")
        ax.spines["bottom"].set_color("#D7DBE7")
        ax.tick_params(colors="#687086")

    draw_candles(ax_price, frame)
    ax_price.plot(x, frame["ma5"], color="#5477C4", linewidth=1.3, label="MA5")
    ax_price.plot(x, frame["ma10"], color="#B8A037", linewidth=1.2, label="MA10")
    ax_price.plot(x, frame["ma20"], color="#804126", linewidth=1.2, label="MA20")
    ax_price.set_ylabel("指数点位", color="#1F2430", fontproperties=font)
    ax_price.legend(loc="upper left", frameon=False, ncol=3, prop=font)

    macd_colors = np.where(frame["macd"] >= 0, "#D84A3A", "#2E9F6E")
    ax_macd.bar(x, frame["macd"], color=macd_colors, width=0.62, alpha=0.72, label="MACD柱")
    ax_macd.plot(x, frame["dif"], color="#F0A202", linewidth=1.4, label="DIF")
    ax_macd.plot(x, frame["dea"], color="#3B6FB6", linewidth=1.4, label="DEA")
    ax_macd.axhline(0, color="#9AA1B3", linewidth=0.8)
    ax_macd.set_ylabel("MACD", color="#1F2430", fontproperties=font)
    ax_macd.legend(loc="upper left", frameon=False, ncol=3, prop=font)

    format_x_axis(ax_macd, frame)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    start_time = frame["trade_time"].iloc[0].strftime("%Y-%m-%d %H:%M")
    end_time = frame["trade_time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
    latest = frame.iloc[-1]
    subtitle = (
        f"最新收盘 {latest['close']:.2f}；"
        f"DIF {latest['dif']:.2f}，DEA {latest['dea']:.2f}，MACD柱 {latest['macd']:.2f}；"
        f"样本 {len(frame)} 根60分钟K线"
    )
    fig.text(
        0.07,
        0.965,
        f"{name} 60分钟K线 + MACD（{start_time} 至 {end_time}）",
        fontsize=18,
        fontproperties=bold_font,
        color="#1F2430",
    )
    fig.text(0.07, 0.925, subtitle, fontsize=11, fontproperties=font, color="#687086")
    fig.text(
        0.07,
        0.035,
        "数据来源：Tushare 60分钟K线；Pro分钟接口限频时自动使用get_k_data备用源；MACD按EMA12/EMA26/DEA9计算。",
        fontsize=9,
        fontproperties=font,
        color="#687086",
    )
    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.90])
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    plt.close(fig)
    return output_png, output_csv


def main() -> None:
    args = parse_args()
    start_date, end_date = date_window(
        args.index_code,
        args.months,
        args.start_date,
        args.end_date,
        args.days,
    )
    fetched_rows = 0
    if not args.skip_sync:
        fetched = fetch_index_60m(args.index_code, start_date, end_date, args.retry_wait)
        fetched_rows = upsert_minute_data(fetched)

    frame = load_cached_60m(args.index_code, start_date, end_date)
    if frame.empty:
        raise RuntimeError(
            f"No cached 60-minute bars for {args.index_code} from {start_date} to {end_date}"
        )
    frame = add_macd(frame)
    png, csv = render_chart(frame, args.index_code)

    latest = frame.iloc[-1]
    print(f"index_code={args.index_code}")
    print(f"start={frame['trade_time'].iloc[0]}")
    print(f"end={frame['trade_time'].iloc[-1]}")
    print(f"rows={len(frame)}")
    print(f"fetched_rows={fetched_rows}")
    print(f"latest_close={latest['close']:.4f}")
    print(f"latest_dif={latest['dif']:.4f}")
    print(f"latest_dea={latest['dea']:.4f}")
    print(f"latest_macd={latest['macd']:.4f}")
    print(f"png={png}")
    print(f"csv={csv}")


if __name__ == "__main__":
    main()
