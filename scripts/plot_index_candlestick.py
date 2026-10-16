from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "a_stock_quant.duckdb"
OUTPUT_DIR = ROOT / "outputs" / "charts"

INDEX_NAMES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot index candlestick chart")
    parser.add_argument("--index-code", default="000001.SH")
    parser.add_argument("--months", type=int, default=3)
    return parser.parse_args()


def load_index(index_code: str, months: int) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        frame = con.execute(
            """
            select trade_date, open, high, low, close, pre_close, pct_chg, amount
            from index_daily
            where ts_code = ?
            order by trade_date
            """,
            [index_code],
        ).fetchdf()
    if frame.empty:
        raise RuntimeError(f"index_daily has no data for {index_code}")

    frame["date"] = pd.to_datetime(frame["trade_date"])
    latest = frame["date"].max()
    start = latest - pd.DateOffset(months=months)
    frame = frame[frame["date"] >= start].copy()
    for window in (5, 10, 20):
        frame[f"ma{window}"] = frame["close"].rolling(window).mean()
    frame["amount_100m"] = frame["amount"] / 100000
    return frame.reset_index(drop=True)


def draw_candles(ax: plt.Axes, frame: pd.DataFrame) -> None:
    x = np.arange(len(frame))
    width = 0.62
    up_color = "#D84A3A"
    down_color = "#2E9F6E"
    flat_color = "#7A828F"
    for i, row in frame.iterrows():
        open_price = float(row["open"])
        close_price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        if close_price > open_price:
            color = up_color
        elif close_price < open_price:
            color = down_color
        else:
            color = flat_color
        ax.vlines(i, low, high, color=color, linewidth=1.1, zorder=2)
        lower = min(open_price, close_price)
        height = abs(close_price - open_price)
        if height < 0.05:
            ax.hlines(close_price, i - width / 2, i + width / 2, color=color, linewidth=1.4)
        else:
            ax.add_patch(
                Rectangle(
                    (i - width / 2, lower),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=3,
                )
            )
    ax.set_xlim(-1, len(frame))


def format_x_axis(ax: plt.Axes, frame: pd.DataFrame) -> None:
    x = np.arange(len(frame))
    dates = frame["date"].dt.strftime("%m-%d").tolist()
    step = max(1, len(frame) // 9)
    ticks = x[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i] for i in ticks], rotation=0)


def plot(frame: pd.DataFrame, index_code: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = INDEX_NAMES.get(index_code, index_code)
    latest_date = frame["trade_date"].iloc[-1]
    start_date = frame["trade_date"].iloc[0]
    output_png = OUTPUT_DIR / f"{index_code.replace('.', '_')}_kline_3m_{latest_date}.png"
    output_csv = OUTPUT_DIR / f"{index_code.replace('.', '_')}_kline_3m_{latest_date}.csv"

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(14, 8), facecolor="#FCFCFD")
    grid = fig.add_gridspec(5, 1, height_ratios=[3.5, 0.05, 1.1, 0.1, 0.25])
    ax_price = fig.add_subplot(grid[0, 0])
    ax_volume = fig.add_subplot(grid[2, 0], sharex=ax_price)

    for ax in (ax_price, ax_volume):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D7DBE7")
        ax.spines["bottom"].set_color("#D7DBE7")
        ax.tick_params(colors="#6F768A")

    draw_candles(ax_price, frame)
    x = np.arange(len(frame))
    ax_price.plot(x, frame["ma5"], color="#5477C4", linewidth=1.6, label="MA5")
    ax_price.plot(x, frame["ma10"], color="#B8A037", linewidth=1.5, label="MA10")
    ax_price.plot(x, frame["ma20"], color="#804126", linewidth=1.5, label="MA20")

    colors = np.where(frame["close"] >= frame["open"], "#D84A3A", "#2E9F6E")
    ax_volume.bar(x, frame["amount_100m"], color=colors, width=0.62, alpha=0.72)
    ax_volume.set_ylabel("成交额(亿元)", color="#1F2430")
    ax_price.set_ylabel("指数点位", color="#1F2430")
    ax_price.legend(loc="upper left", frameon=False, ncol=3)

    format_x_axis(ax_volume, frame)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    start_close = frame["close"].iloc[0]
    end_close = frame["close"].iloc[-1]
    period_return = (end_close / start_close - 1) * 100
    high = frame["high"].max()
    low = frame["low"].min()
    latest_pct = frame["pct_chg"].iloc[-1]
    title = f"{name}近三个月K线图（{start_date} 至 {latest_date}）"
    subtitle = (
        f"最新收盘 {end_close:.2f}，当日涨跌幅 {latest_pct:+.2f}%；"
        f"区间涨跌幅 {period_return:+.2f}%；区间最高 {high:.2f}，最低 {low:.2f}"
    )
    fig.text(0.08, 0.965, title, fontsize=18, fontweight="bold", color="#1F2430")
    fig.text(0.08, 0.925, subtitle, fontsize=11, color="#6F768A")
    fig.text(
        0.08,
        0.045,
        "数据来源：本地Tushare index_daily；红色为上涨K线，绿色为下跌K线。",
        fontsize=9,
        color="#6F768A",
    )
    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.90])
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    plt.close(fig)
    return output_png, output_csv


def main() -> None:
    args = parse_args()
    frame = load_index(args.index_code, args.months)
    png, csv = plot(frame, args.index_code)
    print(f"rows={len(frame)}")
    print(f"start={frame['trade_date'].iloc[0]}")
    print(f"end={frame['trade_date'].iloc[-1]}")
    print(f"latest_close={frame['close'].iloc[-1]:.2f}")
    print(f"latest_pct={frame['pct_chg'].iloc[-1]:.4f}")
    print(f"period_return={(frame['close'].iloc[-1] / frame['close'].iloc[0] - 1) * 100:.4f}")
    print(f"png={png}")
    print(f"csv={csv}")


if __name__ == "__main__":
    main()
