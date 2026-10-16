from __future__ import annotations

import html
import textwrap
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "a_stock_quant.duckdb"
OUTPUT_DIR = ROOT / "outputs" / "first_board_event_study"
REPORT_PATH = ROOT / "outputs" / "reports" / "first_board_event_study.html"
ASSET_DIR = ROOT / "outputs" / "reports" / "first_board_event_study_assets"

SURFACE = "#FCFCFD"
PANEL = "#FFFFFF"
INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"
AXIS = "#D7DBE7"
BLUE = "#A3BEFA"
BLUE_DARK = "#2E4780"
ORANGE = "#F0986E"
ORANGE_DARK = "#804126"
GOLD = "#FFE15B"
OLIVE = "#A3D576"
NEUTRAL = "#C5CAD3"


def compounded_return(values: pd.Series, window: int) -> pd.Series:
    return (
        (1 + values / 100)
        .rolling(window, min_periods=window)
        .apply(np.prod, raw=True)
        * 100
        - 100
    )


def consecutive_positive(values: pd.Series) -> pd.Series:
    result: list[int] = []
    count = 0
    for value in values:
        count = count + 1 if value > 0 else 0
        result.append(count)
    return pd.Series(result, index=values.index)


def price_limit_rate(codes: pd.Series) -> np.ndarray:
    return np.where(
        codes.str.endswith(".BJ"),
        0.30,
        np.where(codes.str.match(r"^(300|301|688|689)"), 0.20, 0.10),
    )


def rounded_price_limit(price: pd.Series, multiplier: np.ndarray) -> np.ndarray:
    return np.floor(price.to_numpy() * multiplier * 100 + 0.5) / 100


def market_state(row: pd.Series) -> str:
    if row["market_return"] >= 1.0 and row["market_up_ratio"] >= 0.60:
        return "强势"
    if (
        row["market_return"] <= -1.0
        and row["market_up_ratio"] <= 0.40
    ) or row["market_broken_rate"] >= 0.45:
        return "退潮"
    if row["market_return"] < -0.30 or row["market_up_ratio"] < 0.45:
        return "弱势"
    return "震荡"


def sector_phase(row: pd.Series) -> str:
    if (
        row["sector_return"] >= 2.0
        and row["sector_up_ratio"] >= 0.65
        and row["sector_limit_count"] >= 2
    ):
        return "高潮"
    if (
        row["sector_return"] > row["market_return"]
        and row["sector_relative_3d"] > 0
        and row["sector_up_ratio"] >= 0.55
    ):
        return "启动"
    if (
        row["sector_return"] < row["market_return"]
        and row["sector_relative_3d"] < 0
        and row["sector_up_ratio"] < 0.45
    ):
        return "退潮"
    return "分歧"


def load_and_engineer() -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    frame = con.execute(
        """
        select
            p.ts_code, p.trade_date, p.open, p.high, p.low, p.close, p.pre_close,
            p.pct_chg, p.vol, p.amount,
            b.name, coalesce(b.industry, '未分类') as industry, b.market, b.list_date,
            t.ma5, t.ma10, t.ma20, t.macd_dif, t.macd_dea, t.macd_hist,
            t.kdj_k, t.kdj_d, t.kdj_j, t.rsi6,
            t.above_ma5, t.above_ma10, t.above_ma20, t.macd_golden_cross,
            v.vol_ma5, v.amount_ma5, v.volume_ratio,
            d.turnover_rate
        from daily_price p
        join stock_basic b using (ts_code)
        left join factor_technical t using (ts_code, trade_date)
        left join factor_volume_price v using (ts_code, trade_date)
        left join daily_basic d using (ts_code, trade_date)
        order by p.ts_code, p.trade_date
        """
    ).fetchdf()
    dates = con.execute(
        "select distinct trade_date from daily_price order by trade_date"
    ).fetchdf()["trade_date"].tolist()
    table_counts = {
        table: con.execute(f"select count(*) from {table}").fetchone()[0]
        for table in ["stock_limit", "moneyflow", "daily_basic", "daily_price"]
    }
    con.close()

    frame = frame.sort_values(["ts_code", "trade_date"]).copy()
    bool_columns = [
        "above_ma5",
        "above_ma10",
        "above_ma20",
        "macd_golden_cross",
    ]
    for column in bool_columns:
        frame[column] = frame[column].fillna(False).astype(bool)

    frame["limit_rate"] = price_limit_rate(frame["ts_code"])
    frame["up_limit_proxy"] = rounded_price_limit(
        frame["pre_close"], 1 + frame["limit_rate"]
    )
    frame["down_limit_proxy"] = rounded_price_limit(
        frame["pre_close"], 1 - frame["limit_rate"]
    )
    frame["limit_up"] = frame["close"] >= frame["up_limit_proxy"] - 0.001
    frame["hit_limit"] = frame["high"] >= frame["up_limit_proxy"] - 0.001
    frame["broken_board"] = frame["hit_limit"] & ~frame["limit_up"]
    frame["limit_down"] = frame["close"] <= frame["down_limit_proxy"] + 0.001
    frame["one_word_board"] = (
        frame["limit_up"]
        & (frame["open"] >= frame["up_limit_proxy"] - 0.001)
        & (frame["low"] >= frame["up_limit_proxy"] - 0.001)
    )
    frame["opening_limit_broken_proxy"] = (
        frame["limit_up"]
        & (frame["open"] >= frame["up_limit_proxy"] - 0.001)
        & (frame["low"] < frame["up_limit_proxy"] - 0.001)
    )

    grouped = frame.groupby("ts_code", sort=False)
    for column in [
        "open",
        "high",
        "low",
        "close",
        "pct_chg",
        "ma5",
        "macd_dif",
        "macd_hist",
        "rsi6",
        "volume_ratio",
        "limit_up",
    ]:
        frame[f"{column}_lag1"] = grouped[column].shift(1)
    for lag in range(1, 6):
        frame[f"limit_lag{lag}"] = grouped["limit_up"].shift(lag).fillna(False)
    frame["first_board_all"] = frame["limit_up"] & ~frame[
        [f"limit_lag{lag}" for lag in range(1, 6)]
    ].any(axis=1)
    frame["consecutive_board"] = frame["limit_up"] & frame["limit_lag1"]

    frame["observations_before"] = grouped.cumcount()
    names = frame["name"].astype(object).fillna("").astype(str)
    frame["bad_name"] = names.str.contains(
        "ST", case=False, regex=False
    ) | names.str.contains("退", regex=False)

    frame["open_return"] = (frame["open"] / frame["pre_close"] - 1) * 100
    frame["high_return"] = (frame["high"] / frame["pre_close"] - 1) * 100
    frame["low_return"] = (frame["low"] / frame["pre_close"] - 1) * 100
    frame["amplitude_calc"] = (
        (frame["high"] - frame["low"]) / frame["pre_close"] * 100
    )
    frame["amount_100m"] = frame["amount"] / 100000
    day_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["close_position_calc"] = (
        (frame["close"] - frame["low"]) / day_range
    ).fillna(0.5)
    frame["upper_shadow_ratio"] = (
        (frame["high"] - frame[["open", "close"]].max(axis=1)) / day_range
    ).fillna(0)
    frame["long_upper_shadow"] = frame["upper_shadow_ratio"] >= 0.45
    frame["high_open_low_close"] = (
        (frame["open"] > frame["pre_close"]) & (frame["close"] < frame["open"])
    )
    frame["intraday_reversal"] = (
        (frame["high_return"] - frame["pct_chg"] >= 3)
        & (frame["close_position_calc"] < 0.50)
    )
    frame["volume_to_ma5"] = frame["vol"] / frame["vol_ma5"].replace(0, np.nan)
    frame["amount_to_ma5"] = frame["amount"] / frame["amount_ma5"].replace(0, np.nan)
    frame["volume_stall"] = (
        (frame["volume_to_ma5"] >= 1.5) & (frame["pct_chg"] < 2)
    )
    frame["five_day_volume_expansion"] = frame["volume_to_ma5"] >= 1.20
    frame["contraction_then_strength"] = (
        (grouped["volume_to_ma5"].shift(1) < 0.80)
        & (frame["volume_to_ma5"] >= 1.20)
        & (frame["pct_chg"] > 0)
    )
    frame["bullish_engulfing"] = (
        (frame["close"] > frame["open"])
        & (frame["close_lag1"] < frame["open_lag1"])
        & (frame["open"] <= frame["close_lag1"])
        & (frame["close"] >= frame["open_lag1"])
    )
    frame["return_3d"] = grouped["pct_chg"].transform(
        lambda values: compounded_return(values, 3)
    )
    frame["return_5d"] = grouped["pct_chg"].transform(
        lambda values: compounded_return(values, 5)
    )
    frame["return_10d"] = grouped["pct_chg"].transform(
        lambda values: compounded_return(values, 10)
    )
    frame["consecutive_up"] = grouped["pct_chg"].apply(
        consecutive_positive
    ).reset_index(level=0, drop=True)
    frame["distance_ma5"] = (frame["close"] / frame["ma5"] - 1) * 100
    frame["distance_ma10"] = (frame["close"] / frame["ma10"] - 1) * 100
    frame["distance_ma20"] = (frame["close"] / frame["ma20"] - 1) * 100
    frame["ma5_up"] = frame["ma5"] > frame["ma5_lag1"]
    frame["dif_up"] = frame["macd_dif"] > frame["macd_dif_lag1"]
    frame["green_shrinking"] = (
        (frame["macd_hist"] < 0) & (frame["macd_hist"] > frame["macd_hist_lag1"])
    )
    frame["rsi_up"] = frame["rsi6"] > frame["rsi6_lag1"]
    prior_5_high = grouped["high"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=5).max()
    )
    prior_10_high = grouped["high"].transform(
        lambda values: values.shift(1).rolling(10, min_periods=10).max()
    )
    frame["breakout_5d_platform"] = frame["close"] > prior_5_high
    frame["breakout_10d_platform"] = frame["close"] > prior_10_high

    strong_amplitude_limit = frame["limit_rate"] * 100 * 0.65
    frame["strong_board_proxy"] = (
        frame["first_board_all"]
        & ~frame["one_word_board"]
        & (frame["open_return"] >= 0)
        & (frame["amplitude_calc"] <= strong_amplitude_limit)
        & (frame["amount_to_ma5"] <= 3.0)
    )
    frame["board_class"] = np.select(
        [
            frame["one_word_board"],
            frame["strong_board_proxy"],
            frame["first_board_all"],
        ],
        ["一字首板", "强首板（日线代理）", "烂板首板（日线代理）"],
        default="非首板",
    )

    market = (
        frame.groupby("trade_date")
        .agg(
            market_return=("pct_chg", "mean"),
            market_up_ratio=("pct_chg", lambda values: (values > 0).mean()),
            market_limit_count=("limit_up", "sum"),
            market_down_limit_count=("limit_down", "sum"),
            market_consecutive_board_count=("consecutive_board", "sum"),
            market_hit_limit_count=("hit_limit", "sum"),
            market_broken_count=("broken_board", "sum"),
            market_amount=("amount", "sum"),
        )
        .reset_index()
        .sort_values("trade_date")
    )
    market["market_broken_rate"] = (
        market["market_broken_count"]
        / market["market_hit_limit_count"].replace(0, np.nan)
    ).fillna(0)
    market["market_amount"] = market["market_amount"] / 100000
    market["market_amount_change"] = market["market_amount"].pct_change() * 100
    market["market_state"] = market.apply(market_state, axis=1)

    sector = (
        frame.groupby(["trade_date", "industry"])
        .agg(
            sector_return=("pct_chg", "mean"),
            sector_up_ratio=("pct_chg", lambda values: (values > 0).mean()),
            sector_limit_count=("limit_up", "sum"),
            sector_broken_count=("broken_board", "sum"),
        )
        .reset_index()
        .merge(market[["trade_date", "market_return"]], on="trade_date", how="left")
        .sort_values(["industry", "trade_date"])
    )
    sector["sector_relative"] = sector["sector_return"] - sector["market_return"]
    sector["sector_return_3d"] = sector.groupby("industry")["sector_return"].transform(
        lambda values: values.rolling(3, min_periods=3).sum()
    )
    sector["sector_return_5d"] = sector.groupby("industry")["sector_return"].transform(
        lambda values: values.rolling(5, min_periods=5).sum()
    )
    sector["sector_relative_3d"] = sector.groupby("industry")[
        "sector_relative"
    ].transform(lambda values: values.rolling(3, min_periods=3).sum())
    sector["sector_relative_5d"] = sector.groupby("industry")[
        "sector_relative"
    ].transform(lambda values: values.rolling(5, min_periods=5).sum())
    sector["sector_leader_limit"] = sector["sector_limit_count"] > 0
    sector["sector_phase"] = sector.apply(sector_phase, axis=1)

    frame = frame.merge(market, on="trade_date", how="left")
    frame = frame.merge(
        sector[
            [
                "trade_date",
                "industry",
                "sector_return",
                "sector_up_ratio",
                "sector_limit_count",
                "sector_broken_count",
                "sector_return_3d",
                "sector_return_5d",
                "sector_relative_3d",
                "sector_relative_5d",
                "sector_leader_limit",
                "sector_phase",
            ]
        ],
        on=["trade_date", "industry"],
        how="left",
    )
    frame["sector_strong"] = frame["sector_return"] > frame["market_return"]
    frame["stock_relative_sector"] = frame["pct_chg"] - frame["sector_return"]
    frame["stock_stronger_sector"] = frame["stock_relative_sector"] > 0
    frame["stock_stronger_market"] = frame["pct_chg"] > frame["market_return"]

    required = [
        "ma5",
        "ma10",
        "ma20",
        "macd_dif",
        "kdj_j",
        "rsi6",
        "volume_to_ma5",
        "amount_to_ma5",
        "sector_relative_3d",
        "sector_relative_5d",
    ]
    frame["eligible"] = (
        ~frame["bad_name"]
        & (frame["observations_before"] >= 60)
        & (frame["vol"] > 0)
        & (frame["amount"] >= 100000)
        & frame[required].notna().all(axis=1)
    )

    prev_columns = [
        "eligible",
        "market_return",
        "market_up_ratio",
        "market_limit_count",
        "market_down_limit_count",
        "market_consecutive_board_count",
        "market_broken_rate",
        "market_amount",
        "market_amount_change",
        "market_state",
        "sector_return",
        "sector_strong",
        "sector_return_3d",
        "sector_return_5d",
        "sector_relative_3d",
        "sector_relative_5d",
        "sector_up_ratio",
        "sector_limit_count",
        "sector_leader_limit",
        "sector_phase",
        "stock_stronger_sector",
        "stock_relative_sector",
        "above_ma5",
        "above_ma10",
        "above_ma20",
        "distance_ma5",
        "distance_ma10",
        "distance_ma20",
        "ma5_up",
        "dif_up",
        "green_shrinking",
        "macd_golden_cross",
        "kdj_j",
        "rsi6",
        "rsi_up",
        "breakout_5d_platform",
        "breakout_10d_platform",
        "bullish_engulfing",
        "long_upper_shadow",
        "high_open_low_close",
        "volume_stall",
        "pct_chg",
        "high_return",
        "amplitude_calc",
        "close_position_calc",
        "volume_ratio",
        "volume_to_ma5",
        "amount_to_ma5",
        "turnover_rate",
        "consecutive_up",
        "five_day_volume_expansion",
        "contraction_then_strength",
        "return_3d",
        "return_5d",
        "return_10d",
    ]
    grouped = frame.groupby("ts_code", sort=False)
    for column in prev_columns:
        frame[f"prev_{column}"] = grouped[column].shift(1)

    for column in [
        "open",
        "high",
        "low",
        "close",
        "pct_chg",
        "limit_up",
        "close_position_calc",
    ]:
        frame[f"next_{column}"] = grouped[column].shift(-1)
    frame["next_open_return"] = (frame["next_open"] / frame["close"] - 1) * 100
    frame["next_high_return"] = (frame["next_high"] / frame["close"] - 1) * 100
    frame["next_low_return"] = (frame["next_low"] / frame["close"] - 1) * 100
    frame["next_close_return"] = frame["next_pct_chg"]
    frame["next_reversal"] = (
        (frame["next_high_return"] - frame["next_close_return"] >= 3)
        & (frame["next_close_position_calc"] < 0.50)
    )
    frame["next_nuclear_proxy"] = (
        (frame["next_open_return"] <= -5) | (frame["next_close_return"] <= -7)
    )

    return frame, dates, table_counts


METRICS: list[tuple[str, str, str]] = [
    ("全A等权涨跌幅", "prev_market_return", "numeric"),
    ("上涨家数占比", "prev_market_up_ratio", "binary"),
    ("涨停家数", "prev_market_limit_count", "numeric"),
    ("跌停家数", "prev_market_down_limit_count", "numeric"),
    ("连板家数", "prev_market_consecutive_board_count", "numeric"),
    ("炸板率", "prev_market_broken_rate", "binary"),
    ("市场成交额（亿元）", "prev_market_amount", "numeric"),
    ("市场成交额变化", "prev_market_amount_change", "numeric"),
    ("行业T-1涨跌幅", "prev_sector_return", "numeric"),
    ("行业强于全A", "prev_sector_strong", "binary"),
    ("行业近3日涨幅", "prev_sector_return_3d", "numeric"),
    ("行业近5日涨幅", "prev_sector_return_5d", "numeric"),
    ("行业近3日相对收益", "prev_sector_relative_3d", "numeric"),
    ("行业上涨家数占比", "prev_sector_up_ratio", "binary"),
    ("行业涨停家数", "prev_sector_limit_count", "numeric"),
    ("行业龙头涨停", "prev_sector_leader_limit", "binary"),
    ("个股强于行业", "prev_stock_stronger_sector", "binary"),
    ("个股相对行业强度", "prev_stock_relative_sector", "numeric"),
    ("站上MA5", "prev_above_ma5", "binary"),
    ("站上MA10", "prev_above_ma10", "binary"),
    ("站上MA20", "prev_above_ma20", "binary"),
    ("MA5乖离", "prev_distance_ma5", "numeric"),
    ("MA10乖离", "prev_distance_ma10", "numeric"),
    ("MA20乖离", "prev_distance_ma20", "numeric"),
    ("MA5拐头向上", "prev_ma5_up", "binary"),
    ("DIF上拐", "prev_dif_up", "binary"),
    ("MACD绿柱缩短", "prev_green_shrinking", "binary"),
    ("MACD刚金叉", "prev_macd_golden_cross", "binary"),
    ("KDJ-J", "prev_kdj_j", "numeric"),
    ("RSI6", "prev_rsi6", "numeric"),
    ("RSI6向上", "prev_rsi_up", "binary"),
    ("突破近5日平台", "prev_breakout_5d_platform", "binary"),
    ("突破近10日平台", "prev_breakout_10d_platform", "binary"),
    ("阳包阴", "prev_bullish_engulfing", "binary"),
    ("长上影", "prev_long_upper_shadow", "binary"),
    ("高开低走", "prev_high_open_low_close", "binary"),
    ("放量滞涨", "prev_volume_stall", "binary"),
    ("T-1日涨跌幅", "prev_pct_chg", "numeric"),
    ("T-1日最高涨幅", "prev_high_return", "numeric"),
    ("T-1日振幅", "prev_amplitude_calc", "numeric"),
    ("收盘位置系数", "prev_close_position_calc", "numeric"),
    ("量比", "prev_volume_to_ma5", "numeric"),
    ("成交量/5日均量", "prev_volume_to_ma5", "numeric"),
    ("成交额/5日均额", "prev_amount_to_ma5", "numeric"),
    ("换手率", "prev_turnover_rate", "numeric"),
    ("连续上涨天数", "prev_consecutive_up", "numeric"),
    ("近5日明显放量", "prev_five_day_volume_expansion", "binary"),
    ("缩量调整后放量转强", "prev_contraction_then_strength", "binary"),
    ("近3日涨幅", "prev_return_3d", "numeric"),
    ("近5日涨幅", "prev_return_5d", "numeric"),
    ("近10日涨幅", "prev_return_10d", "numeric"),
]


def build_study(
    frame: pd.DataFrame, dates: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    study_dates = dates[-120:]
    universe = frame[
        frame["trade_date"].isin(study_dates)
        & frame["eligible"]
        & frame["prev_eligible"].fillna(False)
    ].copy()
    universe["practical_first_board"] = (
        universe["first_board_all"] & ~universe["one_word_board"]
    )
    practical_universe = universe[~universe["one_word_board"]].copy()
    events_all = universe[universe["first_board_all"]].copy()
    events = practical_universe[practical_universe["practical_first_board"]].copy()
    return universe, practical_universe, events_all, study_dates


def match_controls(
    events: pd.DataFrame, pool: pd.DataFrame, group_name: str
) -> pd.DataFrame:
    used: set[int] = set()
    matched: list[pd.Series] = []
    ordered_events = events.sort_values(["trade_date", "industry", "amount"])
    for event_index, event in ordered_events.iterrows():
        same_industry = pool[
            (pool["trade_date"] == event["trade_date"])
            & (pool["industry"] == event["industry"])
            & ~pool.index.isin(used)
        ]
        candidates = same_industry
        exact = True
        if candidates.empty:
            candidates = pool[
                (pool["trade_date"] == event["trade_date"])
                & ~pool.index.isin(used)
            ]
            exact = False
        if candidates.empty:
            continue
        target = np.log1p(event["amount"])
        chosen_index = (np.log1p(candidates["amount"]) - target).abs().idxmin()
        chosen = candidates.loc[chosen_index].copy()
        chosen["matched_event_index"] = event_index
        chosen["exact_date_industry_match"] = exact
        chosen["control_group"] = group_name
        matched.append(chosen)
        used.add(chosen_index)
    return pd.DataFrame(matched)


def build_controls(
    events: pd.DataFrame, practical_universe: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    non_limit = practical_universe[~practical_universe["limit_up"]].copy()
    pools = {
        "对照组A": non_limit,
        "对照组B": non_limit[non_limit["pct_chg"].between(3, 8)],
        "对照组C": non_limit[non_limit["broken_board"]],
    }
    return {
        name: match_controls(events, pool, name)
        for name, pool in pools.items()
    }


def standardized_difference(left: pd.Series, right: pd.Series) -> float:
    left = pd.to_numeric(left, errors="coerce").dropna()
    right = pd.to_numeric(right, errors="coerce").dropna()
    if left.empty or right.empty:
        return np.nan
    pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
    return (left.mean() - right.mean()) / pooled if pooled > 0 else 0.0


def comparison_table(
    events: pd.DataFrame, controls: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    combined_control = pd.concat(list(controls.values()), ignore_index=True)
    rows: list[dict[str, object]] = []
    for label, column, kind in METRICS:
        event_values = pd.to_numeric(events[column], errors="coerce")
        control_values = {
            name: pd.to_numeric(control[column], errors="coerce")
            for name, control in controls.items()
        }
        smds = {
            name: standardized_difference(event_values, values)
            for name, values in control_values.items()
        }
        finite_smds = [value for value in smds.values() if pd.notna(value)]
        signs = {np.sign(value) for value in finite_smds if abs(value) >= 0.05}
        robust_score = (
            float(np.mean([abs(value) for value in finite_smds]))
            if finite_smds
            else np.nan
        )
        combined_median = pd.to_numeric(
            combined_control[column], errors="coerce"
        ).median()
        difference = event_values.median() - combined_median
        distinguishable = robust_score >= 0.20 and len(signs) <= 1
        rows.append(
            {
                "指标": label,
                "字段": column,
                "类型": kind,
                "首板实验组均值": event_values.mean(),
                "对照组A均值": control_values["对照组A"].mean(),
                "对照组B均值": control_values["对照组B"].mean(),
                "对照组C均值": control_values["对照组C"].mean(),
                "实验组中位数": event_values.median(),
                "对照组中位数": combined_median,
                "SMD_A": smds["对照组A"],
                "SMD_B": smds["对照组B"],
                "SMD_C": smds["对照组C"],
                "稳健区分度": robust_score,
                "差异方向": "首板组更高" if difference > 0 else "首板组更低",
                "是否有区分度": "是" if distinguishable else "否",
            }
        )
    unavailable = {
        "封板时间": "本地库无分钟级或涨停事件明细",
        "封单金额": "本地库无涨停事件明细",
        "开板次数": "日线无法识别盘中开板次数",
        "尾盘抢筹": "日线无法识别尾盘成交",
        "主力净流入": "moneyflow表为空",
        "大单净额": "moneyflow表为空",
        "超大单净额": "moneyflow表为空",
        "小单净额": "moneyflow表为空",
        "最近3日主力净流入天数": "moneyflow表为空",
        "放量但主力净流出": "moneyflow表为空",
        "资金流入与上涨一致": "moneyflow表为空",
    }
    for label, reason in unavailable.items():
        rows.append(
            {
                "指标": label,
                "字段": "",
                "类型": "unavailable",
                "差异方向": reason,
                "是否有区分度": "不可验证",
            }
        )
    return pd.DataFrame(rows).sort_values(
        "稳健区分度", key=lambda values: values.abs(), ascending=False
    )


BIN_SPECS: dict[str, tuple[str, list[float], list[str], list[str]]] = {
    "T-1日涨幅": (
        "prev_pct_chg",
        [-np.inf, 0, 3, 5, 7, 10, np.inf],
        ["<0%", "0%-3%", "3%-5%", "5%-7%", "7%-10%", "≥10%"],
        ["<0%", "0%-3%", "3%-5%", "5%-7%", "7%-10%"],
    ),
    "T-1日最高涨幅": (
        "prev_high_return",
        [-np.inf, 0, 3, 5, 7, 10, np.inf],
        ["<0%", "0%-3%", "3%-5%", "5%-7%", "7%-10%", ">10%"],
        ["0%-3%", "3%-5%", "5%-7%", "7%-10%", ">10%"],
    ),
    "收盘位置系数": (
        "prev_close_position_calc",
        [-np.inf, 0.5, 0.7, 0.8, 0.9, 1.000001],
        ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
        ["0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
    ),
    "量比": (
        "prev_volume_to_ma5",
        [-np.inf, 0.8, 1.2, 1.8, 2.5, np.inf],
        ["<0.8", "0.8-1.2", "1.2-1.8", "1.8-2.5", ">2.5"],
        ["<0.8", "0.8-1.2", "1.2-1.8", "1.8-2.5", ">2.5"],
    ),
    "换手率": (
        "prev_turnover_rate",
        [-np.inf, 3, 8, 15, 25, np.inf],
        ["<3%", "3%-8%", "8%-15%", "15%-25%", ">25%"],
        ["<3%", "3%-8%", "8%-15%", "15%-25%", ">25%"],
    ),
    "MA5乖离": (
        "prev_distance_ma5",
        [-np.inf, 0, 3, 6, 10, np.inf],
        ["<0%", "0%-3%", "3%-6%", "6%-10%", ">10%"],
        ["<0%", "0%-3%", "3%-6%", "6%-10%", ">10%"],
    ),
    "RSI6": (
        "prev_rsi6",
        [-np.inf, 40, 55, 70, 85, np.inf],
        ["<40", "40-55", "55-70", "70-85", ">85"],
        ["<40", "40-55", "55-70", "70-85", ">85"],
    ),
    "KDJ-J": (
        "prev_kdj_j",
        [-np.inf, 50, 70, 90, 110, np.inf],
        ["<50", "50-70", "70-90", "90-110", ">110"],
        ["<50", "50-70", "70-90", "90-110", ">110"],
    ),
    "连续上涨天数": (
        "prev_consecutive_up",
        [-0.5, 0.5, 1.5, 2.5, np.inf],
        ["0天", "1天", "2天", "≥3天"],
        ["0天", "1天", "2天", "≥3天"],
    ),
}


RISK_ALLOWED_BINS = {
    "T-1日涨幅": ["<0%", "0%-3%", "3%-5%", "5%-7%"],
    "T-1日最高涨幅": ["0%-3%", "3%-5%", "5%-7%", "7%-10%"],
    "收盘位置系数": ["0.7-0.8", "0.8-0.9", "0.9-1.0"],
    "量比": ["<0.8", "0.8-1.2", "1.2-1.8", "1.8-2.5"],
    "MA5乖离": ["<0%", "0%-3%", "3%-6%", "6%-10%"],
    "RSI6": ["<40", "40-55", "55-70", "70-85"],
    "KDJ-J": ["<50", "50-70", "70-90", "90-110"],
    "连续上涨天数": ["0天", "1天", "2天"],
}


def apply_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for label, (column, edges, labels, _) in BIN_SPECS.items():
        result[f"bin_{label}"] = pd.cut(
            result[column],
            bins=edges,
            labels=labels,
            right=False,
            include_lowest=True,
        )
    return result


def interval_table(practical_universe: pd.DataFrame) -> pd.DataFrame:
    binned = apply_bins(practical_universe)
    rows: list[dict[str, object]] = []
    for metric, (_, _, _, requested_labels) in BIN_SPECS.items():
        source_column = BIN_SPECS[metric][0]
        coverage = binned[source_column].notna().mean() * 100
        for interval in requested_labels:
            group = binned[binned[f"bin_{metric}"].astype(str) == interval]
            rows.append(
                {
                    "指标": metric,
                    "区间": interval,
                    "样本数": len(group),
                    "数据覆盖率%": coverage,
                    "次日首板概率%": group["practical_first_board"].mean() * 100,
                    "次日涨停概率%": group["limit_up"].mean() * 100,
                    "次日平均涨跌幅%": group["pct_chg"].mean(),
                    "次日上涨率%": (group["pct_chg"] > 0).mean() * 100,
                    "次日跑赢大盘率%": group["stock_stronger_market"].mean() * 100,
                    "次日最大回撤风险%": group["low_return"].min(),
                    "次日低点收益P5%": group["low_return"].quantile(0.05),
                    "说明": "历史换手率仅覆盖极少日期，不用于策略优化"
                    if metric == "换手率"
                    else "",
                }
            )
    return pd.DataFrame(rows)


def choose_training_bins(
    practical_universe: pd.DataFrame, study_dates: list[str]
) -> dict[str, str]:
    available_dates = sorted(practical_universe["trade_date"].unique())
    split = max(1, int(len(available_dates) * 0.70))
    train_dates = available_dates[:split]
    train = apply_bins(
        practical_universe[practical_universe["trade_date"].isin(train_dates)]
    )
    best: dict[str, str] = {}
    for metric, (_, _, _, _) in BIN_SPECS.items():
        if metric == "换手率":
            continue
        candidates = []
        for interval in RISK_ALLOWED_BINS[metric]:
            group = train[train[f"bin_{metric}"].astype(str) == interval]
            if len(group) >= 200 and group["pct_chg"].mean() > -0.5:
                candidates.append(
                    (
                        group["practical_first_board"].mean(),
                        group["pct_chg"].mean(),
                        len(group),
                        interval,
                    )
                )
        if candidates:
            best[metric] = max(candidates)[3]
    return best


def validation_table(
    practical_universe: pd.DataFrame, best_bins: dict[str, str]
) -> tuple[pd.DataFrame, int]:
    binned = apply_bins(practical_universe)
    score = pd.Series(0, index=binned.index)
    for metric, interval in best_bins.items():
        score += (binned[f"bin_{metric}"].astype(str) == interval).astype(int)
    binned["interval_score"] = score
    binned["environment_gate"] = (
        (binned["prev_market_state"] != "退潮")
        & (binned["prev_sector_phase"] != "退潮")
        & (binned["prev_stock_relative_sector"] > 0)
    )
    available_dates = sorted(binned["trade_date"].unique())
    split = max(1, int(len(available_dates) * 0.70))
    train_dates = available_dates[:split]
    holdout_dates = available_dates[split:]
    train = binned[binned["trade_date"].isin(train_dates)]
    thresholds = []
    minimum_train_selection = max(1000, int(len(train) * 0.01))
    for threshold in range(1, len(best_bins) + 1):
        selected = train[
            train["environment_gate"] & (train["interval_score"] >= threshold)
        ]
        if len(selected) >= minimum_train_selection:
            thresholds.append(
                (
                    selected["practical_first_board"].mean(),
                    selected["pct_chg"].mean(),
                    len(selected),
                    threshold,
                )
            )
    selected_threshold = max(thresholds)[3] if thresholds else 1

    rows: list[dict[str, object]] = []
    for split_name, dates in [
        ("训练集", train_dates),
        ("验证集", holdout_dates),
        ("全部样本", available_dates),
    ]:
        base = binned[binned["trade_date"].isin(dates)]
        selected = base[
            base["environment_gate"]
            & (base["interval_score"] >= selected_threshold)
        ]
        for rule_name, group in [
            ("全体可交易样本", base),
            (f"环境过滤且得分≥{selected_threshold}", selected),
        ]:
            rows.append(
                {
                    "数据集": split_name,
                    "筛选规则": rule_name,
                    "样本数": len(group),
                    "首板概率%": group["practical_first_board"].mean() * 100,
                    "涨停概率%": group["limit_up"].mean() * 100,
                    "平均涨跌幅%": group["pct_chg"].mean(),
                    "上涨率%": (group["pct_chg"] > 0).mean() * 100,
                    "跑赢大盘率%": group["stock_stronger_market"].mean() * 100,
                    "最大回撤风险%": group["low_return"].min(),
                }
            )
    return pd.DataFrame(rows), selected_threshold


def tplus1_summary(events_all: pd.DataFrame) -> pd.DataFrame:
    complete = events_all[events_all["next_close_return"].notna()].copy()
    complete["next_limit_up"] = complete["next_limit_up"].fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    for board_class, group in complete.groupby("board_class"):
        rows.append(
            {
                "首板分类": board_class,
                "首板数量": len(group),
                "T+1连板率%": group["next_limit_up"].mean() * 100,
                "T+1平均开盘涨幅%": group["next_open_return"].mean(),
                "T+1平均最高涨幅%": group["next_high_return"].mean(),
                "T+1平均收盘涨幅%": group["next_close_return"].mean(),
                "T+1上涨率%": (group["next_close_return"] > 0).mean() * 100,
                "T+1冲高回落比例%": group["next_reversal"].mean() * 100,
                "T+1低开/核按钮代理%": group["next_nuclear_proxy"].mean() * 100,
                "T+1最大回撤风险%": group["next_low_return"].min(),
            }
        )
    all_row = {
        "首板分类": "全部首板",
        "首板数量": len(complete),
        "T+1连板率%": complete["next_limit_up"].mean() * 100,
        "T+1平均开盘涨幅%": complete["next_open_return"].mean(),
        "T+1平均最高涨幅%": complete["next_high_return"].mean(),
        "T+1平均收盘涨幅%": complete["next_close_return"].mean(),
        "T+1上涨率%": (complete["next_close_return"] > 0).mean() * 100,
        "T+1冲高回落比例%": complete["next_reversal"].mean() * 100,
        "T+1低开/核按钮代理%": complete["next_nuclear_proxy"].mean() * 100,
        "T+1最大回撤风险%": complete["next_low_return"].min(),
    }
    return pd.concat([pd.DataFrame([all_row]), pd.DataFrame(rows)], ignore_index=True)


def continuation_feature_table(events_all: pd.DataFrame) -> pd.DataFrame:
    complete = events_all[events_all["next_close_return"].notna()].copy()
    complete["next_limit_up"] = complete["next_limit_up"].fillna(False).astype(bool)
    continued = complete[complete["next_limit_up"]]
    failed = complete[~complete["next_limit_up"]]
    fields = [
        ("T日开盘涨幅", "open_return"),
        ("T日振幅", "amplitude_calc"),
        ("T日量比", "volume_to_ma5"),
        ("T日成交额/5日均额", "amount_to_ma5"),
        ("T日行业涨停家数", "sector_limit_count"),
        ("T日市场涨停家数", "market_limit_count"),
        ("T日市场炸板率", "market_broken_rate"),
        ("一字首板比例", "one_word_board"),
        ("强首板代理比例", "strong_board_proxy"),
        ("开盘涨停后开板代理比例", "opening_limit_broken_proxy"),
    ]
    rows = []
    for label, column in fields:
        left = pd.to_numeric(continued[column], errors="coerce")
        right = pd.to_numeric(failed[column], errors="coerce")
        rows.append(
            {
                "指标": label,
                "连板组均值": left.mean(),
                "未连板组均值": right.mean(),
                "标准化差异": standardized_difference(left, right),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "标准化差异", key=lambda values: values.abs(), ascending=False
    )


def add_chart_header(
    fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str
) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(
        left,
        0.97,
        textwrap.fill(title, 80),
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        left,
        0.915,
        textwrap.fill(subtitle, 115),
        ha="left",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    sns.despine(ax=ax)


def set_chart_style() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": SURFACE,
            "axes.facecolor": PANEL,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "grid.color": GRID,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "Microsoft YaHei",
        },
    )


def save_charts(
    comparison: pd.DataFrame,
    intervals: pd.DataFrame,
    tplus1: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    set_chart_style()

    top = (
        comparison[comparison["类型"] != "unavailable"]
        .head(20)
        .sort_values("稳健区分度")
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(
        data=top,
        x="稳健区分度",
        y="指标",
        color=BLUE,
        edgecolor=BLUE_DARK,
        linewidth=0.7,
        ax=ax,
    )
    ax.axvline(0.20, color=ORANGE_DARK, linestyle="--", linewidth=1)
    ax.set_xlabel("三类对照平均绝对标准化差异")
    ax.set_ylabel("")
    add_chart_header(
        fig,
        ax,
        "首板前一日最有区分度的20项特征",
        "稳健区分度为实验组相对A/B/C三类对照的平均绝对SMD；虚线为0.20参考阈值",
    )
    fig.savefig(ASSET_DIR / "top_feature_differences.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    heat = intervals[intervals["指标"] != "换手率"].pivot(
        index="指标", columns="区间", values="次日首板概率%"
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        heat,
        cmap=sns.light_palette(BLUE_DARK, as_cmap=True),
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor=GRID,
        cbar_kws={"label": "次日首板概率 (%)"},
        ax=ax,
    )
    ax.set_xlabel("区间")
    ax.set_ylabel("")
    add_chart_header(
        fig,
        ax,
        "T-1特征区间与次日首板概率",
        "基于全部可交易合格股票日样本；一字板不计入首板成功事件",
    )
    fig.savefig(ASSET_DIR / "interval_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    class_rows = tplus1[tplus1["首板分类"] != "全部首板"].copy()
    long_tplus1 = class_rows.melt(
        id_vars="首板分类",
        value_vars=["T+1连板率%", "T+1上涨率%", "T+1冲高回落比例%"],
        var_name="指标",
        value_name="比例",
    )
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.barplot(
        data=long_tplus1,
        x="首板分类",
        y="比例",
        hue="指标",
        palette={
            "T+1连板率%": BLUE,
            "T+1上涨率%": OLIVE,
            "T+1冲高回落比例%": ORANGE,
        },
        edgecolor=INK,
        linewidth=0.5,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("比例 (%)")
    ax.legend(title="", frameon=False, loc="upper right")
    add_chart_header(
        fig,
        ax,
        "不同首板代理分类的T+1表现",
        "强板与烂板由日线代理分类；缺少封板时间、封单金额与盘中开板次数",
    )
    fig.savefig(ASSET_DIR / "tplus1_by_class.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=validation,
        x="数据集",
        y="首板概率%",
        hue="筛选规则",
        palette=[NEUTRAL, BLUE],
        edgecolor=INK,
        linewidth=0.5,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("首板概率 (%)")
    ax.legend(title="", frameon=False, loc="upper left")
    add_chart_header(
        fig,
        ax,
        "策略A的训练集与时间外验证",
        "最优区间仅在前70%有效日期选择，后30%日期只用于验证",
    )
    fig.savefig(ASSET_DIR / "strategy_a_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt_table(
    frame: pd.DataFrame, percent_columns: list[str] | None = None
) -> str:
    display = frame.copy()
    percent_columns = percent_columns or []
    for column in display.columns:
        if column in percent_columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}%"
            )
        elif pd.api.types.is_numeric_dtype(display[column]):
            if column not in {"样本数", "首板数量"}:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{value:.3f}"
                )
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def make_report(
    universe: pd.DataFrame,
    practical_universe: pd.DataFrame,
    events_all: pd.DataFrame,
    events: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    study_dates: list[str],
    table_counts: dict[str, int],
    comparison: pd.DataFrame,
    intervals: pd.DataFrame,
    best_bins: dict[str, str],
    validation: pd.DataFrame,
    threshold: int,
    tplus1: pd.DataFrame,
    continuation: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top = comparison[comparison["类型"] != "unavailable"].head(20)
    selected_validation = validation[
        (validation["数据集"] == "验证集")
        & validation["筛选规则"].str.startswith("环境过滤")
    ].iloc[0]
    base_validation = validation[
        (validation["数据集"] == "验证集")
        & (validation["筛选规则"] == "全体可交易样本")
    ].iloc[0]
    validation_passed = (
        selected_validation["首板概率%"] > base_validation["首板概率%"]
        and selected_validation["平均涨跌幅%"] >= base_validation["平均涨跌幅%"]
    )
    validation_message = (
        "通过了本地时间外验证"
        if validation_passed
        else "未通过本地时间外验证，不能作为已验证策略部署"
    )
    tplus_all = tplus1[tplus1["首板分类"] == "全部首板"].iloc[0]
    class_best = tplus1[
        tplus1["首板分类"].isin(["强首板（日线代理）", "烂板首板（日线代理）", "一字首板"])
    ].sort_values("T+1连板率%", ascending=False).iloc[0]
    distinct = comparison[comparison["是否有区分度"] == "是"].head(12)
    best_lines = "".join(
        f"<li><strong>{html.escape(metric)}</strong>：{html.escape(interval)}</li>"
        for metric, interval in best_bins.items()
    )
    control_counts = "；".join(
        f"{name} {len(control):,}条" for name, control in controls.items()
    )
    exact_rates = "；".join(
        f"{name}同行业精确匹配{control['exact_date_industry_match'].mean() * 100:.1f}%"
        for name, control in controls.items()
        if len(control)
    )

    comparison_display = comparison[
        [
            "指标",
            "首板实验组均值",
            "对照组A均值",
            "对照组B均值",
            "对照组C均值",
            "实验组中位数",
            "对照组中位数",
            "差异方向",
            "稳健区分度",
            "是否有区分度",
        ]
    ].head(32)
    interval_display = intervals.copy()
    tplus_display = tplus1.copy()
    validation_display = validation.copy()

    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A股首板事件研究：从T日首板倒推T-1特征</title>
<style>
body {{ margin:0; background:{SURFACE}; color:{INK}; font-family:"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.65; }}
main {{ max-width:1180px; margin:0 auto; padding:36px 28px 80px; }}
h1 {{ font-size:30px; margin:0 0 12px; }} h2 {{ margin-top:42px; font-size:22px; }} h3 {{ margin-top:28px; font-size:17px; }}
.muted {{ color:{MUTED}; }} .summary {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:24px 0; }}
.card {{ background:white; border:1px solid {GRID}; border-radius:12px; padding:18px; }}
.value {{ font-size:25px; font-weight:700; color:{BLUE_DARK}; }} .label {{ color:{MUTED}; font-size:13px; }}
.callout {{ border-left:4px solid {BLUE_DARK}; background:#EAF1FE; padding:14px 18px; margin:20px 0; }}
.warning {{ border-left-color:{ORANGE_DARK}; background:#FFEDDE; }}
.chart {{ width:100%; background:white; border:1px solid {GRID}; border-radius:12px; margin:16px 0; }}
.data-table {{ border-collapse:collapse; width:100%; font-size:13px; background:white; }}
.data-table th,.data-table td {{ border-bottom:1px solid {GRID}; padding:8px 10px; text-align:right; white-space:nowrap; }}
.data-table th:first-child,.data-table td:first-child {{ text-align:left; }}
.table-wrap {{ overflow-x:auto; border:1px solid {GRID}; border-radius:10px; margin:14px 0; }}
code {{ background:#F4F5F7; padding:2px 5px; border-radius:4px; }}
@media(max-width:800px) {{ .summary {{ grid-template-columns:repeat(2,1fr); }} main {{ padding:22px 14px 60px; }} }}
</style>
</head>
<body><main>
<h1>A股首板事件研究：从T日首板倒推T-1特征</h1>
<p class="muted">本地Tushare行情库 · {study_dates[0]} 至 {study_dates[-1]} · 三类分层对照 · 一字板单列 · T+1接力分析</p>

<h2>技术摘要</h2>
<div class="summary">
<div class="card"><div class="value">{len(events_all):,}</div><div class="label">全部首板事件</div></div>
<div class="card"><div class="value">{len(events):,}</div><div class="label">非一字可交易首板</div></div>
<div class="card"><div class="value">{len(events_all[events_all['one_word_board']]):,}</div><div class="label">一字首板，单独统计</div></div>
<div class="card"><div class="value">{practical_universe['practical_first_board'].mean() * 100:.2f}%</div><div class="label">全体可交易样本首板基准概率</div></div>
</div>
<div class="callout">
本地库只有 <strong>{len(study_dates)}个交易日</strong>，少于用户要求的120/250日；为了严格剔除上市不足60个交易日的新股，本研究要求每只股票在本地库内至少已有60条历史交易记录，因此实际首板事件从 <strong>{events_all['trade_date'].min()}</strong> 开始。样本数量已超过300，但结论仍需更长历史验证。
</div>
<p><strong>核心结论：</strong>三类对照下最稳健的前一日差异集中在
{html.escape("、".join(distinct["指标"].head(6).tolist()))}。市场和行业情绪更多承担风险过滤作用，个股T-1相对行业强度、价格位置和量价状态更适合排序。策略A在时间外验证集的首板概率为
<strong>{selected_validation['首板概率%']:.2f}%</strong>，基准为 <strong>{base_validation['首板概率%']:.2f}%</strong>，并且<strong>{validation_message}</strong>。该结果是描述性事件研究，不等同于可成交实盘收益。</p>
<p><strong>首板后T+1：</strong>全部首板次日连板率为 <strong>{tplus_all['T+1连板率%']:.2f}%</strong>，平均开盘涨幅
<strong>{tplus_all['T+1平均开盘涨幅%']:.2f}%</strong>，平均最高涨幅 <strong>{tplus_all['T+1平均最高涨幅%']:.2f}%</strong>，平均收盘涨幅
<strong>{tplus_all['T+1平均收盘涨幅%']:.2f}%</strong>。日线代理分类中连板率最高的是
<strong>{html.escape(class_best['首板分类'])}</strong>，但缺少封板时间、开板次数和封单强度，不能把代理分类当作真实板质量结论。</p>

<h2>样本、首板定义与数据缺口</h2>
<ul>
<li>涨停识别优先级本应使用涨停价表，但本地 <code>stock_limit</code> 为 {table_counts['stock_limit']} 条，因此按主板10%、创业板/科创板20%、北交所30%的理论涨停价并按分位四舍五入近似识别。</li>
<li>首板定义为T日收盘涨停且前5个交易日没有涨停；这比“前两日非涨停”更保守，用于近似“本轮第一次涨停”。</li>
<li>一字首板定义为开、高、低、收均处于理论涨停价；共 {len(events_all[events_all['one_word_board']]):,} 条，不纳入策略A与区间命中率。</li>
<li>强首板/烂板仅为日线代理：非一字、开盘不弱、振幅相对涨停幅度较小且不过度放量定义为强首板代理；其他非一字首板归为烂板代理。</li>
<li>封板时间、封单金额、盘中开板次数和尾盘抢筹无法从日线识别；moneyflow表为 {table_counts['moneyflow']} 条，资金面不可验证；历史换手率仅覆盖少量日期，不用于策略优化。</li>
</ul>

<h2>实验组与三类对照</h2>
<p>实验组为非一字首板。{control_counts}。匹配优先使用同日期、同行业、成交额最接近的股票，缺少同业候选时回退到同日期；{exact_rates}。</p>
<div class="callout warning">对照组C的同行业精确匹配率明显低于A/B组，因此涉及炸板组的行业层面差异可信度较低；稳健结论应优先看A/B方向一致、且C组方向不冲突的指标。</div>
<ul>
<li><strong>对照组A：</strong>同日同行业未涨停股票，用于识别首板与普通股票的差异。</li>
<li><strong>对照组B：</strong>T日上涨3%—8%但未涨停股票，用于识别“强但未封板”的差异。</li>
<li><strong>对照组C：</strong>T日触及理论涨停价但收盘未封住的炸板股票，用于识别封板与失败冲板的差异。</li>
</ul>

<h2>首板前一日真正不同的特征</h2>
<p>下图按实验组相对A/B/C三类对照的平均绝对标准化差异排序。区分度要求平均绝对SMD至少0.20，且三类对照的方向基本一致。</p>
<img class="chart" src="first_board_event_study_assets/top_feature_differences.png" alt="Top feature differences">
<div class="table-wrap">{fmt_table(comparison_display)}</div>

<h2>哪些T-1区间更容易出现首板</h2>
<p>区间统计使用全部非一字可交易股票日样本。最大回撤风险为该区间T日最低收益的最差值，并同时保留低点收益P5，避免仅由单个极端值决定判断。换手率历史覆盖不足，仅展示，不用于最优区间选择。</p>
<img class="chart" src="first_board_event_study_assets/interval_heatmap.png" alt="Interval heatmap">
<div class="table-wrap">{fmt_table(interval_display, ['数据覆盖率%','次日首板概率%','次日涨停概率%','次日平均涨跌幅%','次日上涨率%','次日跑赢大盘率%','次日最大回撤风险%','次日低点收益P5%'])}</div>

<h2>策略A：T-1收盘后预测次日首板</h2>
<p>最优区间只在前70%的有效日期中选择，并排除量比、乖离、RSI、KDJ和连续上涨天数的过热尾部；后30%日期仅用于验证。训练集自动选择出的风险控制区间为：</p>
<ul>{best_lines}</ul>
<img class="chart" src="first_board_event_study_assets/strategy_a_validation.png" alt="Strategy A validation">
<div class="table-wrap">{fmt_table(validation_display, ['首板概率%','涨停概率%','平均涨跌幅%','上涨率%','跑赢大盘率%','最大回撤风险%'])}</div>
<div class="callout warning"><strong>验证结论：</strong>{validation_message}。以下公式只能作为待继续验证的研究观察池，不能解释为已经稳定提高首板概率的可部署模型。训练期与验证期市场收益分布明显不同，说明首板模式存在显著的市场阶段漂移。</div>
<ol>
<li><strong>大盘环境：</strong>T-1市场状态不得为退潮；若炸板率高、跌停家数扩张或市场宽度低于40%，降低观察等级。</li>
<li><strong>板块环境：</strong>T-1行业不得处于退潮阶段；行业强于全A、行业涨停家数增加、行业近3日相对收益为正作为加分项。</li>
<li><strong>个股位置：</strong>命中上述训练期最优风险控制区间，且T-1个股相对行业强度大于0。</li>
<li><strong>技术与交易面：</strong>优先站上MA5/MA10/MA20、MA5向上、DIF上拐；剔除长上影、放量滞涨、高开低走和明显过热；成交额必须不低于1亿元。</li>
<li><strong>资金面：</strong>当前不可验证。未来moneyflow可用后，再单独测试主力/大单净流入能否提高时间外首板概率。</li>
<li><strong>打分：</strong>{len(best_bins)}项区间条件每项1分，同时通过大盘、行业、相对行业强度过滤；训练期自动阈值为至少 <strong>{threshold}分</strong>。</li>
<li><strong>分级：</strong>研究A级=通过全部环境过滤且得分≥{threshold + 1}；研究B级=通过过滤且得分={threshold}；其余不进入观察池。由于时间外验证未通过，当前所有等级均只用于复盘研究。</li>
</ol>

<h2>首板后的T+1表现与策略B</h2>
<p>下图和表格把一字首板单列。强板与烂板是日线代理，不包含真实封板时间、封单金额或开板次数。</p>
<p><strong>关键结果：</strong>日线强首板代理与烂板代理的连板率接近，不能单独用于接力判断。连板组相对未连板组的T日振幅更低（{continuation.loc[continuation['指标'] == 'T日振幅', '连板组均值'].iloc[0]:.2f}% 对 {continuation.loc[continuation['指标'] == 'T日振幅', '未连板组均值'].iloc[0]:.2f}%）、量比更低（{continuation.loc[continuation['指标'] == 'T日量比', '连板组均值'].iloc[0]:.2f} 对 {continuation.loc[continuation['指标'] == 'T日量比', '未连板组均值'].iloc[0]:.2f}）、成交额/5日均额更低（{continuation.loc[continuation['指标'] == 'T日成交额/5日均额', '连板组均值'].iloc[0]:.2f} 对 {continuation.loc[continuation['指标'] == 'T日成交额/5日均额', '未连板组均值'].iloc[0]:.2f}）。这支持“封板过程更克制”而不是“越放量越好”。</p>
<img class="chart" src="first_board_event_study_assets/tplus1_by_class.png" alt="T plus one by class">
<div class="table-wrap">{fmt_table(tplus_display, ['T+1连板率%','T+1平均开盘涨幅%','T+1平均最高涨幅%','T+1平均收盘涨幅%','T+1上涨率%','T+1冲高回落比例%','T+1低开/核按钮代理%','T+1最大回撤风险%'])}</div>
<h3>连板组与未连板组的T日差异</h3>
<div class="table-wrap">{fmt_table(continuation)}</div>
<ol>
<li><strong>首板质量：</strong>日线强/烂代理本身没有明显区分度；候选观察条件改为T日振幅约不高于10%、量比约不高于1.5、成交额/5日均额约不高于1.6，并要求不是异常爆量封板。一字首板只统计，不作为普通接力候选。</li>
<li><strong>封板时间、开板次数、封单强度：</strong>当前本地数据无法验证阈值。获得涨停事件明细后，应要求较早封板、较少开板、封单金额相对流通市值充足，并重新做样本外检验。</li>
<li><strong>换手率：</strong>历史覆盖不足，当前不能给出可信最优区间；实战前必须补齐后再验证。</li>
<li><strong>板块与辨识度：</strong>行业涨停家数在连板组与未连板组间没有明显区分度，因此只作为风险过滤；行业退潮时放弃接力，不因板块高潮而机械加分。</li>
<li><strong>次日观察条件：</strong>开盘不过度一致、竞价未显著弱于板块、开盘后承接正常才继续观察；大幅低开、快速跌破首板日关键价位或板块集体退潮时放弃。</li>
<li><strong>风险剔除：</strong>一字不可成交、尾盘勉强回封、T日异常放量、T+1高开低走、板块炸板率上升或市场退潮时不进入接力池。</li>
</ol>

<h2>限制、稳健性与后续验证</h2>
<ul>
<li>本地历史不足120日，且严格60交易日上市过滤压缩了有效事件窗口；应扩展至至少250交易日后复跑。</li>
<li>理论涨停价是近似识别；补齐stock_limit、limit_list_d、limit_step或盘口数据后，应核对特殊涨跌幅规则和真实封板质量。</li>
<li>当前只验证了事件概率和日线收益，没有模拟集合竞价成交、涨停不可买、手续费、滑点与仓位约束。</li>
<li>策略A的时间外验证仍只有后30%本地有效日期，需要滚动走样本和不同市场阶段复核。</li>
<li>报告仅用于量化复盘和短线观察池研究，不构成投资建议。</li>
</ul>
</main></body></html>"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def save_outputs(
    universe: pd.DataFrame,
    practical_universe: pd.DataFrame,
    events_all: pd.DataFrame,
    events: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    intervals: pd.DataFrame,
    best_bins: dict[str, str],
    validation: pd.DataFrame,
    tplus1: pd.DataFrame,
    continuation: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event_columns = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "board_class",
        "pct_chg",
        "open_return",
        "high_return",
        "amplitude_calc",
        "one_word_board",
        "opening_limit_broken_proxy",
        "amount",
        "amount_100m",
        "turnover_rate",
        "prev_market_state",
        "prev_sector_phase",
        "next_open_return",
        "next_high_return",
        "next_close_return",
        "next_limit_up",
        "next_reversal",
        "next_nuclear_proxy",
    ] + sorted({column for _, column, _ in METRICS})
    events_all[event_columns].to_csv(
        OUTPUT_DIR / "first_board_events_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    events[event_columns].to_csv(
        OUTPUT_DIR / "first_board_events_tradable.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for name, control in controls.items():
        control.to_csv(
            OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig"
        )
    comparison.to_csv(
        OUTPUT_DIR / "experiment_vs_controls.csv", index=False, encoding="utf-8-sig"
    )
    intervals.to_csv(
        OUTPUT_DIR / "interval_performance.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [{"指标": metric, "风险控制最优区间": interval} for metric, interval in best_bins.items()]
    ).to_csv(OUTPUT_DIR / "optimal_intervals.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(
        OUTPUT_DIR / "strategy_a_validation.csv", index=False, encoding="utf-8-sig"
    )
    tplus1.to_csv(
        OUTPUT_DIR / "tplus1_by_board_class.csv", index=False, encoding="utf-8-sig"
    )
    continuation.to_csv(
        OUTPUT_DIR / "tplus1_continuation_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compact = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "pct_chg",
        "high_return",
        "low_return",
        "limit_up",
        "first_board_all",
        "one_word_board",
        "practical_first_board",
        "market_return",
        "sector_return",
        "stock_relative_sector",
    ] + sorted({column for _, column, _ in METRICS})
    practical_universe[compact].to_csv(
        OUTPUT_DIR / "eligible_practical_universe.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    frame, dates, table_counts = load_and_engineer()
    universe, practical_universe, events_all, study_dates = build_study(frame, dates)
    events = events_all[~events_all["one_word_board"]].copy()
    controls = build_controls(events, practical_universe)
    comparison = comparison_table(events, controls)
    intervals = interval_table(practical_universe)
    best_bins = choose_training_bins(practical_universe, study_dates)
    validation, threshold = validation_table(practical_universe, best_bins)
    tplus1 = tplus1_summary(events_all)
    continuation = continuation_feature_table(events_all)

    save_outputs(
        universe,
        practical_universe,
        events_all,
        events,
        controls,
        comparison,
        intervals,
        best_bins,
        validation,
        tplus1,
        continuation,
    )
    save_charts(comparison, intervals, tplus1, validation)
    make_report(
        universe,
        practical_universe,
        events_all,
        events,
        controls,
        study_dates,
        table_counts,
        comparison,
        intervals,
        best_bins,
        validation,
        threshold,
        tplus1,
        continuation,
    )

    print(f"study_dates={study_dates[0]}..{study_dates[-1]} ({len(study_dates)})")
    print(f"eligible_universe={len(universe)}")
    print(f"practical_universe={len(practical_universe)}")
    print(f"first_board_all={len(events_all)}")
    print(f"first_board_tradable={len(events)}")
    print(f"one_word={events_all['one_word_board'].sum()}")
    print({name: len(control) for name, control in controls.items()})
    print(f"best_bins={best_bins}")
    print(f"threshold={threshold}")
    print(validation.to_string(index=False))
    print(tplus1.to_string(index=False))


if __name__ == "__main__":
    main()
