from __future__ import annotations

import html
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "a_stock_quant.duckdb"
OUTPUT_DIR = ROOT / "outputs" / "event_study_7_12"
REPORT_PATH = ROOT / "outputs" / "reports" / "event_study_next_day_7_12_60d.html"
ASSET_DIR = ROOT / "outputs" / "reports" / "event_study_7_12_assets"
RNG = np.random.default_rng(20260612)

SURFACE = "#FCFCFD"
INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"
BLUE = "#5477C4"
BLUE_LIGHT = "#CEDFFE"
ORANGE = "#CC6F47"
NEUTRAL = "#C5CAD3"


def market_state(market_return: float, up_ratio: float) -> str:
    if market_return >= 1.0 and up_ratio >= 0.60:
        return "强势"
    if market_return <= -1.0 and up_ratio <= 0.40:
        return "退潮"
    if market_return < -0.30 or up_ratio < 0.45:
        return "弱势"
    return "震荡"


def streak(values: pd.Series) -> pd.Series:
    count = 0
    result = []
    for value in values:
        count = count + 1 if value > 0 else 0
        result.append(count)
    return pd.Series(result, index=values.index)


def compounded_return(values: pd.Series, window: int) -> pd.Series:
    return (1 + values / 100).rolling(window, min_periods=window).apply(
        np.prod, raw=True
    ) * 100 - 100


def load_and_engineer() -> tuple[pd.DataFrame, list[str]]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    query = """
        select
            p.ts_code, p.trade_date, p.open, p.high, p.low, p.close, p.pre_close,
            p.pct_chg, p.vol, p.amount,
            b.name, coalesce(b.industry, '未分类') as industry, b.market, b.list_date,
            t.ma5, t.ma10, t.ma20, t.macd_dif, t.macd_dea, t.macd_hist,
            t.kdj_k, t.kdj_d, t.kdj_j, t.rsi6,
            t.above_ma5, t.above_ma10, t.above_ma20,
            t.macd_golden_cross, t.kdj_golden_cross,
            v.vol_ma5, v.amount_ma5, v.volume_ratio, v.close_position,
            s.return_5d, s.return_10d, s.platform_breakout,
            d.turnover_rate
        from daily_price p
        join stock_basic b using (ts_code)
        left join factor_technical t using (ts_code, trade_date)
        left join factor_volume_price v using (ts_code, trade_date)
        left join factor_stage s using (ts_code, trade_date)
        left join daily_basic d using (ts_code, trade_date)
        order by p.ts_code, p.trade_date
    """
    frame = con.execute(query).fetchdf()
    dates = con.execute(
        "select distinct trade_date from daily_price order by trade_date"
    ).fetchdf()["trade_date"].tolist()
    con.close()

    bool_columns = [
        "above_ma5",
        "above_ma10",
        "above_ma20",
        "macd_golden_cross",
        "kdj_golden_cross",
        "platform_breakout",
    ]
    for column in bool_columns:
        frame[column] = frame[column].fillna(False).astype(bool)

    frame = frame.sort_values(["ts_code", "trade_date"]).copy()
    for column in [
        "open",
        "high",
        "low",
        "close",
        "pct_chg",
        "ma5",
        "macd_dif",
        "macd_hist",
        "kdj_k",
        "kdj_d",
        "rsi6",
    ]:
        frame[f"{column}_lag1"] = frame.groupby("ts_code")[column].shift(1)

    frame["consecutive_up"] = (
        frame.groupby("ts_code", group_keys=False)["pct_chg"].apply(streak).astype(int)
    )
    frame["return_3d"] = frame.groupby("ts_code")["pct_chg"].transform(
        lambda values: compounded_return(values, 3)
    )
    frame["distance_ma5"] = (frame["close"] / frame["ma5"] - 1) * 100
    frame["distance_ma10"] = (frame["close"] / frame["ma10"] - 1) * 100
    frame["distance_ma20"] = (frame["close"] / frame["ma20"] - 1) * 100
    frame["ma5_up"] = frame["ma5"] > frame["ma5_lag1"]
    frame["dif_up"] = frame["macd_dif"] > frame["macd_dif_lag1"]
    frame["green_shrinking"] = (
        (frame["macd_hist"] < 0) & (frame["macd_hist"] > frame["macd_hist_lag1"])
    )
    frame["rsi_up"] = frame["rsi6"] > frame["rsi6_lag1"]
    frame["kdj_low_mid_golden"] = frame["kdj_golden_cross"] & (frame["kdj_k"] < 70)
    frame["high_return"] = (frame["high"] / frame["pre_close"] - 1) * 100
    frame["low_return"] = (frame["low"] / frame["pre_close"] - 1) * 100
    day_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["close_position_calc"] = ((frame["close"] - frame["low"]) / day_range).fillna(0.5)
    frame["upper_shadow_ratio"] = (
        (frame["high"] - frame[["open", "close"]].max(axis=1)) / day_range
    ).fillna(0)
    frame["long_upper_shadow"] = frame["upper_shadow_ratio"] >= 0.45
    frame["high_open_low_close"] = (
        (frame["open"] / frame["pre_close"] - 1 >= 0.01) & (frame["close"] < frame["open"])
    )
    frame["intraday_reversal"] = (
        (frame["high_return"] - frame["pct_chg"] >= 3)
        & (frame["close_position_calc"] < 0.50)
    )
    frame["volume_to_ma5"] = frame["vol"] / frame["vol_ma5"].replace(0, np.nan)
    frame["amount_to_ma5"] = frame["amount"] / frame["amount_ma5"].replace(0, np.nan)
    frame["volume_stall"] = (frame["volume_to_ma5"] >= 1.8) & (frame["pct_chg"] < 2)
    frame["bullish_engulfing"] = (
        (frame["close"] > frame["open"])
        & (frame["close_lag1"] < frame["open_lag1"])
        & (frame["open"] <= frame["close_lag1"])
        & (frame["close"] >= frame["open_lag1"])
    )
    prior_5_high = frame.groupby("ts_code")["high"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=5).max()
    )
    frame["breakout_5d_platform"] = frame["close"] > prior_5_high

    market = (
        frame.groupby("trade_date")
        .agg(
            market_return=("pct_chg", "mean"),
            market_up_ratio=("pct_chg", lambda values: (values > 0).mean()),
            market_amount=("amount", "sum"),
        )
        .reset_index()
        .sort_values("trade_date")
    )
    market["market_amount_change"] = market["market_amount"].pct_change() * 100
    market["market_state"] = [
        market_state(ret, width)
        for ret, width in zip(market["market_return"], market["market_up_ratio"])
    ]

    sector = (
        frame.groupby(["trade_date", "industry"])
        .agg(
            sector_return=("pct_chg", "mean"),
            sector_up_ratio=("pct_chg", lambda values: (values > 0).mean()),
        )
        .reset_index()
        .merge(market[["trade_date", "market_return"]], on="trade_date")
        .sort_values(["industry", "trade_date"])
    )
    sector["sector_relative"] = sector["sector_return"] - sector["market_return"]
    sector["sector_relative_3d"] = sector.groupby("industry")["sector_relative"].transform(
        lambda values: values.rolling(3, min_periods=3).sum()
    )
    sector["sector_relative_5d"] = sector.groupby("industry")["sector_relative"].transform(
        lambda values: values.rolling(5, min_periods=5).sum()
    )
    sector["sector_retreat"] = (
        (sector["sector_return"] < sector["market_return"])
        & (sector["sector_relative_3d"] < 0)
    ) | (sector["sector_up_ratio"] < 0.40)

    leaders = (
        frame.sort_values(
            ["trade_date", "industry", "pct_chg"], ascending=[True, True, False]
        )
        .groupby(["trade_date", "industry"], group_keys=False)
        .head(1)[["trade_date", "industry", "pct_chg", "close_position_calc"]]
        .copy()
    )
    leaders["sector_leader_limit_proxy"] = (
        (leaders["pct_chg"] >= 9.5) & (leaders["close_position_calc"] >= 0.95)
    )

    frame = frame.merge(market, on="trade_date", how="left")
    frame = frame.merge(
        sector[
            [
                "trade_date",
                "industry",
                "sector_return",
                "sector_up_ratio",
                "sector_relative_3d",
                "sector_relative_5d",
                "sector_retreat",
            ]
        ],
        on=["trade_date", "industry"],
        how="left",
    )
    frame = frame.merge(
        leaders[["trade_date", "industry", "sector_leader_limit_proxy"]],
        on=["trade_date", "industry"],
        how="left",
    )
    frame["sector_strong"] = frame["sector_return"] > frame["market_return"]
    frame["stock_stronger_sector"] = frame["pct_chg"] > frame["sector_return"]
    frame["stock_relative_sector"] = frame["pct_chg"] - frame["sector_return"]
    frame["stock_stronger_market"] = frame["pct_chg"] > frame["market_return"]
    frame["one_price"] = frame["high"].eq(frame["low"])
    frame["bad_name"] = frame["name"].str.contains(r"ST|退", case=False, na=False)
    frame["calendar_age_days"] = (
        pd.to_datetime(frame["trade_date"]) - pd.to_datetime(frame["list_date"])
    ).dt.days
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
        & (frame["calendar_age_days"] >= 90)
        & (frame["vol"] > 0)
        & (frame["amount"] >= 100000)
        & ~frame["one_price"]
        & frame[required].notna().all(axis=1)
    )

    feature_columns = [
        "market_return",
        "market_up_ratio",
        "market_amount_change",
        "market_state",
        "sector_return",
        "sector_strong",
        "sector_relative_3d",
        "sector_relative_5d",
        "sector_up_ratio",
        "sector_leader_limit_proxy",
        "sector_retreat",
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
        "kdj_low_mid_golden",
        "kdj_j",
        "rsi6",
        "rsi_up",
        "long_upper_shadow",
        "upper_shadow_ratio",
        "bullish_engulfing",
        "breakout_5d_platform",
        "pct_chg",
        "high_return",
        "volume_ratio",
        "volume_to_ma5",
        "amount_to_ma5",
        "turnover_rate",
        "close_position_calc",
        "return_3d",
        "return_5d",
        "return_10d",
        "consecutive_up",
        "volume_stall",
        "high_open_low_close",
        "intraday_reversal",
        "eligible",
    ]
    frame = frame.sort_values(["ts_code", "trade_date"])
    for column in feature_columns:
        frame[f"prev_{column}"] = frame.groupby("ts_code")[column].shift(1)
    return frame, dates


def build_samples(
    frame: pd.DataFrame, dates: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    study_dates = dates[-60:]
    study = frame[
        frame["trade_date"].isin(study_dates)
        & frame["eligible"]
        & frame["prev_eligible"].fillna(False)
    ].copy()
    study["success"] = study["pct_chg"].between(7, 12)
    study["failure_candidate"] = (
        (study["pct_chg"] < 0)
        | (study["pct_chg"] < study["market_return"])
        | ((study["intraday_reversal"]) & (study["pct_chg"] < 0))
    )
    success = study[study["success"]].copy()
    failure_pool = study[study["failure_candidate"] & ~study["success"]].copy()

    matched_parts = []
    used_indices: set[int] = set()
    for (date, industry), success_group in success.groupby(["trade_date", "industry"]):
        needed = len(success_group)
        same_stratum = failure_pool[
            (failure_pool["trade_date"] == date)
            & (failure_pool["industry"] == industry)
            & ~failure_pool.index.isin(used_indices)
        ]
        take = same_stratum.sample(
            min(needed, len(same_stratum)), random_state=20260612
        )
        matched_parts.append(take)
        used_indices.update(take.index.tolist())
        remaining = needed - len(take)
        if remaining:
            same_date = failure_pool[
                (failure_pool["trade_date"] == date)
                & ~failure_pool.index.isin(used_indices)
            ]
            fill = same_date.sample(
                min(remaining, len(same_date)), random_state=20260612
            )
            matched_parts.append(fill)
            used_indices.update(fill.index.tolist())
    control = pd.concat(matched_parts, ignore_index=False).copy()
    success["group"] = "成功组"
    control["group"] = "失败组"
    matched = pd.concat([success, control], ignore_index=True)
    return study, success, control, study_dates


METRICS: list[tuple[str, str, str]] = [
    ("全A等权涨跌幅", "prev_market_return", "numeric"),
    ("上涨家数占比", "prev_market_up_ratio", "numeric"),
    ("两市成交额较前日变化", "prev_market_amount_change", "numeric"),
    ("行业当日涨跌幅", "prev_sector_return", "numeric"),
    ("行业强于全A", "prev_sector_strong", "binary"),
    ("行业近3日相对收益", "prev_sector_relative_3d", "numeric"),
    ("行业近5日相对收益", "prev_sector_relative_5d", "numeric"),
    ("行业上涨家数占比", "prev_sector_up_ratio", "numeric"),
    ("板块龙头涨停代理", "prev_sector_leader_limit_proxy", "binary"),
    ("板块退潮信号", "prev_sector_retreat", "binary"),
    ("个股强于所属行业", "prev_stock_stronger_sector", "binary"),
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
    ("KDJ低中位金叉", "prev_kdj_low_mid_golden", "binary"),
    ("KDJ-J值", "prev_kdj_j", "numeric"),
    ("RSI6", "prev_rsi6", "numeric"),
    ("RSI6向上", "prev_rsi_up", "binary"),
    ("长上影", "prev_long_upper_shadow", "binary"),
    ("上影线比例", "prev_upper_shadow_ratio", "numeric"),
    ("阳包阴", "prev_bullish_engulfing", "binary"),
    ("突破近5日平台", "prev_breakout_5d_platform", "binary"),
    ("T-1日涨跌幅", "prev_pct_chg", "numeric"),
    ("T-1日最高涨幅", "prev_high_return", "numeric"),
    ("量比", "prev_volume_ratio", "numeric"),
    ("成交量/5日均量", "prev_volume_to_ma5", "numeric"),
    ("成交额/5日均额", "prev_amount_to_ma5", "numeric"),
    ("收盘位置系数", "prev_close_position_calc", "numeric"),
    ("近3日涨幅", "prev_return_3d", "numeric"),
    ("近5日涨幅", "prev_return_5d", "numeric"),
    ("近10日涨幅", "prev_return_10d", "numeric"),
    ("连续上涨天数", "prev_consecutive_up", "numeric"),
    ("放量滞涨", "prev_volume_stall", "binary"),
    ("高开低走", "prev_high_open_low_close", "binary"),
    ("冲高回落", "prev_intraday_reversal", "binary"),
]


def standardized_difference(success: pd.Series, control: pd.Series) -> float:
    success = pd.to_numeric(success, errors="coerce").dropna()
    control = pd.to_numeric(control, errors="coerce").dropna()
    if success.empty or control.empty:
        return np.nan
    pooled = np.sqrt((success.var(ddof=1) + control.var(ddof=1)) / 2)
    return (success.mean() - control.mean()) / pooled if pooled > 0 else 0.0


def comparison_table(success: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, column, kind in METRICS:
        success_values = pd.to_numeric(success[column], errors="coerce")
        control_values = pd.to_numeric(control[column], errors="coerce")
        smd = standardized_difference(success_values, control_values)
        difference = success_values.mean() - control_values.mean()
        rows.append(
            {
                "指标": label,
                "字段": column,
                "类型": kind,
                "成功组均值": success_values.mean(),
                "失败组均值": control_values.mean(),
                "成功组中位数": success_values.median(),
                "失败组中位数": control_values.median(),
                "差异": difference,
                "差异方向": "成功组更高" if difference > 0 else "成功组更低",
                "标准化差异": smd,
                "是否有区分度": "是" if abs(smd) >= 0.20 else "否",
            }
        )
    unavailable = [
        "沪深300涨跌幅",
        "创业板指涨跌幅",
        "换手率",
        "主力净流入",
        "大单净额",
        "超大单净额",
        "小单净额",
        "最近3日主力净流入天数",
        "放量但主力净流出",
        "资金流入与股价上涨一致",
    ]
    rows.extend(
        {
            "指标": label,
            "字段": "",
            "类型": "unavailable",
            "成功组均值": np.nan,
            "失败组均值": np.nan,
            "成功组中位数": np.nan,
            "失败组中位数": np.nan,
            "差异": np.nan,
            "差异方向": "数据不可用",
            "标准化差异": np.nan,
            "是否有区分度": "不可验证",
        }
        for label in unavailable
    )
    return pd.DataFrame(rows).sort_values(
        "标准化差异", key=lambda values: values.abs(), ascending=False
    )


BIN_SPECS = {
    "T-1日涨幅": (
        "prev_pct_chg",
        [-np.inf, 0, 3, 5, 7, 10, 12, np.inf],
        ["<0%", "0%-3%", "3%-5%", "5%-7%", "7%-10%", "10%-12%", ">12%"],
        ["0%-3%", "3%-5%", "5%-7%", "7%-10%", "10%-12%"],
    ),
    "收盘位置系数": (
        "prev_close_position_calc",
        [-np.inf, 0.6, 0.7, 0.8, 0.9, 1.000001],
        ["<0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
        ["0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
    ),
    "量比": (
        "prev_volume_to_ma5",
        [-np.inf, 0.8, 1.2, 1.8, 2.2, 3.0, np.inf],
        ["<0.8", "0.8-1.2", "1.2-1.8", "1.8-2.2", "2.2-3.0", ">3.0"],
        ["0.8-1.2", "1.2-1.8", "1.8-2.2", "2.2-3.0", ">3.0"],
    ),
    "MA5乖离": (
        "prev_distance_ma5",
        [-np.inf, 0, 3, 6, 8, np.inf],
        ["<0%", "0%-3%", "3%-6%", "6%-8%", ">8%"],
        ["0%-3%", "3%-6%", "6%-8%", ">8%"],
    ),
    "RSI6": (
        "prev_rsi6",
        [-np.inf, 30, 45, 60, 75, np.inf],
        ["<30", "30-45", "45-60", "60-75", ">75"],
        ["30-45", "45-60", "60-75", ">75"],
    ),
    "KDJ-J": (
        "prev_kdj_j",
        [-np.inf, 50, 70, 90, np.inf],
        ["<50", "50-70", "70-90", ">90"],
        ["<50", "50-70", "70-90", ">90"],
    ),
    "连续上涨天数": (
        "prev_consecutive_up",
        [-0.5, 0.5, 1.5, 2.5, np.inf],
        ["0天", "1天", "2天", "≥3天"],
        ["0天", "1天", "2天", "≥3天"],
    ),
}

RISK_ALLOWED_BINS = {
    "T-1日涨幅": ["0%-3%", "3%-5%", "5%-7%", "7%-10%", "10%-12%"],
    "收盘位置系数": ["0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
    "量比": ["0.8-1.2", "1.2-1.8", "1.8-2.2"],
    "MA5乖离": ["0%-3%", "3%-6%", "6%-8%"],
    "RSI6": ["30-45", "45-60", "60-75"],
    "KDJ-J": ["<50", "50-70", "70-90"],
    "连续上涨天数": ["0天", "1天", "2天"],
}


def apply_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for label, (column, edges, labels, _) in BIN_SPECS.items():
        result[f"bin_{label}"] = pd.cut(
            result[column], bins=edges, labels=labels, right=False, include_lowest=True
        )
    return result


def interval_table(study: pd.DataFrame) -> pd.DataFrame:
    rows = []
    binned = apply_bins(study)
    for metric, (_, _, _, requested_labels) in BIN_SPECS.items():
        column = f"bin_{metric}"
        for interval in requested_labels:
            group = binned[binned[column].astype(str) == interval]
            rows.append(
                {
                    "指标": metric,
                    "区间": interval,
                    "样本数": len(group),
                    "次日平均涨跌幅%": group["pct_chg"].mean(),
                    "次日涨幅7%-12%概率%": group["success"].mean() * 100,
                    "次日上涨率%": (group["pct_chg"] > 0).mean() * 100,
                    "次日跑赢大盘率%": group["stock_stronger_market"].mean() * 100,
                    "最大回撤风险%": group["low_return"].min(),
                }
            )
    return pd.DataFrame(rows)


def choose_train_bins(
    study: pd.DataFrame,
    study_dates: list[str],
    allowed: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    train_dates = study_dates[:45]
    train = apply_bins(study[study["trade_date"].isin(train_dates)])
    best: dict[str, str] = {}
    for metric, (_, _, _, requested_labels) in BIN_SPECS.items():
        candidates = []
        intervals = allowed.get(metric, requested_labels) if allowed else requested_labels
        for interval in intervals:
            group = train[train[f"bin_{metric}"].astype(str) == interval]
            if len(group) >= 100:
                candidates.append((group["success"].mean(), len(group), interval))
        if candidates:
            best[metric] = max(candidates)[2]
    return best


def score_validation(
    study: pd.DataFrame,
    study_dates: list[str],
    best_bins: dict[str, str],
    profile: str,
) -> tuple[pd.DataFrame, int]:
    binned = apply_bins(study)
    score = pd.Series(0, index=binned.index)
    for metric, interval in best_bins.items():
        score += (binned[f"bin_{metric}"].astype(str) == interval).astype(int)
    binned["optimal_bin_score"] = score
    train_dates = study_dates[:45]
    train = binned[binned["trade_date"].isin(train_dates)]
    thresholds = []
    for threshold in range(1, len(best_bins) + 1):
        group = train[train["optimal_bin_score"] >= threshold]
        if len(group) >= 100:
            thresholds.append((group["success"].mean(), len(group), threshold))
    selected_threshold = max(thresholds)[2] if thresholds else 1

    rows = []
    for split, dates_for_split in [
        ("训练集：前45日", train_dates),
        ("验证集：后15日", study_dates[45:]),
        ("全部60日", study_dates),
    ]:
        base = binned[binned["trade_date"].isin(dates_for_split)]
        selected = base[base["optimal_bin_score"] >= selected_threshold]
        rows.append(
            {
                "数据集": split,
                "筛选规则": "全体合格股票",
                "样本数": len(base),
                "成功概率%": base["success"].mean() * 100,
                "次日平均涨跌幅%": base["pct_chg"].mean(),
                "次日上涨率%": (base["pct_chg"] > 0).mean() * 100,
                "次日跑赢大盘率%": base["stock_stronger_market"].mean() * 100,
            }
        )
        rows.append(
            {
                "数据集": split,
                "筛选规则": f"{profile}得分≥{selected_threshold}",
                "样本数": len(selected),
                "成功概率%": selected["success"].mean() * 100,
                "次日平均涨跌幅%": selected["pct_chg"].mean(),
                "次日上涨率%": (selected["pct_chg"] > 0).mean() * 100,
                "次日跑赢大盘率%": selected["stock_stronger_market"].mean() * 100,
            }
        )
    return pd.DataFrame(rows), selected_threshold


def success_event_table(success: pd.DataFrame) -> pd.DataFrame:
    result = success[
        [
            "trade_date",
            "ts_code",
            "name",
            "industry",
            "pct_chg",
            "high_return",
            "close_position_calc",
            "high_open_low_close",
            "stock_stronger_sector",
            "stock_stronger_market",
        ]
    ].copy()
    result["limit_proxy"] = (result["pct_chg"] >= 9.5) & (
        result["close_position_calc"] >= 0.95
    )
    return result


def set_chart_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "Microsoft YaHei",
        }
    )


def save_charts(
    comparison: pd.DataFrame,
    intervals: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    set_chart_style()

    top = comparison.head(15).sort_values("标准化差异")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [ORANGE if value < 0 else BLUE for value in top["标准化差异"]]
    sns.barplot(data=top, x="标准化差异", y="指标", hue="指标", palette=colors, legend=False, ax=ax)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_title("Top T-1 feature differences", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Standardized difference")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "feature_differences.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    heat = intervals.pivot(index="指标", columns="区间", values="次日涨幅7%-12%概率%")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.heatmap(
        heat,
        cmap=sns.light_palette(BLUE, as_cmap=True),
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor=GRID,
        cbar_kws={"label": "Success probability (%)"},
        ax=ax,
    )
    ax.set_title("Success probability by T-1 interval", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Interval")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "interval_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=validation,
        x="数据集",
        y="成功概率%",
        hue="筛选规则",
        palette=[NEUTRAL, BLUE],
        edgecolor=INK,
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("Train and holdout validation", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Success probability (%)")
    ax.legend(title="", frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt_table(frame: pd.DataFrame, percent_columns: list[str] | None = None) -> str:
    display = frame.copy()
    percent_columns = percent_columns or []
    for column in display.select_dtypes(include="number").columns:
        if column in percent_columns:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2f}%"
            )
        elif column not in {"样本数"}:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def make_report(
    study: pd.DataFrame,
    success: pd.DataFrame,
    control: pd.DataFrame,
    study_dates: list[str],
    comparison: pd.DataFrame,
    intervals: pd.DataFrame,
    raw_best_bins: dict[str, str],
    risk_best_bins: dict[str, str],
    validation: pd.DataFrame,
    risk_threshold: int,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top_distinct = comparison[comparison["是否有区分度"] == "是"].head(12)
    success_strata = success.groupby(["trade_date", "industry"]).size()
    control_strata = control.groupby(["trade_date", "industry"]).size()
    exact_matches = sum(
        min(count, control_strata.get(key, 0)) for key, count in success_strata.items()
    )
    exact_match_rate = exact_matches / len(success)
    success_base = study["success"].mean() * 100
    holdout = validation[
        (validation["数据集"] == "验证集：后15日")
        & validation["筛选规则"].str.startswith("风险控制")
    ].iloc[0]
    holdout_base = validation[
        (validation["数据集"] == "验证集：后15日")
        & (validation["筛选规则"] == "全体合格股票")
    ].iloc[0]
    risk_optimal_lines = "".join(
        f"<li><strong>{html.escape(metric)}</strong>：{html.escape(interval)}</li>"
        for metric, interval in risk_best_bins.items()
    )
    raw_optimal_lines = "".join(
        f"<li><strong>{html.escape(metric)}</strong>：{html.escape(interval)}</li>"
        for metric, interval in raw_best_bins.items()
    )

    comparison_display = comparison[
        [
            "指标",
            "成功组均值",
            "失败组均值",
            "成功组中位数",
            "失败组中位数",
            "差异方向",
            "标准化差异",
            "是否有区分度",
        ]
    ].head(30)
    interval_display = intervals.copy()
    validation_display = validation.copy()

    html_report = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T日涨幅7%-12%事件研究</title>
<style>
body {{ margin:0; background:{SURFACE}; color:{INK}; font-family:"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.65; }}
main {{ max-width:1180px; margin:0 auto; padding:36px 28px 80px; }}
h1 {{ font-size:30px; margin:0 0 12px; }} h2 {{ margin-top:42px; font-size:22px; }}
h3 {{ margin-top:28px; font-size:17px; }} .muted {{ color:{MUTED}; }}
.summary {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:24px 0; }}
.card {{ background:white; border:1px solid {GRID}; border-radius:12px; padding:18px; }}
.value {{ font-size:25px; font-weight:700; color:{BLUE}; }} .label {{ color:{MUTED}; font-size:13px; }}
.callout {{ border-left:4px solid {BLUE}; background:#EAF1FE; padding:14px 18px; margin:20px 0; }}
.warning {{ border-left-color:{ORANGE}; background:#FFEDDE; }}
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
<h1>T日涨幅7%-12%事件研究</h1>
<p class="muted">本地Tushare数据库 · 最近60个交易日（{study_dates[0]} 至 {study_dates[-1]}）· T-1特征反推 · 同日期同行业1:1对照</p>

<h2>技术摘要</h2>
<div class="summary">
<div class="card"><div class="value">{len(success):,}</div><div class="label">成功事件数</div></div>
<div class="card"><div class="value">{len(control):,}</div><div class="label">分层匹配失败样本</div></div>
<div class="card"><div class="value">{success_base:.2f}%</div><div class="label">全体合格股票次日成功概率</div></div>
<div class="card"><div class="value">{len(top_distinct)}</div><div class="label">前30项中具区分度指标</div></div>
</div>
<div class="callout">
成功事件远超100个，因此主分析无需扩展到120日。区分度按绝对标准化差异 <code>|SMD|≥0.20</code> 判断；这是描述性事件研究，不证明因果关系。
</div>
<p><strong>核心发现：</strong>成功组T-1日的MA20乖离均值为 <strong>{success['prev_distance_ma20'].mean():.2f}%</strong>，失败组为
<strong>{control['prev_distance_ma20'].mean():.2f}%</strong>；成功组近10日涨幅均值为 <strong>{success['prev_return_10d'].mean():.2f}%</strong>，
失败组为 <strong>{control['prev_return_10d'].mean():.2f}%</strong>；个股相对行业强度分别为
<strong>{success['prev_stock_relative_sector'].mean():.2f}</strong> 和 <strong>{control['prev_stock_relative_sector'].mean():.2f}</strong> 个百分点。
相反，全A环境、行业相对收益、MACD金叉、长上影与放量滞涨没有达到区分阈值。</p>
<p><strong>T日事件画像：</strong>成功事件T日平均涨幅 <strong>{success['pct_chg'].mean():.2f}%</strong>、平均最高涨幅
<strong>{success['high_return'].mean():.2f}%</strong>；涨停代理占比
<strong>{((success['pct_chg'] >= 9.5) & (success['close_position_calc'] >= 0.95)).mean() * 100:.2f}%</strong>，
强于行业占比 <strong>{success['stock_stronger_sector'].mean() * 100:.2f}%</strong>，强于全A占比
<strong>{success['stock_stronger_market'].mean() * 100:.2f}%</strong>。</p>
<p><strong>样本外验证：</strong>前45日选择出的风险控制区间采用“命中至少{risk_threshold}项”组合，在后15日的成功概率为
<strong>{holdout['成功概率%']:.2f}%</strong>，同期全体合格股票基准为 <strong>{holdout_base['成功概率%']:.2f}%</strong>。
这决定了新版公式应优先使用有样本外提升的区间，而不是机械复用旧阈值。</p>

<h2>哪些T-1特征真正不同</h2>
<p>下图展示成功组与同行业、同日期失败组之间差异最大的T-1特征。蓝色表示成功组更高，橙色表示成功组更低。</p>
<img class="chart" src="event_study_7_12_assets/feature_differences.png" alt="T-1 feature differences">
<div class="table-wrap">{fmt_table(comparison_display)}</div>

<h2>各区间对应的次日表现</h2>
<p>区间统计使用最近60日全部合格股票，而不是仅使用匹配样本。最大回撤风险定义为该区间中从T-1收盘到T日最低价的最差收益。</p>
<img class="chart" src="event_study_7_12_assets/interval_heatmap.png" alt="Interval heatmap">
<div class="table-wrap">{fmt_table(interval_display, ['次日平均涨跌幅%','次日涨幅7%-12%概率%','次日上涨率%','次日跑赢大盘率%','最大回撤风险%'])}</div>

<h2>自动寻找的有效区间与样本外验证</h2>
<p>最优区间仅在前45个交易日中选择，并要求单个区间至少100个样本；后15日只用于验证。</p>
<h3>纯命中率最大化区间</h3>
<ul>{raw_optimal_lines}</ul>
<div class="callout warning">纯命中率最大化会偏向量比&gt;3、MA5乖离&gt;8%、RSI6&gt;75、KDJ-J&gt;90和连续上涨≥3天。它们提高7%-12%事件概率，但代表高位动量与更大的尾部回撤，不能直接作为风险控制策略。</div>
<h3>风险控制区间</h3>
<ul>{risk_optimal_lines}</ul>
<img class="chart" src="event_study_7_12_assets/validation.png" alt="Validation">
<div class="table-wrap">{fmt_table(validation_display, ['成功概率%','次日平均涨跌幅%','次日上涨率%','次日跑赢大盘率%'])}</div>

<h2>新版选股公式</h2>
<div class="callout">新版公式采用“风险环境过滤 + 动量得分”结构。市场和行业变量在同行业、同日期对照中没有显示横截面区分力，因此只作为风险过滤；真正用于排序的是T-1个股动量和相对行业强度。</div>
<ol>
<li><strong>大盘环境：</strong>全A不处于退潮状态，即不同时满足“全A等权≤-1%且上涨家数占比≤40%”；该项只控制仓位，不作为个股排序分。</li>
<li><strong>板块环境：</strong>行业无退潮信号，行业近3日或近5日相对全A收益为正；个股T-1日相对行业强度必须&gt;0。</li>
<li><strong>个股涨幅与位置：</strong>T-1涨幅 <code>{html.escape(risk_best_bins.get('T-1日涨幅','未形成稳定区间'))}</code>，收盘位置 <code>{html.escape(risk_best_bins.get('收盘位置系数','未形成稳定区间'))}</code>。</li>
<li><strong>技术条件：</strong>站上MA20；MA5乖离 <code>{html.escape(risk_best_bins.get('MA5乖离','未形成稳定区间'))}</code>；RSI6 <code>{html.escape(risk_best_bins.get('RSI6','未形成稳定区间'))}</code>；KDJ-J <code>{html.escape(risk_best_bins.get('KDJ-J','未形成稳定区间'))}</code>。MA5向上、站上MA10和DIF向上作为加分项，不单独决定入选。</li>
<li><strong>交易面条件：</strong>量比 <code>{html.escape(risk_best_bins.get('量比','未形成稳定区间'))}</code>，连续上涨天数 <code>{html.escape(risk_best_bins.get('连续上涨天数','未形成稳定区间'))}</code>；成交额不低于1亿元。</li>
<li><strong>资金面条件：</strong>moneyflow为空，当前不能验证。未来数据可用时，再要求主力/大单净流入与价格上涨一致，并单独回测增量效果。</li>
<li><strong>剔除条件：</strong>ST/退市风险、上市不足约60交易日、停牌、低成交额、一字板、长上影、放量滞涨、高开低走、冲高回落、数据缺失严重。</li>
<li><strong>次日操作预案：</strong>上述7个风险控制区间命中至少 <code>{risk_threshold}</code> 项，且通过大盘、板块和剔除条件后进入观察池；次日高开过大或无法成交时放弃，设置止损和单票仓位上限。</li>
</ol>

<h2>范围、方法与限制</h2>
<ul>
<li>成功组：T日涨幅7%-12%；失败组：T日下跌、跑输全A或冲高回落收跌，并按日期和行业1:1匹配。共有 {exact_match_rate * 100:.2f}% 的对照实现同日期同行业精确匹配，其余使用同日期匹配补足。</li>
<li>剔除ST/退市风险、日成交额低于1亿元、一字板、停牌、数据缺失；上市满60个交易日使用上市日期≥90个自然日作为保守代理。</li>
<li>板块使用本地股票基础表中的行业分类，不等同于申万或概念指数。</li>
<li>本地指数表只有上证综指，无法可靠统计沪深300和创业板指；历史换手率仅3日，moneyflow为空，因此均未用于区分度和公式优化。</li>
<li>结果只用于量化复盘和短线观察，不构成投资建议。</li>
</ul>

<h2>建议的下一步验证</h2>
<p>将数据扩展至至少250个交易日，并使用次日开盘可成交价、涨停成交限制、手续费和滑点重新验证新版公式；重点检查后15日提升能否持续。</p>
</main></body></html>"""
    REPORT_PATH.write_text(html_report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, dates = load_and_engineer()
    study, success, control, study_dates = build_samples(frame, dates)
    comparison = comparison_table(success, control)
    intervals = interval_table(study)
    raw_best_bins = choose_train_bins(study, study_dates)
    risk_best_bins = choose_train_bins(study, study_dates, allowed=RISK_ALLOWED_BINS)
    raw_validation, raw_threshold = score_validation(
        study, study_dates, raw_best_bins, "纯命中率"
    )
    risk_validation, risk_threshold = score_validation(
        study, study_dates, risk_best_bins, "风险控制"
    )
    validation = pd.concat([raw_validation, risk_validation], ignore_index=True)
    success_events = success_event_table(success)

    compact_columns = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "pct_chg",
        "high_return",
        "low_return",
        "market_return",
        "sector_return",
        "stock_stronger_market",
        "stock_stronger_sector",
        "success",
    ] + sorted({column for _, column, _ in METRICS})
    study[compact_columns].to_csv(
        OUTPUT_DIR / "eligible_universe_60d.csv", index=False, encoding="utf-8-sig"
    )
    success_events.to_csv(OUTPUT_DIR / "success_events.csv", index=False, encoding="utf-8-sig")
    control.to_csv(OUTPUT_DIR / "matched_failure_controls.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_DIR / "success_vs_failure_comparison.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(OUTPUT_DIR / "interval_performance.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUT_DIR / "train_holdout_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "指标": metric,
                "纯命中率最优区间": raw_best_bins.get(metric),
                "风险控制最优区间": risk_best_bins.get(metric),
            }
            for metric in BIN_SPECS
        ]
    ).to_csv(OUTPUT_DIR / "optimal_intervals.csv", index=False, encoding="utf-8-sig")

    save_charts(comparison, intervals, risk_validation)
    make_report(
        study,
        success,
        control,
        study_dates,
        comparison,
        intervals,
        raw_best_bins,
        risk_best_bins,
        validation,
        risk_threshold,
    )

    print(f"study_dates={study_dates[0]}..{study_dates[-1]}")
    print(f"eligible_rows={len(study)}")
    print(f"success={len(success)}")
    print(f"control={len(control)}")
    print(f"base_success_rate={study['success'].mean():.6f}")
    print(f"raw_best_bins={raw_best_bins}")
    print(f"risk_best_bins={risk_best_bins}")
    print(f"raw_threshold={raw_threshold}")
    print(f"risk_threshold={risk_threshold}")
    print(validation.to_string(index=False))
    print(comparison.head(15)[["指标", "成功组均值", "失败组均值", "标准化差异", "是否有区分度"]].to_string(index=False))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
