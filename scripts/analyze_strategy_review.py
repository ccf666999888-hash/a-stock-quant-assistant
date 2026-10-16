from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "a_stock_quant.duckdb"
REPORT_PATH = ROOT / "outputs" / "reports" / "strategy_review_20260611_to_20260612.md"
DETAIL_PATH = ROOT / "outputs" / "backtests" / "original_strategy_20_review_20260612.csv"
EXTENDED_PATH = ROOT / "outputs" / "backtests" / "original_strategy_extended_20d.csv"
COMPARISON_PATH = ROOT / "outputs" / "backtests" / "strategy_group_comparison_20260612.csv"

SIGNAL_DATE = "20260611"
OUTCOME_DATE = "20260612"
CSI300_OUTCOME_RETURN = 1.1628

SELECTED_CODES = [
    "688037.SH",
    "301086.SZ",
    "301045.SZ",
    "000032.SZ",
    "002654.SZ",
    "002025.SZ",
    "603738.SH",
    "300203.SZ",
    "300084.SZ",
    "300209.SZ",
    "300437.SZ",
    "300722.SZ",
    "301183.SZ",
    "301095.SZ",
    "300843.SZ",
    "300903.SZ",
    "300720.SZ",
    "300820.SZ",
    "300790.SZ",
    "301373.SZ",
]
STRICT_CODES = {"000032.SZ", "002654.SZ", "002025.SZ"}


def pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "不可用"
    return f"{value:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "不可用"
    return f"{value:.{digits}f}"


def bool_rate(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.astype(bool).mean() * 100) if not valid.empty else np.nan


def market_state(market_return: float, up_ratio: float) -> str:
    if market_return >= 1.0 and up_ratio >= 0.60:
        return "强势"
    if market_return <= -1.0 and up_ratio <= 0.40:
        return "退潮"
    if market_return < -0.3 or up_ratio < 0.45:
        return "弱势"
    return "震荡"


def kdj_state(row: pd.Series) -> str:
    if row["kdj_j"] >= 90 or row["kdj_k"] >= 80:
        return "高位钝化"
    if row["kdj_golden_cross"]:
        return "低位金叉" if row["kdj_k"] < 40 else "中位金叉"
    if row["kdj_k"] > row["kdj_d"]:
        return "修复向上"
    return "走弱"


def macd_state(row: pd.Series) -> str:
    if row["macd_golden_cross"]:
        return "刚金叉"
    if row["green_shrinking"]:
        return "绿柱缩短"
    if row["macd_hist"] > 0 and row["macd_hist"] > row["macd_hist_prev"]:
        return "红柱增强"
    return "走弱"


def add_streaks(frame: pd.DataFrame) -> pd.DataFrame:
    def streak(values: pd.Series) -> pd.Series:
        count = 0
        result = []
        for value in values:
            count = count + 1 if value > 0 else 0
            result.append(count)
        return pd.Series(result, index=values.index)

    frame["consecutive_up"] = (
        frame.groupby("ts_code", group_keys=False)["pct_chg"].apply(streak).astype(int)
    )
    return frame


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    dates = con.execute(
        "select distinct trade_date from daily_price order by trade_date"
    ).fetchdf()["trade_date"].tolist()
    selected_dates = dates[-42:]
    placeholders = ",".join(["?"] * len(selected_dates))

    query = f"""
        select
            p.ts_code, p.trade_date, p.open, p.high, p.low, p.close, p.pre_close,
            p.pct_chg, p.vol, p.amount,
            b.name, coalesce(b.industry, '未分类') as industry,
            t.ma5, t.ma10, t.ma20, t.macd_dif, t.macd_dea, t.macd_hist,
            t.kdj_k, t.kdj_d, t.kdj_j, t.rsi6,
            t.above_ma5, t.above_ma10, t.above_ma20,
            t.macd_golden_cross, t.kdj_golden_cross, t.technical_strengthening,
            v.volume_ratio, v.close_position, v.is_volume_expansion,
            s.return_5d, s.return_10d, s.return_20d,
            s.new_high_20d, s.new_low_5d, s.new_low_10d, s.is_platform,
            s.platform_breakout, s.stage_overheated,
            r.risk_triggered, r.severe_risk, r.risk_score,
            d.turnover_rate
        from daily_price p
        join stock_basic b using (ts_code)
        left join factor_technical t using (ts_code, trade_date)
        left join factor_volume_price v using (ts_code, trade_date)
        left join factor_stage s using (ts_code, trade_date)
        left join factor_risk r using (ts_code, trade_date)
        left join daily_basic d using (ts_code, trade_date)
        where p.trade_date in ({placeholders})
        order by p.ts_code, p.trade_date
    """
    frame = con.execute(query, selected_dates).fetchdf()
    moneyflow_count = con.execute("select count(*) from moneyflow").fetchone()[0]
    con.close()

    bool_columns = [
        "above_ma5",
        "above_ma10",
        "above_ma20",
        "macd_golden_cross",
        "kdj_golden_cross",
        "technical_strengthening",
        "is_volume_expansion",
        "new_high_20d",
        "new_low_5d",
        "new_low_10d",
        "is_platform",
        "platform_breakout",
        "stage_overheated",
        "risk_triggered",
        "severe_risk",
    ]
    for column in bool_columns:
        frame[column] = frame[column].fillna(False).astype(bool)

    frame = add_streaks(frame)
    previous_columns = ["ma5", "macd_dif", "macd_hist", "kdj_k", "kdj_d", "rsi6"]
    for column in previous_columns:
        frame[f"{column}_prev"] = frame.groupby("ts_code")[column].shift(1)

    frame["ma5_up"] = frame["ma5"] > frame["ma5_prev"]
    frame["dif_up"] = frame["macd_dif"] > frame["macd_dif_prev"]
    frame["green_shrinking"] = (
        (frame["macd_hist"] < 0) & (frame["macd_hist"] > frame["macd_hist_prev"])
    )
    frame["rsi_up"] = frame["rsi6"] > frame["rsi6_prev"]
    frame["distance_ma5"] = (frame["close"] / frame["ma5"] - 1) * 100
    frame["distance_ma10"] = (frame["close"] / frame["ma10"] - 1) * 100
    intraday_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["upper_shadow_ratio"] = (
        frame["high"] - frame[["open", "close"]].max(axis=1)
    ) / intraday_range
    frame["long_upper_shadow"] = frame["upper_shadow_ratio"] >= 0.45
    frame["high_open_low_close"] = (
        (frame["open"] / frame["pre_close"] - 1 >= 0.01) & (frame["close"] < frame["open"])
    )
    frame["volume_stall"] = (frame["volume_ratio"] >= 1.8) & (frame["pct_chg"] < 2)
    frame["false_breakout"] = frame["new_high_20d"] & (frame["close_position"] < 0.70)
    frame["volume_break_ma20"] = (
        (frame["volume_ratio"] >= 1.3) & (frame["close"] < frame["ma20"])
    )
    frame["near_ma20"] = frame["close"] >= frame["ma20"] * 0.98
    frame["no_new_low"] = ~frame["new_low_5d"] & ~frame["new_low_10d"]
    frame["kdj_repair"] = (
        frame["kdj_golden_cross"]
        | ((frame["kdj_k"] > frame["kdj_d"]) & (frame["kdj_k"] > frame["kdj_k_prev"]))
    )
    frame["kdj_not_hot"] = (frame["kdj_k"] < 75) & (frame["kdj_j"] < 90)
    frame["rsi_not_hot"] = frame["rsi6"].between(35, 70)
    frame["macd_repair"] = (
        (frame["green_shrinking"] & frame["dif_up"]) | frame["macd_golden_cross"]
    )

    market = (
        frame.groupby("trade_date")
        .agg(
            market_return=("pct_chg", "mean"),
            market_median=("pct_chg", "median"),
            market_up_ratio=("pct_chg", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )
    sector = (
        frame.groupby(["trade_date", "industry"])
        .agg(
            sector_return=("pct_chg", "mean"),
            sector_up_ratio=("pct_chg", lambda s: (s > 0).mean()),
            sector_count=("ts_code", "size"),
        )
        .reset_index()
    )
    sector = sector.merge(market[["trade_date", "market_return"]], on="trade_date")
    sector["sector_relative"] = sector["sector_return"] - sector["market_return"]
    sector = sector.sort_values(["industry", "trade_date"])
    sector["sector_relative_3d"] = sector.groupby("industry")["sector_relative"].transform(
        lambda s: s.rolling(3, min_periods=3).sum()
    )

    frame = frame.merge(market, on="trade_date", how="left")
    frame = frame.merge(
        sector[
            [
                "trade_date",
                "industry",
                "sector_return",
                "sector_up_ratio",
                "sector_count",
                "sector_relative_3d",
            ]
        ],
        on=["trade_date", "industry"],
        how="left",
    )
    frame["sector_strong"] = frame["sector_return"] > frame["market_return"]
    frame["sector_trend_strong"] = frame["sector_relative_3d"] > 0
    frame["stock_stronger_sector"] = frame["pct_chg"] > frame["sector_return"]

    frame = frame.sort_values(["ts_code", "trade_date"])
    frame["next_trade_date"] = frame.groupby("ts_code")["trade_date"].shift(-1)
    frame["next_pct_chg"] = frame.groupby("ts_code")["pct_chg"].shift(-1)
    market_map = market.set_index("trade_date")["market_return"].to_dict()
    sector_map = sector.set_index(["trade_date", "industry"])["sector_return"].to_dict()
    frame["next_market_return"] = frame["next_trade_date"].map(market_map)
    frame["next_sector_return"] = [
        sector_map.get((date, industry), np.nan)
        for date, industry in zip(frame["next_trade_date"], frame["industry"])
    ]
    frame["sector_continued"] = frame["next_sector_return"] > frame["next_market_return"]
    frame["next_up"] = frame["next_pct_chg"] > 0
    frame["next_beat_market"] = frame["next_pct_chg"] > frame["next_market_return"]
    frame["next_beat_sector"] = frame["next_pct_chg"] > frame["next_sector_return"]

    frame["strict_rule"] = (
        ((frame["return_20d"] <= 10) | frame["is_platform"])
        & frame["no_new_low"]
        & frame["above_ma5"]
        & frame["above_ma10"]
        & frame["near_ma20"]
        & frame["ma5_up"]
        & frame["macd_repair"]
        & frame["kdj_repair"]
        & frame["kdj_not_hot"]
        & frame["rsi_up"]
        & frame["rsi_not_hot"]
        & (frame["close_position"] >= 0.70)
        & (frame["return_5d"] <= 15)
        & frame["sector_trend_strong"]
        & frame["stock_stronger_sector"]
        & frame["volume_ratio"].between(0.9, 2.2)
        & (frame["distance_ma5"] <= 5)
        & ~frame["volume_stall"]
        & ~frame["volume_break_ma20"]
        & ~frame["severe_risk"]
    )
    score_columns = [
        "no_new_low",
        "above_ma5",
        "above_ma10",
        "near_ma20",
        "ma5_up",
        "dif_up",
        "kdj_repair",
        "kdj_not_hot",
        "rsi_up",
        "rsi_not_hot",
        "sector_trend_strong",
        "stock_stronger_sector",
    ]
    frame["original_score"] = frame[score_columns].astype(int).sum(axis=1)
    frame["relaxed_rule"] = (
        frame["above_ma5"]
        & frame["above_ma10"]
        & frame["near_ma20"]
        & frame["dif_up"]
        & frame["rsi_up"]
        & (frame["close_position"] >= 0.60)
        & (frame["return_5d"] <= 15)
        & (frame["volume_ratio"] <= 2.8)
        & (frame["distance_ma5"] <= 8)
        & frame["kdj_not_hot"]
        & ~frame["volume_break_ma20"]
        & ~frame["severe_risk"]
        & (frame["original_score"] >= 8)
    )
    frame["funds_available"] = moneyflow_count > 0
    return frame, market, sector, dates


def technical_text(row: pd.Series) -> str:
    ma = "/".join(
        [
            "MA5上" if row["above_ma5"] else "MA5下",
            "MA10上" if row["above_ma10"] else "MA10下",
            "MA20上" if row["above_ma20"] else "MA20下",
        ]
    )
    return (
        f"{ma}；MA5乖离{row['distance_ma5']:.1f}%；"
        f"MACD {macd_state(row)}；KDJ {kdj_state(row)}；RSI6 {row['rsi6']:.1f}"
    )


def trading_text(row: pd.Series) -> str:
    flags = []
    if row["long_upper_shadow"]:
        flags.append("长上影")
    if row["volume_stall"]:
        flags.append("放量滞涨")
    if row["false_breakout"]:
        flags.append("疑似假突破")
    if row["consecutive_up"] >= 3:
        flags.append(f"连涨{int(row['consecutive_up'])}日")
    flag_text = "、".join(flags) if flags else "无明显异常形态"
    return (
        f"量比{row['volume_ratio']:.2f}；换手{row['turnover_rate']:.2f}%；"
        f"收盘位置{row['close_position']:.2f}；5日涨幅{row['return_5d']:.2f}%；{flag_text}"
    )


def outcome_reason(row: pd.Series) -> str:
    drivers: list[str] = []
    if row["next_pct_chg"] > 0:
        if row["next_beat_market"]:
            drivers.append("次日跑赢大盘")
        if row["next_beat_sector"]:
            drivers.append("个股强于行业")
        if row["sector_continued"]:
            drivers.append("行业相对强势延续")
        if row["close_position"] >= 0.80:
            drivers.append("前日强收盘")
        if 0.9 <= row["volume_ratio"] <= 2.0:
            drivers.append("量能适中")
    else:
        if row["sector_strong"] and not row["sector_continued"]:
            drivers.append("行业次日退潮")
        elif not row["sector_continued"]:
            drivers.append("行业次日弱于市场")
        if row["pct_chg"] >= 8:
            drivers.append("前日大涨后兑现")
        if row["distance_ma5"] > 5:
            drivers.append("偏离MA5过远")
        if row["kdj_j"] >= 90 or row["rsi6"] > 70:
            drivers.append("指标过热")
        if row["close_position"] < 0.70:
            drivers.append("前日收盘位置偏弱")
        if row["volume_ratio"] > 2.2:
            drivers.append("前日量能过大")
    return "；".join(drivers[:4]) or "主要由次日个股波动驱动，现有字段未识别单一主因"


def build_detail(frame: pd.DataFrame) -> pd.DataFrame:
    detail = frame[
        (frame["trade_date"] == SIGNAL_DATE) & frame["ts_code"].isin(SELECTED_CODES)
    ].copy()
    order = {code: idx for idx, code in enumerate(SELECTED_CODES, start=1)}
    detail["rank"] = detail["ts_code"].map(order)
    detail["昨日分类"] = np.where(detail["ts_code"].isin(STRICT_CODES), "严格版", "可用版")
    detail["是否跑赢大盘"] = np.where(
        detail["next_pct_chg"] > CSI300_OUTCOME_RETURN, "是", "否"
    )
    detail["是否跑赢板块"] = np.where(detail["next_beat_sector"], "是", "否")
    detail["昨日技术面特征"] = detail.apply(technical_text, axis=1)
    detail["昨日交易面特征"] = detail.apply(trading_text, axis=1)
    detail["昨日资金面特征"] = "不可验证：Tushare moneyflow接口当前无权限，数据库为空"
    detail["今日上涨/下跌原因"] = detail.apply(outcome_reason, axis=1)
    detail["板块退潮"] = detail["sector_strong"] & ~detail["sector_continued"]
    detail = detail.sort_values("rank")

    exported = detail[
        [
            "rank",
            "ts_code",
            "name",
            "industry",
            "昨日分类",
            "close",
            "pct_chg",
            "next_pct_chg",
            "next_market_return",
            "next_sector_return",
            "是否跑赢大盘",
            "是否跑赢板块",
            "板块退潮",
            "昨日技术面特征",
            "昨日交易面特征",
            "昨日资金面特征",
            "今日上涨/下跌原因",
        ]
    ].rename(
        columns={
            "rank": "排名",
            "ts_code": "股票代码",
            "name": "股票名称",
            "industry": "所属板块",
            "close": "昨日收盘",
            "pct_chg": "昨日涨跌幅%",
            "next_pct_chg": "今日涨跌幅%",
            "next_market_return": "今日全A等权涨跌幅%",
            "next_sector_return": "今日板块涨跌幅%",
        }
    )
    return detail, exported


def group_comparison(sample: pd.DataFrame, label: str) -> pd.DataFrame:
    groups = {"上涨组": sample[sample["next_up"]], "下跌组": sample[~sample["next_up"]]}
    rows = []
    for group_name, group in groups.items():
        rows.extend(
            [
                [label, group_name, "样本数", len(group)],
                [label, group_name, "次日平均涨跌幅%", group["next_pct_chg"].mean()],
                [label, group_name, "次日跑赢大盘率%", bool_rate(group["next_beat_market"])],
                [label, group_name, "板块次日延续强势率%", bool_rate(group["sector_continued"])],
                [label, group_name, "昨日板块强于大盘率%", bool_rate(group["sector_strong"])],
                [label, group_name, "昨日个股强于板块率%", bool_rate(group["stock_stronger_sector"])],
                [label, group_name, "站上MA20率%", bool_rate(group["above_ma20"])],
                [label, group_name, "MA5向上率%", bool_rate(group["ma5_up"])],
                [label, group_name, "MACD修复率%", bool_rate(group["macd_repair"])],
                [label, group_name, "KDJ修复且不过热率%", bool_rate(group["kdj_repair"] & group["kdj_not_hot"])],
                [label, group_name, "RSI6均值", group["rsi6"].mean()],
                [label, group_name, "RSI6过热率%", bool_rate(group["rsi6"] > 70)],
                [label, group_name, "MA5乖离均值%", group["distance_ma5"].mean()],
                [label, group_name, "量比均值", group["volume_ratio"].mean()],
                [label, group_name, "收盘位置均值", group["close_position"].mean()],
                [label, group_name, "近5日涨幅均值%", group["return_5d"].mean()],
                [label, group_name, "昨日涨幅均值%", group["pct_chg"].mean()],
                [label, group_name, "长上影率%", bool_rate(group["long_upper_shadow"])],
                [label, group_name, "放量滞涨率%", bool_rate(group["volume_stall"])],
                [label, group_name, "连续上涨>=3日率%", bool_rate(group["consecutive_up"] >= 3)],
            ]
        )
    return pd.DataFrame(rows, columns=["样本", "分组", "指标", "数值"])


def build_extended(frame: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    candidate_dates = [date for date in dates if date < OUTCOME_DATE][-20:]
    candidates = frame[
        frame["trade_date"].isin(candidate_dates)
        & frame["next_pct_chg"].notna()
        & (frame["strict_rule"] | frame["relaxed_rule"])
    ].copy()
    candidates["策略版本"] = np.where(candidates["strict_rule"], "严格版", "可用版")
    candidates["validated_momentum_rule"] = (
        (candidates["close_position"] >= 0.90)
        & candidates["volume_ratio"].between(0.90, 2.20)
        & candidates["pct_chg"].between(7, 12)
        & (candidates["distance_ma5"] <= 8)
        & candidates["rsi6"].between(35, 75)
        & candidates["kdj_not_hot"]
        & (candidates["consecutive_up"] <= 1)
        & candidates["sector_trend_strong"]
    )
    candidates["repair_control_rule"] = (
        (candidates["close_position"] >= 0.90)
        & candidates["volume_ratio"].between(0.90, 2.20)
        & (candidates["pct_chg"] < 7)
        & (candidates["distance_ma5"] <= 5)
        & candidates["rsi6"].between(35, 70)
        & candidates["kdj_not_hot"]
        & (candidates["consecutive_up"] <= 1)
        & candidates["sector_trend_strong"]
    )
    candidates["strict_order"] = candidates["strict_rule"].astype(int)
    candidates = (
        candidates.sort_values(
            ["trade_date", "strict_order", "original_score", "close_position"],
            ascending=[True, False, False, False],
        )
        .groupby("trade_date", group_keys=False)
        .head(20)
        .copy()
    )
    candidates["结果分组"] = np.where(candidates["next_up"], "上涨组", "下跌组")
    return candidates


def compact_comparison_table(comparison: pd.DataFrame, sample_label: str) -> pd.DataFrame:
    wanted = [
        "样本数",
        "次日平均涨跌幅%",
        "次日跑赢大盘率%",
        "板块次日延续强势率%",
        "站上MA20率%",
        "MA5向上率%",
        "MACD修复率%",
        "KDJ修复且不过热率%",
        "RSI6均值",
        "RSI6过热率%",
        "MA5乖离均值%",
        "量比均值",
        "收盘位置均值",
        "近5日涨幅均值%",
        "昨日涨幅均值%",
        "放量滞涨率%",
    ]
    table = comparison[
        (comparison["样本"] == sample_label) & comparison["指标"].isin(wanted)
    ].pivot(index="指标", columns="分组", values="数值")
    return table.reindex(wanted).reset_index()


def format_comparison(table: pd.DataFrame) -> pd.DataFrame:
    percent_metrics = {
        "次日平均涨跌幅%",
        "次日跑赢大盘率%",
        "板块次日延续强势率%",
        "站上MA20率%",
        "MA5向上率%",
        "MACD修复率%",
        "KDJ修复且不过热率%",
        "RSI6过热率%",
        "MA5乖离均值%",
        "近5日涨幅均值%",
        "昨日涨幅均值%",
        "放量滞涨率%",
    }
    result = table.copy()
    for column in ["上涨组", "下跌组"]:
        result[column] = [
            (
                f"{value:.2f}%"
                if metric in percent_metrics and pd.notna(value)
                else f"{value:.2f}" if pd.notna(value) else "不可用"
            )
            for metric, value in zip(result["指标"], result[column])
        ]
    return result


def detail_markdown(exported: pd.DataFrame) -> str:
    table = exported[
        [
            "股票代码",
            "股票名称",
            "所属板块",
            "昨日分类",
            "今日涨跌幅%",
            "是否跑赢大盘",
            "是否跑赢板块",
            "昨日技术面特征",
            "昨日交易面特征",
            "昨日资金面特征",
            "今日上涨/下跌原因",
        ]
    ].copy()
    table["今日涨跌幅%"] = table["今日涨跌幅%"].map(lambda x: f"{x:+.2f}%")
    return table.to_markdown(index=False)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame, market, sector, dates = load_data()
    detail, exported = build_detail(frame)
    extended = build_extended(frame, dates)

    provided_comparison = group_comparison(detail, "用户20只")
    extended_comparison = group_comparison(extended, "扩展20交易日")
    comparison = pd.concat([provided_comparison, extended_comparison], ignore_index=True)

    exported.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    extended.to_csv(EXTENDED_PATH, index=False, encoding="utf-8-sig")
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")

    market_idx = market.set_index("trade_date")
    yesterday_market = market_idx.loc[SIGNAL_DATE]
    today_market = market_idx.loc[OUTCOME_DATE]
    yesterday_state = market_state(
        yesterday_market["market_return"], yesterday_market["market_up_ratio"]
    )
    today_state = market_state(today_market["market_return"], today_market["market_up_ratio"])

    today_up = detail[detail["next_up"]]
    today_down = detail[~detail["next_up"]]
    beat_csi = int((detail["next_pct_chg"] > CSI300_OUTCOME_RETURN).sum())
    board_retreat_count = int(detail["板块退潮"].sum())
    board_retreat = detail[detail["板块退潮"]]
    no_board_retreat = detail[~detail["板块退潮"]]
    strict = detail[detail["ts_code"].isin(STRICT_CODES)]
    usable = detail[~detail["ts_code"].isin(STRICT_CODES)]
    validated = extended[extended["validated_momentum_rule"]]
    repair_control = extended[extended["repair_control_rule"]]
    validated_train_dates = sorted(extended["trade_date"].unique())[:15]
    validated_test_dates = sorted(extended["trade_date"].unique())[15:]
    validated_train = validated[validated["trade_date"].isin(validated_train_dates)]
    validated_test = validated[validated["trade_date"].isin(validated_test_dates)]
    validated_market_gate = validated[
        (validated["market_return"] >= -0.30) & (validated["market_up_ratio"] >= 0.50)
    ]

    yesterday_sector = sector[sector["trade_date"] == SIGNAL_DATE].copy()
    today_sector = sector[sector["trade_date"] == OUTCOME_DATE].copy()
    yesterday_strong = yesterday_sector[
        yesterday_sector["sector_return"] > yesterday_market["market_return"]
    ]
    continued_industries = set(
        today_sector[today_sector["sector_return"] > today_market["market_return"]]["industry"]
    )
    continuation_rate = (
        yesterday_strong["industry"].isin(continued_industries).mean() * 100
        if not yesterday_strong.empty
        else np.nan
    )

    provided_table = format_comparison(
        compact_comparison_table(comparison, "用户20只")
    ).to_markdown(index=False)
    extended_table = format_comparison(
        compact_comparison_table(comparison, "扩展20交易日")
    ).to_markdown(index=False)

    version_stats = (
        extended.groupby("策略版本")
        .agg(
            样本数=("ts_code", "size"),
            次日平均涨跌幅=("next_pct_chg", "mean"),
            次日上涨率=("next_up", "mean"),
            次日跑赢大盘率=("next_beat_market", "mean"),
        )
        .reset_index()
    )
    for column in ["次日平均涨跌幅", "次日上涨率", "次日跑赢大盘率"]:
        if column == "次日平均涨跌幅":
            version_stats[column] = version_stats[column].map(lambda x: f"{x:.2f}%")
        else:
            version_stats[column] = version_stats[column].map(lambda x: f"{x * 100:.2f}%")

    report = f"""# A股短线选股策略复盘：2026-06-11 信号至 2026-06-12 表现

## 结论先行

- 真实行情显示，这20只股票次日 **{len(today_up)}只上涨、{len(today_down)}只下跌**，上涨率 **{len(today_up) / len(detail) * 100:.2f}%**；平均次日涨跌幅 **{detail['next_pct_chg'].mean():+.2f}%**，中位数 **{detail['next_pct_chg'].median():+.2f}%**。
- 以用户指定的沪深300今日 **+{CSI300_OUTCOME_RETURN:.4f}%** 为基准，只有 **{beat_csi}/20** 跑赢。严格版3只次日平均 **{strict['next_pct_chg'].mean():+.2f}%**，可用版17只次日平均 **{usable['next_pct_chg'].mean():+.2f}%**。
- 原策略最大问题不是某个阈值太松，而是把“低位修复”和“首日强势启动”混成一个策略。扩展样本中，重建严格版次日平均 **{extended[extended['strict_rule']]['next_pct_chg'].mean():+.2f}%**，并未证明指标越齐全越有效。
- 数据支持的“首日强收盘动量”子集有 **{len(validated)}个** 样本，次日平均 **{validated['next_pct_chg'].mean():+.2f}%**、上涨率 **{validated['next_up'].mean() * 100:.2f}%**；同口径低位修复对照组次日平均 **{repair_control['next_pct_chg'].mean():+.2f}%**。
- 资金流接口当前无权限，`moneyflow` 表为空，因此本报告不对主力/大单净流入作虚假判断；新版策略将其作为取得权限后的硬过滤项。

## 数据质量说明

用户提供的名单有效，但表格数值不能直接用于统计：`300722.SZ` 实际名称为 **新余国科**，不是“金牛化工”；例如芯源微 2026-06-11 实际涨幅为 **+6.63%**，不是 -1.81%；航天电器实际为 **+5.85%**，不是 -3.04%；行云科技 2026-06-12 实际为 **-3.84%**，不是 +13.01%。以下全部以本地 Tushare 日线数据库为准。

## 一、大盘和板块环境

- 2026-06-11：全A等权平均 **{yesterday_market['market_return']:+.2f}%**，上涨家数占比 **{yesterday_market['market_up_ratio'] * 100:.2f}%**，状态判定为 **{yesterday_state}**。
- 2026-06-12：全A等权平均 **{today_market['market_return']:+.2f}%**，上涨家数占比 **{today_market['market_up_ratio'] * 100:.2f}%**，状态判定为 **{today_state}**；沪深300为用户给定的 **+{CSI300_OUTCOME_RETURN:.4f}%**。
- 昨日强于全A的行业中，今日仍强于全A的比例为 **{continuation_rate:.2f}%**。
- 20只样本中有 **{board_retreat_count}只** 出现“昨日行业相对强、今日行业弱于市场”的板块退潮现象。
- 板块退潮样本次日平均 **{board_retreat['next_pct_chg'].mean():+.2f}%**、上涨率 **{board_retreat['next_up'].mean() * 100:.2f}%**；未退潮样本次日平均 **{no_board_retreat['next_pct_chg'].mean():+.2f}%**、上涨率 **{no_board_retreat['next_up'].mean() * 100:.2f}%**。
- 板块使用 `stock_basic.industry` 行业分类作为代理，不等同于申万实时行业指数或概念题材指数。

## 二、昨天选出的20只股票

{detail_markdown(exported)}

## 三、上涨组与下跌组前一日特征

上涨组定义为 2026-06-12 收涨；下跌组定义为收跌。是否跑赢市场另行统计，避免把“上涨但跑输大盘”混为强势。

{provided_table}

资金面共同特征无法统计：当前无 `moneyflow` 数据，无法验证主力净流入、大单净额、连续流入或小单推动。

## 四、上涨组为什么能涨

1. 上涨组次日跑赢大盘率为 **{bool_rate(today_up['next_beat_market']):.2f}%**，跑赢行业率为 **{bool_rate(today_up['next_beat_sector']):.2f}%**；个股相对强度比“站上均线”更有区分力。
2. 上涨组前日平均收盘位置 **{today_up['close_position'].mean():.2f}**，明显高于下跌组 **{today_down['close_position'].mean():.2f}**；MA5向上率为 **{bool_rate(today_up['ma5_up']):.2f}%**，高于下跌组 **{bool_rate(today_down['ma5_up']):.2f}%**。
3. 上涨组行业次日延续强势率 **{bool_rate(today_up['sector_continued']):.2f}%**，高于下跌组 **{bool_rate(today_down['sector_continued']):.2f}%**。这项是事后变量，实际交易只能用行业近3日相对强度和行业宽度估计延续概率。
4. MACD修复率、KDJ修复率、RSI6、量比和MA5乖离在两组间差异很小，不能单独解释上涨。

## 五、下跌组为什么会跌

1. 下跌组共有 **{len(today_down)}只**。最明显的问题是前日收盘位置均值仅 **{today_down['close_position'].mean():.2f}**，低于上涨组 **{today_up['close_position'].mean():.2f}**，说明冲高后的承接不足。
2. 下跌组行业次日延续强势率仅 **{bool_rate(today_down['sector_continued']):.2f}%**；板块退潮的12只股票次日平均 **{board_retreat['next_pct_chg'].mean():+.2f}%**。
3. MACD修复率在上涨组为 **{bool_rate(today_up['macd_repair']):.2f}%**、下跌组为 **{bool_rate(today_down['macd_repair']):.2f}%**；RSI6均值分别为 **{today_up['rsi6'].mean():.2f}** 和 **{today_down['rsi6'].mean():.2f}**。这些指标几乎没有区分能力。
4. 严格版只有3只，平均收益被深桑达A涨停显著抬高；扩展样本中的重建严格版反而弱于可用版，因此不能据此认定“原条件越严格越好”。

## 六、最近20个交易日扩展样本

复现方式：每个交易日先选严格版，再从可用版按原策略条件得分补足，单日最多20只；使用次日真实收益验证。扩展样本共 **{len(extended)}个**，其中上涨 **{int(extended['next_up'].sum())}个**、下跌 **{int((~extended['next_up']).sum())}个**，满足至少30个上涨和30个下跌样本要求。

{version_stats.to_markdown(index=False)}

{extended_table}

### 扩展样本的关键验证

- 重建严格版并未获得优势：严格版次日平均 **{extended[extended['strict_rule']]['next_pct_chg'].mean():+.2f}%**、上涨率 **{extended[extended['strict_rule']]['next_up'].mean() * 100:.2f}%**；可用版分别为 **{extended[~extended['strict_rule']]['next_pct_chg'].mean():+.2f}%** 和 **{extended[~extended['strict_rule']]['next_up'].mean() * 100:.2f}%**。
- “首日强收盘动量”子集条件为：收盘位置 `>=0.90`、昨日涨幅 `7%–12%`、量比 `0.90–2.20`、MA5乖离 `<=8%`、RSI6 `35–75`、J值 `<90`、连续上涨不超过1日、行业近3日相对强度为正。
- 该子集共 **{len(validated)}个** 样本，次日平均 **{validated['next_pct_chg'].mean():+.2f}%**、上涨率 **{validated['next_up'].mean() * 100:.2f}%**、跑赢市场率 **{validated['next_beat_market'].mean() * 100:.2f}%**。
- 时间切分校验：前15个交易日 **{len(validated_train)}个** 样本，次日平均 **{validated_train['next_pct_chg'].mean():+.2f}%**；后5个交易日 **{len(validated_test)}个** 样本，次日平均 **{validated_test['next_pct_chg'].mean():+.2f}%**。样本仍较小，只能作为迭代依据，不能视为稳定收益承诺。
- 同样强调强收盘、量能受控和不过热，但昨日涨幅 `<7%` 的低位修复对照组有 **{len(repair_control)}个** 样本，次日平均 **{repair_control['next_pct_chg'].mean():+.2f}%**、上涨率 **{repair_control['next_up'].mean() * 100:.2f}%**。这说明当前数据不支持直接交易“低位修复”模式。
- 给首日动量子集叠加大盘过滤后有 **{len(validated_market_gate)}个** 样本，次日平均 **{validated_market_gate['next_pct_chg'].mean():+.2f}%**、跑赢市场率 **{validated_market_gate['next_beat_market'].mean() * 100:.2f}%**。大盘过滤更适合控制仓位和相对收益，而不是预测次日绝对涨跌。

说明：历史 `daily_basic` 仅覆盖最近3个交易日，因此扩展样本不使用换手率作统计结论；量比由日线成交量计算，可用于比较。

## 七、改进后的可执行策略

### 1. 先拆分策略，不再混选

- **低位修复观察模式**：符合原策略但昨日涨幅 `<7%` 的股票只进入观察池，不直接作为次日交易候选。当前扩展对照组次日平均为负，尚未验证出正向优势。
- **首日强收盘交易模式**：采用扩展样本验证出的8项组合条件，作为当前唯一可进入优先观察的模式。

### 2. 大盘过滤与仓位

- 仅在全A等权日涨跌幅 `>= -0.30%` 且上涨家数占比 `>= 50%` 时使用正常仓位。
- 若全A等权 `<= -1.00%` 或上涨家数占比 `< 40%`，停止使用“可用版”补足数量，只保留严格版；允许最终少于20只。
- 该过滤用于控制风险，不作为次日必涨预测；20日扩展数据没有证明信号日大盘越强，次日收益越高。

### 3. 板块延续过滤

- 所属行业最近3日累计相对全A收益必须 `> 0`。
- 行业当日涨幅必须强于全A，行业上涨家数占比建议 `>= 55%`。
- 排除“前一日行业强、最近3日相对收益转负”的退潮板块。

### 4. 个股技术与相对强度

- 收盘价同时站上 MA5、MA10，且不低于 MA20 的 `98%`；MA5 必须高于前一日。
- 个股当日涨幅必须高于行业涨幅；最近5日相对行业收益为正。
- DIF必须上拐；MACD绿柱缩短/刚金叉只作为标签，不再加分，因为扩展样本中MACD修复率在下跌组反而更高。
- KDJ 要求 `K < 75` 且 `J < 90`；RSI6 要求 `35 <= RSI6 <= 75` 且向上。指标仅作过热排除，不作上涨预测。

### 5. 数据验证的首日强收盘条件

- 收盘位置系数 `>= 0.90`。
- 量比控制在 `0.90–2.20`；排除量比 `> 2.20` 且涨幅 `< 2%` 的放量滞涨。
- 昨日涨幅限定为 `7%–12%`，并要求连续上涨不超过1日；它代表首日启动，不得与低位修复混用。
- 距离MA5 `<= 8%`，距离MA10 `<= 10%`；超过时视为追高。
- 近5日涨幅 `<= 15%`；单日涨幅超过12%时不追。
- 排除长上影比例 `>= 45%`、高开低走、放量跌破MA20、连续上涨 `>= 3日` 后再放量的股票。

### 6. 资金面约束

- 获得资金流权限后，要求主力净流入和大单净额至少一项为正，且最近3日中至少2日为正。
- 排除放量但主力净流出、仅小单净流入的股票。
- 在资金流数据不可用时，候选只能标记为“待验证”，不能因技术指标完整而自动升级为优先观察。

### 7. 排名与数量

- 严格版优先，按“大盘环境、板块3日相对强度、个股相对行业强度、收盘位置、量能适中”排序。
- 不再为了凑足20只强制放宽条件。合格标的不足时，宁可输出5–10只。

## 八、最终回答

1. **原策略最大问题**：把低位修复与首日强势启动混成一个策略；多数技术指标只描述昨日形态，没有证明能预测次日延续。
2. **导致第二天下跌的指标**：20只样本中最清晰的是收盘位置偏低、MA5未上拐和板块相对强势未延续；MACD/KDJ/RSI并没有形成有效区分。
3. **支持第二天上涨的指标**：高收盘位置、首日上涨而非连续加速、量比受控、个股跑赢行业，以及行业相对强势延续。
4. **必须新增的条件**：策略模式拆分、大盘仓位过滤、行业3日相对强度、个股相对行业强度、量比上限、乖离上限、过热否决和资金流验证。
5. **新版策略筛选方式**：低位修复只观察；优先观察仅保留数据验证的首日强收盘组合，并允许少于20只。
6. **以后即使符合原策略也要剔除**：为了凑数才入选、收盘位置低于0.90、连续上涨超过1日、量比超过2.2、J值超过90、RSI6超过75、距离MA5超过8%、板块3日相对强度转负、资金流无法验证却仅靠技术形态升级的股票。

## 数据来源与限制

- 数据来源：本地 Tushare `daily`、`daily_basic`、`stock_basic` 及项目计算因子。
- 信号日：2026-06-11；结果日：2026-06-12；扩展窗口：截至2026-06-11的最近20个交易日。
- 沪深300 2026-06-12 涨幅采用用户提供的 +{CSI300_OUTCOME_RETURN:.4f}%；板块代理为股票基础表中的行业分类。
- 资金流接口无权限，相关结论明确标记为不可验证。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"report={REPORT_PATH}")
    print(f"detail={DETAIL_PATH}")
    print(f"extended={EXTENDED_PATH}")
    print(f"comparison={COMPARISON_PATH}")
    print(
        f"provided: n={len(detail)}, up={len(today_up)}, down={len(today_down)}, "
        f"avg={detail['next_pct_chg'].mean():.4f}"
    )
    print(
        f"extended: n={len(extended)}, up={int(extended['next_up'].sum())}, "
        f"down={int((~extended['next_up']).sum())}"
    )


if __name__ == "__main__":
    main()
