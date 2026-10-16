from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_strategy_review import CSI300_OUTCOME_RETURN, load_data


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs" / "reports" / "first_day_momentum_backtest_20260611.md"
STRICT_PATH = ROOT / "outputs" / "backtests" / "first_day_momentum_strict_20260611.csv"
COUNTERFACTUAL_PATH = (
    ROOT / "outputs" / "backtests" / "first_day_momentum_no_market_gate_20260611.csv"
)
CURRENT_PATH = ROOT / "outputs" / "stock_pools" / "first_day_momentum_watchlist_20260612.csv"
BACKTEST_PATH = ROOT / "outputs" / "backtests" / "first_day_momentum_backtest_20d.csv"
FUNNEL_PATH = ROOT / "outputs" / "backtests" / "first_day_momentum_funnel_20260611.csv"

SIGNAL_DATE = "20260611"
OUTCOME_DATE = "20260612"


def add_strategy_features(
    frame: pd.DataFrame, sector: pd.DataFrame
) -> pd.DataFrame:
    sector = sector.sort_values(["industry", "trade_date"]).copy()
    sector["sector_return_5d"] = sector.groupby("industry")["sector_return"].transform(
        lambda values: (1 + values / 100)
        .rolling(5, min_periods=5)
        .apply(np.prod, raw=True)
        * 100
        - 100
    )
    result = frame.merge(
        sector[["trade_date", "industry", "sector_return_5d"]],
        on=["trade_date", "industry"],
        how="left",
    )
    result["relative_industry_5d"] = result["return_5d"] - result["sector_return_5d"]

    leaders = (
        result.sort_values(
            ["trade_date", "industry", "pct_chg"], ascending=[True, True, False]
        )
        .groupby(["trade_date", "industry"], group_keys=False)
        .head(1)
        .copy()
    )
    leaders["leader_bad_proxy"] = leaders["long_upper_shadow"] | (
        (leaders["pct_chg"] >= 9.5) & (leaders["close_position"] < 0.80)
    )
    leader_map = leaders.set_index(["trade_date", "industry"])[
        "leader_bad_proxy"
    ].to_dict()
    result["leader_ok_proxy"] = [
        not leader_map.get((date, industry), True)
        for date, industry in zip(result["trade_date"], result["industry"])
    ]

    result["market_gate"] = (
        (result["market_return"] >= -0.30)
        & (result["market_up_ratio"] >= 0.50)
        & ~((result["market_return"] <= -1.00) & (result["market_up_ratio"] <= 0.40))
    )
    result["sector_gate"] = (
        result["sector_strong"]
        & result["sector_trend_strong"]
        & (result["sector_up_ratio"] >= 0.55)
        & result["leader_ok_proxy"]
    )
    result["stock_gate"] = (
        result["pct_chg"].between(7, 12)
        & (result["consecutive_up"] <= 1)
        & (result["close_position"] >= 0.90)
        & result["volume_ratio"].between(0.90, 2.20)
        & (result["distance_ma5"] <= 8)
        & (result["distance_ma10"] <= 10)
        & (result["return_5d"] <= 15)
        & result["stock_stronger_sector"]
        & (result["relative_industry_5d"] > 0)
    )
    result["technical_gate"] = (
        result["ma5_up"]
        & result["above_ma5"]
        & result["above_ma10"]
        & result["near_ma20"]
        & result["dif_up"]
        & result["rsi6"].between(35, 75)
        & (result["kdj_j"] < 90)
    )
    result["exclusion_gate"] = (
        ~result["long_upper_shadow"]
        & ~result["volume_stall"]
        & ~result["high_open_low_close"]
        & ~result["volume_break_ma20"]
    )
    result["strict_signal"] = (
        result["market_gate"]
        & result["sector_gate"]
        & result["stock_gate"]
        & result["technical_gate"]
        & result["exclusion_gate"]
    )
    result["no_market_gate_signal"] = (
        result["sector_gate"]
        & result["stock_gate"]
        & result["technical_gate"]
        & result["exclusion_gate"]
    )
    result = result.sort_values(["ts_code", "trade_date"])
    result["next_open"] = result.groupby("ts_code")["open"].shift(-1)
    result["next_close_price"] = result.groupby("ts_code")["close"].shift(-1)
    result["next_gap"] = (result["next_open"] / result["close"] - 1) * 100
    result["next_open_to_close"] = (
        result["next_close_price"] / result["next_open"] - 1
    ) * 100
    return result


def detail_columns() -> list[str]:
    return [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "pct_chg",
        "next_trade_date",
        "next_pct_chg",
        "next_gap",
        "next_open_to_close",
        "next_market_return",
        "next_sector_return",
        "next_beat_market",
        "next_beat_sector",
        "market_return",
        "market_up_ratio",
        "sector_return",
        "sector_relative_3d",
        "sector_up_ratio",
        "relative_industry_5d",
        "consecutive_up",
        "close_position",
        "volume_ratio",
        "distance_ma5",
        "distance_ma10",
        "return_5d",
        "ma5_up",
        "above_ma5",
        "above_ma10",
        "near_ma20",
        "dif_up",
        "rsi6",
        "kdj_j",
        "leader_ok_proxy",
        "long_upper_shadow",
        "volume_stall",
        "high_open_low_close",
        "volume_break_ma20",
    ]


def format_watchlist(frame: pd.DataFrame) -> pd.DataFrame:
    def risk_notes(row: pd.Series) -> str:
        notes = []
        if row["pct_chg"] >= 9.5:
            notes.append("接近涨停，次日成交风险")
        if row["volume_ratio"] >= 2.0:
            notes.append("量比接近上限")
        if row["distance_ma5"] >= 7:
            notes.append("MA5乖离接近上限")
        if row["rsi6"] >= 70:
            notes.append("RSI接近上限")
        if row["kdj_j"] >= 85:
            notes.append("J值接近上限")
        if row["relative_industry_5d"] <= 1:
            notes.append("5日相对行业优势较薄")
        return "；".join(notes) if notes else "未触及临界阈值"

    frame = frame.copy()
    frame["risk_notes"] = frame.apply(risk_notes, axis=1)
    columns = [
        "ts_code",
        "name",
        "industry",
        "pct_chg",
        "sector_return",
        "sector_relative_3d",
        "sector_up_ratio",
        "relative_industry_5d",
        "close_position",
        "volume_ratio",
        "distance_ma5",
        "distance_ma10",
        "return_5d",
        "consecutive_up",
        "rsi6",
        "kdj_j",
        "risk_notes",
    ]
    return frame[columns].rename(
        columns={
            "ts_code": "股票代码",
            "name": "股票名称",
            "industry": "所属行业",
            "pct_chg": "当日涨幅%",
            "sector_return": "行业涨幅%",
            "sector_relative_3d": "行业近3日相对全A收益%",
            "sector_up_ratio": "行业上涨家数占比",
            "relative_industry_5d": "个股近5日相对行业收益%",
            "close_position": "收盘位置系数",
            "volume_ratio": "量比",
            "distance_ma5": "距离MA5%",
            "distance_ma10": "距离MA10%",
            "return_5d": "近5日涨幅%",
            "consecutive_up": "连续上涨天数",
            "rsi6": "RSI6",
            "kdj_j": "KDJ_J",
            "risk_notes": "临界风险提示",
        }
    )


def markdown_watchlist(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无。"
    table = format_watchlist(frame).copy()
    for column in [
        "当日涨幅%",
        "行业涨幅%",
        "行业近3日相对全A收益%",
        "个股近5日相对行业收益%",
        "距离MA5%",
        "距离MA10%",
        "近5日涨幅%",
    ]:
        table[column] = table[column].map(lambda value: f"{value:.2f}%")
    table["行业上涨家数占比"] = table["行业上涨家数占比"].map(
        lambda value: f"{value * 100:.2f}%"
    )
    for column in ["收盘位置系数", "量比", "RSI6", "KDJ_J"]:
        table[column] = table[column].map(lambda value: f"{value:.2f}")
    return table.to_markdown(index=False)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame, market, sector, dates = load_data()
    frame = add_strategy_features(frame, sector)

    signal_day = frame[frame["trade_date"] == SIGNAL_DATE].copy()
    strict_signal = signal_day[signal_day["strict_signal"]].copy()
    no_market_gate = signal_day[signal_day["no_market_gate_signal"]].copy()
    current_watchlist = frame[
        (frame["trade_date"] == OUTCOME_DATE) & frame["strict_signal"]
    ].copy()

    recent_signal_dates = [date for date in dates if date < OUTCOME_DATE][-20:]
    backtest = frame[
        frame["trade_date"].isin(recent_signal_dates)
        & frame["strict_signal"]
        & frame["next_pct_chg"].notna()
    ].copy()

    funnel = pd.DataFrame(
        [
            ["全A股票池", len(signal_day)],
            ["通过大盘环境", int(signal_day["market_gate"].sum())],
            ["通过板块环境", int(signal_day["sector_gate"].sum())],
            ["通过个股条件", int(signal_day["stock_gate"].sum())],
            ["通过技术辅助", int(signal_day["technical_gate"].sum())],
            ["通过剔除条件", int(signal_day["exclusion_gate"].sum())],
            [
                "不考虑大盘、其余条件全部通过",
                int(signal_day["no_market_gate_signal"].sum()),
            ],
            ["严格公式最终入选", int(signal_day["strict_signal"].sum())],
        ],
        columns=["筛选层级", "股票数量"],
    )

    strict_signal[detail_columns()].to_csv(STRICT_PATH, index=False, encoding="utf-8-sig")
    no_market_gate[detail_columns()].to_csv(
        COUNTERFACTUAL_PATH, index=False, encoding="utf-8-sig"
    )
    format_watchlist(current_watchlist).to_csv(
        CURRENT_PATH, index=False, encoding="utf-8-sig"
    )
    backtest[detail_columns()].to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    funnel.to_csv(FUNNEL_PATH, index=False, encoding="utf-8-sig")

    market_by_date = market.set_index("trade_date")
    signal_market = market_by_date.loc[SIGNAL_DATE]
    current_market = market_by_date.loc[OUTCOME_DATE]

    counterfactual_table = no_market_gate[
        [
            "ts_code",
            "name",
            "industry",
            "pct_chg",
            "next_pct_chg",
            "next_gap",
            "next_open_to_close",
            "next_market_return",
            "next_sector_return",
            "next_beat_market",
            "next_beat_sector",
        ]
    ].rename(
        columns={
            "ts_code": "股票代码",
            "name": "股票名称",
            "industry": "所属行业",
            "pct_chg": "6月11日涨幅%",
            "next_pct_chg": "6月12日涨幅%",
            "next_gap": "6月12日开盘缺口%",
            "next_open_to_close": "6月12日开盘至收盘%",
            "next_market_return": "6月12日全A等权涨幅%",
            "next_sector_return": "6月12日行业涨幅%",
            "next_beat_market": "是否跑赢全A",
            "next_beat_sector": "是否跑赢行业",
        }
    )
    for column in [
        "6月11日涨幅%",
        "6月12日涨幅%",
        "6月12日开盘缺口%",
        "6月12日开盘至收盘%",
        "6月12日全A等权涨幅%",
        "6月12日行业涨幅%",
    ]:
        counterfactual_table[column] = counterfactual_table[column].map(
            lambda value: f"{value:+.2f}%"
        )
    for column in ["是否跑赢全A", "是否跑赢行业"]:
        counterfactual_table[column] = counterfactual_table[column].map(
            lambda value: "是" if value else "否"
        )

    backtest_summary = pd.DataFrame(
        [
            ["信号样本数", len(backtest)],
            ["出现信号的交易日数", backtest["trade_date"].nunique()],
            ["次日平均涨跌幅%", backtest["next_pct_chg"].mean()],
            ["次日涨跌幅中位数%", backtest["next_pct_chg"].median()],
            ["次日上涨率%", backtest["next_up"].mean() * 100],
            ["次日平均开盘缺口%", backtest["next_gap"].mean()],
            ["次日开盘至收盘平均收益%", backtest["next_open_to_close"].mean()],
            [
                "次日开盘至收盘上涨率%",
                (backtest["next_open_to_close"] > 0).mean() * 100,
            ],
            ["次日跑赢全A率%", backtest["next_beat_market"].mean() * 100],
            ["次日跑赢行业率%", backtest["next_beat_sector"].mean() * 100],
        ],
        columns=["指标", "数值"],
    )
    backtest_summary["数值"] = backtest_summary.apply(
        lambda row: (
            f"{int(row['数值'])}"
            if row["指标"] in {"信号样本数", "出现信号的交易日数"}
            else f"{row['数值']:.2f}%"
        ),
        axis=1,
    )

    report = f"""# 首日强收盘动量策略回测与筛选

## 结论

- 2026-06-11 全A等权涨跌幅为 **{signal_market['market_return']:+.2f}%**，上涨家数占比仅 **{signal_market['market_up_ratio'] * 100:.2f}%**，属于明显退潮，未通过策略的大盘环境硬门槛。
- 因此，严格执行公式后，2026-06-11 **入选股票为0只**。策略在当天正确动作是空仓或停止新增首日动量仓位。
- 若仅为了诊断而移除大盘门槛，其余所有条件全部通过的股票只有 **{len(no_market_gate)}只**；该结果不能视为严格策略信号。
- 最近20个交易日的严格公式回测共有 **{len(backtest)}个** 信号，次日收盘对收盘平均 **{backtest['next_pct_chg'].mean():+.2f}%**，但更可执行的次日开盘至收盘平均为 **{backtest['next_open_to_close'].mean():+.2f}%**、上涨率 **{(backtest['next_open_to_close'] > 0).mean() * 100:.2f}%**。当前样本没有证明该公式具备稳定的可交易收益优势。

## 规则实现口径

- 所有6月11日筛选条件只使用6月11日收盘时可见数据，6月12日数据仅用于事后验证，不存在前视。
- “板块”使用 `stock_basic.industry` 行业分类代理。
- “板块龙头”定义为行业当日涨幅第一名。由于当前没有涨停价和炸板明细权限，龙头炸板使用代理判断：龙头长上影，或涨幅达到9.5%以上但收盘位置低于0.80。
- 长上影定义为上影线占当日振幅比例 `>=45%`；高开低走定义为开盘较昨收高至少1%且收盘低于开盘；放量滞涨定义为量比 `>=1.8` 且涨幅 `<2%`。

## 2026-06-11 筛选漏斗

{funnel.to_markdown(index=False)}

大盘门槛在第一层即否决了当天所有候选。其余条件分别计算的通过数量不能直接相加，因为同一股票可能同时通过或未通过多个条件。

## 6月11日严格入选及6月12日验证

严格入选为 **0只**，因此没有严格策略持仓可供6月12日收益验证。不能把事后上涨股票补录成当日信号。

## 移除大盘门槛后的诊断样本

以下仅用于判断大盘过滤的影响，不属于严格策略入选：

{counterfactual_table.to_markdown(index=False) if not counterfactual_table.empty else '无。'}

该诊断样本6月12日表现较强，但样本只有{len(no_market_gate)}只，不能据此删除大盘过滤。

## 最近20个交易日严格公式回测

{backtest_summary.to_markdown(index=False)}

- 20个信号日中，仅 **{backtest['trade_date'].nunique()}日** 出现严格信号，说明公式交易频率较低。
- 次日收盘对收盘平均收益接近0；次日开盘至收盘平均 **{backtest['next_open_to_close'].mean():+.2f}%**、上涨率 **{(backtest['next_open_to_close'] > 0).mean() * 100:.2f}%**，说明隔夜高开消耗了部分收益。
- 当前回测仍未计入手续费、滑点、涨停无法成交和流动性约束。部分首日涨停股实际无法在信号日收盘买入。

## 使用2026-06-12最新数据筛选的下一个交易日观察池

6月12日全A等权涨跌幅 **{current_market['market_return']:+.2f}%**，上涨家数占比 **{current_market['market_up_ratio'] * 100:.2f}%**，大盘环境通过。按同一严格公式筛得 **{len(current_watchlist)}只**：

{markdown_watchlist(current_watchlist)}

这些股票尚无下一交易日结果，只能作为观察池，不代表必涨或买入建议。

## 下一步优化建议

1. 保留大盘过滤作为仓位控制，但不要将其解释为次日上涨预测器。
2. 增加“次日可成交性”规则：首日接近涨停的股票，必须用次日开盘价而不是首日收盘价回测。
3. 将回测窗口扩展到至少120个交易日，并按主板、创业板、科创板和北交所分别统计。
4. 增加手续费、滑点、涨跌停无法成交、最大回撤和盈亏比后，再决定是否用于实盘观察。
5. 当前资金流接口无权限，主力和大单条件尚未纳入，不能用空值代替资金验证。

## 输出文件

- 6月11日严格入选：`{STRICT_PATH}`
- 6月11日移除大盘门槛诊断：`{COUNTERFACTUAL_PATH}`
- 最近20日严格回测：`{BACKTEST_PATH}`
- 6月12日最新观察池：`{CURRENT_PATH}`
- 6月11日筛选漏斗：`{FUNNEL_PATH}`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"strict_20260611={len(strict_signal)}")
    print(f"counterfactual_20260611={len(no_market_gate)}")
    print(f"backtest_20d={len(backtest)}")
    print(f"current_watchlist_20260612={len(current_watchlist)}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
