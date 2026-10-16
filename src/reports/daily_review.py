from __future__ import annotations

from datetime import datetime

import pandas as pd

from reports.stock_pool_report import stock_pool_markdown


DISCLAIMER = "仅供复盘研究和交易计划参考，不构成投资建议。"


def infer_market_state(indices: pd.DataFrame) -> str:
    if indices.empty or "pct_chg" not in indices:
        return "数据不足"
    mean_return = pd.to_numeric(indices["pct_chg"], errors="coerce").mean()
    if mean_return >= 1:
        return "强势"
    if mean_return >= -0.3:
        return "震荡"
    if mean_return >= -1:
        return "弱势"
    return "退潮"


def add_trade_plans(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    close = pd.to_numeric(result.get("close", 0), errors="coerce").fillna(0)
    ma5 = pd.to_numeric(result.get("ma5", close), errors="coerce").fillna(close)
    ma10 = pd.to_numeric(result.get("ma10", close), errors="coerce").fillna(close)
    ma20 = pd.to_numeric(result.get("ma20", close), errors="coerce").fillna(close)
    boll_upper = pd.to_numeric(result.get("boll_upper", close), errors="coerce").fillna(close)
    result["support"] = pd.concat([ma5, ma10, ma20], axis=1).min(axis=1).round(2)
    result["resistance"] = pd.concat([boll_upper, close * 1.05], axis=1).max(axis=1).round(2)
    result["upside_logic"] = "模型延续且量价未恶化时继续观察"
    result["high_open_plan"] = "高开不追价，等待回踩确认承接与量能"
    result["low_open_plan"] = "低开先观察支撑位，跌破则取消积极观察"
    result["spike_reversal_plan"] = "冲高回落且收盘位置转弱时降低观察等级"
    result["stop_loss_condition"] = "放量跌破支撑位或触发严重风险"
    result["take_profit_condition"] = "接近压力位且量价背离时分批止盈评估"
    return result


def build_daily_review(
    *,
    trade_date: str,
    screening: pd.DataFrame,
    indices: pd.DataFrame | None = None,
    strong_sectors: list[str] | None = None,
    weak_sectors: list[str] | None = None,
) -> str:
    index_frame = indices if indices is not None else pd.DataFrame()
    state = infer_market_state(index_frame)
    index_table = "暂无指数数据"
    if not index_frame.empty:
        columns = [column for column in ["ts_code", "pct_chg", "amount"] if column in index_frame]
        index_table = index_frame[columns].to_markdown(index=False)
    strong_text = "、".join(strong_sectors or []) or "待补充行业/概念数据"
    weak_text = "、".join(weak_sectors or []) or "待补充行业/概念数据"
    planned = add_trade_plans(screening)
    detailed_columns = [
        column for column in [
            "ts_code", "name", "classification", "models", "total_score",
            "support", "resistance", "upside_logic", "risk_reasons",
            "high_open_plan", "low_open_plan", "spike_reversal_plan",
            "stop_loss_condition", "take_profit_condition",
        ] if column in planned
    ]
    details = planned[detailed_columns].head(60).to_markdown(index=False) if not planned.empty else "暂无样本"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""# A股短线量化复盘 {trade_date}

> {DISCLAIMER}

- 生成时间：{generated_at}
- 市场状态：**{state}**
- 今日强势板块：{strong_text}
- 今日弱势板块：{weak_text}

## 指数与成交额

{index_table}

## 明日观察池

{stock_pool_markdown(planned, "优先观察")}

{stock_pool_markdown(planned, "谨慎观察")}

{stock_pool_markdown(planned, "回避")}

## 个股观察预案

{details}

## 风险声明

{DISCLAIMER} 所有结论均基于历史数据、指标和规则，需结合次日市场环境复核。
"""

