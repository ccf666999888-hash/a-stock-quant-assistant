from __future__ import annotations

import pandas as pd


def stock_pool_markdown(frame: pd.DataFrame, classification: str, limit: int = 30) -> str:
    if frame.empty or "classification" not in frame:
        return f"### {classification}\n\n暂无符合规则的样本。\n"
    selected = frame[frame.get("classification", "") == classification].head(limit)
    if selected.empty:
        return f"### {classification}\n\n暂无符合规则的样本。\n"
    columns = [
        column
        for column in [
            "ts_code", "name", "models", "technical_score", "market_score",
            "stage_score", "risk_deduction", "total_score", "reasons", "risk_reasons",
        ]
        if column in selected
    ]
    return f"### {classification}\n\n{selected[columns].to_markdown(index=False)}\n"
