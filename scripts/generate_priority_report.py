from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reports.markdown_writer import write_markdown


DISCLAIMER = "仅供复盘研究和交易计划参考，不构成投资建议。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate priority observation report")
    parser.add_argument("--trade-date", required=True)
    return parser.parse_args()


def money(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.2f}"


def main() -> None:
    args = parse_args()
    database = ROOT / "data" / "db" / "a_stock_quant.duckdb"
    with duckdb.connect(str(database), read_only=True) as connection:
        priority = connection.execute(
            """
            SELECT
                s.ts_code, s.name, b.industry, s.models, s.technical_score,
                s.market_score, s.stage_score, s.risk_deduction, s.total_score,
                p.close, p.pct_chg, p.high, p.low, p.amount,
                v.turnover_rate, v.volume_ratio, v.close_position,
                t.ma5, t.ma10, t.ma20, t.ma60, t.macd_hist, t.kdj_k, t.kdj_d,
                t.rsi6, t.boll_upper, g.return_5d, g.return_20d,
                g.new_high_20d, g.platform_breakout, s.reasons, s.risk_reasons
            FROM screening_result s
            JOIN daily_price p USING (ts_code, trade_date)
            LEFT JOIN stock_basic b USING (ts_code)
            LEFT JOIN factor_volume_price v USING (ts_code, trade_date)
            LEFT JOIN factor_technical t USING (ts_code, trade_date)
            LEFT JOIN factor_stage g USING (ts_code, trade_date)
            WHERE s.trade_date = ? AND s.classification = '优先观察'
            ORDER BY s.total_score DESC, s.ts_code
            """,
            [args.trade_date],
        ).fetchdf()
        coverage = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM daily_price WHERE trade_date = ?) AS daily_rows,
                (SELECT COUNT(*) FROM daily_basic WHERE trade_date = ?) AS daily_basic_rows,
                (SELECT COUNT(*) FROM moneyflow WHERE trade_date = ?) AS moneyflow_rows,
                (SELECT COUNT(*) FROM index_daily WHERE trade_date = ?) AS index_rows
            """,
            [args.trade_date] * 4,
        ).fetchone()

    if priority.empty:
        content = f"# 优先观察股票报告 {args.trade_date}\n\n暂无符合规则的优先观察股票。\n\n> {DISCLAIMER}\n"
    else:
        priority["support"] = priority[["ma5", "ma10", "ma20"]].min(axis=1).round(2)
        priority["resistance"] = pd.concat(
            [priority["boll_upper"], priority["close"] * 1.05], axis=1
        ).max(axis=1).round(2)
        priority["market_note"] = "量价数据可用；资金流无权限，未纳入正向判断"
        priority["next_session_plan"] = (
            "高开不追，观察回踩承接；低开守支撑；冲高回落且收盘位置转弱则降级"
        )
        priority["risk_note"] = priority["ts_code"].apply(
            lambda code: "北交所标的，涨跌幅与流动性特征需单独评估"
            if str(code).endswith(".BJ")
            else "未触发规则内严重风险，仍需观察次日量价"
        )

        summary_columns = [
            "ts_code", "name", "industry", "models", "total_score", "close",
            "pct_chg", "volume_ratio", "turnover_rate", "close_position",
            "return_5d", "support", "resistance",
        ]
        summary = priority[summary_columns].copy()
        for column in [
            "total_score", "close", "pct_chg", "volume_ratio", "turnover_rate",
            "close_position", "return_5d", "support", "resistance",
        ]:
            summary[column] = summary[column].round(2)

        sections: list[str] = []
        for _, row in priority.iterrows():
            sections.append(
                f"""### {row['name']}（{row['ts_code']}）

- 分类与模型：优先观察；{row['models']}
- 综合得分：{row['total_score']:.2f}（技术 {row['technical_score']:.0f} / 行情 {row['market_score']:.0f} / 阶段 {row['stage_score']:.0f} / 风险扣分 {row['risk_deduction']:.0f}）
- 收盘与涨跌：{money(row['close'])}；{row['pct_chg']:.2f}%
- 量价：量比 {row['volume_ratio']:.2f}；换手率 {row['turnover_rate']:.2f}%；收盘位置 {row['close_position']:.2f}
- 技术指标：MA5 {money(row['ma5'])} / MA10 {money(row['ma10'])} / MA20 {money(row['ma20'])}；RSI6 {row['rsi6']:.2f}；KDJ-K/D {row['kdj_k']:.2f}/{row['kdj_d']:.2f}
- 阶段表现：5 日 {row['return_5d']:.2f}%；20 日 {row['return_20d']:.2f}%；20 日新高：{'是' if row['new_high_20d'] else '否'}
- 观察区间：支撑约 {money(row['support'])}；压力约 {money(row['resistance'])}
- 次日预案：{row['next_session_plan']}
- 风险提示：{row['risk_note']}。{row['market_note']}。
"""
            )

        content = f"""# 优先观察股票报告 {args.trade_date}

> {DISCLAIMER}

- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 优先观察数量：{len(priority)}
- 数据覆盖：日线 {coverage[0]} 只；每日指标 {coverage[1]} 只；资金流 {coverage[2]} 只；指数 {coverage[3]} 条
- 入选规则：综合得分不低于 75、命中积极模型、未触发严重风险
- 数据限制：资金流接口无权限，指数接口当日额度不足；相关维度不作为正向依据

## 汇总

{summary.to_markdown(index=False)}

## 个股分析

{''.join(sections)}

## 使用原则

优先观察不代表确定性上涨。次日若出现放量跌破支撑、突破失败、收盘位置显著转弱，应降低观察等级或回避。

> {DISCLAIMER}
"""

    output = ROOT / "outputs" / "reports" / f"priority_watch_report_{args.trade_date}.md"
    write_markdown(output, content)
    priority.to_csv(
        ROOT / "outputs" / "stock_pools" / f"priority_watch_{args.trade_date}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

