from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")


def _fmt(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "无"
    if np.isnan(number):
        return "无"
    return f"{number:,.{digits}f}"


def _image_data(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _equity_chart(equity: pd.DataFrame, initial_cash: float) -> str:
    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    frame = equity.copy()
    if frame.empty:
        frame = pd.DataFrame(
            {"trade_date": ["启动日"], "total_equity": [initial_cash],
             "benchmark_cumulative_return_pct": [0.0]}
        )
    x = np.arange(len(frame))
    account_curve = frame["total_equity"].astype(float) / initial_cash * 100
    benchmark = 100 + pd.to_numeric(
        frame.get("benchmark_cumulative_return_pct", 0), errors="coerce"
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=145)
    ax.plot(x, account_curve, color="#315a9b", linewidth=2.2, label="模拟账户")
    ax.plot(x, benchmark, color="#d98c28", linewidth=1.7, label="基准")
    ax.axhline(100, color="#9ca7b6", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.18)
    ax.set_ylabel("归一化净值", fontproperties=font)
    labels = frame["trade_date"].astype(str).tolist()
    ticks = x if len(x) <= 8 else x[:: max(1, len(x) // 8)]
    ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=25, ha="right", fontproperties=font)
    ax.legend(prop=font, loc="best")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _table(frame: pd.DataFrame, columns: list[str], empty_text: str) -> str:
    if frame.empty:
        return f'<div class="empty">{html.escape(empty_text)}</div>'
    shown = frame[[column for column in columns if column in frame.columns]].copy()
    shown = shown.fillna("无")
    return '<div class="table-wrap">' + shown.to_html(
        index=False, border=0, escape=True, classes="data-table"
    ) + "</div>"


def write_paper_report(
    output: Path,
    snapshot: dict[str, Any],
    data_date: str,
    report_date: str,
    context: dict[str, Any],
    status_note: str,
    market_chart: Path | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    account = snapshot["account"]
    positions = snapshot["positions"].copy()
    pending = snapshot["pending"].copy()
    trades = snapshot["trades"].copy()
    signals = snapshot["signals"].copy()
    equity = snapshot["equity"].copy()
    initial_cash = float(account["initial_cash"])
    latest_equity = float(equity.iloc[-1]["total_equity"]) if not equity.empty else initial_cash
    cash = float(account["cash"])
    cumulative = (latest_equity / initial_cash - 1) * 100
    max_drawdown = float(equity["drawdown_pct"].min()) if not equity.empty else 0.0
    sells = trades[trades["side"] == "SELL"] if not trades.empty else pd.DataFrame()
    win_rate = (sells["realized_pnl"] > 0).mean() * 100 if not sells.empty else np.nan
    win_rate_text = f"{win_rate:.2f}%" if pd.notna(win_rate) else "暂无"
    realized = float(sells["realized_pnl"].sum()) if not sells.empty else 0.0

    if not positions.empty:
        positions["市值"] = positions["quantity"] * positions["last_price"]
        positions["浮动盈亏"] = positions["quantity"] * (
            positions["last_price"] - positions["avg_cost"]
        )
        positions["收益率%"] = (positions["last_price"] / positions["avg_cost"] - 1) * 100
        positions = positions.rename(
            columns={"ts_code": "代码", "name": "名称", "sector": "板块", "quantity": "股数",
                     "avg_cost": "成本", "last_price": "收盘价", "days_held": "持有日"}
        )
    if not pending.empty:
        pending["方向"] = pending["side"].map({"BUY": "买入", "SELL": "卖出"})
        pending = pending.rename(
            columns={"signal_date": "信号日", "ts_code": "代码", "name": "名称",
                     "sector": "板块", "quantity": "计划股数", "reference_price": "参考价",
                     "reason": "原因", "note": "成交约束"}
        )
    if not signals.empty:
        latest_signal_date = signals["trade_date"].max()
        signals = signals[signals["trade_date"] == latest_signal_date].copy()
        signals["计划"] = signals["selected"].map({True: "已生成待成交订单", False: "不下单"})
        signals = signals.rename(
            columns={"rank": "排名", "ts_code": "代码", "name": "名称", "sector": "板块",
                     "score": "评分", "close": "收盘价", "eligibility": "规则检查",
                     "reason": "入选依据"}
        )
    if not trades.empty:
        trades["方向"] = trades["side"].map({"BUY": "买入", "SELL": "卖出"})
        trades = trades.rename(
            columns={"trade_date": "成交日", "ts_code": "代码", "name": "名称",
                     "quantity": "股数", "fill_price": "成交价", "fees": "费用",
                     "realized_pnl": "已实现盈亏", "reason": "交易原因"}
        )

    chart_src = _image_data(market_chart)
    chart_html = (
        f'<img class="chart" src="{chart_src}" alt="上证指数60分钟K线">'
        if chart_src else '<div class="empty">本次启动报告未生成新的市场图，使用最近完成交易日数据。</div>'
    )
    equity_src = _equity_chart(equity, initial_cash)
    pending_buys = len(pending[pending["方向"] == "买入"]) if not pending.empty else 0
    regime_class = {"强势": "good", "震荡": "watch", "弱势": "caution", "退潮": "risk"}.get(
        str(context.get("regime")), "neutral"
    )
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股日线模拟盘 {html.escape(report_date)}</title>
<style>
:root{{--bg:#f3f6fa;--card:#fff;--ink:#172033;--muted:#687386;--line:#dfe5ed;--blue:#315a9b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei",sans-serif;line-height:1.6}}
.page{{max-width:1420px;margin:auto;padding:28px}} .hero{{background:linear-gradient(135deg,#17243b,#315a9b);color:#fff;padding:30px;border-radius:18px}}
.hero h1{{margin:0 0 8px}} .hero p{{margin:4px 0;color:#e7edf8}} .section{{background:#fff;margin-top:20px;padding:23px;border-radius:15px;box-shadow:0 5px 18px rgba(28,39,58,.06)}}
.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:18px}} .metric{{background:#f7f9fc;border:1px solid var(--line);padding:13px;border-radius:11px}}
.metric span{{display:block;color:var(--muted);font-size:13px}} .metric strong{{font-size:20px}} .badge{{padding:4px 10px;border-radius:99px;font-size:13px}}
.good{{background:#e7f6ef;color:#15704d}} .watch{{background:#eaf1ff;color:#315a9b}} .caution{{background:#fff4dc;color:#986515}} .risk{{background:#fdebea;color:#a63831}}
.note{{border-left:4px solid #d69e2e;background:#fff8e8;padding:13px 15px;border-radius:6px}} .chart{{width:100%;border:1px solid var(--line);border-radius:12px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}} table{{border-collapse:collapse;width:100%;font-size:13px;min-width:900px}}
th{{background:#f4f7fb;text-align:left}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}} .empty{{border:1px dashed #cbd2dc;padding:17px;border-radius:10px;color:var(--muted)}}
.rules{{columns:2;column-gap:34px}} .footer{{color:var(--muted);font-size:13px;padding:20px 3px}} @media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.rules{{columns:1}}}}
</style></head><body><main class="page">
<header class="hero"><h1>A股日线模拟盘</h1><p>报告日：{html.escape(report_date)}　数据完成日：{html.escape(data_date)}　策略：daily-pullback-v1</p><p>{html.escape(status_note)}</p></header>
<section class="section"><h2>账户概览</h2><div class="metrics">
<div class="metric"><span>初始资金</span><strong>¥{_fmt(initial_cash)}</strong></div>
<div class="metric"><span>账户权益</span><strong>¥{_fmt(latest_equity)}</strong></div>
<div class="metric"><span>可用现金</span><strong>¥{_fmt(cash)}</strong></div>
<div class="metric"><span>累计收益</span><strong>{cumulative:+.2f}%</strong></div>
<div class="metric"><span>最大回撤</span><strong>{max_drawdown:.2f}%</strong></div>
<div class="metric"><span>持仓/待买</span><strong>{len(positions)}/{pending_buys}</strong></div>
</div><p class="note">真实记账从 {html.escape(str(account['start_wall_date']))} 开始。不会把启动前已经发生的开盘价回填成模拟成交。</p>
<img class="chart" src="{equity_src}" alt="模拟账户净值曲线"></section>
<section class="section"><h2>市场过滤</h2><p><span class="badge {regime_class}">{html.escape(str(context.get('regime','数据不足')))}</span>
 全A等权 <strong>{_fmt(context.get('equal_pct'))}%</strong>，上涨家数占比 <strong>{_fmt(context.get('breadth'))}%</strong>，
涨停/炸板/跌停 <strong>{context.get('limit_up',0)}/{context.get('broken',0)}/{context.get('limit_down',0)}</strong>，成交额 <strong>{_fmt(context.get('amount_yi'))}亿元</strong>。</p>{chart_html}</section>
<section class="section"><h2>当前持仓</h2>{_table(positions, ['代码','名称','板块','股数','成本','收盘价','市值','浮动盈亏','收益率%','持有日'], '当前空仓。空仓是规则结果，不强制开仓。')}</section>
<section class="section"><h2>下一交易日待处理订单</h2>{_table(pending, ['方向','信号日','代码','名称','板块','计划股数','参考价','原因','成交约束'], '没有待成交订单。')}</section>
<section class="section"><h2>最新候选与规则检查</h2>{_table(signals, ['排名','代码','名称','板块','评分','收盘价','规则检查','计划','入选依据'], '尚未产生启动后的有效收盘信号。')}</section>
<section class="section"><h2>成交记录</h2><p>已实现盈亏：<strong>¥{_fmt(realized)}</strong>；已完成卖出交易胜率：<strong>{win_rate_text}</strong>。</p>{_table(trades.head(30), ['成交日','方向','代码','名称','股数','成交价','费用','已实现盈亏','交易原因'], '尚无成交。')}</section>
<section class="section"><h2>固定执行规则</h2><div class="rules"><p>1. T日收盘生成信号，T+1开盘模拟成交。</p><p>2. 开盘跳空低于-2%或高于+3%取消买入。</p><p>3. 最多持有2只，震荡市场最多1只，单只目标仓位45%。</p><p>4. 买入100股整数倍，保留不少于10%现金。</p><p>5. 收盘亏损达到5%、跌破MA10或板块转弱，次日开盘退出。</p><p>6. 盈利8%后跌破MA5执行保护，最长持有10日。</p><p>7. 加入0.10%滑点、最低5元佣金和卖出印花税模拟参数。</p><p>8. 不做T、不补录成交、不因候选不足降低门槛。</p></div></section>
<footer class="footer">这是规则驱动的虚拟资金模拟盘，不连接券商、不构成投资建议，也不代表未来收益。</footer>
</main></body></html>"""
    output.write_text(content, encoding="utf-8")
    return output
