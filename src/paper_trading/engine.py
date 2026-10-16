from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml


STATE_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS paper_account (
        account_id VARCHAR PRIMARY KEY, created_at TIMESTAMP, start_wall_date VARCHAR,
        initial_cash DOUBLE, cash DOUBLE, status VARCHAR, strategy_version VARCHAR,
        last_processed_date VARCHAR, cooldown_remaining INTEGER, inception_note VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_positions (
        account_id VARCHAR, ts_code VARCHAR, name VARCHAR, sector VARCHAR,
        quantity INTEGER, avg_cost DOUBLE, entry_date VARCHAR, entry_signal_date VARCHAR,
        entry_fee DOUBLE, last_price DOUBLE, high_close DOUBLE, days_held INTEGER,
        PRIMARY KEY(account_id, ts_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_orders (
        order_id VARCHAR PRIMARY KEY, account_id VARCHAR, created_at TIMESTAMP,
        signal_date VARCHAR, side VARCHAR, ts_code VARCHAR, name VARCHAR, sector VARCHAR,
        quantity INTEGER, reference_price DOUBLE, reason VARCHAR, status VARCHAR,
        fill_date VARCHAR, fill_price DOUBLE, gross_amount DOUBLE, fees DOUBLE, note VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_trades (
        trade_id VARCHAR PRIMARY KEY, order_id VARCHAR, account_id VARCHAR,
        trade_date VARCHAR, side VARCHAR, ts_code VARCHAR, name VARCHAR,
        quantity INTEGER, fill_price DOUBLE, gross_amount DOUBLE, fees DOUBLE,
        realized_pnl DOUBLE, reason VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_signals (
        account_id VARCHAR, trade_date VARCHAR, rank INTEGER, ts_code VARCHAR,
        name VARCHAR, sector VARCHAR, score DOUBLE, close DOUBLE, selected BOOLEAN,
        eligibility VARCHAR, reason VARCHAR,
        PRIMARY KEY(account_id, trade_date, ts_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_equity (
        account_id VARCHAR, trade_date VARCHAR, cash DOUBLE, market_value DOUBLE,
        total_equity DOUBLE, daily_return_pct DOUBLE, cumulative_return_pct DOUBLE,
        drawdown_pct DOUBLE, benchmark_code VARCHAR, benchmark_close DOUBLE,
        benchmark_cumulative_return_pct DOUBLE,
        PRIMARY KEY(account_id, trade_date)
    )
    """,
]


@dataclass(frozen=True)
class ExecutionCost:
    fill_price: float
    gross: float
    fees: float


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def execution_cost(
    side: str,
    open_price: float,
    quantity: int,
    config: dict[str, Any],
) -> ExecutionCost:
    settings = config["execution"]
    slippage = float(settings["slippage_rate"])
    fill_price = float(open_price) * (1 + slippage if side == "BUY" else 1 - slippage)
    gross = fill_price * int(quantity)
    commission = max(
        float(settings["minimum_commission"]),
        gross * float(settings["commission_rate"]),
    )
    stamp = gross * float(settings["sell_stamp_rate"]) if side == "SELL" else 0.0
    return ExecutionCost(round(fill_price, 4), round(gross, 2), round(commission + stamp, 2))


def opening_gap_pct(open_price: float, pre_close: float) -> float:
    if not pre_close:
        return np.nan
    return (float(open_price) / float(pre_close) - 1) * 100


def filter_entry_candidates(
    candidates: pd.DataFrame,
    config: dict[str, Any],
    market_regime: str,
    market_breadth: float,
) -> pd.DataFrame:
    columns = list(candidates.columns) + ["eligibility"]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    entry = config["entry"]
    portfolio = config["portfolio"]
    work = candidates.copy()
    numeric = [
        "score", "close", "pct_chg", "volume_ratio", "close_position", "rsi6",
        "kdj_j", "bias5", "relative_sector", "return_5d",
    ]
    for column in numeric:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    market_ok = (
        market_regime in set(entry["allowed_market_regimes"])
        and float(market_breadth) >= float(entry["minimum_market_breadth_pct"])
    )
    conditions = {
        "市场过滤": pd.Series(market_ok, index=work.index),
        "评分": work["score"] >= float(entry["minimum_score"]),
        "板块状态": work["sector_state"].isin(entry["allowed_sector_states"]),
        "价格": work["close"] <= float(portfolio["maximum_stock_price"]),
        "当日涨幅": work["pct_chg"].between(entry["pct_chg_min"], entry["pct_chg_max"]),
        "量比": work["volume_ratio"].between(
            entry["volume_ratio_min"], entry["volume_ratio_max"]
        ),
        "收盘位置": work["close_position"] >= float(entry["close_position_min"]),
        "RSI6": work["rsi6"].between(entry["rsi6_min"], entry["rsi6_max"]),
        "KDJ-J": work["kdj_j"] < float(entry["kdj_j_max"]),
        "MA5乖离": work["bias5"].between(entry["bias5_min"], entry["bias5_max"]),
        "近5日涨幅": work["return_5d"] <= float(entry["return_5d_max"]),
        "相对行业": work["relative_sector"] > 0,
        "风险": work.get("risk_reasons", pd.Series("", index=work.index))
        .fillna("")
        .isin(["", "无严重量价风险"]),
        "资金": ~work.get("fund_status", pd.Series("", index=work.index))
        .fillna("")
        .isin(["强净流出"]),
    }
    passed = pd.DataFrame(conditions)
    work["eligible"] = passed.all(axis=1)
    work["eligibility"] = passed.apply(
        lambda row: "通过" if bool(row.all()) else "未通过：" + "、".join(row.index[~row]),
        axis=1,
    )
    return work.sort_values(["eligible", "score"], ascending=[False, False])


class PaperTradingEngine:
    def __init__(
        self,
        market_db: str | Path,
        state_db: str | Path,
        config_path: str | Path,
        output_root: str | Path,
    ):
        self.market_db = Path(market_db)
        self.state_db = Path(state_db)
        self.config_path = Path(config_path)
        self.output_root = Path(output_root)
        self.config = load_config(self.config_path)
        self.account_id = str(self.config["account"]["account_id"])
        self.state_db.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    def initialize_schema(self) -> None:
        with duckdb.connect(str(self.state_db)) as con:
            for sql in STATE_SCHEMA:
                con.execute(sql)

    def initialize_account(self, start_wall_date: str, inception_note: str) -> None:
        account = self.config["account"]
        with duckdb.connect(str(self.state_db)) as con:
            exists = con.execute(
                "select count(1) from paper_account where account_id = ?", [self.account_id]
            ).fetchone()[0]
            if exists:
                return
            con.execute(
                """
                insert into paper_account values (?, ?, ?, ?, ?, 'ACTIVE', ?, null, 0, ?)
                """,
                [
                    self.account_id,
                    datetime.now(),
                    start_wall_date,
                    float(account["initial_cash"]),
                    float(account["initial_cash"]),
                    str(account["strategy_version"]),
                    inception_note,
                ],
            )

    def account(self, con: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
        owns = con is None
        connection = con or duckdb.connect(str(self.state_db))
        try:
            frame = connection.execute(
                "select * from paper_account where account_id = ?", [self.account_id]
            ).fetchdf()
            if frame.empty:
                raise RuntimeError("Paper account is not initialized")
            return frame.iloc[0].to_dict()
        finally:
            if owns:
                connection.close()

    def latest_market_date(self) -> str:
        with duckdb.connect(str(self.market_db), read_only=True) as con:
            value = con.execute("select max(trade_date) from daily_price").fetchone()[0]
        if not value:
            raise RuntimeError("daily_price has no completed trade date")
        return str(value)

    def market_context(self, trade_date: str) -> dict[str, Any]:
        with duckdb.connect(str(self.market_db), read_only=True) as con:
            row = con.execute(
                """
                select avg(pct_chg), median(pct_chg),
                       avg(case when pct_chg > 0 then 1.0 else 0.0 end) * 100,
                       sum(amount) / 100000.0, count(1)
                from daily_price where trade_date = ?
                """,
                [trade_date],
            ).fetchone()
            index = con.execute(
                "select close, pct_chg from index_daily where ts_code='000001.SH' and trade_date=?",
                [trade_date],
            ).fetchone()
            events = con.execute(
                """
                select sum(case when event_type='limit_up' then 1 else 0 end),
                       sum(case when event_type='broken_board' then 1 else 0 end),
                       sum(case when event_type='limit_down' then 1 else 0 end)
                from ak_limit_event where trade_date=?
                """,
                [trade_date],
            ).fetchone()
        equal_pct, median_pct, breadth, amount_yi, stocks = row
        index_close, index_pct = index if index else (np.nan, np.nan)
        limit_up, broken, limit_down = [int(value or 0) for value in events]
        attempts = limit_up + broken
        broken_rate = broken / attempts * 100 if attempts else np.nan
        retreat = (
            float(equal_pct) <= -1.0
            or float(breadth) < 35
            or (pd.notna(index_pct) and float(index_pct) <= -1.2)
            or (pd.notna(broken_rate) and broken_rate >= 40 and limit_down >= 10)
            or limit_down > limit_up
        )
        if retreat:
            regime = "退潮"
        elif float(equal_pct) < -0.3 or float(breadth) < 45:
            regime = "弱势"
        elif float(equal_pct) >= 0.5 and float(breadth) >= 60 and float(index_pct or 0) >= 0:
            regime = "强势"
        else:
            regime = "震荡"
        return {
            "trade_date": trade_date,
            "equal_pct": float(equal_pct),
            "median_pct": float(median_pct),
            "breadth": float(breadth),
            "amount_yi": float(amount_yi),
            "stocks": int(stocks),
            "index_close": float(index_close),
            "index_pct": float(index_pct),
            "limit_up": limit_up,
            "broken": broken,
            "broken_rate": broken_rate,
            "limit_down": limit_down,
            "regime": regime,
        }

    def load_candidates(self, trade_date: str, context: dict[str, Any]) -> pd.DataFrame:
        report_dir = self.output_root.parent / "after_close" / trade_date
        watchlist_path = report_dir / f"stock_watchlist_{trade_date}.csv"
        if not watchlist_path.exists():
            return pd.DataFrame()
        candidates = pd.read_csv(watchlist_path, encoding="utf-8-sig")
        if candidates.empty:
            return candidates
        with duckdb.connect(str(self.market_db), read_only=True) as con:
            extras = con.execute(
                """
                select s.ts_code, s.return_5d, t.ma5, t.ma10, t.ma20,
                       t.above_ma5, t.above_ma10, t.above_ma20
                from factor_stage s
                left join factor_technical t using(ts_code, trade_date)
                where s.trade_date = ?
                """,
                [trade_date],
            ).fetchdf()
        candidates = candidates.merge(extras, on="ts_code", how="left")
        candidates = candidates[
            candidates[["above_ma5", "above_ma10", "above_ma20"]].fillna(False).all(axis=1)
        ].copy()
        return filter_entry_candidates(
            candidates,
            self.config,
            str(context["regime"]),
            float(context["breadth"]),
        )

    def _price_row(self, trade_date: str, ts_code: str) -> dict[str, Any] | None:
        with duckdb.connect(str(self.market_db), read_only=True) as con:
            frame = con.execute(
                "select * from daily_price where trade_date=? and ts_code=?",
                [trade_date, ts_code],
            ).fetchdf()
        return None if frame.empty else frame.iloc[0].to_dict()

    def _fill_pending_orders(self, trade_date: str, con: duckdb.DuckDBPyConnection) -> list[str]:
        orders = con.execute(
            """
            select * from paper_orders
            where account_id=? and status='PENDING' and signal_date < ?
            order by case when side='SELL' then 0 else 1 end, created_at
            """,
            [self.account_id, trade_date],
        ).fetchdf()
        messages: list[str] = []
        for order in orders.to_dict("records"):
            with duckdb.connect(str(self.market_db), read_only=True) as market:
                next_session = market.execute(
                    "select min(cal_date) from trade_calendar where cal_date > ? and is_open = 1",
                    [order["signal_date"]],
                ).fetchone()[0]
            if str(next_session) != trade_date:
                note = (
                    f"错过下一交易日{next_session}开盘，未补录成交"
                    if next_session else "缺少下一交易日历，未补录成交"
                )
                con.execute(
                    "update paper_orders set status='CANCELLED', note=? where order_id=?",
                    [note, order["order_id"]],
                )
                messages.append(f"{order['name']}取消{order['side']}：{note}")
                continue
            price = self._price_row(trade_date, str(order["ts_code"]))
            if price is None or not price.get("open") or not price.get("pre_close"):
                con.execute(
                    "update paper_orders set status='CANCELLED', note=? where order_id=?",
                    ["下一交易日缺少有效开盘数据", order["order_id"]],
                )
                continue
            gap = opening_gap_pct(price["open"], price["pre_close"])
            if order["side"] == "BUY":
                lower = float(self.config["execution"]["buy_gap_min_pct"])
                upper = float(self.config["execution"]["buy_gap_max_pct"])
                if not lower <= gap <= upper:
                    note = f"开盘跳空{gap:+.2f}%，超出[{lower:.1f}%, {upper:.1f}%]"
                    con.execute(
                        "update paper_orders set status='CANCELLED', note=? where order_id=?",
                        [note, order["order_id"]],
                    )
                    messages.append(f"{order['name']}取消买入：{note}")
                    continue
                existing = con.execute(
                    "select count(1) from paper_positions where account_id=?",
                    [self.account_id],
                ).fetchone()[0]
                if existing >= int(self.config["portfolio"]["max_positions"]):
                    con.execute(
                        "update paper_orders set status='CANCELLED', note='持仓数量已达上限' where order_id=?",
                        [order["order_id"]],
                    )
                    continue
                account = self.account(con)
                quantity = int(order["quantity"])
                cost = execution_cost("BUY", price["open"], quantity, self.config)
                while quantity >= 100 and cost.gross + cost.fees > float(account["cash"]):
                    quantity -= 100
                    cost = execution_cost("BUY", price["open"], quantity, self.config)
                if quantity < 100:
                    con.execute(
                        "update paper_orders set status='CANCELLED', note='现金不足一手' where order_id=?",
                        [order["order_id"]],
                    )
                    continue
                avg_cost = (cost.gross + cost.fees) / quantity
                con.execute(
                    "update paper_account set cash=cash-? where account_id=?",
                    [cost.gross + cost.fees, self.account_id],
                )
                con.execute(
                    """
                    insert into paper_positions values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    [
                        self.account_id, order["ts_code"], order["name"], order["sector"],
                        quantity, avg_cost, trade_date, order["signal_date"], cost.fees,
                        float(price["close"]), float(price["close"]),
                    ],
                )
                realized = 0.0
            else:
                position = con.execute(
                    "select * from paper_positions where account_id=? and ts_code=?",
                    [self.account_id, order["ts_code"]],
                ).fetchdf()
                if position.empty:
                    con.execute(
                        "update paper_orders set status='CANCELLED', note='持仓已不存在' where order_id=?",
                        [order["order_id"]],
                    )
                    continue
                pos = position.iloc[0]
                quantity = int(pos["quantity"])
                cost = execution_cost("SELL", price["open"], quantity, self.config)
                proceeds = cost.gross - cost.fees
                realized = proceeds - quantity * float(pos["avg_cost"])
                con.execute(
                    "update paper_account set cash=cash+? where account_id=?",
                    [proceeds, self.account_id],
                )
                con.execute(
                    "delete from paper_positions where account_id=? and ts_code=?",
                    [self.account_id, order["ts_code"]],
                )
            con.execute(
                """
                update paper_orders set status='FILLED', fill_date=?, fill_price=?,
                       gross_amount=?, fees=?, quantity=?, note=? where order_id=?
                """,
                [trade_date, cost.fill_price, cost.gross, cost.fees, quantity,
                 f"开盘跳空{gap:+.2f}%", order["order_id"]],
            )
            trade_id = stable_id("trade", order["order_id"], trade_date)
            con.execute(
                """
                insert or replace into paper_trades values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [trade_id, order["order_id"], self.account_id, trade_date, order["side"],
                 order["ts_code"], order["name"], quantity, cost.fill_price, cost.gross,
                 cost.fees, realized, order["reason"]],
            )
            messages.append(
                f"{order['name']} {order['side']} {quantity}股 @ {cost.fill_price:.2f}"
            )
        return messages

    def _mark_positions(self, trade_date: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        positions = con.execute(
            "select * from paper_positions where account_id=?", [self.account_id]
        ).fetchdf()
        for position in positions.to_dict("records"):
            price = self._price_row(trade_date, str(position["ts_code"]))
            if price is None:
                continue
            with duckdb.connect(str(self.market_db), read_only=True) as market:
                sessions_held = market.execute(
                    """
                    select count(*) from trade_calendar
                    where cal_date between ? and ? and is_open = 1
                    """,
                    [position["entry_date"], trade_date],
                ).fetchone()[0]
            con.execute(
                """
                update paper_positions set last_price=?, high_close=greatest(high_close, ?),
                       days_held=?
                where account_id=? and ts_code=?
                """,
                [price["close"], price["close"], sessions_held,
                 self.account_id, position["ts_code"]],
            )
        return con.execute(
            "select * from paper_positions where account_id=?", [self.account_id]
        ).fetchdf()

    def _create_exit_orders(
        self,
        trade_date: str,
        positions: pd.DataFrame,
        sector_states: dict[str, str],
        con: duckdb.DuckDBPyConnection,
    ) -> None:
        exit_cfg = self.config["exit"]
        with duckdb.connect(str(self.market_db), read_only=True) as market:
            technical = market.execute(
                "select ts_code, ma5, ma10 from factor_technical where trade_date=?",
                [trade_date],
            ).fetchdf().set_index("ts_code")
        for pos in positions.to_dict("records"):
            code = str(pos["ts_code"])
            if con.execute(
                "select count(1) from paper_orders where account_id=? and ts_code=? and status='PENDING'",
                [self.account_id, code],
            ).fetchone()[0]:
                continue
            close = float(pos["last_price"])
            pnl_pct = (close / float(pos["avg_cost"]) - 1) * 100
            tech = technical.loc[code] if code in technical.index else pd.Series(dtype=float)
            reasons: list[str] = []
            if pnl_pct <= float(exit_cfg["hard_stop_pct"]):
                reasons.append(f"收盘亏损{pnl_pct:.2f}%触发硬止损")
            if pd.notna(tech.get("ma10")) and close < float(tech["ma10"]):
                reasons.append("收盘跌破MA10")
            if (
                pnl_pct >= float(exit_cfg["profit_protect_trigger_pct"])
                and pd.notna(tech.get("ma5"))
                and close < float(tech["ma5"])
            ):
                reasons.append("盈利保护后跌破MA5")
            if (
                int(pos["days_held"]) >= int(exit_cfg["time_stop_days"])
                and pnl_pct < float(exit_cfg["time_stop_min_return_pct"])
            ):
                reasons.append("时间止损")
            if int(pos["days_held"]) >= int(exit_cfg["maximum_holding_days"]):
                reasons.append("达到最长持有期")
            if sector_states.get(str(pos["sector"])) in {"逃顶预警", "弱势回避", "下跌延续"}:
                reasons.append("所属板块转弱")
            if not reasons:
                continue
            order_id = stable_id(self.account_id, trade_date, "SELL", code)
            con.execute(
                """
                insert or ignore into paper_orders values (?, ?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?,
                    'PENDING', null, null, null, null, null)
                """,
                [order_id, self.account_id, datetime.now(), trade_date, code, pos["name"],
                 pos["sector"], int(pos["quantity"]), close, "；".join(reasons)],
            )

    def _record_signals_and_buy_order(
        self,
        trade_date: str,
        candidates: pd.DataFrame,
        context: dict[str, Any],
        con: duckdb.DuckDBPyConnection,
    ) -> None:
        con.execute(
            "delete from paper_signals where account_id=? and trade_date=?",
            [self.account_id, trade_date],
        )
        eligible = candidates[candidates.get("eligible", False)].copy() if not candidates.empty else pd.DataFrame()
        positions_count = con.execute(
            "select count(1) from paper_positions where account_id=?", [self.account_id]
        ).fetchone()[0]
        pending_buys = con.execute(
            "select count(1) from paper_orders where account_id=? and side='BUY' and status='PENDING'",
            [self.account_id],
        ).fetchone()[0]
        account = self.account(con)
        equity = self.current_equity(con)
        high_water = con.execute(
            "select max(total_equity) from paper_equity where account_id=?", [self.account_id]
        ).fetchone()[0]
        drawdown = (1 - equity / high_water) * 100 if high_water else 0.0
        cooldown = int(account.get("cooldown_remaining") or 0)
        max_positions = int(self.config["portfolio"]["max_positions"])
        if context["regime"] == "震荡" or drawdown >= float(self.config["risk"]["drawdown_reduce_pct"]):
            max_positions = min(max_positions, 1)
        can_order = (
            positions_count + pending_buys < max_positions
            and drawdown < float(self.config["risk"]["drawdown_pause_pct"])
            and cooldown == 0
        )
        selected_code = str(eligible.iloc[0]["ts_code"]) if can_order and not eligible.empty else None
        for rank, row in enumerate(candidates.to_dict("records"), start=1):
            selected = str(row["ts_code"]) == selected_code
            con.execute(
                """
                insert into paper_signals values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self.account_id, trade_date, rank, row["ts_code"], row["name"], row["sector"],
                 row["score"], row["close"], selected, row["eligibility"], row.get("reasons", "")],
            )
        if not selected_code:
            if cooldown > 0:
                con.execute(
                    "update paper_account set cooldown_remaining=0 where account_id=?",
                    [self.account_id],
                )
            return
        row = eligible.iloc[0]
        target_value = equity * float(self.config["portfolio"]["target_position_pct"])
        reserved = con.execute(
            """
            select coalesce(sum(quantity * reference_price), 0) from paper_orders
            where account_id=? and side='BUY' and status='PENDING'
            """,
            [self.account_id],
        ).fetchone()[0]
        available = max(0.0, float(account["cash"]) - float(reserved))
        target_value = min(target_value, available * (1 - float(self.config["portfolio"]["minimum_cash_pct"])))
        quantity = int(target_value // (float(row["close"]) * 100)) * 100
        if quantity < 100:
            return
        order_id = stable_id(self.account_id, trade_date, "BUY", selected_code)
        reason = (
            f"评分{float(row['score']):.1f}；市场{context['regime']}；"
            f"板块{row['sector_state']}；收盘位置{float(row['close_position']):.2f}；"
            f"量比{float(row['volume_ratio']):.2f}"
        )
        con.execute(
            """
            insert or ignore into paper_orders values (?, ?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?,
                'PENDING', null, null, null, null, '下一交易日开盘验证跳空区间')
            """,
            [order_id, self.account_id, datetime.now(), trade_date, selected_code, row["name"],
             row["sector"], quantity, float(row["close"]), reason],
        )

    def _activate_loss_cooldown(
        self, trade_date: str, con: duckdb.DuckDBPyConnection
    ) -> None:
        sold_today = con.execute(
            """
            select count(1) from paper_trades
            where account_id=? and trade_date=? and side='SELL'
            """,
            [self.account_id, trade_date],
        ).fetchone()[0]
        if not sold_today:
            return
        loss_count = int(self.config["risk"]["consecutive_loss_pause_count"])
        recent = con.execute(
            """
            select realized_pnl from paper_trades
            where account_id=? and side='SELL'
            order by trade_date desc, trade_id desc limit ?
            """,
            [self.account_id, loss_count],
        ).fetchdf()
        if len(recent) == loss_count and (recent["realized_pnl"] < 0).all():
            con.execute(
                "update paper_account set cooldown_remaining=1 where account_id=?",
                [self.account_id],
            )

    def current_equity(self, con: duckdb.DuckDBPyConnection) -> float:
        account = self.account(con)
        market_value = con.execute(
            "select coalesce(sum(quantity * last_price), 0) from paper_positions where account_id=?",
            [self.account_id],
        ).fetchone()[0]
        return float(account["cash"]) + float(market_value)

    def _record_equity(self, trade_date: str, con: duckdb.DuckDBPyConnection) -> None:
        account = self.account(con)
        market_value = con.execute(
            "select coalesce(sum(quantity * last_price), 0) from paper_positions where account_id=?",
            [self.account_id],
        ).fetchone()[0]
        total = float(account["cash"]) + float(market_value)
        history = con.execute(
            "select trade_date,total_equity from paper_equity where account_id=? order by trade_date",
            [self.account_id],
        ).fetchdf()
        previous = float(history.iloc[-1]["total_equity"]) if not history.empty else float(account["initial_cash"])
        high_water = max([float(account["initial_cash"]), total] + history["total_equity"].astype(float).tolist())
        daily_return = (total / previous - 1) * 100 if previous else 0.0
        cumulative = (total / float(account["initial_cash"]) - 1) * 100
        drawdown = (total / high_water - 1) * 100
        with duckdb.connect(str(self.market_db), read_only=True) as market:
            benchmark = market.execute(
                """
                select ts_code, close from index_daily
                where trade_date=? and ts_code in ('000300.SH','000001.SH')
                order by case when ts_code='000300.SH' then 0 else 1 end limit 1
                """,
                [trade_date],
            ).fetchone()
        benchmark_code, benchmark_close = benchmark if benchmark else ("", np.nan)
        first_benchmark = con.execute(
            """
            select benchmark_close from paper_equity where account_id=? and benchmark_code=?
            and benchmark_close is not null order by trade_date limit 1
            """,
            [self.account_id, benchmark_code],
        ).fetchone()
        baseline = float(first_benchmark[0]) if first_benchmark else float(benchmark_close)
        benchmark_return = (
            (float(benchmark_close) / baseline - 1) * 100
            if pd.notna(benchmark_close) and baseline
            else np.nan
        )
        con.execute(
            "delete from paper_equity where account_id=? and trade_date=?",
            [self.account_id, trade_date],
        )
        con.execute(
            "insert into paper_equity values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [self.account_id, trade_date, account["cash"], market_value, total, daily_return,
             cumulative, drawdown, benchmark_code, benchmark_close, benchmark_return],
        )

    def process_trade_date(self, trade_date: str, candidates: pd.DataFrame) -> dict[str, Any]:
        context = self.market_context(trade_date)
        filtered = self.load_candidates(trade_date, context) if candidates.empty else filter_entry_candidates(
            candidates, self.config, context["regime"], context["breadth"]
        )
        sector_path = self.output_root.parent / "after_close" / trade_date / f"sector_summary_{trade_date}.csv"
        sector_states: dict[str, str] = {}
        if sector_path.exists():
            sectors = pd.read_csv(sector_path, encoding="utf-8-sig")
            sector_states = dict(zip(sectors["板块"].astype(str), sectors["状态"].astype(str)))
        with duckdb.connect(str(self.state_db)) as con:
            account = self.account(con)
            if str(account["start_wall_date"]) > trade_date:
                raise RuntimeError("Refusing to process data before the paper account inception date")
            already = con.execute(
                "select count(1) from paper_equity where account_id=? and trade_date=?",
                [self.account_id, trade_date],
            ).fetchone()[0]
            if already:
                return {"context": context, "candidates": filtered, "fills": [], "idempotent": True}
            fills = self._fill_pending_orders(trade_date, con)
            self._activate_loss_cooldown(trade_date, con)
            positions = self._mark_positions(trade_date, con)
            self._create_exit_orders(trade_date, positions, sector_states, con)
            self._record_equity(trade_date, con)
            self._record_signals_and_buy_order(trade_date, filtered, context, con)
            con.execute(
                "update paper_account set last_processed_date=? where account_id=?",
                [trade_date, self.account_id],
            )
        return {"context": context, "candidates": filtered, "fills": fills, "idempotent": False}

    def snapshot(self) -> dict[str, pd.DataFrame | dict[str, Any]]:
        with duckdb.connect(str(self.state_db), read_only=True) as con:
            account = self.account(con)
            positions = con.execute(
                "select * from paper_positions where account_id=? order by ts_code",
                [self.account_id],
            ).fetchdf()
            pending = con.execute(
                "select * from paper_orders where account_id=? and status='PENDING' order by created_at",
                [self.account_id],
            ).fetchdf()
            trades = con.execute(
                "select * from paper_trades where account_id=? order by trade_date desc",
                [self.account_id],
            ).fetchdf()
            signals = con.execute(
                "select * from paper_signals where account_id=? order by trade_date desc,rank",
                [self.account_id],
            ).fetchdf()
            equity = con.execute(
                "select * from paper_equity where account_id=? order by trade_date",
                [self.account_id],
            ).fetchdf()
        return {
            "account": account,
            "positions": positions,
            "pending": pending,
            "trades": trades,
            "signals": signals,
            "equity": equity,
        }
