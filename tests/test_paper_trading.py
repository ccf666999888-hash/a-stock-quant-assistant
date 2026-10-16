from __future__ import annotations

import pandas as pd
import duckdb

from paper_trading.engine import (
    PaperTradingEngine,
    execution_cost,
    filter_entry_candidates,
    load_config,
    opening_gap_pct,
    stable_id,
)


def config():
    return load_config("configs/paper_trading.yaml")


def candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "示例股份",
                "sector": "示例行业",
                "sector_state": "强势延续",
                "score": 82.0,
                "close": 20.0,
                "pct_chg": 2.0,
                "volume_ratio": 1.2,
                "close_position": 0.82,
                "rsi6": 58.0,
                "kdj_j": 72.0,
                "bias5": 2.5,
                "relative_sector": 1.2,
                "return_5d": 6.0,
                "risk_reasons": "",
                "fund_status": "温和净流入",
            }
        ]
    )


def test_execution_cost_includes_slippage_minimum_commission_and_stamp():
    buy = execution_cost("BUY", 10.0, 400, config())
    sell = execution_cost("SELL", 10.0, 400, config())
    assert buy.fill_price == 10.01
    assert buy.fees == 5.0
    assert sell.fill_price == 9.99
    assert sell.fees == 7.0


def test_opening_gap_uses_pre_close():
    assert round(opening_gap_pct(10.2, 10.0), 2) == 2.0


def test_candidate_passes_only_when_market_gate_is_open():
    passed = filter_entry_candidates(candidate_frame(), config(), "震荡", 52.0)
    blocked = filter_entry_candidates(candidate_frame(), config(), "弱势", 42.0)
    assert bool(passed.iloc[0]["eligible"])
    assert not bool(blocked.iloc[0]["eligible"])
    assert "市场过滤" in blocked.iloc[0]["eligibility"]


def test_overheated_candidate_is_rejected():
    frame = candidate_frame()
    frame.loc[0, "rsi6"] = 78.0
    result = filter_entry_candidates(frame, config(), "强势", 65.0)
    assert not bool(result.iloc[0]["eligible"])
    assert "RSI6" in result.iloc[0]["eligibility"]


def test_no_severe_risk_label_is_not_misread_as_a_risk():
    frame = candidate_frame()
    frame.loc[0, "risk_reasons"] = "无严重量价风险"
    result = filter_entry_candidates(frame, config(), "强势", 65.0)
    assert bool(result.iloc[0]["eligible"])


def test_stable_ids_are_idempotent():
    assert stable_id("a", "20260831", "BUY") == stable_id("a", "20260831", "BUY")


def test_missed_next_session_cancels_order_without_a_late_fill(tmp_path):
    market_db = tmp_path / "market.duckdb"
    with duckdb.connect(str(market_db)) as market:
        market.execute("create table trade_calendar (cal_date varchar, is_open integer)")
        market.execute("insert into trade_calendar values ('20261013', 1), ('20261014', 1)")
    engine = PaperTradingEngine(
        market_db, tmp_path / "paper.duckdb", "configs/paper_trading.yaml", tmp_path / "out"
    )
    engine.initialize_account("20261012", "test")
    with duckdb.connect(str(engine.state_db)) as paper:
        paper.execute(
            """
            insert into paper_orders values
            ('missed', ?, current_timestamp, '20261012', 'SELL', '601360.SH',
             '三六零', '互联网', 400, 8.99, '跌破MA10', 'PENDING',
             null, null, null, null, null)
            """,
            [engine.account_id],
        )
        messages = engine._fill_pending_orders("20261014", paper)
        status, note = paper.execute(
            "select status, note from paper_orders where order_id='missed'"
        ).fetchone()
        assert status == "CANCELLED"
        assert "20261013" in note
        assert paper.execute("select count(*) from paper_trades").fetchone()[0] == 0
        assert "取消" in messages[0]


def test_holding_sessions_include_unprocessed_trading_days(tmp_path):
    market_db = tmp_path / "market.duckdb"
    with duckdb.connect(str(market_db)) as market:
        market.execute("create table trade_calendar (cal_date varchar, is_open integer)")
        market.execute(
            "insert into trade_calendar values "
            "('20261012', 1), ('20261013', 1), ('20261014', 1)"
        )
        market.execute(
            "create table daily_price "
            "(ts_code varchar, trade_date varchar, open double, pre_close double, close double)"
        )
        market.execute(
            "insert into daily_price values ('601360.SH', '20261014', 8.87, 8.92, 8.80)"
        )
    engine = PaperTradingEngine(
        market_db, tmp_path / "paper.duckdb", "configs/paper_trading.yaml", tmp_path / "out"
    )
    engine.initialize_account("20261012", "test")
    with duckdb.connect(str(engine.state_db)) as paper:
        paper.execute(
            """
            insert into paper_positions values
            (?, '601360.SH', '三六零', '互联网', 400, 9.07,
             '20261012', '20261009', 5.0, 8.99, 9.05, 1)
            """,
            [engine.account_id],
        )
        marked = engine._mark_positions("20261014", paper)
        assert int(marked.iloc[0]["days_held"]) == 3
        assert float(marked.iloc[0]["last_price"]) == 8.80
