from __future__ import annotations

BASE_TABLE_SQL: dict[str, str] = {
    "stock_basic": """
        CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code VARCHAR, symbol VARCHAR, name VARCHAR, area VARCHAR,
            industry VARCHAR, market VARCHAR, list_date VARCHAR
        )
    """,
    "trade_calendar": """
        CREATE TABLE IF NOT EXISTS trade_calendar (
            exchange VARCHAR, cal_date VARCHAR, is_open INTEGER, pretrade_date VARCHAR
        )
    """,
    "daily_price": """
        CREATE TABLE IF NOT EXISTS daily_price (
            ts_code VARCHAR, trade_date VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, pre_close DOUBLE, change DOUBLE, pct_chg DOUBLE,
            vol DOUBLE, amount DOUBLE
        )
    """,
    "daily_basic": """
        CREATE TABLE IF NOT EXISTS daily_basic (
            ts_code VARCHAR, trade_date VARCHAR, close DOUBLE, turnover_rate DOUBLE,
            turnover_rate_f DOUBLE, volume_ratio DOUBLE, pe DOUBLE, pb DOUBLE,
            total_mv DOUBLE, circ_mv DOUBLE
        )
    """,
    "moneyflow": """
        CREATE TABLE IF NOT EXISTS moneyflow (
            ts_code VARCHAR, trade_date VARCHAR, buy_sm_amount DOUBLE, sell_sm_amount DOUBLE,
            buy_lg_amount DOUBLE, sell_lg_amount DOUBLE, net_mf_amount DOUBLE
        )
    """,
    "stock_limit": """
        CREATE TABLE IF NOT EXISTS stock_limit (
            trade_date VARCHAR, ts_code VARCHAR, pre_close DOUBLE,
            up_limit DOUBLE, down_limit DOUBLE
        )
    """,
    "ak_market_fund_flow": """
        CREATE TABLE IF NOT EXISTS ak_market_fund_flow (
            trade_date VARCHAR, sh_close DOUBLE, sh_pct_chg DOUBLE,
            sz_close DOUBLE, sz_pct_chg DOUBLE,
            main_net_amount DOUBLE, main_net_ratio DOUBLE,
            super_large_net_amount DOUBLE, super_large_net_ratio DOUBLE,
            large_net_amount DOUBLE, large_net_ratio DOUBLE,
            medium_net_amount DOUBLE, medium_net_ratio DOUBLE,
            small_net_amount DOUBLE, small_net_ratio DOUBLE,
            net_inflow_amount DOUBLE, net_inflow_ratio DOUBLE, flow_metric VARCHAR,
            source VARCHAR, fetched_at TIMESTAMP, data_status VARCHAR, is_proxy BOOLEAN
        )
    """,
    "ak_sector_fund_flow": """
        CREATE TABLE IF NOT EXISTS ak_sector_fund_flow (
            trade_date VARCHAR, provider_sector VARCHAR, canonical_sector VARCHAR,
            sector_type VARCHAR, pct_chg DOUBLE,
            main_net_amount DOUBLE, main_net_ratio DOUBLE,
            super_large_net_amount DOUBLE, super_large_net_ratio DOUBLE,
            large_net_amount DOUBLE, large_net_ratio DOUBLE,
            medium_net_amount DOUBLE, medium_net_ratio DOUBLE,
            small_net_amount DOUBLE, small_net_ratio DOUBLE,
            net_inflow_amount DOUBLE, net_inflow_ratio DOUBLE, flow_metric VARCHAR,
            largest_inflow_stock VARCHAR, source VARCHAR, fetched_at TIMESTAMP,
            data_status VARCHAR, is_proxy BOOLEAN
        )
    """,
    "ak_stock_fund_flow": """
        CREATE TABLE IF NOT EXISTS ak_stock_fund_flow (
            trade_date VARCHAR, ts_code VARCHAR, name VARCHAR,
            close DOUBLE, pct_chg DOUBLE,
            main_net_amount DOUBLE, main_net_ratio DOUBLE,
            super_large_net_amount DOUBLE, super_large_net_ratio DOUBLE,
            large_net_amount DOUBLE, large_net_ratio DOUBLE,
            medium_net_amount DOUBLE, medium_net_ratio DOUBLE,
            small_net_amount DOUBLE, small_net_ratio DOUBLE,
            net_inflow_amount DOUBLE, net_inflow_ratio DOUBLE,
            turnover_amount DOUBLE, flow_metric VARCHAR,
            source VARCHAR, fetched_at TIMESTAMP, data_status VARCHAR, is_proxy BOOLEAN
        )
    """,
    "ak_limit_event": """
        CREATE TABLE IF NOT EXISTS ak_limit_event (
            trade_date VARCHAR, ts_code VARCHAR, name VARCHAR,
            event_type VARCHAR, pct_chg DOUBLE, close DOUBLE, limit_price DOUBLE,
            amount DOUBLE, float_mv DOUBLE, total_mv DOUBLE, turnover_rate DOUBLE,
            seal_amount DOUBLE, first_seal_time VARCHAR, last_seal_time VARCHAR,
            open_count INTEGER, consecutive_boards INTEGER, limit_stats VARCHAR,
            amplitude DOUBLE, industry VARCHAR, source VARCHAR, fetched_at TIMESTAMP,
            data_status VARCHAR, is_proxy BOOLEAN
        )
    """,
    "top_list": """
        CREATE TABLE IF NOT EXISTS top_list (
            trade_date VARCHAR, ts_code VARCHAR, name VARCHAR, close DOUBLE,
            pct_change DOUBLE, turnover_rate DOUBLE, amount DOUBLE, net_amount DOUBLE
        )
    """,
    "top_inst": """
        CREATE TABLE IF NOT EXISTS top_inst (
            trade_date VARCHAR, ts_code VARCHAR, exalter VARCHAR,
            buy DOUBLE, sell DOUBLE, net_buy DOUBLE
        )
    """,
    "index_daily": """
        CREATE TABLE IF NOT EXISTS index_daily (
            ts_code VARCHAR, trade_date VARCHAR, close DOUBLE, open DOUBLE, high DOUBLE,
            low DOUBLE, pre_close DOUBLE, change DOUBLE, pct_chg DOUBLE,
            vol DOUBLE, amount DOUBLE
        )
    """,
}

FACTOR_TABLE_SQL: dict[str, str] = {
    "factor_technical": """
        CREATE TABLE IF NOT EXISTS factor_technical (
            ts_code VARCHAR, trade_date VARCHAR, close DOUBLE,
            ma5 DOUBLE, ma10 DOUBLE, ma20 DOUBLE, ma60 DOUBLE,
            macd_dif DOUBLE, macd_dea DOUBLE, macd_hist DOUBLE,
            kdj_k DOUBLE, kdj_d DOUBLE, kdj_j DOUBLE,
            rsi6 DOUBLE, rsi12 DOUBLE, boll_upper DOUBLE, boll_mid DOUBLE,
            boll_lower DOUBLE, cci DOUBLE, bias5 DOUBLE, bias10 DOUBLE,
            bias20 DOUBLE, wr DOUBLE, mtm DOUBLE, above_ma5 BOOLEAN,
            above_ma10 BOOLEAN, above_ma20 BOOLEAN, ma5_cross_ma10 BOOLEAN,
            macd_golden_cross BOOLEAN, kdj_golden_cross BOOLEAN,
            technical_strengthening BOOLEAN
        )
    """,
    "factor_volume_price": """
        CREATE TABLE IF NOT EXISTS factor_volume_price (
            ts_code VARCHAR, trade_date VARCHAR, pct_chg DOUBLE, amplitude DOUBLE,
            turnover_rate DOUBLE, vol DOUBLE, amount DOUBLE, vol_ma5 DOUBLE,
            amount_ma5 DOUBLE, volume_ratio DOUBLE, close_position DOUBLE,
            main_net_inflow DOUBLE, large_net_inflow DOUBLE, small_net_inflow DOUBLE,
            is_volume_expansion BOOLEAN, is_volume_contraction BOOLEAN,
            stronger_than_market BOOLEAN, stronger_than_sector BOOLEAN
        )
    """,
    "factor_stage": """
        CREATE TABLE IF NOT EXISTS factor_stage (
            ts_code VARCHAR, trade_date VARCHAR, return_5d DOUBLE, return_10d DOUBLE,
            return_20d DOUBLE, return_60d DOUBLE, turnover_5d DOUBLE,
            turnover_20d DOUBLE, amplitude_5d DOUBLE, amplitude_20d DOUBLE,
            new_high_5d BOOLEAN, new_high_10d BOOLEAN, new_high_20d BOOLEAN,
            new_low_5d BOOLEAN, new_low_10d BOOLEAN, is_platform BOOLEAN,
            platform_breakout BOOLEAN, stage_volume_expansion BOOLEAN,
            stage_volume_contraction BOOLEAN, stage_overheated BOOLEAN
        )
    """,
    "factor_risk": """
        CREATE TABLE IF NOT EXISTS factor_risk (
            ts_code VARCHAR, trade_date VARCHAR, risk_triggered BOOLEAN,
            severe_risk BOOLEAN, risk_score DOUBLE, risk_reasons VARCHAR
        )
    """,
}

RESULT_TABLE_SQL: dict[str, str] = {
    "screening_result": """
        CREATE TABLE IF NOT EXISTS screening_result (
            ts_code VARCHAR, trade_date VARCHAR, name VARCHAR, models VARCHAR,
            technical_score DOUBLE, market_score DOUBLE, stage_score DOUBLE,
            risk_deduction DOUBLE, total_score DOUBLE, classification VARCHAR,
            reasons VARCHAR, risk_reasons VARCHAR
        )
    """,
    "event_study_result": """
        CREATE TABLE IF NOT EXISTS event_study_result (
            ts_code VARCHAR, trade_date VARCHAR, next_pct_chg DOUBLE,
            group_name VARCHAR, metrics_json VARCHAR
        )
    """,
    "daily_report_log": """
        CREATE TABLE IF NOT EXISTS daily_report_log (
            trade_date VARCHAR, generated_at TIMESTAMP, report_path VARCHAR,
            watchlist_path VARCHAR, status VARCHAR
        )
    """,
}

ALL_TABLE_SQL = {**BASE_TABLE_SQL, **FACTOR_TABLE_SQL, **RESULT_TABLE_SQL}

MIGRATION_SQL = [
    "ALTER TABLE ak_market_fund_flow ADD COLUMN IF NOT EXISTS net_inflow_amount DOUBLE",
    "ALTER TABLE ak_market_fund_flow ADD COLUMN IF NOT EXISTS net_inflow_ratio DOUBLE",
    "ALTER TABLE ak_market_fund_flow ADD COLUMN IF NOT EXISTS flow_metric VARCHAR",
    "ALTER TABLE ak_sector_fund_flow ADD COLUMN IF NOT EXISTS net_inflow_amount DOUBLE",
    "ALTER TABLE ak_sector_fund_flow ADD COLUMN IF NOT EXISTS net_inflow_ratio DOUBLE",
    "ALTER TABLE ak_sector_fund_flow ADD COLUMN IF NOT EXISTS flow_metric VARCHAR",
    "ALTER TABLE ak_stock_fund_flow ADD COLUMN IF NOT EXISTS net_inflow_amount DOUBLE",
    "ALTER TABLE ak_stock_fund_flow ADD COLUMN IF NOT EXISTS net_inflow_ratio DOUBLE",
    "ALTER TABLE ak_stock_fund_flow ADD COLUMN IF NOT EXISTS turnover_amount DOUBLE",
    "ALTER TABLE ak_stock_fund_flow ADD COLUMN IF NOT EXISTS flow_metric VARCHAR",
]
