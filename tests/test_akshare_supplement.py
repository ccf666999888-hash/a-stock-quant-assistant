from __future__ import annotations

from datetime import datetime

import pandas as pd

from data.akshare_client import (
    canonical_sector_name,
    normalize_limit_pool,
    normalize_stock_fund_flow,
    normalize_ths_sector_fund_flow,
    normalize_ths_stock_fund_flow,
    parse_number,
    standardize_stock_code,
)


def test_standardize_stock_code_handles_all_three_markets() -> None:
    assert standardize_stock_code(1) == "000001.SZ"
    assert standardize_stock_code("600000") == "600000.SH"
    assert standardize_stock_code("920082") == "920082.BJ"


def test_parse_number_supports_chinese_amount_units() -> None:
    assert parse_number("1.5亿") == 150_000_000
    assert parse_number("-23.4万") == -234_000
    assert parse_number("3.2%") == 3.2


def test_stock_fund_flow_normalization_preserves_provider_definition() -> None:
    raw = pd.DataFrame(
        [
            {
                "代码": "000001",
                "名称": "平安银行",
                "最新价": 12.3,
                "今日涨跌幅": 1.2,
                "今日主力净流入-净额": 1000,
                "今日主力净流入-净占比": 2.5,
                "今日超大单净流入-净额": 700,
                "今日大单净流入-净额": 300,
            }
        ]
    )
    result = normalize_stock_fund_flow(raw, "20260828", datetime(2026, 8, 29, 16, 0))

    assert result.loc[0, "ts_code"] == "000001.SZ"
    assert result.loc[0, "main_net_ratio"] == 2.5
    assert result.loc[0, "source"] == "eastmoney_via_akshare"
    assert result.loc[0, "is_proxy"] == False  # noqa: E712


def test_limit_pool_normalization_keeps_board_quality_fields() -> None:
    raw = pd.DataFrame(
        [
            {
                "代码": "600001",
                "名称": "示例股份",
                "涨跌幅": 10.01,
                "最新价": 11.0,
                "封板资金": 50_000_000,
                "首次封板时间": 92500,
                "最后封板时间": 143501,
                "炸板次数": 2,
                "连板数": 1,
                "所属行业": "电网设备",
            }
        ]
    )
    result = normalize_limit_pool(raw, "20260828", "limit_up")

    assert result.loc[0, "first_seal_time"] == "09:25:00"
    assert result.loc[0, "last_seal_time"] == "14:35:01"
    assert result.loc[0, "open_count"] == 2
    assert result.loc[0, "consecutive_boards"] == 1


def test_sector_aliases_match_existing_report_names() -> None:
    assert canonical_sector_name("电子元件") == "元器件"
    assert canonical_sector_name("电网设备") == "电力设备"
    assert canonical_sector_name("旅游酒店") == "旅游"


def test_ths_stock_flow_stays_generic_not_main_flow() -> None:
    raw = pd.DataFrame(
        [{"股票代码": 1, "股票简称": "平安银行", "最新价": 12, "涨跌幅": "1.2%", "净额": "3000万", "成交额": "3亿"}]
    )
    result = normalize_ths_stock_fund_flow(raw, "20260828")

    assert result.loc[0, "net_inflow_amount"] == 30_000_000
    assert result.loc[0, "net_inflow_ratio"] == 10
    assert pd.isna(result.loc[0, "main_net_amount"])
    assert result.loc[0, "flow_metric"] == "total_flow_net"
    assert result.loc[0, "source"] == "tonghuashun_via_akshare"


def test_ths_sector_flow_converts_yi_to_yuan() -> None:
    raw = pd.DataFrame(
        [{"行业": "旅游酒店", "行业-涨跌幅": 2.1, "流入资金": 12.0, "流出资金": 8.0, "净额": 4.0, "领涨股": "示例"}]
    )
    result = normalize_ths_sector_fund_flow(raw, "20260828")

    assert result.loc[0, "canonical_sector"] == "旅游"
    assert result.loc[0, "net_inflow_amount"] == 400_000_000
    assert result.loc[0, "net_inflow_ratio"] == 20
