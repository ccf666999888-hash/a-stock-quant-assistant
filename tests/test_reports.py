from __future__ import annotations

import pandas as pd

from reports.daily_review import build_daily_review


def test_daily_report_handles_empty_screening_results():
    report = build_daily_review(
        trade_date="20260612",
        screening=pd.DataFrame(),
        indices=pd.DataFrame(),
    )
    assert "暂无符合规则的样本" in report
    assert "不构成投资建议" in report

