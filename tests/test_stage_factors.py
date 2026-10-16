from __future__ import annotations

from factors.stage import calculate_stage_factors


def test_stage_factors_have_expected_columns(sample_prices):
    factors = calculate_stage_factors(sample_prices)
    assert {"return_5d", "return_20d", "new_high_20d", "stage_overheated"}.issubset(factors.columns)

