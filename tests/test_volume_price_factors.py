from __future__ import annotations

from factors.volume_price import calculate_volume_price_factors


def test_close_position_is_bounded(sample_prices):
    factors = calculate_volume_price_factors(sample_prices)
    assert factors["close_position"].between(0, 1).all()


def test_equal_high_low_uses_neutral_close_position(sample_prices):
    frame = sample_prices.head(1).copy()
    frame["high"] = frame["low"] = frame["close"]
    factors = calculate_volume_price_factors(frame)
    assert factors["close_position"].iloc[0] == 0.5


def test_missing_market_benchmark_does_not_award_relative_strength(sample_prices):
    factors = calculate_volume_price_factors(sample_prices)
    assert not factors["stronger_than_market"].any()
