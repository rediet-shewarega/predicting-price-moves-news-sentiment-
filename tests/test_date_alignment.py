"""Tests for trading-day alignment helpers."""

from __future__ import annotations

import pandas as pd

from src.date_alignment import align_news_to_next_trading_day


def test_weekend_rolls_forward_to_next_session():
    trading = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    news = pd.Series(pd.to_datetime(["2024-01-06 18:00:00"]))  # Saturday
    aligned = align_news_to_next_trading_day(news, pd.DatetimeIndex(trading))
    assert aligned.iloc[0] == pd.Timestamp("2024-01-08").normalize()


def test_holiday_gap_uses_next_available():
    trading = pd.to_datetime(["2024-01-02", "2024-01-05"])
    news = pd.Series(pd.to_datetime(["2024-01-03 10:00:00"]))
    aligned = align_news_to_next_trading_day(news, pd.DatetimeIndex(trading))
    assert aligned.iloc[0] == pd.Timestamp("2024-01-05").normalize()
