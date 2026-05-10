"""Align news publication timestamps to equity trading sessions."""

from __future__ import annotations

import pandas as pd


def align_news_to_next_trading_day(
    news_dates: pd.Series,
    trading_days: pd.DatetimeIndex,
) -> pd.Series:
    """
    Map each calendar datetime to the trading day used for return matching.

    Articles on weekends or market holidays roll forward to the next day
    that exists in ``trading_days`` (typically the next open session).
    """
    td = trading_days.normalize().sort_values()
    out = []
    for ts in pd.to_datetime(news_dates, utc=False):
        d = ts.normalize()
        if d in td:
            out.append(d)
            continue
        future = td[td > d]
        if len(future):
            out.append(future[0])
        else:
            out.append(pd.NaT)
    return pd.Series(out, index=news_dates.index, dtype="datetime64[ns]")
