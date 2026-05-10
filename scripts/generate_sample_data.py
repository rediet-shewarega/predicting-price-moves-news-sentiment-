#!/usr/bin/env python3
"""Generate small CSVs matching FNSPID + OHLCV schemas for local / CI demos."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

PUBLISHERS = [
    "news@reuters.com",
    "tips@benzinga.com",
    "editor@marketwatch.com",
    "alerts@thefly.com",
    "Reuters",
    "Bloomberg",
]
HEADLINE_PARTS = [
    ("earnings", "beat", "estimates", "shares rise"),
    ("price target", "raised", "analyst", "bullish"),
    ("FDA", "approval", "pipeline", "stock jumps"),
    ("downgrade", "concerns", "margin", "pressure"),
    ("merger", "talks", "speculation", "volatile"),
    ("dividend", "increased", "yield", "income"),
    ("SEC", "filing", "insider", "buying"),
    ("revenue", "growth", "cloud", "strong"),
]


def _random_headline(rng: np.random.Generator) -> str:
    n = int(rng.integers(3, 6))
    idx = rng.choice(len(HEADLINE_PARTS), size=n, replace=False)
    tokens: list[str] = []
    for i in idx:
        tokens.extend(HEADLINE_PARTS[int(i)])
    return " ".join(tokens).title() + " — Update"


def build_news(n: int, rng: np.random.Generator, symbols: list[str]) -> pd.DataFrame:
    rows = []
    start = datetime(2023, 1, 3, 14, 0, 0)
    for i in range(n):
        sym = rng.choice(symbols)
        # Bias times toward US market hours (UTC-4 in spec; store as naive ET-like)
        hour = int(rng.choice([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21]))
        minute = int(rng.integers(0, 60))
        ts = start + timedelta(days=int(i // 8), hours=hour, minutes=minute)
        pub = str(rng.choice(PUBLISHERS))
        h = _random_headline(rng)
        rows.append(
            {
                "headline": h,
                "url": f"https://example.com/article/{i}",
                "publisher": pub,
                "date": ts.strftime("%Y-%m-%d %H:%M:%S UTC-4"),
                "stock": sym,
            }
        )
    return pd.DataFrame(rows)


def build_prices(symbols: list[str], days: int, rng: np.random.Generator) -> pd.DataFrame:
    all_rows = []
    start = datetime(2023, 1, 3).date()
    d0 = start
    for sym in symbols:
        price = float(rng.uniform(80, 350))
        d = d0
        for _ in range(days):
            # Skip weekends
            while d.weekday() >= 5:
                d += timedelta(days=1)
            ret = rng.normal(0, 0.015)
            o = price
            c = price * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.005)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.005)))
            vol = int(rng.integers(1_000_000, 50_000_000))
            all_rows.append(
                {
                    "Date": d.isoformat(),
                    "stock": sym,
                    "Open": round(o, 4),
                    "High": round(h, 4),
                    "Low": round(l, 4),
                    "Close": round(c, 4),
                    "Adj Close": round(c, 4),
                    "Volume": vol,
                }
            )
            price = c
            d += timedelta(days=1)
    df = pd.DataFrame(all_rows)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--news-rows", type=int, default=800)
    p.add_argument("--price-days", type=int, default=320)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    RAW.mkdir(parents=True, exist_ok=True)
    news = build_news(args.news_rows, rng, symbols)
    news.to_csv(RAW / "fnspid_sample.csv", index=False)
    prices = build_prices(symbols, args.price_days, rng)
    prices.to_csv(RAW / "stock_prices_sample.csv", index=False)
    print(f"Wrote {RAW / 'fnspid_sample.csv'} and {RAW / 'stock_prices_sample.csv'}")


if __name__ == "__main__":
    main()
