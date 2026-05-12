---
title: "Predicting Price Moves from News Sentiment"
subtitle: "Week 1 Interim Summary · Nova Financial Solutions"
author: "Rediet Shewarega"
date: "10 May 2026"
---

# Interim Summary

## What Was Built

A two-notebook analytical pipeline linking **financial news sentiment** to **stock price movements**, covering Tasks 1 and 2 of the Nova Financial Solutions challenge.

---

## Task 1 — Exploratory Data Analysis

**Dataset:** 800 news articles · 3 tickers (AAPL, MSFT, GOOGL) · Jan 2023 onwards

| Finding | Detail |
|---------|--------|
| Headline length | Mean **138 chars / 19 words** — uniform, wire-service style |
| Top publishers | benzinga.com (17%), reuters.com (16%), thefly.com (16%), marketwatch.com (15%) — top 4 cover **64%** of articles |
| Publication timing | Bimodal: peaks at **07:00–09:00 ET** (pre-market) and **15:00–16:00 ET** (close) |
| Zero data loss | 0% unparseable timestamps after stripping `UTC-4` suffix |

**4 LDA topics identified in headlines:**

1. **Biotech catalyst** — FDA approval, pipeline, stock jumps  
2. **Earnings surprise** — beat estimates, shares rise, downgrade  
3. **Analyst upgrade** — price target raised, bullish  
4. **Income / yield** — dividend increased, yield, income  

> These themes map directly to known market microstructure patterns and will serve as categorical features in Task 3 correlation modelling.

---

## Task 2 — Technical Indicators (TA-Lib)

**Dataset:** AAPL · 320 trading days · Price range $280 – $422

| Indicator | Parameters | Last Reading (25 Mar 2024) |
|-----------|-----------|---------------------------|
| SMA | 20-day, 50-day | $381.77 / $370.66 — price well above both |
| EMA | 20-day | $389.00 — faster trend tracking |
| RSI | 14-day | **67.3** — pulling back from overbought (peaked 75.2) |
| MACD | 12/26/9 | **+13.71** — positive momentum, no reversal signal yet |

**Key signal:** Price is in a strong uptrend with RSI recently overbought and MACD histogram expanding — momentum is intact but a short-term correction is plausible.

---

## Plan for Final Submission (12 May 2026)

| Task | Action |
|------|--------|
| Task 1 (full data) | Swap sample CSVs for full FNSPID slice; annotate publication spikes with FOMC/CPI dates |
| Task 2 (expand) | Add Bollinger Bands and ATR |
| **Task 3 (new)** | VADER sentiment scores → date alignment → Pearson correlation with daily returns → investment strategy recommendations |

---

*GitHub: [rediet-shewarega/predicting-price-moves-news-sentiment-](https://github.com/rediet-shewarega/predicting-price-moves-news-sentiment-) · Branch: `task-1`*
