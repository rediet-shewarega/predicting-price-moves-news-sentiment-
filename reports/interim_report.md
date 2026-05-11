# Interim Report — Predicting Price Moves from News Sentiment
**Nova Financial Solutions · Week 1 Interim Submission**  
**Author:** Rediet Shewarega · **Date:** 10 May 2026

---

## 1. Executive Summary

This interim report documents progress on Tasks 1 and 2 of the Nova Financial Solutions challenge. Task 1 delivered a complete exploratory analysis of an 800-article financial news dataset, uncovering publisher concentration patterns, intraday publication timing, and four distinct narrative themes in the headlines. Task 2 produced a quantitative analysis of AAPL historical price data using TA-Lib, computing SMA, EMA, RSI, and MACD indicators and visualising their relationship to price action. The analytical pipeline is functional end-to-end and ready for the full FNSPID dataset.

---

## 2. Data Loading and Cleaning

### 2.1 News Dataset (FNSPID-style)

The dataset (`fnspid_sample.csv`) was loaded into a pandas DataFrame with five fields: `headline`, `url`, `publisher`, `date`, and `stock`. The dataset contains **800 articles** covering three ticker symbols (AAPL, MSFT, GOOGL) across a date range beginning January 2023.

**Cleaning steps applied:**
- Timestamp parsing: the `date` field contains a `UTC-4` timezone suffix in plain text (e.g. `2023-01-04 06:39:00 UTC-4`). The suffix was stripped with a regex before passing to `pd.to_datetime`. Result: **0% unparseable timestamps** — no rows were lost.
- Derived columns added: `headline_len` (character count) and `headline_words` (whitespace-split token count) to support descriptive statistics.
- Publisher normalisation: a regex extractor pulled domain names from email-format publishers (e.g. `editor@marketwatch.com` → `marketwatch.com`).

### 2.2 Stock Price Dataset (OHLCV)

The price dataset (`stock_prices_sample.csv`) was loaded with columns `Date`, `stock`, `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`. AAPL was isolated for Task 2 analysis, yielding **320 trading-day rows**.

**Cleaning steps applied:**
- All numeric columns cast to `float64` via `pd.to_numeric(errors='coerce')`.
- Rows missing `Adj Close` or `Close` were dropped — **0 rows removed**.
- Sanity assertion confirmed no negative prices.
- `Date` parsed and set as a `DatetimeIndex`; rows sorted chronologically.

---

## 3. Task 1: EDA Findings

### 3.1 Headline Length Distribution

| Statistic | Characters | Words |
|-----------|-----------|-------|
| Mean      | 138.5     | 19.4  |
| Std       | 26.9      | 3.6   |
| Min       | 94        | 14    |
| Median    | 139       | 19    |
| Max       | 183       | 25    |

Headlines are moderate-length and fairly uniform (std ≈ 27 chars). This suggests a templated wire-service style, with limited vocabulary variation — a useful prior for choosing a lightweight tokeniser (TF-IDF bigrams are sufficient; transformer tokenisers would add overhead without clear gain at this scale).

### 3.2 Publisher Activity

The four active publishing domains and their article counts are:

| Domain | Articles |
|--------|---------|
| benzinga.com | 137 (17.1%) |
| reuters.com | 131 (16.4%) |
| thefly.com | 130 (16.3%) |
| marketwatch.com | 117 (14.6%) |

The top four sources account for **64% of all articles**. This concentration creates source bias risk: if one publisher systematically reports with more positive or negative framing, sentiment scores will inherit that bias. Planned mitigation: publisher fixed-effects controls in the correlation model (Task 3).

### 3.3 Publication Calendar and Intraday Timing

Monthly article volume is relatively stable across the sample period, with no pronounced seasonal spike visible in synthetic data. On the **full FNSPID dataset**, this chart will serve as a macro-event detector — peaks aligned with FOMC announcements, CPI releases, or mega-cap earnings clusters signal newsworthy periods that warrant closer correlation analysis.

The intraday hour distribution is bimodal, with peaks concentrated in early US pre-market hours (07:00–09:00 ET) and again at market close (15:00–16:00 ET). This timing pattern informs **alignment strategy in Task 3**: articles published after 16:00 ET should be mapped to the *next* trading day's returns, not the same-day close.

### 3.4 Keywords and Topic Analysis

**TF-IDF top terms** (mean score across corpus): `update`, `margin pressure`, `downgrade`, `dividend increased`, `merger talks`, `speculation volatile`, `analyst bullish`, `price target raised`.

**LDA (4 topics)** surfaced four coherent narrative clusters:

| Topic | Core Terms | Financial Interpretation |
|-------|-----------|--------------------------|
| 1 | FDA, approval, pipeline, stock jumps | Biotech/pharma catalyst events |
| 2 | earnings beat, estimates, shares rise, downgrade | Earnings surprise / analyst reaction |
| 3 | price target raised, analyst, bullish | Analyst upgrade cycle |
| 4 | dividend increased, yield, income | Income / yield-seeking narrative |

These topics align with well-known market microstructure patterns: earnings surprises and analyst upgrades are associated with sharp short-term price reactions, while dividend narratives drive slower mean-reversion moves. Topic assignment will serve as a categorical feature in downstream sentiment–return modelling.

---

## 4. Task 2: Technical Indicators

Analysis was performed on **AAPL** over **320 trading days** (Jan 2023 – Mar 2024). All indicators were computed using TA-Lib 0.6.8 on Adjusted Close prices.

### 4.1 Price Statistics

| Metric | Value |
|--------|-------|
| Mean Adj Close | $335.44 |
| Std | $26.93 |
| Min | $279.96 |
| Max | $422.51 |
| Observations | 320 |

### 4.2 Indicators Computed

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| SMA | 20-day, 50-day | Trend identification; crossover signals |
| EMA | 20-day | Faster-reacting trend line, weighted to recent prices |
| RSI | 14-day | Overbought (>70) / oversold (<30) momentum oscillator |
| MACD | 12/26/9 | Momentum shifts; signal-line crossovers for entries/exits |

### 4.3 Indicator Readings (Final 5 Sessions)

As of 25 March 2024, the final period of the sample:

| Date | Adj Close | SMA20 | EMA20 | RSI14 | MACD |
|------|-----------|-------|-------|-------|------|
| 19 Mar | $405.98 | $371.34 | $375.31 | 70.3 | 7.90 |
| 20 Mar | $421.43 | $373.70 | $379.70 | 74.9 | 10.58 |
| 21 Mar | $422.51 | $376.00 | $383.78 | 75.2 | 12.65 |
| 22 Mar | $412.77 | $378.69 | $386.54 | 67.6 | 13.34 |
| 25 Mar | $412.42 | $381.77 | $389.00 | 67.3 | 13.71 |

**Key observations:**
- Price is trading **well above both SMAs** — a strongly trending bull market in this sample window.
- **RSI peaked at 75.2** on 21 March, signalling overbought conditions. The pullback on 22–25 March (–2.3%, –0.08%) is consistent with a mild correction from overbought levels.
- **MACD histogram is positive and growing** through the period, indicating sustained bullish momentum. No signal-line crossover (bearish reversal signal) has occurred.

---

## 5. Challenges Encountered

| Challenge | Resolution |
|-----------|-----------|
| `pynance` fails to import on Python 3.11 due to a `pandas_datareader` API break (`deprecate_kwarg` signature mismatch) | Wrapped in `try/except`; falling back to `pandas.pct_change()`, which computes an identical simple return | 
| TA-Lib C library not pre-installed | Installed via `brew install ta-lib` then `pip install TA-Lib` |
| FNSPID `date` column contains plain-text timezone suffix `UTC-4` rather than ISO format | Stripped suffix with `str.replace(r'\s*UTC-4\s*$', '', regex=True)` before parsing |
| Git push denied (credential mismatch) | Commits staged locally; re-authentication with `gh auth refresh --scopes repo` required before pushing |

---

## 6. Plan for Final Submission (12 May 2026)

| Step | Notebook | Target |
|------|----------|--------|
| Replace sample CSVs with full FNSPID slice | 01 + 03 | Real headline vocabulary and volume |
| Re-run EDA on full dataset; annotate publication spikes with known macro events (FOMC, CPI, earnings) | 01 | Richer time-series findings |
| Add Bollinger Bands and ATR to Task 2 | 02 | Broader indicator coverage |
| Implement Task 3: VADER sentiment scoring, date alignment, daily return calculation, Pearson correlation | 03 | Core deliverable |
| Write investment strategy recommendations based on sentiment–return correlation | Report | Final blog-post |

The sentiment–return correlation pipeline (Task 3) is the analytical centrepiece. The date-alignment logic is already outlined: weekend/holiday articles will be forwarded to the next trading day open. VADER will be the primary scorer (robust to financial jargon, no training required); TextBlob scores will be computed as a secondary cross-check.

---

*Repository: [github.com/rediet-shewarega/predicting-price-moves-news-sentiment-](https://github.com/rediet-shewarega/predicting-price-moves-news-sentiment-) · Branch: `task-1`*
