# News sentiment and price moves — Nova Financial Solutions

Rigorous pipeline: **EDA on financial headlines**, **technical indicators** (TA-Lib + PyNance), and **Pearson correlation** between daily sentiment and returns.

## Repository layout

See the brief’s recommended tree: `data/raw/`, `notebooks/`, `src/`, `tests/`, `scripts/`, `.github/workflows/`.

## Environment

```bash
cd news-sentiment-analysis
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### TA-Lib (system library)

TA-Lib’s Python wheel depends on the C library.

- **macOS**: `brew install ta-lib` then `pip install TA-Lib`
- **Linux**: install `ta-lib` from your distro or build from [ta-lib sources](http://ta-lib.org/)
- **CI**: GitHub Actions workflow compiles TA-Lib on Ubuntu before `pip install`

## Sample data

The full FNSPID export is not bundled. For a reproducible demo, generate small CSVs:

```bash
python scripts/generate_sample_data.py
```

This writes `data/raw/fnspid_sample.csv` and `data/raw/stock_prices_sample.csv`. Replace them with your real news and price files; notebooks assume the column names described in the assignment.

## Branches (assignment workflow)

- `task-1` — EDA notebook + CI
- `task-2` — merge Task 1 via PR, then technical indicators notebook
- `task-3` — sentiment + correlation notebook

```bash
git checkout -b task-1
# … commit work …
git push -u origin task-1
```

Use [Conventional Commits](https://www.conventionalcommits.org/) for messages (e.g. `feat(ed): add publisher volume plot`).

## Tests & CI

```bash
pytest tests/ -v
```

`.github/workflows/unittests.yml` runs the same on push/PR to `main` and task branches.

## Notebooks

| File | Purpose |
|------|---------|
| `notebooks/01_task1_eda.ipynb` | Task 1 — EDA |
| `notebooks/02_task2_technical_indicators.ipynb` | Task 2 — indicators |
| `notebooks/03_task3_sentiment_correlation.ipynb` | Task 3 — sentiment & correlation |

## PyNance note

The course references [mqandil/pynance](https://github.com/mqandil/pynance). This repo pins the PyPI package `pynance` for one-command installs; if your facilitator requires the GitHub fork, install with:

`pip install git+https://github.com/mqandil/pynance.git`

and adjust imports per that project’s docs.

## Interim submission checklist (Sunday 10 May 2026, 8:00 PM UTC)

- [ ] GitHub repo public (or shared with facilitators) on branch **`task-1`**
- [ ] `01_task1_eda.ipynb` run end-to-end on your **real** FNSPID slice (replace sample CSVs)
- [ ] `02_task2_technical_indicators.ipynb` with **at least one** indicator on real prices (SMA/RSI/MACD)
- [ ] Interim report (≤3 pages): loading/cleaning, EDA highlights with figures, indicator preview, challenges & plan

## License

Educational use — Nova Financial Solutions challenge.
