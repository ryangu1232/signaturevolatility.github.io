# Volatility Forecasting: Signature-HAR vs HAR-RV

A research pipeline comparing path-signature-based volatility models against classical HAR-RV benchmarks. Data is fetched live from Yahoo Finance via `yfinance`.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Reference](#cli-reference)
4. [Tickers](#tickers)
5. [Frequencies](#frequencies)
6. [Models](#models)
7. [Output Files](#output-files)
8. [Data Limitations](#data-limitations)
9. [Project Structure](#project-structure)

---

## Installation

**Core dependencies** (required):

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `yfinance`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`

**Optional — exact path signatures** (recommended for research runs):

```bash
pip install iisignature
```

Without `iisignature`, the pipeline falls back to `--signature-mode approximate`, which uses a hand-computed truncated signature (level-1 + level-2 iterated integrals). With `iisignature`, `--signature-mode exact` computes the full piecewise-linear Stratonovich signature to any level.

---

## Quick Start

Run the default configuration (SPY, all 5 frequencies, approximate signatures, level 2):

```bash
python -m src.main
```

Run with exact signatures at truncation level 3:

```bash
python -m src.main --signature-mode exact --sig-level 3
```

Run on multiple tickers at once:

```bash
python -m src.main --tickers SPY TSLA NVDA --signature-mode exact
```

Run on daily frequency only (fastest, most data history):

```bash
python -m src.main --freqs 1d
```

---

## CLI Reference

```
python -m src.main [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--tickers` | `SPY` | One or more ticker symbols. Accepts preset names (see below) or any valid yfinance symbol. |
| `--freqs` | `1m 5m 10m 1h 1d` | Frequencies to run. Any subset of: `1m 5m 10m 1h 1d`. |
| `--outdir` | `output` | Root output directory. Per-ticker subdirectories are created automatically. |
| `--n-splits` | `5` | Number of `TimeSeriesSplit` folds for out-of-sample evaluation. |
| `--signature-mode` | `approximate` | `approximate` uses a fast hand-computed signature. `exact` requires `iisignature` and computes the full Stratonovich signature. |
| `--sig-level` | `2` | Truncation level for the path signature (only meaningful with `--signature-mode exact`). Higher levels capture more path geometry but increase feature dimensionality. |

### Examples

```bash
# Exact signatures, level 2, daily only, save to custom folder
python -m src.main --signature-mode exact --sig-level 2 --freqs 1d --outdir results/daily

# Cross-ticker comparison: low vs high volatility
python -m src.main --tickers spy gme tsla --freqs 1d 1h --signature-mode exact

# Intraday only with 10 OOS folds
python -m src.main --freqs 1m 5m 10m --n-splits 10

# Single ticker, all frequencies, exact signatures level 3
python -m src.main --tickers nvda --signature-mode exact --sig-level 3
```

---

## Tickers

You can pass preset shorthand names or any raw yfinance symbol:

| Preset | Symbol | Profile |
|---|---|---|
| `spy` | `SPY` | S&P 500 ETF — low volatility |
| `qqq` | `QQQ` | Nasdaq 100 ETF — medium volatility |
| `gld` | `GLD` | Gold ETF — low volatility |
| `xom` | `XOM` | Energy — medium volatility |
| `meta` | `META` | Post-2022 tech reset |
| `tsla` | `TSLA` | High volatility |
| `nvda` | `NVDA` | High volatility |
| `gme` | `GME` | Extreme volatility / frequent halts |
| `gspc` | `^GSPC` | S&P 500 index (long daily history) |
| `btc` | `BTC-USD` | Bitcoin — 24/7, no market gaps |

Any other valid yfinance symbol (e.g. `AAPL`, `ETH-USD`, `^VIX`) can be passed directly.

---

## Frequencies

Each frequency has a fixed data history window and frequency-specific lookback windows for the signature models:

| Freq | yfinance interval | History | Sig windows (short / med / long) | Horizon |
|---|---|---|---|---|
| `1m` | 1 min | 7 days | 30 / 78 / 195 bars | 5 bars ahead |
| `5m` | 5 min | 60 days | 6 / 16 / 78 bars | 5 bars ahead |
| `10m` | 5 min resampled | 60 days | 3 / 8 / 39 bars | 5 bars ahead |
| `1h` | 60 min | 730 days | 2 / 7 / 35 bars | 5 bars ahead |
| `1d` | 1 day | 10 years | 5 / 22 / 60 bars | 5 bars ahead |

The windows are chosen to correspond to natural market timescales at each frequency (e.g., at 1d: 1 week / 1 month / 3 months — the classic HAR scales).

---

## Models

Five models are run for each ticker × frequency combination:

### `har_rv_linear`
Classical HAR-RV benchmark. Features are `[RV_1, RV_5, RV_22]` where `RV_w = sqrt(mean(r²))` over the last `w` bars. Fitted by OLS with no regularization. Serves as the primary baseline.

### `sig_har_lasso_krr_{mode}`
Main signature model. At each time step, computes path signature features at all three lookback windows (short, medium, long), concatenates them into one feature vector, then fits a two-stage model:
1. **LassoCV** — selects the sparse subset of signature features most predictive of future RV
2. **Kernel Ridge Regression (RBF)** — fits a non-linear model on the selected features, with `(alpha, gamma)` tuned by inner time-series cross-validation

This is the HAR multi-timescale idea applied in signature feature space.

### `sig_har_lasso_krr_wf_{mode}` (walk-forward)
Identical pipeline to `sig_har_lasso_krr` but refitted every `max(sig_windows)` new observations instead of 5 times total. Hyperparameters adapt continuously as the volatility regime evolves.

### `sig_lasso_krr_{mode}`
Ablation of `sig_har_lasso_krr`: uses only the single longest lookback window (no multi-scale concatenation). Isolates the contribution of the HAR structure by removing it.

### `range_har_linear`
HAR variant using the Parkinson (1980) high-low range estimator instead of squared close-to-close returns:
```
park_var_t = (ln(H_t / L_t))² / (4 ln 2)
```
The Parkinson estimator is ~5× more statistically efficient than squared returns under Brownian motion, since it uses the full intraday price range. Features are `[sqrt(mean(park_var))]` at each HAR window.

For a detailed technical description of each model, see [src/documents/models_overview.md](src/documents/models_overview.md).

---

## Output Files

All outputs are written to `{outdir}/{TICKER}-{sig_level}/` for each ticker. A combined summary is also written to `{outdir}/`.

### Per-ticker outputs

| File | Description |
|---|---|
| `data_used_{freq}.csv` | Raw OHLC + log returns used for that frequency |
| `metrics_summary.csv` | All model metrics for this ticker across all frequencies |
| `volatility_summary.csv` | Realized volatility profile: mean, std, median RV, annualized vol % |
| `predictions_{freq}.csv` | Aligned OOS predictions from sig-HAR and HAR-RV for scatter plots |
| `scatter_{freq}.png` | Scatter plot: true RV vs predicted RV for both main models |
| `lead_lag_path_{freq}.png` | Lead-lag phase space path from the last valid window |
| `bar_comparison_*.png` | Bar charts comparing model metrics per frequency |

### Combined outputs (multi-ticker runs)

| File | Description |
|---|---|
| `metrics_summary.csv` | All metrics across all tickers and frequencies |
| `cross_ticker_{model}.png` | Line chart of each metric vs frequency, one line per ticker |

### Metrics reported

| Metric | Description |
|---|---|
| `MSE` | Mean squared error |
| `RMSE` | Root mean squared error |
| `MAE` | Mean absolute error |
| `R2` | Coefficient of determination |
| `QLIKE` | Quasi-likelihood loss: `mean(log(ŷ) + y/ŷ)` — standard in volatility forecasting |

---

## Data Limitations

This pipeline uses Yahoo Finance data via `yfinance`, which imposes hard history limits on intraday data:

| Frequency | Max history available |
|---|---|
| `1m` | ~7 days |
| `5m`, `10m` | ~60 days |
| `1h` | ~730 days (2 years) |
| `1d` | Up to 10 years (or full listing history) |

**Practical implications**:
- Intraday results (`1m`, `5m`, `10m`) are prototype-grade — small sample sizes mean high variance in OOS metrics.
- Daily results are the most reliable for drawing conclusions.
- For publication-grade intraday analysis, a commercial data source (e.g. Polygon, Refinitiv, Bloomberg) is needed.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── config.py          # FreqConfig dataclass, FREQ_CONFIGS, ticker presets
│   ├── data.py            # yfinance download + OHLC resampling
│   ├── features.py        # Lead-lag transform, signature features, dataset builders
│   ├── models.py          # All model training/evaluation wrappers
│   ├── evaluate.py        # MSE, RMSE, MAE, R², QLIKE
│   ├── plots.py           # All figure generation
│   ├── main.py            # CLI entry point, orchestrates the full pipeline
│   └── documents/
│       └── models_overview.md   # Detailed technical description of all models
├── visuals/
│   ├── lead_lag_demo.py   # Standalone script generating research paper figures
│   └── figures/           # Output figures from lead_lag_demo.py
└── output/                # Generated at runtime
    └── {TICKER}-{sig_level}/
        ├── metrics_summary.csv
        ├── volatility_summary.csv
        └── ...
```
