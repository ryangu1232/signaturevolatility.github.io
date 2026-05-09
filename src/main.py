from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_HAR_WINDOWS,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    FREQ_CONFIGS,
    TICKER_PRESETS,
)
from .data import load_frequency_data
from .evaluate import evaluate_predictions
from .features import (
    build_garch_dataset,
    build_har_dataset,
    build_signature_dataset,
    lead_lag_transform,
    prepare_returns,
)
from .models import (
    rolling_oos_predictions_garch,
    rolling_oos_predictions_lasso_krr_walkforward,
    rolling_oos_predictions_linear,
    rolling_oos_predictions_xgboost,
)
from .plots import plot_all_metrics_comparison, plot_combined_scatter, plot_lead_lag_path


def _strip_tz(ts):
    """Normalize timestamps to tz-naive for consistent merging across frequencies."""
    s = pd.to_datetime(ts)
    if hasattr(s, "dt"):
        return s.dt.tz_localize(None).to_numpy()
    try:
        return s.tz_localize(None).to_numpy()
    except Exception:
        return s.to_numpy()


def _align_predictions(raw_preds: dict) -> pd.DataFrame:
    """Inner-join all model predictions on timestamp.

    raw_preds: {model_name: (timestamps, y_true, y_pred)}

    Returns a DataFrame with columns [timestamp, y_true, <model1>, <model2>, ...]
    containing only rows whose timestamp appears in every model's predictions.
    y_true is taken from the first model (all models share the same target series).
    """
    first_name = next(iter(raw_preds))
    ts0, yt0, yp0 = raw_preds[first_name]
    master = pd.DataFrame({
        "timestamp": pd.to_datetime(np.asarray(ts0)),
        "y_true": yt0,
        first_name: yp0,
    })
    for name, (ts, _, yp) in raw_preds.items():
        if name == first_name:
            continue
        other = pd.DataFrame({
            "timestamp": pd.to_datetime(np.asarray(ts)),
            name: yp,
        })
        master = master.merge(other, on="timestamp", how="inner")
    return master.sort_values("timestamp").reset_index(drop=True)


def run_frequency(
    freq: str,
    ticker: str,
    outdir: Path,
    har_windows,
    n_splits: int,
    signature_mode: str,
    sig_level: int,
):
    raw = load_frequency_data(freq, ticker)
    df = prepare_returns(raw)
    df.to_csv(outdir / f"data_used_{freq}.csv", index=False)

    cfg = FREQ_CONFIGS[freq]
    horizon_bars = cfg.horizon_bars
    sig_windows = cfg.sig_windows

    # raw_preds collects {model_name: (timestamps, y_true, y_pred)} for alignment.
    # lookback_meta stores per-model metadata that isn't derived from predictions.
    raw_preds = {}
    lookback_meta = {}

    # ------------------------------------------------------------------ #
    # Phase 1: run all models, collect raw predictions with timestamps.   #
    # Metrics are computed after aligning all models on common timestamps #
    # so that every model is evaluated on the exact same observations.    #
    # ------------------------------------------------------------------ #

    # sig_har_lasso_krr variants (multi-scale) are omitted — empirically identical
    # to single-scale at sig_level=2, and the walk-forward version has catastrophic
    # QLIKE at 1h and 1d. Re-enable by restoring build_multiscale_signature_dataset
    # and the model calls from git history if needed at higher sig_level.

    meta_sig_single, X_sig_single, y_sig_single = build_signature_dataset(
        df=df,
        lookback_bars=max(sig_windows),
        horizon_bars=horizon_bars,
        signature_mode=signature_mode,
        sig_level=sig_level,
    )
    if len(y_sig_single) >= 200:
        y_true_single_wf, y_pred_single_wf, idx_single_wf = rolling_oos_predictions_lasso_krr_walkforward(
            X_sig_single, y_sig_single,
            min_train_size=200,
            refit_every=max(sig_windows),
            window_size=5 * max(sig_windows),
            random_state=DEFAULT_RANDOM_STATE,
        )
        raw_preds["SIG"] = (
            _strip_tz(meta_sig_single.iloc[idx_single_wf]["timestamp"]),
            y_true_single_wf, y_pred_single_wf,
        )
        lookback_meta["SIG"] = int(max(sig_windows))

    meta_har, X_har, y_har = build_har_dataset(
        df=df,
        horizon_bars=horizon_bars,
        windows=har_windows,
    )
    if len(y_har) < 200:
        raise ValueError(f"Not enough samples for HAR model at {freq}: {len(y_har)}")

    y_true_har, y_pred_har, idx_har = rolling_oos_predictions_linear(
        X_har, y_har, n_splits=n_splits
    )
    raw_preds["HAR_RV"] = (
        _strip_tz(meta_har.iloc[idx_har]["timestamp"]), y_true_har, y_pred_har
    )
    lookback_meta["HAR_RV"] = None

    # --- XGBoost on HAR features (same inputs as har_rv_linear, nonlinear model) ---
    try:
        y_true_xgb, y_pred_xgb, idx_xgb = rolling_oos_predictions_xgboost(
            X_har, y_har, n_splits=n_splits, random_state=DEFAULT_RANDOM_STATE
        )
        raw_preds["XGBOOST"] = (
            _strip_tz(meta_har.iloc[idx_xgb]["timestamp"]),
            y_true_xgb, y_pred_xgb,
        )
        lookback_meta["XGBOOST"] = None
    except ImportError:
        print(f"  [{freq}] XGBoost skipped — install with: pip install xgboost")

    # --- GARCH-X(1,1) benchmark ---
    try:
        meta_garch, returns_garch, y_garch = build_garch_dataset(df, horizon_bars)
        if len(y_garch) >= 200:
            y_true_garch, y_pred_garch, idx_garch = rolling_oos_predictions_garch(
                returns_garch, y_garch, n_splits=n_splits
            )
            raw_preds["GARCH"] = (
                _strip_tz(meta_garch.iloc[idx_garch]["timestamp"]),
                y_true_garch, y_pred_garch,
            )
            lookback_meta["GARCH"] = None
    except Exception as e:
        print(f"  Warning: GARCH failed for {freq}: {e}")

    # ------------------------------------------------------------------ #
    # Align all models on common timestamps, then compute metrics.        #
    # Every model is now evaluated on the identical set of observations.  #
    # ------------------------------------------------------------------ #
    aligned = _align_predictions(raw_preds)
    n_common = len(aligned)

    results = []
    for model_name in raw_preds:
        metrics = evaluate_predictions(aligned["y_true"].values, aligned[model_name].values)
        metrics.update({
            "ticker": ticker,
            "frequency": freq,
            "model": model_name,
            "n_samples": n_common,
            "lookback_bars": lookback_meta[model_name],
            "target_horizon_bars": int(horizon_bars),
        })
        results.append(metrics)

    # --- Realized-volatility statistics (full dataset, not just eval period) ---
    rv_mean = float(np.mean(y_sig_single))
    rv_std = float(np.std(y_sig_single))
    rv_median = float(np.median(y_sig_single))
    ann_vol_pct = rv_mean * float(np.sqrt(cfg.annual_bars / cfg.horizon_bars))
    vol_stats = {
        "ticker": ticker,
        "frequency": freq,
        "n_bars_total": len(df),
        "n_rv_samples": int(len(y_sig_single)),
        "rv_mean": round(rv_mean, 8),
        "rv_std": round(rv_std, 8),
        "rv_median": round(rv_median, 8),
        "ann_vol_pct": round(ann_vol_pct, 4),
    }

    # --- Lead-lag path visualisation (last valid window, largest scale) ---
    try:
        r = df["log_return"].to_numpy()
        max_w = max(sig_windows)
        last_t = len(df) - horizon_bars - 2
        window = r[last_t - max_w + 1 : last_t + 1]
        if np.all(np.isfinite(window)):
            ll = lead_lag_transform(window)
            plot_lead_lag_path(ll, outdir / f"lead_lag_path_{freq}.png", freq)
    except Exception as e:
        print(f"  Warning: could not save lead-lag plot for {freq}: {e}")

    # ------------------------------------------------------------------ #
    # Phase 2: combined predictions CSV and scatter plot (aligned data).  #
    # ------------------------------------------------------------------ #
    try:
        aligned.to_csv(outdir / f"predictions_{freq}.csv", index=False)
    except Exception as e:
        print(f"  Warning: could not save predictions CSV for {freq}: {e}")

    try:
        scatter_df = aligned[["timestamp", "y_true", "SIG", "HAR_RV"]].rename(
            columns={"SIG": "y_pred_sig_har", "HAR_RV": "y_pred_har"}
        )
        plot_combined_scatter(scatter_df, outdir / f"scatter_{freq}.png", freq)
    except Exception as e:
        print(f"  Warning: could not save scatter plot for {freq}: {e}")

    return results, vol_stats


def _safe_ticker_dirname(ticker: str) -> str:
    """Convert a ticker symbol to a filesystem-safe directory name."""
    return ticker.replace("^", "").replace("-", "_").upper()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Volatility forecasting: Sig-HAR vs HAR-RV vs RV-GARCH.\n\n"
            "Preset ticker names you can pass to --tickers:\n"
            + "\n".join(f"  {k:8s} -> {v}" for k, v in TICKER_PRESETS.items())
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY"],
        metavar="TICKER",
        help=(
            "One or more ticker symbols (e.g. SPY TSLA GME ^GSPC BTC-USD). "
            "Each ticker is run independently across all --freqs. "
            "Use preset names or any valid yfinance symbol."
        ),
    )
    parser.add_argument("--freqs", nargs="+", default=["1m", "5m", "10m", "1h", "1d"], choices=list(FREQ_CONFIGS.keys()))
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--signature-mode", type=str, default="approximate", choices=["approximate", "exact"])
    parser.add_argument("--sig-level", type=int, default=2)
    args = parser.parse_args()

    # Resolve preset names to actual symbols
    tickers = [TICKER_PRESETS.get(t.lower(), t) for t in args.tickers]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for ticker in tickers:
        ticker_dir = outdir / f"{_safe_ticker_dirname(ticker)}-{args.sig_level}"
        ticker_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Ticker: {ticker} ===")

        ticker_metrics = []
        ticker_vol_stats = []
        for freq in args.freqs:
            print(f"  [{ticker}] Running {freq}...")
            try:
                results, vol_stats = run_frequency(
                    freq=freq,
                    ticker=ticker,
                    outdir=ticker_dir,
                    har_windows=DEFAULT_HAR_WINDOWS,
                    n_splits=args.n_splits,
                    signature_mode=args.signature_mode,
                    sig_level=args.sig_level,
                )
                ticker_metrics.extend(results)
                all_metrics.extend(results)
                ticker_vol_stats.append(vol_stats)
                print(f"  [{ticker}] Finished {freq}")
            except Exception as e:
                print(f"  [{ticker}] Failed on {freq}: {e}")

        # Per-ticker summary
        if ticker_metrics:
            t_df = pd.DataFrame(ticker_metrics)
            cols = ["ticker", "frequency", "model", "n_samples", "lookback_bars",
                    "target_horizon_bars", "MSE", "RMSE", "MAE", "R2", "QLIKE"]
            t_df = t_df[[c for c in cols if c in t_df.columns]]
            t_df.to_csv(ticker_dir / "metrics_summary.csv", index=False)
            plot_all_metrics_comparison(t_df, ticker_dir)
            print(t_df.to_string(index=False))

        # Per-ticker volatility summary
        if ticker_vol_stats:
            vol_df = pd.DataFrame(ticker_vol_stats)
            vol_df.to_csv(ticker_dir / "volatility_summary.csv", index=False)
            print(f"\n  [{ticker}] Realized-volatility profile (ann_vol_pct uses equity trading calendar):")
            print(vol_df[["frequency", "n_bars_total", "rv_mean", "rv_std", "rv_median", "ann_vol_pct"]].to_string(index=False))

    # Combined summary across all tickers
    if all_metrics:
        all_df = pd.DataFrame(all_metrics)
        cols = ["ticker", "frequency", "model", "n_samples", "lookback_bars",
                "target_horizon_bars", "MSE", "RMSE", "MAE", "R2", "QLIKE"]
        all_df = all_df[[c for c in cols if c in all_df.columns]]
        all_df.to_csv(outdir / "metrics_summary.csv", index=False)
        print(f"\nSaved combined metrics to {outdir / 'metrics_summary.csv'}")

        if len(tickers) > 1:
            _plot_cross_ticker_comparison(all_df, outdir)
    else:
        print("No experiment completed.")


def _plot_cross_ticker_comparison(metrics_df: pd.DataFrame, outdir: Path):
    """For multi-ticker runs: one figure per metric showing model performance across tickers × frequencies."""
    import matplotlib.pyplot as plt

    metrics = ["RMSE", "MAE", "QLIKE"]
    present = [m for m in metrics if m in metrics_df.columns]
    if not present:
        return

    models = metrics_df["model"].unique()

    ncols = 2
    nrows = (len(present) + 1) // 2

    for model in models:
        sub = metrics_df[metrics_df["model"] == model]
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)
        for i, metric in enumerate(present):
            ax = axes[i // ncols][i % ncols]
            try:
                pivot = sub.pivot(index="frequency", columns="ticker", values=metric)
                freq_order = ["1m", "5m", "10m", "1h", "1d"]
                pivot = pivot.reindex([f for f in freq_order if f in pivot.index])
            except Exception:
                continue
            for col in pivot.columns:
                ax.plot(pivot.index, pivot[col], marker="o", label=col)
            ax.set_xlabel("Frequency")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} — {model}")
            ax.legend(fontsize=8)

        for i in range(len(present), nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        fig.tight_layout()
        safe_model = model.replace("/", "_")
        fig.savefig(outdir / f"cross_ticker_{safe_model}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
