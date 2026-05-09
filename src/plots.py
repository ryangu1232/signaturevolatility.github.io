from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

_FREQ_ORDER = ["1m", "5m", "10m", "1h", "1d"]

_MODEL_LABELS = {
    "y_pred_sig_har": "Sig-HAR",
    "y_pred_har": "HAR-RV",
    "y_pred_garch": "RV-GARCH(1,1)",
}


def plot_lead_lag_path(ll_path: np.ndarray, outpath: Path, freq: str):
    """2D phase-space plot of a lead-lag transformed path, coloured by time.

    Each original step i→i+1 produces two axis-aligned segments:
      (x_{i-1}, x_{i-1}) → (x_i, x_{i-1})   [horizontal]
      (x_i, x_{i-1})     → (x_i, x_i)        [vertical]
    Colouring by segment index makes the L-shaped structure and
    the time ordering simultaneously visible.
    """
    ll_path = np.asarray(ll_path, dtype=float)
    x1, x2 = ll_path[:, 0], ll_path[:, 1]
    n_segs = len(ll_path) - 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: 2D phase-space path coloured by time ---
    ax = axes[0]
    colors = cm.viridis(np.linspace(0, 1, n_segs))
    for i in range(n_segs):
        ax.plot(x1[i:i+2], x2[i:i+2], color=colors[i], linewidth=0.8)
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, n_segs))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Path step")
    mn = min(x1.min(), x2.min())
    mx = max(x1.max(), x2.max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.6, alpha=0.4, label="diagonal")
    ax.set_xlabel("$X^1$ (lead)")
    ax.set_ylabel("$X^2$ (lag)")
    ax.set_title(f"Lead-lag path — {freq} (phase space)")
    ax.legend(fontsize=8)

    # --- right: X1 and X2 vs path index ---
    ax2 = axes[1]
    idx = np.arange(len(ll_path))
    ax2.plot(idx, x1, color="steelblue", linewidth=0.8, label="$X^1$ (lead)")
    ax2.plot(idx, x2, color="tomato", linewidth=0.8, label="$X^2$ (lag)")
    ax2.set_xlabel("Path step")
    ax2.set_ylabel("Value")
    ax2.set_title(f"Lead-lag components vs time — {freq}")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_combined_scatter(preds_df: pd.DataFrame, outpath: Path, freq: str):
    """One figure with one scatter subplot per model, all on the same scale."""
    model_cols = [c for c in preds_df.columns if c.startswith("y_pred_")]
    if not model_cols or preds_df.empty:
        return

    n = len(model_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    for ax, col in zip(axes[0], model_cols):
        label = _MODEL_LABELS.get(col, col.replace("y_pred_", "").replace("_", " "))
        ax.scatter(preds_df["y_true"], preds_df[col], alpha=0.4, s=10)
        mn = min(preds_df["y_true"].min(), preds_df[col].min())
        mx = max(preds_df["y_true"].max(), preds_df[col].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        ax.set_xlabel("True RV")
        ax.set_ylabel("Predicted RV")
        ax.set_title(f"{label} — {freq}")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_all_metrics_comparison(metrics_df: pd.DataFrame, outdir: Path):
    """One figure with subplots for RMSE, MAE, QLIKE across frequencies and models."""
    metrics = ["RMSE", "MAE", "QLIKE"]
    present = [m for m in metrics if m in metrics_df.columns]
    if not present:
        return

    ncols = 2
    nrows = (len(present) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows), squeeze=False)

    for i, metric in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        try:
            plot_df = metrics_df.pivot(index="frequency", columns="model", values=metric)
            ordered = [f for f in _FREQ_ORDER if f in plot_df.index]
            plot_df = plot_df.reindex(ordered)
        except Exception:
            continue
        for col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col], marker="o", label=col)
        ax.set_xlabel("Frequency")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=8)

    for i in range(len(present), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "metrics_comparison.png", dpi=150)
    plt.close(fig)
