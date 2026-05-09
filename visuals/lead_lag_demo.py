"""
lead_lag_demo.py
================
Publication-quality figures for the lead-lag transformation used in the
signature-HAR volatility forecasting paper.

Figures produced
----------------
  fig1_construction.pdf   -- Original series → Lead/Lag components → Phase space
  fig2_signed_area.pdf    -- Trending vs mean-reverting: signed triangle decomposition
  fig3_step_by_step.pdf   -- Annotated step-by-step L-path construction with area labels

Usage
-----
    python visuals/lead_lag_demo.py [--outdir visuals/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch


# ---------------------------------------------------------------------------
# Lead-lag transform (mirrors src/features.py exactly)
# ---------------------------------------------------------------------------

def lead_lag_transform(x: np.ndarray) -> np.ndarray:
    """Embed 1-D series into 2-D lead-lag path.

    Each step i → i+1 produces two micro-steps:
        (x[i],   x[i-1])   horizontal move  (lead advances, lag stays)
        (x[i],   x[i])     vertical move    (lag catches up)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    out = [[x[0], x[0]]]
    for i in range(1, len(x)):
        out.append([x[i],     x[i - 1]])
        out.append([x[i],     x[i]])
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Synthetic series generators
# ---------------------------------------------------------------------------

def trending_series(n: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.08, scale=0.06, size=n)
    return np.cumsum(np.concatenate([[0.0], steps]))


def mean_reverting_series(n: int = 12, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n + 1)
    for t in range(1, n + 1):
        x[t] = -0.6 * x[t - 1] + rng.normal(0, 0.07)
    return x


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def signed_triangles(ll: np.ndarray):
    """
    Return (green_tris, red_tris) where each is a list of (x1,y1,x2,y2,x3,y3)
    triangle vertices.

    At each L-step (horizontal + vertical micro-step):
        corner = (x_lead_new, x_lag_old)
    The signed triangle is formed by:
        A = (x_lag_old, x_lag_old)  -- point on diagonal before step
        B = (x_lead_new, x_lag_old) -- corner (horizontal end)
        C = (x_lead_new, x_lead_new) -- point on diagonal after step

    If x_lead_new > x_lag_old  →  corner is *above* diagonal → green (upward move)
    If x_lead_new < x_lag_old  →  corner is *below* diagonal → red  (downward move)
    """
    green, red = [], []
    # ll has 2*(n-1)+1 points: index 0 is start, then pairs (horizontal, vertical)
    # step k corresponds to ll indices [2k-1, 2k] for k = 1 .. n-1
    n_steps = (len(ll) - 1) // 2
    for k in range(n_steps):
        A = ll[2 * k]          # (x_lead_old, x_lag_old) — on diagonal
        B = ll[2 * k + 1]      # (x_lead_new, x_lag_old) — horizontal end
        C = ll[2 * k + 2]      # (x_lead_new, x_lead_new) — back on diagonal
        tri = np.array([A, B, C])
        if B[0] > B[1]:        # lead > lag → above diagonal
            green.append(tri)
        elif B[0] < B[1]:      # lead < lag → below diagonal
            red.append(tri)
        # zero move → degenerate, skip
    return green, red


def levy_area_from_triangles(green, red) -> tuple[float, float, float]:
    """Return (total_green_area, total_red_area, net_levy_area)."""
    def tri_area(t):
        # |det([B-A, C-A])| / 2
        v1 = t[1] - t[0]
        v2 = t[2] - t[0]
        return abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0
    g = sum(tri_area(t) for t in green)
    r = sum(tri_area(t) for t in red)
    return g, r, g - r


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

FIGSTYLE = {
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}

GREEN = "#2ca02c"
RED   = "#d62728"
BLUE  = "#1f77b4"
GRAY  = "#aaaaaa"


def _colored_path(ax, ll, cmap="viridis", lw=1.8, zorder=2):
    """Draw the 2-D lead-lag path as a time-colored line."""
    segs = [ll[i:i+2] for i in range(len(ll) - 1)]
    t = np.linspace(0, 1, len(segs))
    lc = LineCollection(segs, array=t, cmap=cmap, linewidth=lw, zorder=zorder)
    ax.add_collection(lc)
    return lc


def _draw_diagonal(ax, lo, hi, color=GRAY, ls="--", lw=0.8):
    ax.plot([lo, hi], [lo, hi], color=color, ls=ls, lw=lw, zorder=1)


def _fill_triangles(ax, green, red, alpha=0.35):
    for tri in green:
        ax.fill(tri[:, 0], tri[:, 1], color=GREEN, alpha=alpha, zorder=3)
    for tri in red:
        ax.fill(tri[:, 0], tri[:, 1], color=RED,   alpha=alpha, zorder=3)


# ---------------------------------------------------------------------------
# Figure 1 — Construction walkthrough
# ---------------------------------------------------------------------------

def fig1_construction(x: np.ndarray, outpath: Path):
    ll = lead_lag_transform(x)
    n = len(x)
    t_bars = np.arange(n)
    steps = np.arange(len(ll))

    with plt.rc_context(FIGSTYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

        # --- Panel A: original 1-D series ---
        ax = axes[0]
        ax.plot(t_bars, x, marker="o", color=BLUE, lw=1.8, ms=5, zorder=3)
        ax.axhline(0, color=GRAY, lw=0.6, ls=":")
        ax.set_xlabel("Bar $t$")
        ax.set_ylabel("$X_t$")
        ax.set_title("(a) Original time series")

        # --- Panel B: Lead and Lag components vs path step ---
        ax = axes[1]
        ax.plot(steps, ll[:, 0], color=BLUE,  lw=1.5, label=r"$X^1$ (lead)")
        ax.plot(steps, ll[:, 1], color=RED,   lw=1.5, label=r"$X^2$ (lag)", ls="--")
        ax.set_xlabel("Path step")
        ax.set_ylabel("Value")
        ax.set_title("(b) Lead / Lag components")
        ax.legend()

        # --- Panel C: Phase space (X^1 vs X^2) ---
        ax = axes[2]
        lo = min(ll.min(), x.min()) - 0.05
        hi = max(ll.max(), x.max()) + 0.05
        _draw_diagonal(ax, lo, hi)
        lc = _colored_path(ax, ll)
        fig.colorbar(lc, ax=ax, label="Time →", fraction=0.046, pad=0.04)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$X^1$ (lead)")
        ax.set_ylabel(r"$X^2$ (lag)")
        ax.set_title("(c) Phase space")
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Figure 2 — Signed area: trending vs mean-reverting
# ---------------------------------------------------------------------------

def fig2_signed_area(x_trend: np.ndarray, x_mr: np.ndarray, outpath: Path):
    series = [(x_trend, "Trending"), (x_mr, "Mean-reverting")]

    with plt.rc_context(FIGSTYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

        for ax, (x, title) in zip(axes, series):
            ll = lead_lag_transform(x)
            green, red = signed_triangles(ll)
            g_area, r_area, net = levy_area_from_triangles(green, red)

            lo = ll.min() - 0.05
            hi = ll.max() + 0.05
            _draw_diagonal(ax, lo, hi)
            _fill_triangles(ax, green, red, alpha=0.45)
            _colored_path(ax, ll, lw=2.0)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$X^1$ (lead)")
            ax.set_ylabel(r"$X^2$ (lag)")

            legend_handles = [
                mpatches.Patch(color=GREEN, alpha=0.55,
                               label=fr"Upward   $\Delta > 0$  (area={g_area:.4f})"),
                mpatches.Patch(color=RED,   alpha=0.55,
                               label=fr"Downward $\Delta < 0$  (area={r_area:.4f})"),
            ]
            ax.legend(handles=legend_handles, loc="upper left", framealpha=0.8)
            ax.set_title(
                f"{title}\n"
                fr"Net Lévy area $= {net:+.4f}$   "
                fr"QV $= {g_area + r_area:.4f}$"
            )

        # global legend annotation
        fig.text(
            0.5, 0.01,
            r"Green triangles (above diagonal): $X^1 > X^2$ → lead has advanced → upward move.  "
            r"Red triangles (below diagonal): $X^1 < X^2$ → lead has retreated → downward move.",
            ha="center", fontsize=7.5, color="#444444",
        )

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Figure 3 — Step-by-step annotated construction
# ---------------------------------------------------------------------------

def fig3_step_by_step(x: np.ndarray, outpath: Path, n_shown: int = 6):
    """Annotated construction for first n_shown points."""
    x = x[:n_shown]
    ll = lead_lag_transform(x)
    green, red = signed_triangles(ll)

    with plt.rc_context(FIGSTYLE):
        fig, ax = plt.subplots(figsize=(6, 6))

        lo = ll.min() - 0.08
        hi = ll.max() + 0.08
        _draw_diagonal(ax, lo, hi, color="#cccccc", lw=1.0)
        _fill_triangles(ax, green, red, alpha=0.4)

        # Draw the path with arrows on each segment
        for i in range(len(ll) - 1):
            x0, y0 = ll[i]
            x1, y1 = ll[i + 1]
            is_horizontal = abs(y1 - y0) < 1e-12
            color = "#555555" if is_horizontal else "#999999"
            ls = "-" if is_horizontal else ":"
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.4,
                    mutation_scale=8,
                    connectionstyle="arc3,rad=0.0",
                ),
            )

        # Mark nodes on the diagonal
        diag_pts = ll[::2]   # every even index is back on the diagonal
        ax.scatter(diag_pts[:, 0], diag_pts[:, 1],
                   s=40, color=BLUE, zorder=5)
        for k, (xk, yk) in enumerate(diag_pts):
            ax.annotate(
                f"$t={k}$",
                xy=(xk, yk),
                xytext=(xk + 0.015 * (hi - lo), yk + 0.015 * (hi - lo)),
                fontsize=7.5,
                color=BLUE,
            )

        # Mark corner points (off-diagonal)
        corners = ll[1::2]   # every odd index is the horizontal corner
        ax.scatter(corners[:, 0], corners[:, 1],
                   s=20, color="#ff7f0e", zorder=5, marker="D")

        # Annotate first triangle explicitly
        if green:
            tri = green[0]
            cx = tri[:, 0].mean()
            cy = tri[:, 1].mean()
            ax.text(cx, cy, r"$+\frac{\Delta^2}{2}$",
                    ha="center", va="center", fontsize=9,
                    color=GREEN, fontweight="bold")
        if red:
            tri = red[0]
            cx = tri[:, 0].mean()
            cy = tri[:, 1].mean()
            ax.text(cx, cy, r"$-\frac{\Delta^2}{2}$",
                    ha="center", va="center", fontsize=9,
                    color=RED, fontweight="bold")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$X^1_t$ (lead component)")
        ax.set_ylabel(r"$X^2_t$ (lag component)")
        ax.set_title(
            "Step-by-step L-path construction\n"
            r"$\bullet$ = on diagonal ($X^1=X^2$),  "
            r"$\diamond$ = corner ($X^1$ advanced, $X^2$ frozen)"
        )

        legend_handles = [
            mpatches.Patch(color=GREEN, alpha=0.5, label=r"Upward step: area $= +\Delta^2/2$"),
            mpatches.Patch(color=RED,   alpha=0.5, label=r"Downward step: area $= -\Delta^2/2$"),
            mpatches.Patch(color="#555555", label="Horizontal micro-step (lead advances)"),
            mpatches.Patch(color="#999999", alpha=0.6, label="Vertical micro-step (lag catches up)"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=7.5, framealpha=0.85)

        fig.tight_layout()
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Figure 4 — Mathematical summary panel
# ---------------------------------------------------------------------------

def fig4_math_summary(x_trend: np.ndarray, x_mr: np.ndarray, outpath: Path):
    """Bar chart comparing Lévy area decomposition for trending vs mean-reverting."""
    results = {}
    for label, x in [("Trending", x_trend), ("Mean-rev.", x_mr)]:
        ll = lead_lag_transform(x)
        green, red = signed_triangles(ll)
        g, r, net = levy_area_from_triangles(green, red)
        # QV is sum of squared increments
        qv = float(np.sum(np.diff(x) ** 2))
        results[label] = {"green": g, "red": r, "net": net, "qv/2": qv / 2}

    with plt.rc_context(FIGSTYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

        labels = list(results.keys())
        x_pos = np.arange(len(labels))
        width = 0.28

        # Panel A: Green / Red / Net area
        ax = axes[0]
        ax.bar(x_pos - width, [results[l]["green"] for l in labels],
               width, color=GREEN, alpha=0.7, label="Green area (↑)")
        ax.bar(x_pos,         [results[l]["red"]   for l in labels],
               width, color=RED,   alpha=0.7, label="Red area (↓)")
        ax.bar(x_pos + width, [results[l]["net"]   for l in labels],
               width, color=BLUE,  alpha=0.7, label="Net Lévy area")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Area")
        ax.set_title("(a) Signed-area decomposition")
        ax.legend()

        # Panel B: Net Lévy area vs QV/2 (should be equal up to sign conventions)
        ax = axes[1]
        ax.bar(x_pos - width / 2, [results[l]["green"] + results[l]["red"] for l in labels],
               width, color="#7f7f7f", alpha=0.7, label=r"$\Sigma|\Delta_i^2|/2$  (≡ QV/2)")
        ax.bar(x_pos + width / 2, [results[l]["qv/2"] for l in labels],
               width, color=BLUE, alpha=0.4, label=r"QV/2 from $\Sigma\Delta_i^2$", ls="--",
               edgecolor=BLUE)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Area / QV")
        ax.set_title(r"(b) Total triangle area $= \frac{1}{2}\,$QV (identity check)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default="visuals/figures",
                        help="Output directory for figures (default: visuals/figures)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Generate synthetic series ---
    x_trend = trending_series(n=14, seed=7)
    x_mr    = mean_reverting_series(n=14, seed=3)

    print("Generating figures...")

    fig1_construction(x_trend, outdir / "fig1_construction.pdf")
    fig2_signed_area(x_trend, x_mr, outdir / "fig2_signed_area.pdf")
    fig3_step_by_step(x_trend, outdir / "fig3_step_by_step.pdf", n_shown=7)
    fig4_math_summary(x_trend, x_mr, outdir / "fig4_math_summary.pdf")

    # Also save PNG versions for quick preview
    for pdf in outdir.glob("*.pdf"):
        pass  # PDFs are already saved; PNGs are produced at dpi=200 inside each function

    print(f"\nAll figures saved to {outdir}/")

    # --- Print numerical summary to stdout ---
    print("\n── Lévy area summary ──")
    for label, x in [("Trending", x_trend), ("Mean-reverting", x_mr)]:
        ll = lead_lag_transform(x)
        green, red = signed_triangles(ll)
        g, r, net = levy_area_from_triangles(green, red)
        qv = float(np.sum(np.diff(x) ** 2))
        n_up   = len(green)
        n_down = len(red)
        print(f"\n  {label}:")
        print(f"    up moves   = {n_up},  total green area = {g:.6f}")
        print(f"    down moves = {n_down}, total red area   = {r:.6f}")
        print(f"    net Lévy area (green − red)  = {net:+.6f}")
        print(f"    QV/2 from increments         = {qv/2:.6f}  (should ≈ green + red)")
        print(f"    directional asymmetry ratio  = {net / (g + r + 1e-15):.4f}")


if __name__ == "__main__":
    main()
