from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_close"] = np.log(df["close"])
    df["log_return"] = df["log_close"].diff()
    df = df.dropna().reset_index(drop=True)
    return df


def realized_volatility_target(returns: pd.Series, horizon_bars: int) -> pd.Series:
    sq = returns.pow(2)
    future_sum = sum(sq.shift(-i) for i in range(1, horizon_bars + 1))
    return np.sqrt(future_sum) * 100.0


def rolling_realized_volatility(returns: pd.Series, window_bars: int) -> pd.Series:
    return np.sqrt((returns.pow(2)).rolling(window_bars).sum())


def lead_lag_transform(path_1d: np.ndarray) -> np.ndarray:
    x = np.asarray(path_1d, dtype=float).reshape(-1)
    if len(x) < 2:
        raise ValueError("Path must have length at least 2.")

    out = [[x[0], x[0]]]
    for i in range(1, len(x)):
        out.append([x[i], x[i - 1]])
        out.append([x[i], x[i]])
    return np.asarray(out, dtype=float)


def approximate_signature_features(path: np.ndarray) -> np.ndarray:
    increments = np.diff(path, axis=0)
    _, d = increments.shape

    level1 = increments.sum(axis=0)

    cumsum = np.zeros(d)
    accum = []
    for dx in increments:
        accum.append(np.outer(cumsum, dx).reshape(-1))
        cumsum += dx
    level2 = np.sum(accum, axis=0) if len(accum) else np.zeros(d * d)

    qv = np.sum(increments ** 2, axis=0)
    abs_sum = np.sum(np.abs(increments), axis=0)
    max_abs = np.max(np.abs(increments), axis=0)

    cross = []
    for i in range(d):
        for j in range(i + 1, d):
            cross.append(np.sum(increments[:, i] * increments[:, j]))
    cross = np.asarray(cross, dtype=float)

    return np.concatenate([level1, level2, qv, abs_sum, max_abs, cross])


def exact_signature_features(path: np.ndarray, sig_level: int = 2) -> np.ndarray:
    try:
        import iisignature  # type: ignore
    except ImportError as e:
        raise ImportError(
            "iisignature is not installed. Install it with `pip install iisignature` "
            "or run with --signature-mode approximate."
        ) from e

    path = np.asarray(path, dtype=float)
    sig = iisignature.sig(path, sig_level)
    return np.asarray(sig, dtype=float)


def signature_features(path: np.ndarray, mode: str = "approximate", sig_level: int = 2) -> np.ndarray:
    if mode == "approximate":
        return approximate_signature_features(path)
    if mode == "exact":
        return exact_signature_features(path, sig_level=sig_level)
    raise ValueError(f"Unknown signature mode: {mode}")


def build_signature_dataset(
    df: pd.DataFrame,
    lookback_bars: int,
    horizon_bars: int,
    signature_mode: str = "approximate",
    sig_level: int = 2,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = df.copy()
    work["target_rv"] = realized_volatility_target(work["log_return"], horizon_bars)

    r = work["log_return"].to_numpy()
    y = work["target_rv"].to_numpy()
    timestamps = work["timestamp"].to_numpy()

    X_list, y_list, ts_list = [], [], []
    last_valid_t = len(work) - horizon_bars - 1

    for t in range(lookback_bars - 1, last_valid_t + 1):
        window = r[t - lookback_bars + 1 : t + 1]
        if np.any(~np.isfinite(window)):
            continue

        ll_path = lead_lag_transform(window)
        feats = signature_features(ll_path, mode=signature_mode, sig_level=sig_level)

        target = y[t]
        if not np.isfinite(target):
            continue

        X_list.append(feats)
        y_list.append(target)
        ts_list.append(timestamps[t])

    X = np.asarray(X_list, dtype=float)
    y_out = np.asarray(y_list, dtype=float)
    meta = pd.DataFrame({"timestamp": pd.to_datetime(ts_list)})
    return meta, X, y_out


def build_har_dataset(
    df: pd.DataFrame,
    horizon_bars: int,
    windows=(1, 5, 22),
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = df.copy()
    work["target_rv"] = realized_volatility_target(work["log_return"], horizon_bars)

    # Proper HAR-RV features: sqrt((1/w) * Σ r²_{t-i})
    # Using mean squared return (variance) then sqrt keeps features in vol units
    # consistent with the target, and avoids the mean(|r|) ≠ sqrt(mean(r²)) bias.
    sq = work["log_return"].pow(2)
    feature_cols = []

    for w in windows:
        col = f"har_rv_{w}"
        work[col] = np.sqrt(sq.rolling(w).mean())
        feature_cols.append(col)

    work = work.dropna().reset_index(drop=True)
    X = work[feature_cols].to_numpy(dtype=float)
    y = work["target_rv"].to_numpy(dtype=float)
    meta = work[["timestamp"]].copy()
    return meta, X, y


def build_range_har_dataset(
    df: pd.DataFrame,
    horizon_bars: int,
    windows: tuple = (1, 5, 22),
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """HAR with Parkinson range-based per-bar variance estimate.

    Uses (ln(H/L))² / (4*ln(2)) as the per-bar variance proxy instead of r².
    Parkinson (1980) showed this is ~5x more efficient than close-to-close for
    Brownian motion, since it uses the full intraday price range.
    Target is still close-to-close realized vol for comparability with HAR-RV.
    """
    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("Range-based HAR requires 'high' and 'low' columns.")

    work = df.copy()
    work["target_rv"] = realized_volatility_target(work["log_return"], horizon_bars)

    ln2_4 = 4.0 * np.log(2.0)
    work["park_var"] = (np.log(work["high"] / work["low"]) ** 2) / ln2_4

    feature_cols = []
    for w in windows:
        col = f"range_rv_{w}"
        work[col] = np.sqrt(work["park_var"].rolling(w).mean())
        feature_cols.append(col)

    work = work.dropna().reset_index(drop=True)
    X = work[feature_cols].to_numpy(dtype=float)
    y = work["target_rv"].to_numpy(dtype=float)
    meta = work[["timestamp"]].copy()
    return meta, X, y


def build_rv_path_dataset(
    df: pd.DataFrame,
    lookback_windows: tuple,
    horizon_bars: int,
    signature_mode: str = "approximate",
    sig_level: int = 2,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Signature on the RV path rather than the return path.

    At each time t, slices the per-bar RV series at multiple lookback scales,
    applies lead-lag transform + signature to each slice, and concatenates.
    Uses Parkinson (H/L range) as the per-bar RV estimator when OHLC is
    available — falling back to |r_t| otherwise.

    The signature then captures how realized volatility itself evolves and
    clusters across time, rather than how raw returns move.
    """
    work = df.copy()
    work["target_rv"] = realized_volatility_target(work["log_return"], horizon_bars)

    if "high" in work.columns and "low" in work.columns:
        ln2_4 = 4.0 * np.log(2.0)
        work["bar_rv"] = np.sqrt((np.log(work["high"] / work["low"]) ** 2) / ln2_4)
    else:
        work["bar_rv"] = work["log_return"].abs()

    rv = work["bar_rv"].to_numpy()
    y = work["target_rv"].to_numpy()
    timestamps = work["timestamp"].to_numpy()

    max_window = max(lookback_windows)
    X_list, y_list, ts_list = [], [], []
    last_valid_t = len(work) - horizon_bars - 1

    for t in range(max_window - 1, last_valid_t + 1):
        window_feats = []
        valid = True
        for w in lookback_windows:
            window = rv[t - w + 1 : t + 1]
            if np.any(~np.isfinite(window)) or np.any(window <= 0):
                valid = False
                break
            ll_path = lead_lag_transform(window)
            feats = signature_features(ll_path, mode=signature_mode, sig_level=sig_level)
            window_feats.append(feats)

        if not valid:
            continue

        target = y[t]
        if not np.isfinite(target):
            continue

        X_list.append(np.concatenate(window_feats))
        y_list.append(target)
        ts_list.append(timestamps[t])

    X = np.asarray(X_list, dtype=float)
    y_out = np.asarray(y_list, dtype=float)
    meta = pd.DataFrame({"timestamp": pd.to_datetime(ts_list)})
    return meta, X, y_out


def build_garch_dataset(
    df: pd.DataFrame,
    horizon_bars: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = df.copy()
    work["target_rv"] = realized_volatility_target(work["log_return"], horizon_bars)
    work = work.dropna().reset_index(drop=True)
    mask = np.isfinite(work["log_return"]) & np.isfinite(work["target_rv"])
    work = work[mask].reset_index(drop=True)
    meta = work[["timestamp"]].copy()
    returns = work["log_return"].to_numpy(dtype=float)
    y = work["target_rv"].to_numpy(dtype=float)
    return meta, returns, y


def build_multiscale_signature_dataset(
    df: pd.DataFrame,
    lookback_windows: Tuple[int, ...],
    horizon_bars: int,
    signature_mode: str = "approximate",
    sig_level: int = 2,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Signature-HAR hybrid: concatenates signature features computed at multiple lookback scales."""
    work = df.copy()
    work["target_rv"] = realized_volatility_target(work["log_return"], horizon_bars)

    r = work["log_return"].to_numpy()
    y = work["target_rv"].to_numpy()
    timestamps = work["timestamp"].to_numpy()

    max_window = max(lookback_windows)
    X_list, y_list, ts_list = [], [], []
    last_valid_t = len(work) - horizon_bars - 1

    for t in range(max_window - 1, last_valid_t + 1):
        window_feats = []
        valid = True
        for w in lookback_windows:
            window = r[t - w + 1 : t + 1]
            if np.any(~np.isfinite(window)):
                valid = False
                break
            ll_path = lead_lag_transform(window)
            feats = signature_features(ll_path, mode=signature_mode, sig_level=sig_level)
            window_feats.append(feats)

        if not valid:
            continue

        target = y[t]
        if not np.isfinite(target):
            continue

        X_list.append(np.concatenate(window_feats))
        y_list.append(target)
        ts_list.append(timestamps[t])

    X = np.asarray(X_list, dtype=float)
    y_out = np.asarray(y_list, dtype=float)
    meta = pd.DataFrame({"timestamp": pd.to_datetime(ts_list)})
    return meta, X, y_out
