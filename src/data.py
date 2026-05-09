from __future__ import annotations

import pandas as pd
import yfinance as yf

from .config import FREQ_CONFIGS


def download_yfinance_bars(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for symbol={symbol}, interval={interval}, period={period}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    time_col = None
    for c in ("Datetime", "Date"):
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("Could not find Datetime or Date column in downloaded data.")

    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    cols = [c for c in ["timestamp", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].rename(columns=str.lower)
    return df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    dfx = df.copy().set_index("timestamp")

    agg = {}
    if "open" in dfx.columns:
        agg["open"] = "first"
    if "high" in dfx.columns:
        agg["high"] = "max"
    if "low" in dfx.columns:
        agg["low"] = "min"
    if "close" in dfx.columns:
        agg["close"] = "last"
    if "volume" in dfx.columns:
        agg["volume"] = "sum"

    out = dfx.resample(rule).agg(agg).dropna().reset_index()
    return out


def load_frequency_data(freq: str, symbol: str) -> pd.DataFrame:
    cfg = FREQ_CONFIGS[freq]
    df = download_yfinance_bars(symbol, cfg.interval, cfg.period)

    if cfg.aggregate_from_5m_to_10m:
        df = resample_ohlc(df, "10min")

    return df
