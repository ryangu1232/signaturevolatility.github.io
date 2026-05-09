from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class FreqConfig:
    name: str
    interval: str
    period: str
    horizon_bars: int
    annual_bars: int       # trading bars per year, equity calendar
    sig_windows: Tuple[int, ...]  # (short, medium, long) lookback windows for signature models
    aggregate_from_5m_to_10m: bool = False


FREQ_CONFIGS: Dict[str, FreqConfig] = {
    "1m": FreqConfig(
        name="1m",
        interval="1m",
        period="7d",
        horizon_bars=5,
        annual_bars=98_280,   # 252 days * 390 min/day
        sig_windows=(30, 78, 195),
        # 30 min | 2 h | ~half trading day (5 h)
    ),
    "5m": FreqConfig(
        name="5m",
        interval="5m",
        period="60d",
        horizon_bars=5,
        annual_bars=19_656,   # 252 days * 78 bars/day
        sig_windows=(6, 16, 78),
        # 30 min | ~1.5 h | 1 full trading day
    ),
    "10m": FreqConfig(
        name="10m",
        interval="5m",
        period="60d",
        horizon_bars=5,
        annual_bars=9_828,    # 252 days * 39 bars/day
        sig_windows=(3, 8, 39),
        # 30 min | ~1.5 h | 1 full trading day
        aggregate_from_5m_to_10m=True,
    ),
    "1h": FreqConfig(
        name="1h",
        interval="60m",
        period="730d",
        horizon_bars=5,
        annual_bars=1_638,    # 252 days * 6.5 hours/day
        sig_windows=(2, 7, 35),
        # 2 h | ~1 trading day | ~1 trading week
    ),
    "1d": FreqConfig(
        name="1d",
        interval="1d",
        period="10y",
        horizon_bars=5,
        annual_bars=252,
        sig_windows=(5, 22, 60),
        # 1 week | 1 month | ~3 months  (original HAR scales, still valid at daily)
    ),
}

# ------------------------------------------------------------------
# Suggested tickers for cross-ticker experiments.
#
# Volatility spectrum
#   Low    : GLD, BND, SPY
#   Medium : AAPL, MSFT, QQQ
#   High   : TSLA, NVDA
#   Extreme: GME, AMC, MSTR
#
# Regime diversity
#   Energy / commodity cycle : XOM, CVX
#   Post-2022 tech reset     : META, NFLX
#   Crypto-adjacent          : MSTR, COIN
#   Pure crypto (24/7, no gaps): BTC-USD, ETH-USD
#
# Data / missing-data stress
#   Long history (30 y daily): ^GSPC, ^DJI, SPY
#   Short / thin history     : SPCE, RIVN
#   Frequent halts / gaps    : GME (Jan 2021), AMC
#   Index (no intraday in yf): ^VIX  — daily only
# ------------------------------------------------------------------
TICKER_PRESETS: Dict[str, str] = {
    "spy":   "SPY",
    "qqq":   "QQQ",
    "tsla":  "TSLA",
    "nvda":  "NVDA",
    "gme":   "GME",
    "gld":   "GLD",
    "xom":   "XOM",
    "meta":  "META",
    "btc":   "BTC-USD",
    "gspc":  "^GSPC",
}

DEFAULT_HAR_WINDOWS = (1, 5, 22)
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 0
