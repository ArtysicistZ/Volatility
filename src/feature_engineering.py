"""
Feature engineering pipeline and target variable computation.
Covers course requirement: Pandas.

CRITICAL: Every rolling/lag feature uses .shift(1) to prevent look-ahead bias.
"""
import logging

import numpy as np
import pandas as pd

from src.config import (
    TARGET_WINDOW, ROLLING_WINDOWS, LAG_STEPS,
    ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Individual Feature Functions
# ═══════════════════════════════════════════════════════════════════════════

def log_returns(close: pd.Series) -> pd.Series:
    """Compute log returns: log(close_t / close_{t-1})."""
    return np.log(close).diff()


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Parkinson (1980) volatility estimator using high-low range.
    Formula: sqrt(1/(4*n*ln2) * sum(ln(H/L)^2))
    Applied as rolling window, then shifted by 1 to avoid leakage.
    """
    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2
    pv = np.sqrt(log_hl_sq.rolling(window=window).mean() / (4 * np.log(2)))
    return pv.shift(1)


def garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    window: int,
) -> pd.Series:
    """
    Garman-Klass (1980) volatility estimator using OHLC.
    7.4x more efficient than close-to-close estimator.
    GK = 0.5 * (ln(H/L))^2 - (2*ln2 - 1) * (ln(C/O))^2
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    result = np.sqrt(gk.rolling(window=window).mean().clip(lower=0))
    return result.shift(1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = ATR_PERIOD) -> pd.Series:
    """
    Average True Range.
    TR = max(H-L, |H-Close_{t-1}|, |L-Close_{t-1}|)
    ATR = EWM of TR.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    result = tr.ewm(span=period, adjust=False).mean()
    return result.shift(1)


def bollinger_width(close: pd.Series, period: int = BOLLINGER_PERIOD,
                    num_std: int = BOLLINGER_STD) -> pd.Series:
    """
    Bollinger Band width: (upper - lower) / middle.
    Measures volatility expansion/contraction.
    """
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma
    return width.shift(1)


def volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    """Current volume / rolling mean volume. Detects unusual activity."""
    rolling_mean = volume.rolling(window=window).mean()
    ratio = volume / rolling_mean
    return ratio.shift(1)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume: cumulative sum of signed volume.
    Volume is added if close > prev_close, subtracted otherwise.
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    result = (direction * volume).cumsum()
    # Normalize OBV to prevent scale issues
    result = (result - result.rolling(window=60, min_periods=1).mean()) / \
             (result.rolling(window=60, min_periods=1).std() + 1e-10)
    return result.shift(1)


def rolling_stats(returns: pd.Series,
                  windows: list[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Rolling statistics for each window: mean, std, max, min, skew, kurtosis.
    ALL shifted by 1 to avoid look-ahead bias.
    """
    frames = {}
    for w in windows:
        roll = returns.rolling(window=w)
        frames[f"ret_mean_{w}"] = roll.mean().shift(1)
        frames[f"ret_std_{w}"] = roll.std().shift(1)
        frames[f"ret_max_{w}"] = roll.max().shift(1)
        frames[f"ret_min_{w}"] = roll.min().shift(1)
        frames[f"ret_skew_{w}"] = roll.skew().shift(1)
        frames[f"ret_kurt_{w}"] = roll.kurt().shift(1)
    return pd.DataFrame(frames, index=returns.index)


def lag_features(returns: pd.Series,
                 lags: list[int] = LAG_STEPS) -> pd.DataFrame:
    """Lag features: returns.shift(lag) for each lag."""
    frames = {f"lag_{lag}": returns.shift(lag) for lag in lags}
    return pd.DataFrame(frames, index=returns.index)


def taker_ratio(taker_buy_vol: pd.Series, total_vol: pd.Series) -> pd.Series:
    """
    Proxy for bid-ask pressure. Klines don't have spread data,
    so we use taker_buy_base_volume / volume instead.
    """
    ratio = taker_buy_vol / (total_vol + 1e-10)
    return ratio.shift(1)


# ═══════════════════════════════════════════════════════════════════════════
# Master Feature Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete feature matrix from cleaned candle DataFrame.
    Expects columns: open, high, low, close, volume, taker_buy_base_volume.
    Returns DataFrame with ~40-50 feature columns, no NaN.

    CRITICAL: Every feature uses .shift(1) or is computed on strictly past data.
    """
    features = pd.DataFrame(index=df.index)

    # --- Log returns ---
    rets = log_returns(df["close"])
    features["log_return"] = rets.shift(1)  # shift so we don't use current return

    # --- Volatility estimators ---
    for w in ROLLING_WINDOWS:
        features[f"parkinson_{w}"] = parkinson_vol(df["high"], df["low"], w)
        features[f"garman_klass_{w}"] = garman_klass_vol(
            df["open"], df["high"], df["low"], df["close"], w
        )

    # --- Technical indicators ---
    features["atr"] = atr(df["high"], df["low"], df["close"])
    features["bollinger_width"] = bollinger_width(df["close"])

    # --- Volume features ---
    for w in [10, 30, 60]:
        features[f"volume_ratio_{w}"] = volume_ratio(df["volume"], w)
    features["obv"] = obv(df["close"], df["volume"])

    # --- Taker ratio (bid-ask proxy) ---
    if "taker_buy_base_volume" in df.columns:
        features["taker_ratio"] = taker_ratio(
            df["taker_buy_base_volume"], df["volume"]
        )

    # --- Rolling statistics of returns ---
    rstats = rolling_stats(rets)
    features = pd.concat([features, rstats], axis=1)

    # --- Lag features ---
    lags = lag_features(rets)
    features = pd.concat([features, lags], axis=1)

    # --- Cyclical time-of-day encoding ---
    hour = df.index.hour + df.index.minute / 60.0
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # --- Drop warmup NaN rows ---
    initial_len = len(features)
    features = features.dropna()
    dropped = initial_len - len(features)
    logger.info(f"Dropped {dropped} warmup rows. Feature matrix: {features.shape}")

    return features


# ═══════════════════════════════════════════════════════════════════════════
# Target Variable
# ═══════════════════════════════════════════════════════════════════════════

def realized_volatility(log_rets: pd.Series,
                        window: int = TARGET_WINDOW) -> pd.Series:
    """
    Forward-looking realized volatility: std(returns[t+1 : t+window]).
    Uses ddof=1 (sample std). Last `window` rows will be NaN.

    Implementation:
    - Compute rolling std of log returns over `window` periods
    - Shift backward by `window` to align with the prediction origin
    """
    # rolling std gives std over the window ending at each point
    # shift(-window) aligns it so value at t = std of returns from t+1 to t+window
    rv = log_rets.rolling(window=window).std()
    rv = rv.shift(-window)
    return rv


def log_transform_target(rv: pd.Series, eps: float = 1e-10) -> pd.Series:
    """Log-transform realized volatility for training stability."""
    return np.log(rv + eps)


def inverse_log_transform(pred: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Invert log transform: exp(pred) - eps."""
    return np.exp(pred) - eps
