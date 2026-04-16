"""
Data collection pipeline: Binance API + Regex cleaning + DuckDB storage.
Covers course requirements: Regex and DuckDB/SQL.
"""
import io
import re
import zipfile
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.config import (
    BINANCE_SYMBOL, BINANCE_INTERVAL, COLLECTION_START, COLLECTION_END,
    KLINE_COLUMNS, DATA_RAW, DATA_PROCESSED, DUCKDB_PATH,
    RE_TIMESTAMP_MS, RE_TRADING_PAIR, RE_NUMERIC,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# BINANCE PUBLIC DATA (no API key, no regional restrictions)
# ═══════════════════════════════════════════════════════════════════════════

# Column names for the CSV files from data.binance.vision
_KLINE_CSV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def fetch_klines(
    symbol: str = BINANCE_SYMBOL,
    interval: str = BINANCE_INTERVAL,
    start_str: str = COLLECTION_START,
    end_str: str = COLLECTION_END,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Download historical klines from Binance public data repository.
    URL pattern: https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/
    Downloads daily ZIP files, each containing a CSV of that day's candles.

    Returns DataFrame with columns matching KLINE_COLUMNS.
    """
    start_date = pd.Timestamp(start_str)
    end_date = pd.Timestamp(end_str)
    dates = pd.date_range(start_date, end_date, freq="D")

    all_frames = []
    failed_dates = []

    logger.info(
        f"Downloading {symbol} {interval} klines from {start_str} to {end_str} "
        f"({len(dates)} days) from data.binance.vision..."
    )

    for date in tqdm(dates, desc="Downloading daily klines"):
        date_str = date.strftime("%Y-%m-%d")
        url = (
            f"https://data.binance.vision/data/spot/daily/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                failed_dates.append(date_str)
                continue

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as csv_file:
                    df_day = pd.read_csv(
                        csv_file, header=None, names=_KLINE_CSV_COLUMNS
                    )
                    all_frames.append(df_day)

        except Exception as e:
            logger.warning(f"Failed to download {date_str}: {e}")
            failed_dates.append(date_str)

    if not all_frames:
        raise RuntimeError("No data downloaded. Check network or date range.")

    df = pd.concat(all_frames, ignore_index=True)

    if failed_dates:
        logger.warning(f"{len(failed_dates)} days failed: {failed_dates[:5]}...")

    logger.info(f"Downloaded {len(df)} candles across {len(all_frames)} days")

    # Save raw data immediately (never need to re-fetch)
    if save_path is None:
        save_path = DATA_RAW / f"{symbol.lower()}_{interval}_raw.parquet"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)
    logger.info(f"Raw data saved to {save_path}")

    return df


def validate_completeness(df: pd.DataFrame) -> dict:
    """
    Check data completeness: row count, gaps, duplicates, nulls.
    """
    result = {
        "total_rows": len(df),
        "duplicates": df["open_time"].duplicated().sum(),
        "nulls": df.isnull().sum().to_dict(),
    }

    # Check for time gaps > 1 minute (60_000 ms)
    if "open_time" in df.columns:
        times = pd.to_numeric(df["open_time"])
        diffs = times.diff().dropna()
        gaps = diffs[diffs > 60_000]
        result["gaps_count"] = len(gaps)
        result["gaps"] = [
            (df["open_time"].iloc[i - 1], df["open_time"].iloc[i], int(d))
            for i, d in gaps.items()
        ]
    else:
        result["gaps_count"] = 0
        result["gaps"] = []

    logger.info(
        f"Validation: {result['total_rows']} rows, "
        f"{result['duplicates']} duplicates, "
        f"{result['gaps_count']} gaps"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# REGEX CLEANING (course requirement)
# ═══════════════════════════════════════════════════════════════════════════

def validate_timestamps(series: pd.Series) -> pd.Series:
    """
    Validate that each value is a 13-digit Unix millisecond timestamp.
    Returns boolean mask of valid entries.
    """
    pattern = re.compile(RE_TIMESTAMP_MS)
    mask = series.astype(str).apply(lambda x: bool(pattern.fullmatch(x)))
    invalid_count = (~mask).sum()
    if invalid_count > 0:
        logger.warning(f"{invalid_count} invalid timestamps detected")
    return mask


def extract_trading_pair(raw_string: str) -> tuple[str, str] | None:
    """
    Extract (base, quote) from a trading pair string using regex.
    E.g. 'BTCUSDT' -> ('BTC', 'USDT'), 'INVALID' -> None.
    """
    match = re.match(RE_TRADING_PAIR, raw_string.upper().strip())
    if match:
        return match.group(1), match.group(2)
    return None


def validate_numeric_fields(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Validate that specified columns contain valid numeric strings.
    Coerce types and drop rows with non-numeric values.
    """
    pattern = re.compile(RE_NUMERIC)
    df = df.copy()

    invalid_rows = pd.Series(False, index=df.index)
    for col in columns:
        mask = df[col].astype(str).apply(lambda x: bool(pattern.fullmatch(x)))
        invalid_in_col = (~mask).sum()
        if invalid_in_col > 0:
            logger.warning(f"Column '{col}': {invalid_in_col} non-numeric values")
        invalid_rows |= ~mask

    total_invalid = invalid_rows.sum()
    if total_invalid > 0:
        logger.warning(f"Dropping {total_invalid} rows with non-numeric fields")
        df = df[~invalid_rows]

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Validate timestamps with regex
    2. Validate numeric OHLCV fields with regex
    3. Cast types (float64 for prices/volumes, int64 for timestamps)
    4. Convert open_time to UTC datetime index
    5. Sort, deduplicate, forward-fill small gaps (<= 2 minutes)
    """
    df = df.copy()

    # 1. Validate timestamps
    ts_valid = validate_timestamps(df["open_time"])
    if not ts_valid.all():
        logger.info(f"Removing {(~ts_valid).sum()} rows with invalid timestamps")
        df = df[ts_valid]

    # 2. Validate numeric fields
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    df = validate_numeric_fields(df, numeric_cols)

    # 3. Cast types
    df["open_time"] = pd.to_numeric(df["open_time"])
    df["close_time"] = pd.to_numeric(df["close_time"])
    df["num_trades"] = pd.to_numeric(df["num_trades"]).astype(int)
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    # 4. Convert to datetime index
    # Handle mixed timestamp formats: 13-digit (ms) vs 16-digit (us)
    # Normalize all to milliseconds first
    ts_numeric = df["open_time"].astype(np.int64)
    # If value > 1e15, it's microseconds — divide by 1000 to get ms
    ts_ms = ts_numeric.where(ts_numeric < 1e15, ts_numeric // 1000)
    logger.info(
        f"Timestamp normalization: {(ts_numeric >= 1e15).sum()} microsecond, "
        f"{(ts_numeric < 1e15).sum()} millisecond timestamps"
    )
    df["timestamp"] = pd.to_datetime(ts_ms, unit="ms", utc=True)
    df = df.set_index("timestamp")

    # Drop helper columns
    df = df.drop(columns=["open_time", "close_time", "ignore"], errors="ignore")

    # 5. Sort and deduplicate
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # 6. Forward-fill small gaps (<= 2 minutes)
    full_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="1min", tz="UTC"
    )
    df = df.reindex(full_range)
    df.index.name = "timestamp"

    # Identify gap sizes
    was_nan = df["close"].isna()
    # Forward-fill only gaps <= 2 rows (2 minutes)
    gap_groups = was_nan.ne(was_nan.shift()).cumsum()
    gap_sizes = was_nan.groupby(gap_groups).transform("sum")
    small_gaps = was_nan & (gap_sizes <= 2)
    large_gaps = was_nan & (gap_sizes > 2)

    if large_gaps.any():
        large_gap_count = large_gaps.sum()
        logger.warning(f"{large_gap_count} rows in gaps > 2 minutes (not filled)")

    # Forward-fill small gaps
    df = df.ffill()
    # Drop any remaining NaN rows (from large gaps or edges)
    df = df.dropna(subset=["close"])

    logger.info(f"Cleaned data: {len(df)} rows, index from {df.index.min()} to {df.index.max()}")

    # Demonstrate extract_trading_pair (course requirement)
    pair = extract_trading_pair(BINANCE_SYMBOL)
    logger.info(f"Trading pair extracted: {pair}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# DUCKDB (course requirement)
# ═══════════════════════════════════════════════════════════════════════════

def init_duckdb(db_path: Path = DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Initialize DuckDB database with schema for candles and predictions.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_candles (
            timestamp TIMESTAMP WITH TIME ZONE PRIMARY KEY,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            quote_volume DOUBLE,
            num_trades INTEGER,
            taker_buy_base_volume DOUBLE,
            taker_buy_quote_volume DOUBLE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            timestamp TIMESTAMP WITH TIME ZONE,
            model_name VARCHAR,
            predicted_vol DOUBLE,
            actual_vol DOUBLE,
            PRIMARY KEY (timestamp, model_name)
        )
    """)

    logger.info(f"DuckDB initialized at {db_path}")
    return conn


def ingest_candles(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Bulk insert cleaned candle data into raw_candles table."""
    # Reset index so timestamp is a column
    insert_df = df.reset_index()
    insert_df = insert_df[[
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_volume", "num_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]]

    conn.execute("DELETE FROM raw_candles")  # idempotent reload
    conn.execute("INSERT INTO raw_candles SELECT * FROM insert_df")
    count = conn.execute("SELECT COUNT(*) FROM raw_candles").fetchone()[0]
    logger.info(f"Ingested {count} candles into DuckDB")


def query_resampled(
    conn: duckdb.DuckDBPyConnection,
    interval: str = "5 minutes",
) -> pd.DataFrame:
    """
    Resample candles using DuckDB SQL time_bucket.
    Demonstrates GROUP BY, aggregation functions.
    """
    query = f"""
        SELECT
            time_bucket(INTERVAL '{interval}', timestamp) AS bucket,
            FIRST(open ORDER BY timestamp)  AS open,
            MAX(high)                       AS high,
            MIN(low)                        AS low,
            LAST(close ORDER BY timestamp)  AS close,
            SUM(volume)                     AS volume,
            SUM(quote_volume)               AS quote_volume,
            SUM(num_trades)                 AS num_trades
        FROM raw_candles
        GROUP BY bucket
        ORDER BY bucket
    """
    return conn.execute(query).fetchdf()


def store_predictions(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
    preds_df: pd.DataFrame,
):
    """
    Store model predictions. preds_df must have columns:
    timestamp, predicted_vol, actual_vol.
    """
    insert_df = preds_df.copy()
    insert_df["model_name"] = model_name

    conn.execute(f"DELETE FROM predictions WHERE model_name = '{model_name}'")
    conn.execute("""
        INSERT INTO predictions (timestamp, model_name, predicted_vol, actual_vol)
        SELECT timestamp, model_name, predicted_vol, actual_vol
        FROM insert_df
    """)
    count = conn.execute(
        f"SELECT COUNT(*) FROM predictions WHERE model_name = '{model_name}'"
    ).fetchone()[0]
    logger.info(f"Stored {count} predictions for {model_name}")


def compare_models_sql(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    SQL JOIN across predictions table to compare per-timestamp errors.
    Demonstrates JOIN for course requirement.
    """
    query = """
        SELECT
            g.timestamp,
            g.actual_vol,
            g.predicted_vol  AS garch_pred,
            x.predicted_vol  AS xgboost_pred,
            l.predicted_vol  AS lstm_pred,
            POWER(g.actual_vol - g.predicted_vol, 2)  AS garch_se,
            POWER(x.actual_vol - x.predicted_vol, 2)  AS xgboost_se,
            POWER(l.actual_vol - l.predicted_vol, 2)  AS lstm_se
        FROM predictions g
        JOIN predictions x
            ON g.timestamp = x.timestamp AND x.model_name = 'xgboost'
        JOIN predictions l
            ON g.timestamp = l.timestamp AND l.model_name = 'lstm'
        WHERE g.model_name = 'garch'
        ORDER BY g.timestamp
    """
    return conn.execute(query).fetchdf()
