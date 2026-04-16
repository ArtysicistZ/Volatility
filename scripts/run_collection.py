"""
Step 1: Data Collection Pipeline
Fetches 90 days of 1-min BTC/USDT candles from Binance,
cleans with regex, and ingests into DuckDB.

Usage:
    conda activate ctestenv
    python scripts/run_collection.py
"""
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import pandas as pd
    from src.data_collection import (
        fetch_klines, validate_completeness, clean_raw_data,
        init_duckdb, ingest_candles, query_resampled,
    )
    from src.config import BINANCE_SYMBOL, BINANCE_INTERVAL, DATA_RAW

    # --- 1. Fetch raw klines (skip if already downloaded) ---
    # Prefer full dataset if available
    raw_full = DATA_RAW / f"{BINANCE_SYMBOL.lower()}_{BINANCE_INTERVAL}_full.parquet"
    raw_default = DATA_RAW / f"{BINANCE_SYMBOL.lower()}_{BINANCE_INTERVAL}_raw.parquet"
    raw_path = raw_full if raw_full.exists() else raw_default
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching klines from Binance")
    logger.info("=" * 60)
    if raw_path.exists():
        logger.info(f"Raw data already exists at {raw_path}, skipping download")
        df_raw = pd.read_parquet(raw_path)
    else:
        df_raw = fetch_klines()
    print(f"\nRaw data shape: {df_raw.shape}")
    print(df_raw.head(3))

    # --- 2. Validate completeness ---
    logger.info("=" * 60)
    logger.info("STEP 2: Validating data completeness")
    logger.info("=" * 60)
    report = validate_completeness(df_raw)
    print(f"\nValidation report:")
    print(f"  Total rows: {report['total_rows']}")
    print(f"  Duplicates: {report['duplicates']}")
    print(f"  Gaps > 1 min: {report['gaps_count']}")

    # --- 3. Clean with regex ---
    logger.info("=" * 60)
    logger.info("STEP 3: Cleaning data (regex validation)")
    logger.info("=" * 60)
    df_clean = clean_raw_data(df_raw)
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    print(df_clean.head(3))

    # Save cleaned data
    processed_path = PROJECT_ROOT / "data" / "processed" / "btcusdt_1m_clean.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(processed_path)
    logger.info(f"Cleaned data saved to {processed_path}")

    # --- 4. Ingest into DuckDB ---
    logger.info("=" * 60)
    logger.info("STEP 4: Ingesting into DuckDB")
    logger.info("=" * 60)
    conn = init_duckdb()
    ingest_candles(conn, df_clean)

    # Demo: resampled query (course SQL requirement)
    df_5min = query_resampled(conn, interval="5 minutes")
    print(f"\n5-min resampled data shape: {df_5min.shape}")
    print(df_5min.head(3))

    conn.close()

    logger.info("=" * 60)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
