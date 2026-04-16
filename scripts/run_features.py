"""
Step 2: Feature Engineering Pipeline
Builds feature matrix and target variable from cleaned candle data.

Usage:
    conda activate ctestenv
    python scripts/run_features.py
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import pandas as pd
    from src.config import DATA_PROCESSED, TARGET_WINDOW
    from src.feature_engineering import (
        build_features, log_returns, realized_volatility, log_transform_target,
    )

    # --- 1. Load cleaned data ---
    clean_path = DATA_PROCESSED / "btcusdt_1m_clean.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} not found. Run scripts/run_collection.py first."
        )

    logger.info("=" * 60)
    logger.info("STEP 1: Loading cleaned data")
    logger.info("=" * 60)
    df = pd.read_parquet(clean_path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # --- 2. Build feature matrix ---
    logger.info("=" * 60)
    logger.info("STEP 2: Building feature matrix")
    logger.info("=" * 60)
    features = build_features(df)
    print(f"\nFeature matrix: {features.shape}")
    print(f"Features: {list(features.columns)}")
    print(features.head(3))
    print(f"\nAny NaN: {features.isna().any().any()}")

    # --- 3. Compute target variable ---
    logger.info("=" * 60)
    logger.info("STEP 3: Computing realized volatility target")
    logger.info("=" * 60)
    rets = log_returns(df["close"])
    rv = realized_volatility(rets, window=TARGET_WINDOW)

    # Align target to feature index
    rv = rv.loc[features.index]
    valid_rv = rv.dropna()
    print(f"\nRealized volatility: {len(valid_rv)} valid values out of {len(rv)}")
    print(f"RV stats: mean={valid_rv.mean():.6f}, std={valid_rv.std():.6f}, "
          f"min={valid_rv.min():.6f}, max={valid_rv.max():.6f}")

    # Log-transform for LSTM
    rv_log = log_transform_target(rv)
    print(f"Log-transformed RV stats: mean={rv_log.dropna().mean():.4f}, "
          f"std={rv_log.dropna().std():.4f}")

    # --- 4. Save outputs ---
    logger.info("=" * 60)
    logger.info("STEP 4: Saving features and target")
    logger.info("=" * 60)
    features.to_parquet(DATA_PROCESSED / "features.parquet")
    rv.to_frame("realized_volatility").to_parquet(DATA_PROCESSED / "target.parquet")
    rv_log.to_frame("rv_log").to_parquet(DATA_PROCESSED / "target_log.parquet")

    logger.info(f"Features saved: {features.shape}")
    logger.info(f"Target saved: {len(valid_rv)} valid rows")

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
