"""
Step 3: Baseline Models (GARCH + XGBoost)
Trains baselines and saves predictions.

Usage:
    conda activate ctestenv
    python scripts/run_baselines.py               # run both
    python scripts/run_baselines.py --garch-only   # run only GARCH
    python scripts/run_baselines.py --xgb-only     # run only XGBoost
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import logging
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(run_garch=True, run_xgb=True):
    import numpy as np
    import pandas as pd
    from src.config import (
        DATA_PROCESSED, OUTPUT_MODELS, TARGET_WINDOW,
        TRAIN_RATIO, VAL_RATIO,
        GARCH_MIN_WINDOW, GARCH_REFIT_FREQ,
        XGBOOST_MIN_WINDOW, XGBOOST_REFIT_FREQ, XGBOOST_OPTUNA_TRIALS,
    )
    from src.feature_engineering import log_returns
    from src.models import (
        test_stationarity, garch_rolling_forecast,
        xgb_optuna_tune, xgb_expanding_predict,
    )

    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    logger.info("Loading data...")
    df_clean = pd.read_parquet(DATA_PROCESSED / "btcusdt_1m_clean.parquet")
    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    target = pd.read_parquet(DATA_PROCESSED / "target.parquet")["realized_volatility"]

    # Align features and target (drop NaN target rows)
    common_idx = features.index.intersection(target.dropna().index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    n = len(features)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    print(f"Total aligned samples: {n}")
    print(f"Train: 0–{train_end}, Val: {train_end}–{val_end}, Test: {val_end}–{n}")

    # ================================================================
    # GARCH(1,1)
    # ================================================================
    garch_preds = None

    if run_garch:
        logger.info("=" * 60)
        logger.info("GARCH(1,1) BASELINE")
        logger.info("=" * 60)

        # GARCH uses raw log returns, not engineered features
        rets = log_returns(df_clean["close"]).dropna()

        # Stationarity test
        stat = test_stationarity(rets)
        print(f"\nADF Test: stat={stat['adf_stat']:.4f}, p={stat['p_value']:.6f}, "
              f"stationary={stat['is_stationary']}")

        # Rolling window forecast on TEST period only
        test_timestamps = features.index[val_end:]
        rets_aligned = rets.loc[:test_timestamps[-1]]
        test_start_pos = rets.index.get_loc(test_timestamps[0])

        logger.info(f"GARCH forecasting from position {test_start_pos} to {len(rets_aligned)} "
                    f"(rolling window={GARCH_MIN_WINDOW}, refit every {GARCH_REFIT_FREQ} steps)...")

        garch_preds = garch_rolling_forecast(
            rets_aligned,
            start_pos=test_start_pos,
            rolling_window=GARCH_MIN_WINDOW,
            refit_freq=GARCH_REFIT_FREQ,
            horizon=TARGET_WINDOW,
        )
        print(f"GARCH predictions: {len(garch_preds)} values")
        print(f"GARCH pred stats: mean={garch_preds.mean():.6f}, std={garch_preds.std():.6f}")

        garch_preds.to_pickle(OUTPUT_MODELS / "garch_predictions.pkl")
        logger.info("GARCH predictions saved")

    # ================================================================
    # XGBoost
    # ================================================================
    xgb_preds = None

    if run_xgb:
        logger.info("=" * 60)
        logger.info("XGBOOST BASELINE")
        logger.info("=" * 60)

        X_train = features.iloc[:train_end].values
        y_train = target.iloc[:train_end].values
        X_val = features.iloc[train_end:val_end].values
        y_val = target.iloc[train_end:val_end].values

        # Hyperparameter tuning with Optuna
        logger.info(f"Tuning XGBoost with {XGBOOST_OPTUNA_TRIALS} Optuna trials...")
        best_params = xgb_optuna_tune(X_train, y_train, X_val, y_val,
                                       n_trials=XGBOOST_OPTUNA_TRIALS)
        print(f"\nBest XGBoost params: {best_params}")

        with open(OUTPUT_MODELS / "xgboost_best_params.pkl", "wb") as f:
            pickle.dump(best_params, f)

        # Expanding window predictions on TEST period
        logger.info(f"XGBoost expanding window on test set "
                    f"(refit every {XGBOOST_REFIT_FREQ} steps)...")

        full_features = features.iloc[:n]
        full_target = target.iloc[:n]

        xgb_preds = xgb_expanding_predict(
            full_features, full_target,
            params=best_params,
            min_window=val_end,
            refit_freq=XGBOOST_REFIT_FREQ,
        )
        print(f"XGBoost predictions: {len(xgb_preds)} values")
        print(f"XGBoost pred stats: mean={xgb_preds.mean():.6f}, std={xgb_preds.std():.6f}")

        xgb_preds.to_pickle(OUTPUT_MODELS / "xgboost_predictions.pkl")
        logger.info("XGBoost predictions saved")

    # ================================================================
    # Summary
    # ================================================================
    logger.info("=" * 60)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("=" * 60)

    test_actual = target.iloc[val_end:]
    for name, preds in [("GARCH", garch_preds), ("XGBoost", xgb_preds)]:
        if preds is not None:
            overlap = test_actual.index.intersection(preds.index)
            if len(overlap) > 0:
                mse = np.mean((test_actual.loc[overlap] - preds.loc[overlap]) ** 2)
                print(f"{name} test MSE: {mse:.10f} ({len(overlap)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--garch-only", action="store_true", help="Run only GARCH")
    parser.add_argument("--xgb-only", action="store_true", help="Run only XGBoost")
    args = parser.parse_args()
    main(run_garch=not args.xgb_only, run_xgb=not args.garch_only)
