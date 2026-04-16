"""
Step 5: Model Evaluation
Computes metrics, runs hypothesis tests, and generates plots for all available models.

Usage:
    conda activate ctestenv
    python scripts/run_evaluation.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


def main():
    import numpy as np
    import pandas as pd
    from src.config import (
        DATA_PROCESSED, OUTPUT_MODELS, OUTPUT_FIGURES, OUTPUT_TABLES,
        TRAIN_RATIO, VAL_RATIO,
    )
    from src.evaluation import (
        all_metrics, metrics_table,
        generate_eda_plots, generate_result_plots,
        plot_residuals,
    )

    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    logger.info("Loading features and target...")
    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    target = pd.read_parquet(DATA_PROCESSED / "target.parquet")["realized_volatility"]

    # Align and split
    common_idx = features.index.intersection(target.dropna().index)
    target = target.loc[common_idx]
    n = len(common_idx)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_actual = target.iloc[val_end:]

    print(f"Test set: {len(test_actual)} samples")
    print(f"Test period: {test_actual.index.min()} to {test_actual.index.max()}")

    # --- Load available model predictions ---
    results = {}

    # GARCH
    garch_path = OUTPUT_MODELS / "garch_predictions.pkl"
    if garch_path.exists():
        logger.info("Loading GARCH predictions...")
        garch_preds = pd.read_pickle(garch_path)
        overlap = test_actual.index.intersection(garch_preds.index)
        actual_g = test_actual.loc[overlap].values
        pred_g = garch_preds.loc[overlap].values
        results["GARCH"] = all_metrics(actual_g, pred_g)
        print(f"\nGARCH: {len(overlap)} overlapping test samples")

    # XGBoost
    xgb_path = OUTPUT_MODELS / "xgboost_predictions.pkl"
    if xgb_path.exists():
        logger.info("Loading XGBoost predictions...")
        xgb_preds = pd.read_pickle(xgb_path)
        overlap = test_actual.index.intersection(xgb_preds.index)
        actual_x = test_actual.loc[overlap].values
        pred_x = xgb_preds.loc[overlap].values
        results["XGBoost"] = all_metrics(actual_x, pred_x)
        print(f"\nXGBoost: {len(overlap)} overlapping test samples")

    # LSTM — load all variants
    import glob
    lstm_files = sorted(glob.glob(str(OUTPUT_MODELS / "lstm_predictions_*.pkl")))
    if not lstm_files:
        # Fallback to single file
        lstm_path = OUTPUT_MODELS / "lstm_predictions.pkl"
        if lstm_path.exists():
            lstm_files = [str(lstm_path)]

    for lstm_file in lstm_files:
        tag = Path(lstm_file).stem.replace("lstm_predictions_", "").replace("lstm_predictions", "LSTM")
        label = f"LSTM_{tag}" if tag != "LSTM" else "LSTM"
        logger.info(f"Loading {label} predictions...")
        lstm_preds = pd.read_pickle(lstm_file)
        overlap = test_actual.index.intersection(lstm_preds.index)
        if len(overlap) > 0:
            actual_l = test_actual.loc[overlap].values
            pred_l = lstm_preds.loc[overlap].values
            results[label] = all_metrics(actual_l, pred_l)
            print(f"\n{label}: {len(overlap)} overlapping test samples")

    # --- Metrics table ---
    if results:
        logger.info("=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)

        mtable = metrics_table(results)
        print(f"\n{mtable.to_string()}")

        # Save table
        mtable.to_csv(OUTPUT_TABLES / "metrics.csv")
        with open(OUTPUT_TABLES / "metrics.tex", "w") as f:
            f.write(mtable.to_latex(float_format="%.6f"))
        logger.info(f"Metrics saved to {OUTPUT_TABLES}")

        # --- Residual plots for each model ---
        if "GARCH" in results:
            overlap = test_actual.index.intersection(garch_preds.index)
            plot_residuals(
                test_actual.loc[overlap].values,
                garch_preds.loc[overlap].values,
                "GARCH",
                OUTPUT_FIGURES / "residuals_garch.png",
            )

        if "XGBoost" in results:
            overlap = test_actual.index.intersection(xgb_preds.index)
            plot_residuals(
                test_actual.loc[overlap].values,
                xgb_preds.loc[overlap].values,
                "XGBoost",
                OUTPUT_FIGURES / "residuals_xgboost.png",
            )

        if "LSTM" in results:
            overlap = test_actual.index.intersection(lstm_preds.index)
            plot_residuals(
                test_actual.loc[overlap].values,
                lstm_preds.loc[overlap].values,
                "LSTM",
                OUTPUT_FIGURES / "residuals_lstm.png",
            )

        logger.info(f"Plots saved to {OUTPUT_FIGURES}")

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
