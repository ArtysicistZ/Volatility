"""
XGBoost ablation analysis: feature importance + feature group ablation.
Uses the already-tuned best params from the 525K dataset.

Usage:
    conda activate ctestenv  (or source venv on cluster)
    python scripts/run_xgb_ablation.py
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
    import xgboost as xgb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.config import (
        DATA_PROCESSED, OUTPUT_MODELS, OUTPUT_FIGURES, OUTPUT_TABLES,
        TRAIN_RATIO, VAL_RATIO,
    )
    from src.evaluation import mse, rmse, mae, r_squared

    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    # --- Load data (525K dataset) ---
    logger.info("Loading data...")
    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    target = pd.read_parquet(DATA_PROCESSED / "target.parquet")["realized_volatility"]

    common_idx = features.index.intersection(target.dropna().index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]

    n = len(features)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train = features.iloc[:train_end].values
    y_train = target.iloc[:train_end].values
    X_val = features.iloc[train_end:val_end].values
    y_val = target.iloc[train_end:val_end].values
    X_test = features.iloc[val_end:].values
    y_test = target.iloc[val_end:].values
    feature_names = list(features.columns)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {len(feature_names)}")

    # --- Load best params ---
    params_path = OUTPUT_MODELS / "xgboost_best_params.pkl"
    if not params_path.exists():
        raise FileNotFoundError("Run run_baselines.py --xgb-only first")
    with open(params_path, "rb") as f:
        best_params = pickle.load(f)
    print(f"\nBest params: {best_params}")

    # ================================================================
    # 1. FEATURE IMPORTANCE
    # ================================================================
    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 60)

    # Train on full train set with best params
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Get importances (gain-based)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\nTop 20 features by importance:")
    print(importance_df.head(20).to_string(index=False))

    importance_df.to_csv(OUTPUT_TABLES / "xgb_feature_importance.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 12))
    imp_sorted = importance_df.sort_values("importance", ascending=True)
    ax.barh(imp_sorted["feature"], imp_sorted["importance"], color="steelblue")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ================================================================
    # 2. FEATURE GROUP ABLATION
    # ================================================================
    logger.info("=" * 60)
    logger.info("FEATURE GROUP ABLATION")
    logger.info("=" * 60)

    # Define feature groups
    feature_groups = {
        "All features": [],  # baseline, remove nothing
        "No volatility estimators": [c for c in feature_names if "parkinson" in c or "garman" in c],
        "No rolling stats": [c for c in feature_names if c.startswith("ret_")],
        "No lag features": [c for c in feature_names if c.startswith("lag_")],
        "No volume features": [c for c in feature_names if "volume" in c or "obv" in c or "taker" in c],
        "No time encoding": [c for c in feature_names if "hour_" in c],
        "No ATR/Bollinger": [c for c in feature_names if c in ("atr", "bollinger_width")],
        "Only lag features": [c for c in feature_names if not c.startswith("lag_")],
        "Only rolling stats": [c for c in feature_names if not c.startswith("ret_")],
    }

    ablation_results = []
    for group_name, removed_features in feature_groups.items():
        kept_cols = [i for i, c in enumerate(feature_names) if c not in removed_features]
        n_kept = len(kept_cols)

        X_tr = X_train[:, kept_cols]
        X_va = X_val[:, kept_cols]
        X_te = X_test[:, kept_cols]

        model_ab = xgb.XGBRegressor(**best_params)
        model_ab.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
        preds = model_ab.predict(X_te)

        r2 = r_squared(y_test, preds)
        mse_val = mse(y_test, preds)
        rmse_val = rmse(y_test, preds)

        ablation_results.append({
            "group": group_name,
            "features_removed": len(removed_features),
            "features_kept": n_kept,
            "R2": r2,
            "MSE": mse_val,
            "RMSE": rmse_val,
        })
        print(f"  {group_name:30s}  removed={len(removed_features):2d}  kept={n_kept:2d}  R2={r2:.4f}")

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(OUTPUT_TABLES / "xgb_ablation.csv", index=False)

    # Plot ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    ab_sorted = ablation_df.sort_values("R2", ascending=True)
    colors = ["green" if g == "All features" else "steelblue" for g in ab_sorted["group"]]
    ax.barh(ab_sorted["group"], ab_sorted["R2"], color=colors)
    ax.set_xlabel("Test R²")
    ax.set_title("XGBoost Feature Group Ablation")
    ax.axvline(ablation_df[ablation_df["group"] == "All features"]["R2"].values[0],
               color="red", linestyle="--", alpha=0.7, label="All features baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "xgb_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ================================================================
    # 3. TOP-K FEATURE ANALYSIS
    # ================================================================
    logger.info("=" * 60)
    logger.info("TOP-K FEATURE ANALYSIS")
    logger.info("=" * 60)

    topk_results = []
    for k in [5, 10, 15, 20, 30, 40, 52]:
        top_features = importance_df.head(k)["feature"].tolist()
        kept_cols = [i for i, c in enumerate(feature_names) if c in top_features]

        X_tr = X_train[:, kept_cols]
        X_va = X_val[:, kept_cols]
        X_te = X_test[:, kept_cols]

        model_k = xgb.XGBRegressor(**best_params)
        model_k.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
        preds = model_k.predict(X_te)

        r2 = r_squared(y_test, preds)
        topk_results.append({"top_k": k, "R2": r2})
        print(f"  Top-{k:2d} features: R2={r2:.4f}")

    topk_df = pd.DataFrame(topk_results)
    topk_df.to_csv(OUTPUT_TABLES / "xgb_topk.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(topk_df["top_k"], topk_df["R2"], "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Number of Features (Top-K)")
    ax.set_ylabel("Test R²")
    ax.set_title("XGBoost: R² vs Number of Features")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "xgb_topk.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("=" * 60)
    logger.info("ABLATION ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
