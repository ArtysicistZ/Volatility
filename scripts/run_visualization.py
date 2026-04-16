"""
Comprehensive visualization script for CIS 5450 Volatility Prediction.
Generates 15 presentation-quality plots covering raw data, EDA, features, and model results.

Usage:
    conda activate ctestenv
    python scripts/run_visualization.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import sys
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from src.config import (
    DATA_PROCESSED, OUTPUT_MODELS, OUTPUT_FIGURES, OUTPUT_TABLES,
    TRAIN_RATIO, VAL_RATIO,
)
from src.evaluation import (
    plot_price_volume, plot_return_dist, plot_qq,
    plot_acf_squared, plot_rv_timeseries, plot_correlation,
    plot_rolling_stats, plot_residuals, plot_metrics_bar, all_metrics,
)
from src.feature_engineering import log_returns

# Consistent style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")


# ═══════════════════════════════════════════════════════════════════════════
# New Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_predictions_multi(pred_df, save_path=None):
    """Overlay predicted vs actual realized volatility for arbitrary models."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(pred_df.index, pred_df["actual"], color="black",
            linewidth=0.5, alpha=0.7, label="Actual")

    model_cols = [c for c in pred_df.columns if c != "actual"]
    colors = sns.color_palette("deep", len(model_cols))

    for col, color in zip(model_cols, colors):
        ax.plot(pred_df.index, pred_df[col], color=color,
                linewidth=0.5, alpha=0.7, label=col)

    ax.set_title("Predicted vs Actual Realized Volatility (Test Set)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Realized Volatility")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cumulative_error_multi(pred_df, save_path=None):
    """Cumulative squared error over time for arbitrary models."""
    fig, ax = plt.subplots(figsize=(14, 5))

    model_cols = [c for c in pred_df.columns if c != "actual"]
    colors = sns.color_palette("deep", len(model_cols))

    for col, color in zip(model_cols, colors):
        cum_se = ((pred_df["actual"] - pred_df[col]) ** 2).cumsum()
        ax.plot(pred_df.index, cum_se, color=color, linewidth=1, label=col)

    ax.set_title("Cumulative Squared Error (Test Set)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative SE")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_heatmap(metrics_df, save_path=None):
    """
    Render full metrics table as a color-coded heatmap.
    Green = better for each metric (inverted for lower-is-better).
    """
    lower_is_better = ["MSE", "RMSE", "MAE", "QLIKE"]
    higher_is_better = ["R2", "DA"]

    # Normalize each column to [0, 1] for coloring
    norm_df = metrics_df.copy()
    for col in metrics_df.columns:
        col_min = metrics_df[col].min()
        col_max = metrics_df[col].max()
        rng = col_max - col_min
        if rng < 1e-20:
            norm_df[col] = 0.5
        elif col in lower_is_better:
            norm_df[col] = 1 - (metrics_df[col] - col_min) / rng  # invert
        else:
            norm_df[col] = (metrics_df[col] - col_min) / rng

    # Build formatted annotation matrix
    annot = metrics_df.copy().astype(str)
    for col in metrics_df.columns:
        if col == "MSE":
            annot[col] = metrics_df[col].apply(lambda v: f"{v:.2e}")
        elif col in ("RMSE", "MAE"):
            annot[col] = metrics_df[col].apply(lambda v: f"{v:.6f}")
        else:
            annot[col] = metrics_df[col].apply(lambda v: f"{v:.4f}")

    fig, ax = plt.subplots(figsize=(12, max(6, len(metrics_df) * 0.55)))
    sns.heatmap(
        norm_df, annot=annot.values, fmt="",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        xticklabels=metrics_df.columns,
        yticklabels=metrics_df.index,
        cbar_kws={"label": "Relative Performance (green = better)"},
        ax=ax,
    )
    ax.set_title("Model Comparison: All Metrics", fontsize=14, pad=12)
    ax.set_ylabel("")
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    plot_num = 0

    # ================================================================
    # SECTION 1: Raw Data (3 plots)
    # ================================================================
    print("\n" + "=" * 60)
    print("SECTION 1: Raw Data (3 plots)")
    print("=" * 60)

    logger.info("Loading cleaned data...")
    df = pd.read_parquet(DATA_PROCESSED / "btcusdt_1m_clean.parquet")
    returns = log_returns(df["close"])

    plot_num += 1
    print(f"  [{plot_num}/15] Generating price + volume plot...")
    plot_price_volume(df, OUTPUT_FIGURES / "01_price_volume.png")
    plt.close("all")

    plot_num += 1
    print(f"  [{plot_num}/15] Generating return distribution plot...")
    plot_return_dist(returns, OUTPUT_FIGURES / "02_return_distribution.png")
    plt.close("all")

    plot_num += 1
    print(f"  [{plot_num}/15] Generating Q-Q plot...")
    plot_qq(returns, OUTPUT_FIGURES / "03_qq_plot.png")
    plt.close("all")

    # ================================================================
    # SECTION 2: Data Processing & EDA (4 plots)
    # ================================================================
    print("\n" + "=" * 60)
    print("SECTION 2: Data Processing & EDA (4 plots)")
    print("=" * 60)

    plot_num += 1
    print(f"  [{plot_num}/15] Generating ACF plot...")
    plot_acf_squared(returns, save_path=OUTPUT_FIGURES / "04_acf_squared.png")
    plt.close("all")

    plot_num += 1
    print(f"  [{plot_num}/15] Generating realized volatility time series...")
    target = pd.read_parquet(DATA_PROCESSED / "target.parquet")["realized_volatility"]
    plot_rv_timeseries(target.dropna(), OUTPUT_FIGURES / "05_realized_volatility.png")
    plt.close("all")

    plot_num += 1
    print(f"  [{plot_num}/15] Generating feature correlation heatmap...")
    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    plot_correlation(features, OUTPUT_FIGURES / "06_correlation.png")
    del features
    gc.collect()
    plt.close("all")

    plot_num += 1
    print(f"  [{plot_num}/15] Generating rolling stats plot...")
    plot_rolling_stats(df, OUTPUT_FIGURES / "07_rolling_stats.png")
    plt.close("all")

    # Free large objects
    del df, returns
    gc.collect()

    # ================================================================
    # SECTION 3: Feature Analysis (3 plots)
    # ================================================================
    print("\n" + "=" * 60)
    print("SECTION 3: Feature Analysis (3 plots)")
    print("=" * 60)

    # --- Feature Importance ---
    plot_num += 1
    print(f"  [{plot_num}/15] Generating XGBoost feature importance...")
    imp_path = OUTPUT_TABLES / "xgb_feature_importance.csv"
    if imp_path.exists():
        importance_df = pd.read_csv(imp_path)
        fig, ax = plt.subplots(figsize=(10, 12))
        imp_sorted = importance_df.sort_values("importance", ascending=True)
        ax.barh(imp_sorted["feature"], imp_sorted["importance"], color="steelblue")
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title("XGBoost Feature Importance")
        fig.tight_layout()
        fig.savefig(OUTPUT_FIGURES / "08_xgb_feature_importance.png",
                    dpi=150, bbox_inches="tight")
        plt.close("all")
    else:
        print("    WARNING: xgb_feature_importance.csv not found, skipping")

    # --- Feature Group Ablation ---
    plot_num += 1
    print(f"  [{plot_num}/15] Generating feature group ablation...")
    abl_path = OUTPUT_TABLES / "xgb_ablation.csv"
    if abl_path.exists():
        ablation_df = pd.read_csv(abl_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        ab_sorted = ablation_df.sort_values("R2", ascending=True)
        colors = ["green" if g == "All features" else "steelblue"
                  for g in ab_sorted["group"]]
        ax.barh(ab_sorted["group"], ab_sorted["R2"], color=colors)
        ax.set_xlabel("Test R²")
        ax.set_title("XGBoost Feature Group Ablation")
        baseline_r2 = ablation_df[ablation_df["group"] == "All features"]["R2"].values
        if len(baseline_r2) > 0:
            ax.axvline(baseline_r2[0], color="red", linestyle="--", alpha=0.7,
                       label="All features baseline")
            ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_FIGURES / "09_xgb_ablation.png",
                    dpi=150, bbox_inches="tight")
        plt.close("all")
    else:
        print("    WARNING: xgb_ablation.csv not found, skipping")

    # --- Top-K Features ---
    plot_num += 1
    print(f"  [{plot_num}/15] Generating top-K features curve...")
    topk_path = OUTPUT_TABLES / "xgb_topk.csv"
    if topk_path.exists():
        topk_df = pd.read_csv(topk_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(topk_df["top_k"], topk_df["R2"], "o-",
                color="steelblue", linewidth=2, markersize=6)
        ax.set_xlabel("Number of Features (Top-K)")
        ax.set_ylabel("Test R²")
        ax.set_title("XGBoost: R² vs Number of Features")
        fig.tight_layout()
        fig.savefig(OUTPUT_FIGURES / "10_xgb_topk.png",
                    dpi=150, bbox_inches="tight")
        plt.close("all")
    else:
        print("    WARNING: xgb_topk.csv not found, skipping")

    # ================================================================
    # SECTION 4: Model Results (5 plots)
    # ================================================================
    print("\n" + "=" * 60)
    print("SECTION 4: Model Results (5 plots)")
    print("=" * 60)

    # --- Load metrics to find best LSTM ---
    metrics_path = OUTPUT_TABLES / "metrics.csv"
    if not metrics_path.exists():
        print("ERROR: metrics.csv not found. Run run_evaluation.py first.")
        return

    full_metrics = pd.read_csv(metrics_path, index_col=0)

    # Find best LSTM (lowest MSE among LSTM variants, excluding transformers)
    lstm_models = [m for m in full_metrics.index if m.startswith("LSTM_seq")]
    if lstm_models:
        best_lstm_name = full_metrics.loc[lstm_models, "MSE"].idxmin()
    else:
        best_lstm_name = None

    print(f"  Best LSTM model: {best_lstm_name} "
          f"(MSE={full_metrics.loc[best_lstm_name, 'MSE']:.2e})" if best_lstm_name else "")

    # --- Load predictions ---
    # Load target and compute test split
    target = pd.read_parquet(DATA_PROCESSED / "target.parquet")["realized_volatility"]
    features_idx = pd.read_parquet(DATA_PROCESSED / "features.parquet",
                                   columns=[]).index
    common_idx = features_idx.intersection(target.dropna().index)
    target = target.loc[common_idx]
    n = len(common_idx)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_actual = target.iloc[val_end:]
    del features_idx
    gc.collect()

    # Load GARCH
    pred_dict = {}
    garch_path = OUTPUT_MODELS / "garch_predictions.pkl"
    if garch_path.exists():
        garch_preds = pd.read_pickle(garch_path)
        pred_dict["GARCH"] = garch_preds

    # Load XGBoost
    xgb_path = OUTPUT_MODELS / "xgboost_predictions.pkl"
    if xgb_path.exists():
        xgb_preds = pd.read_pickle(xgb_path)
        pred_dict["XGBoost"] = xgb_preds

    # Load best LSTM
    best_lstm_preds = None
    if best_lstm_name:
        # Convert model name to filename: "LSTM_seq120_attn" -> "lstm_predictions_seq120_attn.pkl"
        tag = best_lstm_name.replace("LSTM_", "")
        lstm_path = OUTPUT_MODELS / f"lstm_predictions_{tag}.pkl"
        if lstm_path.exists():
            best_lstm_preds = pd.read_pickle(lstm_path)
            pred_dict[f"Best LSTM ({tag})"] = best_lstm_preds

    # Build aligned pred_df
    if pred_dict:
        pred_df = pd.DataFrame({"actual": test_actual})
        for name, preds in pred_dict.items():
            pred_df[name] = preds
        pred_df = pred_df.dropna()
        print(f"  Aligned test predictions: {len(pred_df)} common timestamps")

        # --- Plot 11: Predictions Overlay ---
        plot_num += 1
        print(f"  [{plot_num}/15] Generating predictions overlay...")
        plot_predictions_multi(pred_df,
                               OUTPUT_FIGURES / "11_predictions_overlay.png")
        plt.close("all")

        # --- Plot 12: Metrics Comparison Bar Chart (3 key models) ---
        plot_num += 1
        print(f"  [{plot_num}/15] Generating metrics comparison bar chart...")
        # Build a subset metrics table for the 3 key models
        subset_results = {}
        for name in pred_dict:
            overlap = test_actual.index.intersection(pred_dict[name].index)
            subset_results[name] = all_metrics(
                test_actual.loc[overlap].values,
                pred_dict[name].loc[overlap].values,
            )
        mtable_subset = pd.DataFrame(subset_results).T
        mtable_subset.index.name = "Model"
        plot_metrics_bar(mtable_subset,
                         OUTPUT_FIGURES / "12_metrics_comparison.png")
        plt.close("all")

        # --- Plot 13: Cumulative Error ---
        plot_num += 1
        print(f"  [{plot_num}/15] Generating cumulative error plot...")
        plot_cumulative_error_multi(pred_df,
                                    OUTPUT_FIGURES / "13_cumulative_error.png")
        plt.close("all")

        # --- Plot 14: Residual Analysis for Best LSTM ---
        plot_num += 1
        if best_lstm_preds is not None:
            print(f"  [{plot_num}/15] Generating residual analysis for best LSTM...")
            overlap = test_actual.index.intersection(best_lstm_preds.index)
            plot_residuals(
                test_actual.loc[overlap].values,
                best_lstm_preds.loc[overlap].values,
                best_lstm_name,
                OUTPUT_FIGURES / "14_residuals_best_lstm.png",
            )
            plt.close("all")
        else:
            print(f"  [{plot_num}/15] Skipping residual analysis (no LSTM predictions)")

    else:
        print("  WARNING: No prediction files found, skipping plots 11–14")
        plot_num += 4

    # --- Plot 15: Full Model Comparison Heatmap ---
    plot_num += 1
    print(f"  [{plot_num}/15] Generating model comparison heatmap...")
    # Exclude GARCH from heatmap if its R2 is extremely negative (distorts scale)
    heatmap_metrics = full_metrics.copy()
    if "GARCH" in heatmap_metrics.index and heatmap_metrics.loc["GARCH", "R2"] < -1:
        # Show GARCH separately — its R2=-14.5 destroys the color scale
        print("    Note: GARCH excluded from heatmap (R²=-14.5 distorts color scale)")
        heatmap_metrics = heatmap_metrics.drop("GARCH")
    plot_metrics_heatmap(heatmap_metrics,
                         OUTPUT_FIGURES / "15_model_comparison_heatmap.png")
    plt.close("all")

    # ================================================================
    # DONE
    # ================================================================
    print("\n" + "=" * 60)
    print(f"COMPLETE: 15 figures saved to {OUTPUT_FIGURES}")
    print("=" * 60)

    # List generated files
    pngs = sorted(OUTPUT_FIGURES.glob("*.png"))
    for p in pngs:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:40s}  {size_kb:>7.0f} KB")


if __name__ == "__main__":
    main()
