"""
Generate the polished, README-grade plots for the resume version of the README.

Conda env: ctestenv
Run:
    conda run -n ctestenv python scripts/run_resume_plots.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import gc
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_PROCESSED, OUTPUT_MODELS, OUTPUT_FIGURES, OUTPUT_TABLES,
    TRAIN_RATIO, VAL_RATIO, COLLECTION_START, COLLECTION_END,
)
from src.feature_engineering import log_returns

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# House style — slightly tighter than the course default, bigger fonts.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "DejaVu Sans",
})
plt.style.use("seaborn-v0_8-whitegrid")
NAVY = "#1f3a68"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
RED = "#c1121f"
GREY = "#6c757d"


# ───────────────────────────────────────────────────────────────────────────
# Hero: BTC price + realized volatility, two stacked panels
# ───────────────────────────────────────────────────────────────────────────

def plot_hero(price: pd.Series, rv: pd.Series, split_idx: pd.Timestamp,
              save_path: Path) -> None:
    # Down-sample minute data to hourly for clean rendering.
    price_h = price.resample("1h").last().dropna()
    # 10-min RV is in return units; annualize to a more readable scale
    rv_h = rv.resample("1h").mean().dropna()

    fig, (ax_p, ax_v) = plt.subplots(
        2, 1, figsize=(13, 6.5),
        gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.08},
        sharex=True,
    )

    ax_p.plot(price_h.index, price_h.values, color=NAVY, linewidth=1.2)
    ax_p.set_ylabel("BTC / USDT  (USD)")
    ax_p.set_title("Bitcoin price and 10-minute realized volatility  ·  Apr 2024 – Mar 2025  ·  525,600 1-min observations")
    ax_p.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax_p.axvline(split_idx, color=RED, linestyle="--", alpha=0.75, linewidth=1.2)
    y_top = ax_p.get_ylim()[1]
    ax_p.annotate("test set →", xy=(split_idx, y_top * 0.93),
                  xytext=(8, 0), textcoords="offset points",
                  color=RED, fontsize=10, va="top", ha="left", fontweight="bold")

    ax_v.fill_between(rv_h.index, 0, rv_h.values * 100, color=TEAL,
                      alpha=0.85, linewidth=0)
    ax_v.set_ylabel("Realized vol  (%)")
    ax_v.set_xlabel("")
    ax_v.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    ax_v.axvline(split_idx, color=RED, linestyle="--", alpha=0.75, linewidth=1.2)

    for ax in (ax_p, ax_v):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.tick_params(axis="x", which="major", length=6)
        ax.tick_params(axis="x", which="minor", length=3)
    ax_p.tick_params(axis="x", which="both", labelbottom=False)
    plt.setp(ax_v.get_xticklabels(), rotation=0, ha="center", fontsize=11)
    ax_v.margins(x=0.005)
    ax_p.margins(x=0.005)

    fig.savefig(save_path)
    plt.close(fig)
    log.info("Wrote %s", save_path.name)


# ───────────────────────────────────────────────────────────────────────────
# Volatility clustering ACF — squared returns
# ───────────────────────────────────────────────────────────────────────────

def plot_acf_clean(returns: pd.Series, save_path: Path, max_lag: int = 60) -> None:
    r = returns.dropna().values
    r2 = r ** 2
    mean = r2.mean()
    var = ((r2 - mean) ** 2).mean()
    acf = np.array([
        ((r2[:-lag] - mean) * (r2[lag:] - mean)).mean() / var
        for lag in range(1, max_lag + 1)
    ])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    n = len(r2)
    conf = 1.96 / np.sqrt(n)
    ax.axhspan(-conf, conf, color=GREY, alpha=0.18, label="95% noise band")

    ax.vlines(range(1, max_lag + 1), 0, acf, color=NAVY, linewidth=1.6, alpha=0.9)
    ax.scatter(range(1, max_lag + 1), acf, color=NAVY, s=18, zorder=3)
    ax.axhline(0, color="black", linewidth=0.7)

    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel("Autocorrelation of  $r_t^2$")
    ax.set_title("Volatility clustering: ACF of squared returns persists for >60 minutes")
    ax.set_xlim(0, max_lag + 1)
    ax.legend(loc="upper right")

    fig.savefig(save_path)
    plt.close(fig)
    log.info("Wrote %s", save_path.name)


# ───────────────────────────────────────────────────────────────────────────
# Hexbin scatter: predicted vs actual for best model
# ───────────────────────────────────────────────────────────────────────────

def plot_fit_scatter(actual: pd.Series, pred: pd.Series, model_name: str,
                     r2: float, save_path: Path) -> None:
    common = actual.index.intersection(pred.index)
    y = actual.loc[common].values
    yhat = pred.loc[common].values

    # Trim extreme actual percentiles to keep the hexbin readable.
    hi = np.quantile(y, 0.999)
    mask = (y < hi) & (yhat < hi)
    y, yhat = y[mask], yhat[mask]

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    hb = ax.hexbin(y, yhat, gridsize=70, cmap="Blues", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="log(count)")

    lim = max(y.max(), yhat.max()) * 1.02
    ax.plot([0, lim], [0, lim], color=RED, linewidth=1.5,
            linestyle="--", label="Perfect prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual realized volatility  (next 10 min)")
    ax.set_ylabel("Predicted realized volatility")
    ax.set_title(f"{model_name}: predicted vs actual  ·  R² = {r2:.3f}")
    ax.legend(loc="lower right")

    fig.savefig(save_path)
    plt.close(fig)
    log.info("Wrote %s", save_path.name)


# ───────────────────────────────────────────────────────────────────────────
# Vol-spike zoom: 36-hour window during peak test-set turbulence
# ───────────────────────────────────────────────────────────────────────────

def plot_vol_spike_zoom(actual: pd.Series, preds: dict, save_path: Path) -> None:
    common = actual.index
    for s in preds.values():
        common = common.intersection(s.index)
    actual = actual.loc[common]

    # Find the densest spike window in actual: max sum over rolling 36h window.
    win = 36 * 60
    spike_center = actual.rolling(win, center=True).sum().idxmax()
    start = spike_center - pd.Timedelta(hours=18)
    end = spike_center + pd.Timedelta(hours=18)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(actual.loc[start:end], color="black", linewidth=1.2,
            alpha=0.85, label="Actual")
    colors = [TEAL, ORANGE, NAVY]
    for (name, p), c in zip(preds.items(), colors):
        ax.plot(p.loc[start:end], color=c, linewidth=1.4, alpha=0.85, label=name)

    ax.set_xlim(start, end)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.3f}%"))
    ax.set_ylabel("10-min realized volatility")
    ax.set_xlabel(f"Time around spike on {spike_center:%Y-%m-%d}")
    ax.set_title(f"Model behaviour during a volatility spike  ·  ±18h around {spike_center:%Y-%m-%d %H:%M} UTC")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.legend(loc="upper right", ncol=len(preds) + 1, framealpha=0.95)

    fig.savefig(save_path)
    plt.close(fig)
    log.info("Wrote %s", save_path.name)


# ───────────────────────────────────────────────────────────────────────────
# Top-20 feature importance
# ───────────────────────────────────────────────────────────────────────────

def plot_top_features(imp_df: pd.DataFrame, save_path: Path, top_n: int = 20) -> None:
    imp = imp_df.sort_values("importance", ascending=False).head(top_n)
    imp = imp.iloc[::-1]  # so largest is on top

    # Category coloring
    cat = []
    for f in imp["feature"]:
        if f.startswith("ret_std"):
            cat.append("Rolling std")
        elif "parkinson" in f or "garman" in f:
            cat.append("Range vol estimator")
        elif f.startswith("ret_"):
            cat.append("Rolling moment")
        elif "bollinger" in f or f == "atr":
            cat.append("Technical")
        elif "volume" in f or f == "obv" or f == "taker_ratio":
            cat.append("Volume")
        else:
            cat.append("Other")
    palette = {
        "Rolling std": NAVY,
        "Range vol estimator": TEAL,
        "Rolling moment": "#6699cc",
        "Technical": ORANGE,
        "Volume": GREY,
        "Other": "#9d4edd",
    }
    colors = [palette[c] for c in cat]

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp["feature"], imp["importance"], color=colors,
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Gain importance")
    ax.set_xlim(left=0)
    ax.set_title(f"Top {top_n} XGBoost features  ·  volatility estimators dominate")
    legend_handles = [Patch(facecolor=palette[c], label=c)
                      for c in palette if c in cat]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95)
    ax.margins(y=0.01)

    fig.savefig(save_path)
    plt.close(fig)
    log.info("Wrote %s", save_path.name)


# ───────────────────────────────────────────────────────────────────────────
# Model comparison heatmap — drops GARCH so the color scale is informative
# ───────────────────────────────────────────────────────────────────────────

def plot_model_heatmap(metrics_df: pd.DataFrame, save_path: Path) -> None:
    df = metrics_df.copy()
    if "GARCH" in df.index:
        df = df.drop("GARCH")
    df = df.sort_values("R2", ascending=False)

    lower_better = ["MSE", "RMSE", "MAE", "QLIKE"]
    higher_better = ["R2", "DA"]
    norm = df.copy().astype(float)
    for col in df.columns:
        col_min, col_max = df[col].min(), df[col].max()
        rng = col_max - col_min
        if rng < 1e-20:
            norm[col] = 0.5
        elif col in lower_better:
            norm[col] = 1 - (df[col] - col_min) / rng
        else:
            norm[col] = (df[col] - col_min) / rng

    annot = df.copy().astype(str)
    for col in df.columns:
        if col == "MSE":
            annot[col] = df[col].apply(lambda v: f"{v:.2e}")
        elif col in ("RMSE", "MAE"):
            annot[col] = df[col].apply(lambda v: f"{v:.1e}")
        else:
            annot[col] = df[col].apply(lambda v: f"{v:.3f}")

    fig, ax = plt.subplots(figsize=(11, max(5, len(df) * 0.45)))
    sns.heatmap(
        norm, annot=annot.values, fmt="",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.6, linecolor="white",
        xticklabels=df.columns, yticklabels=df.index,
        cbar_kws={"label": "Relative performance  (green = better)"},
        ax=ax,
    )
    ax.set_title("Model leaderboard  ·  11 deep + ML models, identical test set", pad=10)
    ax.set_ylabel("")
    plt.yticks(rotation=0)
    plt.xticks(fontsize=11)

    fig.savefig(save_path)
    plt.close(fig)
    log.info("Wrote %s", save_path.name)


# ───────────────────────────────────────────────────────────────────────────
# Walk-forward concept diagram
# ───────────────────────────────────────────────────────────────────────────

def plot_walkforward_concept(save_path: Path) -> None:
    """
    Stack expanding training spans (top) with their corresponding test window (bottom).
    """
    n_iter = 5
    fig, ax = plt.subplots(figsize=(11, 3.6))

    for k in range(n_iter):
        train_end = 1.0 + k * 0.5
        ax.barh(n_iter - k - 0.5, train_end, left=0, height=0.55,
                color=NAVY, alpha=0.85, edgecolor="white")
        ax.barh(n_iter - k - 0.5, 0.5, left=train_end, height=0.55,
                color=ORANGE, alpha=0.95, edgecolor="white")
        ax.text(train_end / 2, n_iter - k - 0.5,
                f"fit  ·  history → step {int((train_end)*1000):d}",
                ha="center", va="center", color="white", fontsize=9.5)
        ax.text(train_end + 0.25, n_iter - k - 0.5,
                "predict\n500 steps",
                ha="center", va="center", color="white", fontsize=8)

    ax.set_xlim(-0.15, 3.6)
    ax.set_ylim(-0.4, n_iter + 0.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    ax.set_title("Walk-forward expanding window  ·  refit every 500 steps  ·  no future leakage",
                 pad=12)
    ax.annotate("", xy=(3.55, -0.2), xytext=(-0.05, -0.2),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(1.7, -0.35, "time", fontsize=10, color=GREY,
            ha="center", va="top")

    # Legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=NAVY, alpha=0.85, label="Training window (all past data)"),
           Patch(facecolor=ORANGE, alpha=0.95, label="Forecast window (held out)")]
    ax.legend(handles=leg, loc="upper right", framealpha=0.95, fontsize=9)

    fig.savefig(save_path)
    plt.close(fig)
    log.info("Wrote %s", save_path.name)


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    log.info("Loading clean candles…")
    df = pd.read_parquet(DATA_PROCESSED / "btcusdt_1m_clean.parquet")

    log.info("Loading target…")
    target = pd.read_parquet(DATA_PROCESSED / "target.parquet")["realized_volatility"]

    # Restrict every series in this script to the resume-framed window
    # (Apr 2024 – Mar 2025, the 525K-row subset). The on-disk parquet
    # contains the full 8-year history; we mustn't draw any of it here.
    window_start = pd.Timestamp(COLLECTION_START, tz="UTC")
    window_end = pd.Timestamp(COLLECTION_END, tz="UTC") + pd.Timedelta(days=1)
    df = df.loc[window_start:window_end]
    target = target.loc[window_start:window_end].dropna()

    price = df["close"]
    returns = log_returns(price)

    # Reproduce the same 85/5/10 split the models used inside this window.
    n = len(target)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_actual = target.iloc[val_end:]
    test_start = test_actual.index[0]
    log.info("Window: %s → %s   (%d minute rows, %d target rows)",
             window_start.date(), window_end.date(), len(df), n)

    # 1. Hero
    plot_hero(price, target.dropna(), test_start,
              OUTPUT_FIGURES / "hero_price_volatility.png")

    # 2. ACF
    plot_acf_clean(returns, OUTPUT_FIGURES / "vol_clustering_acf.png")

    # 3. Top-20 importance
    imp_path = OUTPUT_TABLES / "xgb_feature_importance.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        plot_top_features(imp_df, OUTPUT_FIGURES / "feature_importance_top20.png", top_n=20)

    # 4. Walk-forward concept
    plot_walkforward_concept(OUTPUT_FIGURES / "walkforward_concept.png")

    # 5. Model leaderboard heatmap
    metrics = pd.read_csv(OUTPUT_TABLES / "metrics.csv", index_col=0)
    plot_model_heatmap(metrics, OUTPUT_FIGURES / "model_leaderboard.png")

    # Load predictions for fit-scatter and zoom
    pred_files = {
        "GARCH": OUTPUT_MODELS / "garch_predictions.pkl",
        "XGBoost": OUTPUT_MODELS / "xgboost_predictions.pkl",
        "LSTM": None,
    }
    # Pick best LSTM by MSE
    lstm_rows = [i for i in metrics.index if i.startswith("LSTM_seq")]
    if lstm_rows:
        best_lstm = metrics.loc[lstm_rows, "MSE"].idxmin()
        tag = best_lstm.replace("LSTM_", "")
        pred_files["LSTM"] = OUTPUT_MODELS / f"lstm_predictions_{tag}.pkl"
        best_lstm_label = f"LSTM ({tag})"
    else:
        best_lstm = None
        best_lstm_label = "LSTM"

    loaded = {}
    for name, p in pred_files.items():
        if p is None or not p.exists():
            continue
        loaded[name] = pd.read_pickle(p)

    # 6. Fit scatter for the best R² model (XGBoost in this run)
    best_overall = metrics["R2"].idxmax()
    if best_overall in loaded:
        plot_fit_scatter(
            test_actual, loaded[best_overall], best_overall,
            float(metrics.loc[best_overall, "R2"]),
            OUTPUT_FIGURES / "fit_scatter_best.png",
        )

    # 7. Vol-spike zoom (skip GARCH so y-axis isn't blown out)
    zoom_preds = {k: v for k, v in loaded.items() if k != "GARCH"}
    if zoom_preds:
        # Relabel best LSTM
        renamed = {}
        for k, v in zoom_preds.items():
            if k == "LSTM":
                renamed[best_lstm_label] = v
            else:
                renamed[k] = v
        plot_vol_spike_zoom(test_actual, renamed,
                            OUTPUT_FIGURES / "vol_spike_zoom.png")

    log.info("Done.")
    for p in sorted(OUTPUT_FIGURES.glob("*.png")):
        size_kb = p.stat().st_size / 1024
        log.info("  %-40s %7.0f KB", p.name, size_kb)


if __name__ == "__main__":
    main()
