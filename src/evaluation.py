"""
Evaluation: metrics, block bootstrap hypothesis testing, and visualization.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import (
    BOOT_REPS, BOOT_BLOCK, BOOT_CONF, RANDOM_SEED, OUTPUT_FIGURES,
)

logger = logging.getLogger(__name__)

# Default style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((actual - predicted) ** 2))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(actual, predicted)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Quasi-Likelihood loss (Patton 2011).
    QLIKE = mean(actual_var / predicted_var - log(actual_var / predicted_var) - 1)

    PITFALL: Requires variance inputs (square volatilities first).
    Predicted variance must be strictly positive.
    """
    # Square to get variances
    actual_var = actual ** 2
    pred_var = predicted ** 2
    # Clip predicted variance to avoid division by zero
    pred_var = np.clip(pred_var, 1e-20, None)
    ratio = actual_var / pred_var
    return float(np.mean(ratio - np.log(ratio) - 1))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R-squared: 1 - SS_res / SS_tot."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot < 1e-20:
        return 0.0
    return float(1 - ss_res / ss_tot)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of times direction of change is correctly predicted."""
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    if len(actual_dir) == 0:
        return 0.0
    return float(np.mean(actual_dir == pred_dir))


def all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute all metrics for a single model."""
    return {
        "MSE": mse(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "QLIKE": qlike(actual, predicted),
        "R2": r_squared(actual, predicted),
        "DA": directional_accuracy(actual, predicted),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Format metrics into comparison table.
    Input: {"GARCH": {metrics}, "XGBoost": {metrics}, "LSTM": {metrics}}
    Returns formatted DataFrame. Also generates LaTeX string.
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df


# ═══════════════════════════════════════════════════════════════════════════
# HYPOTHESIS TESTING (Block Bootstrap)
# ═══════════════════════════════════════════════════════════════════════════

def block_bootstrap_test(
    baseline_errors: np.ndarray,
    lstm_errors: np.ndarray,
    block_size: int = BOOT_BLOCK,
    n_reps: int = BOOT_REPS,
    confidence: float = BOOT_CONF,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Test H0: MSE_LSTM >= MSE_baseline vs H1: MSE_LSTM < MSE_baseline.

    Method:
    1. d_t = baseline_error_t^2 - lstm_error_t^2
    2. MovingBlockBootstrap on d_t series
    3. For each bootstrap sample, compute mean(d_t*)
    4. p-value = fraction of bootstrap deltas <= 0
    5. If 0 outside CI, reject H0

    Returns dict with observed_delta, CI, p_value, reject_null, bootstrap_deltas.
    """
    from arch.bootstrap import MovingBlockBootstrap

    # Squared errors
    d = baseline_errors ** 2 - lstm_errors ** 2
    observed_delta = float(np.mean(d))

    # Bootstrap
    rng = np.random.RandomState(seed)
    bs = MovingBlockBootstrap(block_size, d, seed=rng)

    boot_deltas = []
    for data, in bs.bootstrap(n_reps):
        boot_deltas.append(float(np.mean(data[0])))
    boot_deltas = np.array(boot_deltas)

    # Confidence interval
    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))

    # p-value: fraction of bootstrap deltas <= 0
    p_value = float(np.mean(boot_deltas <= 0))

    reject_null = p_value < alpha

    result = {
        "observed_delta": observed_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "reject_null": reject_null,
        "bootstrap_deltas": boot_deltas,
    }

    logger.info(
        f"Bootstrap test: delta={observed_delta:.6f}, "
        f"CI=[{ci_lower:.6f}, {ci_upper:.6f}], "
        f"p={p_value:.4f}, reject_H0={reject_null}"
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# EDA PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_price_volume(df: pd.DataFrame, save_path: Path | None = None):
    """Dual-axis plot: BTC close price + volume."""
    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(df.index, df["close"], color="steelblue", linewidth=0.5, label="Close")
    ax1.set_ylabel("Price (USDT)", color="steelblue")
    ax1.set_xlabel("Time")

    ax2 = ax1.twinx()
    ax2.bar(df.index, df["volume"], alpha=0.3, color="orange", width=0.001, label="Volume")
    ax2.set_ylabel("Volume", color="orange")

    ax1.set_title("BTC/USDT Price and Volume (1-min)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_return_dist(returns: pd.Series, save_path: Path | None = None):
    """Histogram + KDE of log returns with normal overlay. Annotate kurtosis."""
    fig, ax = plt.subplots(figsize=(10, 5))

    returns_clean = returns.dropna()
    ax.hist(returns_clean, bins=200, density=True, alpha=0.6, color="steelblue",
            label="Log Returns")

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(returns_clean)
    x_range = np.linspace(returns_clean.min(), returns_clean.max(), 500)
    ax.plot(x_range, kde(x_range), color="darkblue", linewidth=2, label="KDE")

    # Normal overlay
    mu, sigma = returns_clean.mean(), returns_clean.std()
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    ax.plot(x_range, normal_pdf, color="red", linestyle="--", linewidth=2,
            label=f"Normal(mu={mu:.6f}, sigma={sigma:.6f})")

    kurt = returns_clean.kurtosis()
    ax.annotate(f"Excess Kurtosis: {kurt:.2f}", xy=(0.02, 0.95),
                xycoords="axes fraction", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_title("Distribution of 1-Minute Log Returns")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_qq(returns: pd.Series, save_path: Path | None = None):
    """Q-Q plot against normal distribution. Fat tails visible."""
    fig, ax = plt.subplots(figsize=(7, 7))
    stats.probplot(returns.dropna(), dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of Log Returns vs Normal Distribution")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_acf_squared(returns: pd.Series, lags: int = 100,
                     save_path: Path | None = None):
    """ACF of squared returns to show volatility clustering."""
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(returns.dropna(), lags=lags, ax=axes[0], title="ACF of Returns")
    plot_acf(returns.dropna() ** 2, lags=lags, ax=axes[1],
             title="ACF of Squared Returns (Volatility Clustering)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rv_timeseries(rv: pd.Series, save_path: Path | None = None):
    """Time series of realized volatility (the target variable)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rv.index, rv.values, linewidth=0.5, color="crimson")
    ax.set_title("Realized Volatility (10-min window)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Realized Volatility")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation(features: pd.DataFrame, save_path: Path | None = None):
    """Feature correlation heatmap."""
    corr = features.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title("Feature Correlation Matrix")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rolling_stats(df: pd.DataFrame, save_path: Path | None = None):
    """Rolling mean and std of returns over different windows."""
    rets = np.log(df["close"]).diff()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for w in [10, 60, 360]:
        axes[0].plot(rets.rolling(w).mean(), linewidth=0.5, label=f"{w}-min mean")
        axes[1].plot(rets.rolling(w).std(), linewidth=0.5, label=f"{w}-min std")

    axes[0].set_title("Rolling Mean of Log Returns")
    axes[0].legend()
    axes[1].set_title("Rolling Std of Log Returns")
    axes[1].legend()
    axes[1].set_xlabel("Time")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_predictions(pred_df: pd.DataFrame, save_path: Path | None = None):
    """Overlay predicted vs actual realized volatility for all models."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(pred_df.index, pred_df["actual"], color="black",
            linewidth=0.5, alpha=0.7, label="Actual")

    colors = {"garch": "blue", "xgboost": "green", "lstm": "red"}
    for col in ["garch", "xgboost", "lstm"]:
        if col in pred_df.columns:
            ax.plot(pred_df.index, pred_df[col], color=colors[col],
                    linewidth=0.5, alpha=0.7, label=col.upper())

    ax.set_title("Predicted vs Actual Realized Volatility")
    ax.set_xlabel("Time")
    ax.set_ylabel("Realized Volatility")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residuals(actual: np.ndarray, predicted: np.ndarray,
                   model_name: str, save_path: Path | None = None):
    """4-panel residual analysis plot."""
    residuals = actual - predicted

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals over time
    axes[0, 0].plot(residuals, linewidth=0.3)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title(f"{model_name}: Residuals over Time")

    # Histogram
    axes[0, 1].hist(residuals, bins=100, density=True, alpha=0.7)
    axes[0, 1].set_title(f"{model_name}: Residual Distribution")

    # Q-Q
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f"{model_name}: Residual Q-Q Plot")

    # ACF
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=50, ax=axes[1, 1],
             title=f"{model_name}: Residual ACF")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(train_losses: list, val_losses: list,
                         best_epoch: int | None = None,
                         save_path: Path | None = None):
    """Training vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=1.5)

    if best_epoch:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.7,
                    label=f"Best Epoch ({best_epoch})")

    ax.set_title("LSTM Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bootstrap_dist(deltas: np.ndarray, observed_delta: float,
                        ci_lower: float, ci_upper: float,
                        save_path: Path | None = None):
    """Bootstrap distribution with CI and observed delta."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(deltas, bins=50, density=True, alpha=0.6, color="steelblue",
            label="Bootstrap Distribution")

    ax.axvline(observed_delta, color="red", linewidth=2, linestyle="-",
               label=f"Observed Delta: {observed_delta:.6f}")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Zero")
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color="green",
               label=f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

    ax.set_title("Block Bootstrap: MSE_baseline - MSE_LSTM")
    ax.set_xlabel("Delta MSE")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_bar(table: pd.DataFrame, save_path: Path | None = None):
    """Grouped bar chart comparing metrics across models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, metric in enumerate(table.columns):
        table[metric].plot(kind="bar", ax=axes[i], color=["blue", "green", "red"])
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis="x", rotation=0)

    fig.suptitle("Model Comparison: All Metrics", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cumulative_error(pred_df: pd.DataFrame, save_path: Path | None = None):
    """Cumulative squared error over time per model."""
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = {"garch": "blue", "xgboost": "green", "lstm": "red"}
    for col in ["garch", "xgboost", "lstm"]:
        if col in pred_df.columns:
            cum_se = ((pred_df["actual"] - pred_df[col]) ** 2).cumsum()
            ax.plot(pred_df.index, cum_se, color=colors[col], linewidth=1,
                    label=col.upper())

    ax.set_title("Cumulative Squared Error")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative SE")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attention_weights(weights: np.ndarray, save_path: Path | None = None):
    """Heatmap of attention weights over time steps."""
    fig, ax = plt.subplots(figsize=(12, 3))

    ax.imshow(weights.reshape(1, -1), aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Timestep (minutes back)")
    ax.set_ylabel("")
    ax.set_title("Average LSTM Attention Weights")
    ax.set_yticks([])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# GENERATE ALL PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def generate_eda_plots(df: pd.DataFrame, features: pd.DataFrame,
                       rv: pd.Series, output_dir: Path = OUTPUT_FIGURES):
    """Generate all EDA plots and save to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rets = np.log(df["close"]).diff()

    plot_price_volume(df, output_dir / "01_price_volume.png")
    plot_return_dist(rets, output_dir / "02_return_distribution.png")
    plot_qq(rets, output_dir / "03_qq_plot.png")
    plot_acf_squared(rets, save_path=output_dir / "04_acf_squared.png")
    plot_rv_timeseries(rv, output_dir / "05_realized_volatility.png")
    plot_correlation(features, output_dir / "06_correlation.png")
    plot_rolling_stats(df, output_dir / "07_rolling_stats.png")

    logger.info(f"EDA plots saved to {output_dir}")
    plt.close("all")


def generate_result_plots(
    pred_df: pd.DataFrame,
    train_losses: list | None = None,
    val_losses: list | None = None,
    best_epoch: int | None = None,
    bootstrap_result: dict | None = None,
    mtable: pd.DataFrame | None = None,
    output_dir: Path = OUTPUT_FIGURES,
):
    """Generate all result plots and save to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_predictions(pred_df, output_dir / "08_predictions.png")

    for model in ["garch", "xgboost", "lstm"]:
        if model in pred_df.columns:
            plot_residuals(
                pred_df["actual"].values, pred_df[model].values,
                model.upper(), output_dir / f"09_residuals_{model}.png"
            )

    if train_losses and val_losses:
        plot_training_curves(train_losses, val_losses, best_epoch,
                             output_dir / "10_training_curves.png")

    if bootstrap_result:
        plot_bootstrap_dist(
            bootstrap_result["bootstrap_deltas"],
            bootstrap_result["observed_delta"],
            bootstrap_result["ci_lower"],
            bootstrap_result["ci_upper"],
            output_dir / "11_bootstrap.png",
        )

    if mtable is not None:
        plot_metrics_bar(mtable, output_dir / "12_metrics_comparison.png")

    plot_cumulative_error(pred_df, output_dir / "13_cumulative_error.png")

    logger.info(f"Result plots saved to {output_dir}")
    plt.close("all")
