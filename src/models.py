"""
Model definitions: GARCH(1,1), XGBoost, LSTM (+ optional Luong attention).
Also includes PyTorch Dataset and DataLoader utilities.
"""
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import (
    GARCH_RETURN_SCALE, GARCH_P, GARCH_Q, GARCH_DIST,
    GARCH_MIN_WINDOW, GARCH_REFIT_FREQ, TARGET_WINDOW,
    XGBOOST_OPTUNA_TRIALS, XGBOOST_MIN_WINDOW, XGBOOST_REFIT_FREQ,
    XGBOOST_PARAM_SPACE,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_BATCH, LSTM_USE_ATTENTION,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GARCH(1,1)
# ═══════════════════════════════════════════════════════════════════════════

def test_stationarity(returns: pd.Series) -> dict:
    """
    Run Augmented Dickey-Fuller test on the return series.
    Log returns should be stationary; this is a sanity check.
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(returns.dropna(), autolag="AIC")
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,
        "critical_values": result[4],
    }


def fit_garch(returns: pd.Series, p: int = GARCH_P, q: int = GARCH_Q,
              dist: str = GARCH_DIST, scale: float = GARCH_RETURN_SCALE):
    """
    Fit GARCH(p,q) with Student-t distribution.
    Returns scaled by `scale` before fitting (convergence aid).
    """
    from arch import arch_model

    scaled_returns = returns * scale
    am = arch_model(scaled_returns, p=p, q=q, mean="constant",
                    vol="GARCH", dist=dist)
    res = am.fit(disp="off")
    return res


def garch_rolling_forecast(
    returns: pd.Series,
    start_pos: int,
    rolling_window: int = GARCH_MIN_WINDOW,
    refit_freq: int = GARCH_REFIT_FREQ,
    horizon: int = TARGET_WINDOW,
    scale: float = GARCH_RETURN_SCALE,
    p: int = GARCH_P,
    q: int = GARCH_Q,
    dist: str = GARCH_DIST,
) -> pd.Series:
    """
    Rolling-window GARCH forecast with proper state tracking.

    Key design: Parameters (omega, alpha, beta) are re-estimated every
    `refit_freq` steps on a rolling window. But the conditional variance h_t
    is updated at EVERY step using the GARCH recursion:
        h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
    This ensures predictions reflect the latest market data, not just the
    last refit point.

    Returns predicted realized volatility on ORIGINAL scale.
    """
    from arch import arch_model
    from tqdm import tqdm

    scaled = returns * scale
    n = len(scaled)
    predictions = pd.Series(index=returns.index, dtype=float)

    # GARCH parameters (re-estimated periodically)
    omega, alpha, beta, mu = None, None, None, None
    h_t = None  # current conditional variance (in scaled units)

    pbar = tqdm(range(start_pos, n), desc="GARCH forecast")

    for t in pbar:
        # Refit parameters periodically on a rolling window
        if omega is None or (t - start_pos) % refit_freq == 0:
            window_start = max(0, t - rolling_window)
            try:
                am = arch_model(scaled.iloc[window_start:t], p=p, q=q,
                                mean="constant", vol="GARCH", dist=dist,
                                rescale=False)
                res = am.fit(disp="off", show_warning=False)
                omega = res.params["omega"]
                alpha = res.params["alpha[1]"]
                beta = res.params["beta[1]"]
                mu = res.params["mu"]
                # Initialize h_t from the model's last conditional variance
                h_t = res.conditional_volatility.iloc[-1] ** 2
            except Exception as e:
                logger.debug(f"GARCH fit failed at t={t}: {e}")
                continue

        # Update h_t with the GARCH recursion using the latest return
        eps = scaled.iloc[t - 1] - mu  # innovation at t-1
        h_t = omega + alpha * eps ** 2 + beta * h_t

        # Multi-step ahead forecast from current h_t
        persistence = alpha + beta
        var_forecasts = []
        h_k = h_t
        for k in range(horizon):
            h_k = omega + persistence * h_k
            var_forecasts.append(h_k)

        # Predicted RV = sqrt(mean of forecasted variances) / scale
        # mean is correct because target = sample_std(individual_returns)
        pred_vol = np.sqrt(np.mean(var_forecasts)) / scale
        predictions.iloc[t] = pred_vol

    predictions = predictions.dropna()
    logger.info(f"GARCH produced {len(predictions)} forecasts")
    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# XGBoost
# ═══════════════════════════════════════════════════════════════════════════

def xgb_optuna_tune(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    n_trials: int = XGBOOST_OPTUNA_TRIALS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Hyperparameter tuning with Optuna (TPE sampler).
    Returns best params dict.
    """
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", *XGBOOST_PARAM_SPACE["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *XGBOOST_PARAM_SPACE["learning_rate"], log=True),
            "n_estimators": trial.suggest_int("n_estimators", *XGBOOST_PARAM_SPACE["n_estimators"]),
            "subsample": trial.suggest_float("subsample", *XGBOOST_PARAM_SPACE["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *XGBOOST_PARAM_SPACE["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *XGBOOST_PARAM_SPACE["min_child_weight"]),
            "gamma": trial.suggest_float("gamma", *XGBOOST_PARAM_SPACE["gamma"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *XGBOOST_PARAM_SPACE["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *XGBOOST_PARAM_SPACE["reg_lambda"]),
            "n_jobs": -1,
            "random_state": seed,
            "objective": "reg:squarederror",
        }
        model = xgb.XGBRegressor(**params)
        model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
        pred = model.predict(val_X)
        mse = np.mean((val_y - pred) ** 2)
        return mse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["n_jobs"] = -1
    best["random_state"] = seed
    best["objective"] = "reg:squarederror"
    logger.info(f"Best XGBoost params: {best}")
    return best


def xgb_expanding_predict(
    features: pd.DataFrame,
    targets: pd.Series,
    params: dict,
    min_window: int = XGBOOST_MIN_WINDOW,
    refit_freq: int = XGBOOST_REFIT_FREQ,
) -> pd.Series:
    """
    Walk-forward expanding window predictions.
    Refit every `refit_freq` steps. No shuffling.
    """
    import xgboost as xgb

    n = len(features)
    predictions = pd.Series(index=features.index, dtype=float)
    X = features.values
    y = targets.values

    model = None
    for t in range(min_window, n):
        if model is None or (t - min_window) % refit_freq == 0:
            model = xgb.XGBRegressor(**params)
            model.fit(X[:t], y[:t], verbose=False)

        pred = model.predict(X[t:t+1])
        predictions.iloc[t] = pred[0]

    predictions = predictions.dropna()
    logger.info(f"XGBoost produced {len(predictions)} predictions")
    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# LSTM
# ═══════════════════════════════════════════════════════════════════════════

class LuongAttention(nn.Module):
    """
    Luong (general) dot-product attention mechanism.
    Computes weighted context vector over LSTM hidden states.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, lstm_outputs: torch.Tensor,
                final_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch, seq_len, hidden_size)
            final_hidden: (batch, hidden_size)
        Returns:
            context: (batch, hidden_size)
            attn_weights: (batch, seq_len)
        """
        # score = lstm_outputs @ W @ final_hidden
        query = self.W(final_hidden).unsqueeze(2)       # (batch, hidden, 1)
        scores = torch.bmm(lstm_outputs, query).squeeze(2)  # (batch, seq_len)
        attn_weights = torch.softmax(scores, dim=1)     # (batch, seq_len)

        # context = weighted sum of lstm_outputs
        context = torch.bmm(
            attn_weights.unsqueeze(1), lstm_outputs
        ).squeeze(1)  # (batch, hidden_size)

        return context, attn_weights


class VolatilityLSTM(nn.Module):
    """
    Stacked LSTM for volatility prediction.

    Architecture:
        LSTM(input_size, hidden, num_layers=2, dropout=0.3, batch_first=True)
        -> [Optional: LuongAttention]
        -> Linear(hidden, 1)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        use_attention: bool = LSTM_USE_ATTENTION,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = LuongAttention(hidden_size) if use_attention else None
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            predictions: (batch,)
        """
        lstm_out, (h_n, _) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)

        if self.attention is not None:
            final_hidden = h_n[-1]  # (batch, hidden_size)
            context, self._attn_weights = self.attention(lstm_out, final_hidden)
            out = self.fc(context)
        else:
            out = self.fc(lstm_out[:, -1, :])  # last timestep

        return out.squeeze(-1)  # (batch,)


# ═══════════════════════════════════════════════════════════════════════════
# PatchTST Transformer
# ═══════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """Convert time series into patches and project to d_model."""

    def __init__(self, input_size: int, patch_len: int, stride: int,
                 d_model: int, dropout: float = 0.15):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Linear projection of each patch (patch_len * input_size) -> d_model
        self.proj = nn.Linear(patch_len * input_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            patches: (batch, n_patches, d_model)
        """
        batch, seq_len, n_feat = x.shape
        # Unfold into patches: (batch, n_patches, patch_len, n_feat)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches shape: (batch, n_patches, n_feat, patch_len) — unfold puts size last
        n_patches = patches.shape[1]
        # Reshape to (batch, n_patches, patch_len * n_feat)
        patches = patches.permute(0, 1, 3, 2).reshape(batch, n_patches, -1)
        # Project to d_model
        patches = self.proj(patches)
        patches = self.dropout(patches)
        return patches


class VolatilityTransformer(nn.Module):
    """
    PatchTST-style Transformer for volatility prediction.

    Architecture:
        Input (batch, seq_len, input_size)
        -> PatchEmbedding (batch, n_patches, d_model)
        -> + Learnable positional encoding
        -> TransformerEncoder (num_layers x [MultiHeadAttention + FFN])
        -> Global average pooling (batch, d_model)
        -> Regression head -> (batch, 1)
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int = 240,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len

        # Patch embedding
        self.patch_embed = PatchEmbedding(input_size, patch_len, stride,
                                          d_model, dropout)

        # Number of patches
        n_patches = (seq_len - patch_len) // stride + 1
        self.n_patches = n_patches

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            predictions: (batch,)
        """
        # Patch + project
        patches = self.patch_embed(x)  # (batch, n_patches, d_model)

        # Add positional encoding
        patches = patches + self.pos_embed

        # Transformer encoder
        out = self.transformer(patches)  # (batch, n_patches, d_model)

        # Global average pooling
        out = out.mean(dim=1)  # (batch, d_model)

        # Regression head
        out = self.head(out)  # (batch, 1)
        return out.squeeze(-1)  # (batch,)


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch Dataset & DataLoader
# ═══════════════════════════════════════════════════════════════════════════

class VolatilityDataset(Dataset):
    """
    Sliding-window dataset for LSTM.
    Sample idx -> x: features[idx : idx+seq_len], y: targets[idx+seq_len]
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 seq_len: int = LSTM_SEQ_LEN):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_len]   # (seq_len, n_features)
        y = self.targets[idx + self.seq_len]            # scalar
        return x, y


def normalize_features(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalization using TRAINING set statistics only.
    Returns: (train_normed, val_normed, test_normed, means, stds)
    """
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    stds[stds < 1e-10] = 1.0  # avoid division by zero

    train_n = (train - means) / stds
    val_n = (val - means) / stds
    test_n = (test - means) / stds

    return train_n, val_n, test_n, means, stds


def create_dataloaders(
    features: pd.DataFrame,
    targets: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seq_len: int = LSTM_SEQ_LEN,
    batch_size: int = LSTM_BATCH,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Full data preparation pipeline:
    1. Align features and targets (drop NaN from target)
    2. Chronological split into train/val/test
    3. Z-score normalize features (training stats only)
    4. Create VolatilityDataset and DataLoader for each split
    Returns: (train_loader, val_loader, test_loader, info_dict)
    """
    # Align features and targets
    common_idx = features.index.intersection(targets.dropna().index)
    feat = features.loc[common_idx].values
    tgt = targets.loc[common_idx].values

    n = len(feat)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_feat = feat[:train_end]
    val_feat = feat[train_end:val_end]
    test_feat = feat[val_end:]

    train_tgt = tgt[:train_end]
    val_tgt = tgt[train_end:val_end]
    test_tgt = tgt[val_end:]

    # Normalize features using training stats
    train_feat_n, val_feat_n, test_feat_n, means, stds = \
        normalize_features(train_feat, val_feat, test_feat)

    # Create datasets
    train_ds = VolatilityDataset(train_feat_n, train_tgt, seq_len)
    val_ds = VolatilityDataset(val_feat_n, val_tgt, seq_len)
    test_ds = VolatilityDataset(test_feat_n, test_tgt, seq_len)

    # Create dataloaders (shuffle=False for all — time series!)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    info = {
        "n_features": feat.shape[1],
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "means": means,
        "stds": stds,
        "test_timestamps": common_idx[val_end + seq_len:],
    }

    logger.info(
        f"DataLoaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
        f"features={feat.shape[1]}"
    )

    return train_loader, val_loader, test_loader, info
