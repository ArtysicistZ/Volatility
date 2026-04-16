"""
Training loop for LSTM, prediction utilities, and seed management.
"""
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    RANDOM_SEED, DEVICE,
    LSTM_EPOCHS, LSTM_LR, LSTM_WEIGHT_DECAY, LSTM_GRAD_CLIP,
    LSTM_LR_PATIENCE, LSTM_LR_FACTOR, LSTM_EARLY_STOP,
    LSTM_LOG_TARGET, OUTPUT_MODELS, OUTPUT_LOGS,
)
from src.feature_engineering import inverse_log_transform

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.MSELoss,
    device: str,
    grad_clip: float = LSTM_GRAD_CLIP,
) -> float:
    """Single training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.MSELoss,
    device: str,
) -> float:
    """Validation pass (no gradients). Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = LSTM_EPOCHS,
    lr: float = LSTM_LR,
    weight_decay: float = LSTM_WEIGHT_DECAY,
    patience: int = LSTM_EARLY_STOP,
    device: str = DEVICE,
    checkpoint_dir: Path = OUTPUT_MODELS,
) -> dict:
    """
    Full LSTM training pipeline:
    - Adam optimizer with L2 regularization (weight_decay)
    - ReduceLROnPlateau scheduler
    - Early stopping with best model checkpointing
    - TensorBoard logging (if tensorboard available)

    Returns dict with model, losses, and best epoch info.
    """
    set_all_seeds()

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=LSTM_LR_PATIENCE, factor=LSTM_LR_FACTOR
    )
    criterion = nn.MSELoss()

    # TensorBoard (optional)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = OUTPUT_LOGS / "lstm"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
    except ImportError:
        pass

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []

    pbar = tqdm(range(1, max_epochs + 1), desc="LSTM Training")
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # TensorBoard logging
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        pbar.set_postfix({
            "train": f"{train_loss:.6f}",
            "val": f"{val_loss:.6f}",
            "lr": f"{current_lr:.2e}",
            "best": f"{best_val_loss:.6f}",
        })

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save(model.state_dict(), checkpoint_dir / "lstm_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_dir / "lstm_best.pt",
                                      weights_only=True))

    if writer:
        writer.close()

    logger.info(
        f"Training complete. Best epoch: {best_epoch}, "
        f"Best val loss: {best_val_loss:.6f}"
    )

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def predict_lstm(
    model: nn.Module,
    loader: DataLoader,
    device: str = DEVICE,
) -> np.ndarray:
    """Run inference on a DataLoader. Returns numpy array of predictions."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())

    return np.concatenate(all_preds)


def inverse_transform(
    preds: np.ndarray,
    log_transformed: bool = LSTM_LOG_TARGET,
) -> np.ndarray:
    """
    Reverse target transformation so predictions are on original RV scale.
    All models must be compared on the same scale.
    """
    if log_transformed:
        return inverse_log_transform(preds)
    return preds


def collect_predictions(
    garch_preds: pd.Series,
    xgb_preds: pd.Series,
    lstm_preds: pd.Series,
    actuals: pd.Series,
    timestamps: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Align all model predictions to common timestamps.
    Returns DataFrame with columns: actual, garch, xgboost, lstm.
    """
    df = pd.DataFrame({"actual": actuals})

    if garch_preds is not None:
        df["garch"] = garch_preds
    if xgb_preds is not None:
        df["xgboost"] = xgb_preds
    if lstm_preds is not None:
        if timestamps is not None and len(timestamps) == len(lstm_preds):
            lstm_s = pd.Series(lstm_preds, index=timestamps, name="lstm")
        else:
            lstm_s = pd.Series(lstm_preds, name="lstm")
        df["lstm"] = lstm_s

    # Keep only rows where all models have predictions
    df = df.dropna()
    logger.info(f"Aligned predictions: {len(df)} common timestamps")
    return df
