"""
Step 4b: Transformer Training
Trains PatchTST-style Transformer with multiple configs.
Run on A100 GPU cluster.

Usage:
    source /home/kevinzyz/hansenzuishuai/.venv/bin/activate
    cd /home/kevinzyz/yincheng/volatility

    python scripts/run_transformer.py                          # run all experiments
    python scripts/run_transformer.py --config small --seq-len 120  # single experiment
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import logging
import pickle
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Transformer configs: small (safe for 446K samples) and full
CONFIGS = {
    "small": {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 256,
        "dropout": 0.2,
        "patch_len": 12,
        "stride": 6,
    },
    "full": {
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 4,
        "d_ff": 1024,
        "dropout": 0.2,
        "patch_len": 12,
        "stride": 6,
    },
}


def run_experiment(seq_len: int, config_name: str):
    """Run a single Transformer experiment."""
    import numpy as np
    import pandas as pd
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from src.config import (
        DATA_PROCESSED, OUTPUT_MODELS, DEVICE,
        TRAIN_RATIO, VAL_RATIO,
        LSTM_BATCH, LSTM_LOG_TARGET, RANDOM_SEED,
    )
    from src.feature_engineering import log_transform_target
    from src.models import VolatilityTransformer, create_dataloaders
    from src.training import (
        set_all_seeds, predict_lstm, inverse_transform,
    )

    cfg = CONFIGS[config_name]
    tag = f"tfm_{config_name}_seq{seq_len}"
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {tag}")
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 60)

    set_all_seeds(RANDOM_SEED)

    # --- Load data ---
    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    target_raw = pd.read_parquet(DATA_PROCESSED / "target.parquet")["realized_volatility"]

    if LSTM_LOG_TARGET:
        target = log_transform_target(target_raw)
    else:
        target = target_raw

    # --- Create dataloaders ---
    train_loader, val_loader, test_loader, info = create_dataloaders(
        features, target,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seq_len=seq_len,
        batch_size=LSTM_BATCH,
    )

    print(f"  Train: {info['train_size']}, Val: {info['val_size']}, "
          f"Test: {info['test_size']}, Features: {info['n_features']}")

    # --- Build model ---
    model = VolatilityTransformer(
        input_size=info["n_features"],
        seq_len=seq_len,
        patch_len=cfg["patch_len"],
        stride=cfg["stride"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Patches per sample: {model.n_patches}")

    # --- Training loop with warmup + cosine decay ---
    import torch.nn as nn
    from tqdm import tqdm

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Warmup + cosine schedule
    max_epochs = 80
    warmup_epochs = 5
    early_stop_patience = 10

    checkpoint_dir = OUTPUT_MODELS / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []

    pbar = tqdm(range(1, max_epochs + 1), desc=f"Transformer [{tag}]")
    for epoch in pbar:
        # Warmup LR: linear increase for first warmup_epochs
        if epoch <= warmup_epochs:
            lr = 1e-4 + (1e-3 - 1e-4) * (epoch / warmup_epochs)
        else:
            # Cosine decay from 1e-3 to 1e-5
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Train
        model.train()
        total_train = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train += loss.item()
            n_batches += 1
        train_loss = total_train / max(n_batches, 1)

        # Validate
        model.eval()
        total_val = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                total_val += loss.item()
                n_val += 1
        val_loss = total_val / max(n_val, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        pbar.set_postfix({
            "train": f"{train_loss:.6f}",
            "val": f"{val_loss:.6f}",
            "lr": f"{lr:.2e}",
            "best": f"{best_val_loss:.6f}",
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", weights_only=True))
    print(f"  Best epoch: {best_epoch}, Best val loss: {best_val_loss:.8f}")

    # --- Predict on test set ---
    preds = predict_lstm(model, test_loader, device=DEVICE)  # works for any model
    preds_original = inverse_transform(preds, log_transformed=LSTM_LOG_TARGET)

    # Align with timestamps
    test_timestamps = info["test_timestamps"]
    n_preds = len(preds_original)
    if len(test_timestamps) > n_preds:
        test_timestamps = test_timestamps[:n_preds]
    elif len(test_timestamps) < n_preds:
        preds_original = preds_original[:len(test_timestamps)]

    preds_series = pd.Series(preds_original, index=test_timestamps, name="transformer")

    # --- Save ---
    preds_series.to_pickle(OUTPUT_MODELS / f"lstm_predictions_{tag}.pkl")

    history = {
        "tag": tag,
        "config": config_name,
        "seq_len": seq_len,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "n_params": n_params,
    }
    with open(OUTPUT_MODELS / f"lstm_history_{tag}.pkl", "wb") as f:
        pickle.dump(history, f)

    logger.info(f"[{tag}] Predictions saved: {len(preds_series)} samples, "
                f"mean={preds_series.mean():.6f}, std={preds_series.std():.6f}")

    return tag, history, preds_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["small", "full"], default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()

    if args.config and args.seq_len:
        experiments = [(args.seq_len, args.config)]
    else:
        experiments = [
            (120, "small"),
            (240, "small"),
            (120, "full"),
            (240, "full"),
        ]

    logger.info(f"Running {len(experiments)} Transformer experiment(s)")

    results_summary = []
    for seq_len, config_name in experiments:
        tag, history, preds = run_experiment(seq_len, config_name)
        results_summary.append({
            "tag": tag,
            "config": config_name,
            "seq_len": seq_len,
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "n_params": history["n_params"],
            "pred_mean": preds.mean(),
            "pred_std": preds.std(),
        })

    import pandas as pd
    from src.config import OUTPUT_TABLES
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(results_summary)
    print("\n" + "=" * 60)
    print("TRANSFORMER EXPERIMENT SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_TABLES / "transformer_experiments.csv", index=False)

    logger.info("ALL TRANSFORMER EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
