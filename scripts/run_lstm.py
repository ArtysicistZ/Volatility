"""
Step 4: LSTM Training
Trains LSTM with multiple seq_len and attention configurations.
Run on A100 GPU cluster for best performance.

Usage:
    source /home/kevinzyz/hansenzuishuai/.venv/bin/activate
    cd /home/kevinzyz/yincheng/volatility

    # Run all experiments (60/120/240 x with/without attention)
    python scripts/run_lstm.py

    # Run a single experiment
    python scripts/run_lstm.py --seq-len 60 --no-attention
    python scripts/run_lstm.py --seq-len 240 --attention
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


def run_experiment(seq_len: int, use_attention: bool):
    """Run a single LSTM experiment with given seq_len and attention config."""
    import numpy as np
    import pandas as pd
    import torch
    from src.config import (
        DATA_PROCESSED, OUTPUT_MODELS, DEVICE,
        TRAIN_RATIO, VAL_RATIO,
        LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT, LSTM_BATCH,
        LSTM_LOG_TARGET,
    )
    from src.feature_engineering import log_transform_target, inverse_log_transform
    from src.models import VolatilityLSTM, create_dataloaders
    from src.training import (
        set_all_seeds, train_lstm, predict_lstm, inverse_transform,
    )

    tag = f"seq{seq_len}_{'attn' if use_attention else 'noattn'}"
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: seq_len={seq_len}, attention={use_attention} [{tag}]")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 60)

    set_all_seeds()

    # --- Load data ---
    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    target_raw = pd.read_parquet(DATA_PROCESSED / "target.parquet")["realized_volatility"]

    # Log-transform target for LSTM
    if LSTM_LOG_TARGET:
        target = log_transform_target(target_raw)
    else:
        target = target_raw

    # --- Create dataloaders with this seq_len ---
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
    model = VolatilityLSTM(
        input_size=info["n_features"],
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
        use_attention=use_attention,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # --- Train ---
    checkpoint_dir = OUTPUT_MODELS / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result = train_lstm(
        model, train_loader, val_loader,
        device=DEVICE,
        checkpoint_dir=checkpoint_dir,
    )

    print(f"  Best epoch: {result['best_epoch']}, "
          f"Best val loss: {result['best_val_loss']:.8f}")

    # --- Predict on test set ---
    preds = predict_lstm(result["model"], test_loader, device=DEVICE)

    # Inverse transform to original scale
    preds_original = inverse_transform(preds, log_transformed=LSTM_LOG_TARGET)

    # Align predictions with test timestamps
    test_timestamps = info["test_timestamps"]
    n_preds = len(preds_original)
    if len(test_timestamps) > n_preds:
        test_timestamps = test_timestamps[:n_preds]
    elif len(test_timestamps) < n_preds:
        preds_original = preds_original[:len(test_timestamps)]

    preds_series = pd.Series(preds_original, index=test_timestamps, name="lstm")

    # --- Save results ---
    preds_series.to_pickle(OUTPUT_MODELS / f"lstm_predictions_{tag}.pkl")

    # Save training history
    history = {
        "seq_len": seq_len,
        "use_attention": use_attention,
        "train_losses": result["train_losses"],
        "val_losses": result["val_losses"],
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "n_params": n_params,
    }
    with open(OUTPUT_MODELS / f"lstm_history_{tag}.pkl", "wb") as f:
        pickle.dump(history, f)

    logger.info(f"[{tag}] Predictions saved: {len(preds_series)} samples")
    logger.info(f"[{tag}] Pred stats: mean={preds_series.mean():.6f}, "
                f"std={preds_series.std():.6f}")

    return tag, history, preds_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Single seq_len to run (default: run all)")
    parser.add_argument("--attention", action="store_true",
                        help="Use attention (for single experiment)")
    parser.add_argument("--no-attention", action="store_true",
                        help="No attention (for single experiment)")
    args = parser.parse_args()

    # Determine experiments to run
    if args.seq_len is not None:
        # Single experiment
        use_attn = args.attention and not args.no_attention
        experiments = [(args.seq_len, use_attn)]
    else:
        # Full experiment grid
        experiments = [
            (60,  False),
            (60,  True),
            (120, False),
            (120, True),
            (240, False),
            (240, True),
        ]

    logger.info(f"Running {len(experiments)} experiment(s)")

    results_summary = []
    for seq_len, use_attention in experiments:
        tag, history, preds = run_experiment(seq_len, use_attention)
        results_summary.append({
            "tag": tag,
            "seq_len": seq_len,
            "attention": use_attention,
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "n_params": history["n_params"],
            "pred_mean": preds.mean(),
            "pred_std": preds.std(),
        })

    # --- Print summary table ---
    import pandas as pd
    from src.config import OUTPUT_TABLES
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(results_summary)
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(OUTPUT_TABLES / "lstm_experiments.csv", index=False)

    # Copy the best model's predictions as the main lstm_predictions.pkl
    best_idx = summary_df["best_val_loss"].idxmin()
    best_tag = summary_df.loc[best_idx, "tag"]
    logger.info(f"Best experiment: {best_tag}")

    import shutil
    from src.config import OUTPUT_MODELS
    best_pred_path = OUTPUT_MODELS / f"lstm_predictions_{best_tag}.pkl"
    main_pred_path = OUTPUT_MODELS / "lstm_predictions.pkl"
    shutil.copy(best_pred_path, main_pred_path)
    logger.info(f"Copied {best_tag} predictions -> lstm_predictions.pkl")

    logger.info("ALL LSTM EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
