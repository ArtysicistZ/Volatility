"""
Central configuration for the CIS 5450 Volatility Prediction project.
All hyperparameters, paths, and constants live here. No magic numbers elsewhere.
"""
from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DUCKDB_PATH = DATA_PROCESSED / "crypto_volatility.duckdb"
OUTPUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOGS = PROJECT_ROOT / "outputs" / "logs"

# ---------------------------------------------------------------------------
# Data Collection (Binance API)
# ---------------------------------------------------------------------------
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"
COLLECTION_START = "2024-04-01"
COLLECTION_END = "2025-03-31"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
TARGET_WINDOW = 10                       # 10-minute forward realized volatility
ROLLING_WINDOWS = [5, 10, 30, 60]        # for rolling statistics
LAG_STEPS = list(range(1, 11))           # lag_1 through lag_10
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# ---------------------------------------------------------------------------
# Data Splitting (strict chronological, no shuffle)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.85
VAL_RATIO = 0.05
TEST_RATIO = 0.10

# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------
GARCH_RETURN_SCALE = 100                 # multiply returns by 100 for convergence
GARCH_P = 1
GARCH_Q = 1
GARCH_DIST = "t"                         # Student-t for crypto fat tails
GARCH_MIN_WINDOW = 5000                  # min observations for first fit
GARCH_REFIT_FREQ = 500                   # refit every N steps (not every step)

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
XGBOOST_OPTUNA_TRIALS = 50
XGBOOST_MIN_WINDOW = 5000
XGBOOST_REFIT_FREQ = 500
XGBOOST_PARAM_SPACE = {
    "max_depth": (3, 8),
    "learning_rate": (0.01, 0.1),
    "n_estimators": (100, 1000),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "gamma": (0.0, 0.3),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.0, 1.0),
}

# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
LSTM_SEQ_LEN = 60                        # 60-minute lookback window
LSTM_HIDDEN = 128                        # hidden units per layer
LSTM_LAYERS = 2                          # stacked LSTM layers
LSTM_DROPOUT = 0.3                       # between-layer dropout
LSTM_BATCH = 512                         # batch size (A100 can handle this)
LSTM_EPOCHS = 100                        # max epochs (early stopping will cut)
LSTM_LR = 1e-3                           # initial learning rate
LSTM_WEIGHT_DECAY = 1e-4                 # L2 regularization
LSTM_GRAD_CLIP = 1.0                     # gradient clipping max norm
LSTM_LR_PATIENCE = 5                     # ReduceLROnPlateau patience
LSTM_LR_FACTOR = 0.5                     # LR reduction factor
LSTM_EARLY_STOP = 10                     # early stopping patience
LSTM_USE_ATTENTION = False               # toggle Luong attention
LSTM_LOG_TARGET = True                   # log-transform target for stability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Hypothesis Testing (Block Bootstrap)
# ---------------------------------------------------------------------------
BOOT_REPS = 1000                         # bootstrap replications
BOOT_BLOCK = 50                          # block size (~n^(1/3) for 130K)
BOOT_CONF = 0.95                         # confidence level

# ---------------------------------------------------------------------------
# Regex Patterns (course requirement)
# ---------------------------------------------------------------------------
RE_TIMESTAMP_MS = r"^\d{13,16}$"                     # 13-digit (ms) or 16-digit (us) Unix
RE_TRADING_PAIR = r"^([A-Z]{2,10})(USDT|BTC|ETH)$"  # e.g. BTCUSDT
RE_NUMERIC = r"^-?\d+\.?\d*$"                        # decimal number
