# CIS 5450 Final Project: Implementation Plan
## Sequential Deep Learning for High-Frequency Crypto Volatility Prediction

---

## Context

Predict high-frequency cryptocurrency volatility using deep learning, testing whether LSTM sequence models outperform traditional GARCH(1,1) and XGBoost baselines. Uses 90 days of 1-minute BTC/USDT candle data from Binance (~130K rows). Target: realized volatility over the next 10-minute window. Must integrate course concepts: Regex, DuckDB/SQL, and Pandas. A100 GPU available for training.

**Constraint**: Final deliverable is a single `.ipynb` notebook for the TA. Development happens in modular `.py` files locally, then everything is consolidated into one well-structured notebook at the end.

---

## 1. File Structure

```
CIS-5450-Volatility/
|
|-- data/
|   |-- raw/                         # Raw parquet from Binance API
|   |-- processed/                   # Cleaned features, DuckDB file
|
|-- src/
|   |-- config.py                    # All constants and hyperparameters
|   |-- data_collection.py           # Binance API fetch + regex cleaning + DuckDB
|   |-- feature_engineering.py       # Feature computation + target variable
|   |-- models.py                    # GARCH, XGBoost, LSTM model definitions
|   |-- training.py                  # LSTM training loop, baseline fitting
|   |-- evaluation.py               # Metrics, hypothesis testing, visualization
|
|-- notebooks/
|   |-- dev.ipynb                    # Development notebook (import from src/)
|
|-- outputs/
|   |-- models/                     # Saved checkpoints
|   |-- figures/                    # Generated plots
|
|-- final_notebook.ipynb            # FINAL DELIVERABLE: self-contained, all code inline
|-- requirements.txt
|-- .gitignore
```

**Development workflow**: Write/debug in `src/*.py` files. Import into `notebooks/dev.ipynb` for interactive experimentation. Once everything works, consolidate all code into `final_notebook.ipynb` with clear markdown section headers -- no imports from `src/`, everything inline.

**Final notebook structure** (single `.ipynb`):
```
1. Setup & Configuration
2. Data Collection (Binance API)
3. Data Cleaning (Regex) 
4. DuckDB Storage & SQL Queries
5. EDA & Visualization
6. Feature Engineering (Pandas)
7. Target Variable (Realized Volatility)
8. Data Preparation (Train/Val/Test Split, Normalization, PyTorch Dataset)
9. Baseline: GARCH(1,1)
10. Baseline: XGBoost
11. Advanced Model: LSTM
12. Model Comparison & Metrics
13. Hypothesis Testing (Block Bootstrap)
14. Results Visualization
15. Conclusion
```

---

## 2. Configuration: `src/config.py`

Single source of truth. Will become the first code cell in final notebook.

```python
import torch
from pathlib import Path

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"
COLLECTION_START = "2025-01-01"
COLLECTION_END = "2025-03-31"

# Features
TARGET_WINDOW = 10
ROLLING_WINDOWS = [5, 10, 30, 60]
LAG_STEPS = list(range(1, 11))
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Splits (strict chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# GARCH
GARCH_RETURN_SCALE = 100
GARCH_DIST = "t"
GARCH_MIN_WINDOW = 5000
GARCH_REFIT_FREQ = 500

# XGBoost
XGBOOST_OPTUNA_TRIALS = 50
XGBOOST_MIN_WINDOW = 5000
XGBOOST_REFIT_FREQ = 500

# LSTM
LSTM_SEQ_LEN = 60
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BATCH = 512
LSTM_EPOCHS = 100
LSTM_LR = 1e-3
LSTM_WEIGHT_DECAY = 1e-4
LSTM_GRAD_CLIP = 1.0
LSTM_LR_PATIENCE = 5
LSTM_LR_FACTOR = 0.5
LSTM_EARLY_STOP = 10
LSTM_USE_ATTENTION = False
LSTM_LOG_TARGET = True

# Hypothesis Testing
BOOT_REPS = 1000
BOOT_BLOCK = 50
BOOT_CONF = 0.95

# Regex (course requirement)
RE_TIMESTAMP_MS = r"^\d{13}$"
RE_TRADING_PAIR = r"^([A-Z]{2,10})(USDT|BTC|ETH)$"
RE_NUMERIC = r"^-?\d+\.?\d*$"
```

---

## 3. Module-by-Module Specification

### 3.1 `src/data_collection.py` -- Data Pipeline (Binance + Regex + DuckDB)

Covers course requirements: **Regex** and **DuckDB/SQL**.

```python
# === BINANCE API ===
def fetch_klines(symbol, interval, start, end) -> pd.DataFrame:
    """python-binance Client('','').get_historical_klines() auto-paginates.
    ~130 requests (1000 candles each). Save to data/raw/btcusdt_1m_raw.parquet."""

def validate_completeness(df) -> dict:
    """Check row count (~130K), gaps, duplicates, nulls."""

# === REGEX CLEANING (course requirement) ===
def validate_timestamps(series) -> pd.Series:
    """re.fullmatch(r'^\d{13}$', str(val)) on each timestamp."""

def extract_trading_pair(s: str) -> tuple | None:
    """re.match(r'^([A-Z]{2,10})(USDT|BTC|ETH)$', s) -> ('BTC','USDT')"""

def validate_numeric_fields(df, cols) -> pd.DataFrame:
    """re.fullmatch(r'^-?\d+\.?\d*$', str(val)) on OHLCV."""

def clean_raw_data(df) -> pd.DataFrame:
    """Full pipeline: regex validate -> cast types -> UTC datetime index
    -> sort -> dedup -> forward-fill gaps <= 2 min."""

# === DUCKDB (course requirement) ===
def init_duckdb(path) -> duckdb.DuckDBPyConnection:
    """CREATE TABLE raw_candles (...); CREATE TABLE predictions (...)"""

def ingest_candles(conn, df):
    """Bulk insert cleaned data."""

def query_resampled(conn, interval='5min') -> pd.DataFrame:
    """SQL with time_bucket() for resampling. GROUP BY, FIRST, MAX, MIN, LAST, SUM."""

def compare_models_sql(conn) -> pd.DataFrame:
    """SQL JOIN across predictions for per-timestamp error comparison."""

def store_predictions(conn, model_name, preds_df):
    """INSERT INTO predictions."""
```

### 3.2 `src/feature_engineering.py` -- Features & Target (Course Pandas Requirement)

```python
# === INDIVIDUAL FEATURES ===
def log_returns(close) -> pd.Series:
    """np.log(close).diff()"""

def parkinson_vol(high, low, window) -> pd.Series:
    """sqrt(1/(4*n*ln2) * rolling_sum(ln(H/L)^2)). Shift(1) applied."""

def garman_klass_vol(open_, high, low, close, window) -> pd.Series:
    """0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2. 7.4x more efficient. Shift(1)."""

def atr(high, low, close, period=14) -> pd.Series:
    """Average True Range. Shift(1)."""

def bollinger_width(close, period=20, std=2) -> pd.Series:
    """(upper - lower) / middle. Shift(1)."""

def volume_ratio(volume, window) -> pd.Series:
    """current / rolling_mean. Shift(1)."""

def obv(close, volume) -> pd.Series:
    """On-Balance Volume. Shift(1)."""

def rolling_stats(returns, windows) -> pd.DataFrame:
    """mean, std, max, min, skew, kurtosis per window. ALL shifted by 1."""

def lag_features(returns, lags) -> pd.DataFrame:
    """returns.shift(lag) for each lag."""

def taker_ratio(taker_buy_vol, total_vol) -> pd.Series:
    """Proxy for bid-ask pressure (klines don't have spread)."""

# === MASTER PIPELINE ===
def build_features(df) -> pd.DataFrame:
    """Calls all above. Adds cyclical time-of-day (sin/cos of hour, minute).
    Drops warmup NaN rows. ~40-50 feature columns. CRITICAL: all .shift(1)."""

# === TARGET ===
def realized_volatility(log_rets, window=10) -> pd.Series:
    """Forward-looking std(returns[t+1:t+window]). Last 10 rows NaN. ddof=1."""

def log_transform_target(rv) -> pd.Series:
    """log(rv + 1e-10) for training stability."""
```

### 3.3 `src/models.py` -- All Model Definitions

```python
# === GARCH ===
def test_stationarity(returns) -> dict:
    """ADF test -> {adf_stat, p_value, is_stationary}"""

def fit_garch(returns, p=1, q=1, dist='t', scale=100):
    """Scale 100x, fit with arch library. Student-t for fat tails."""

def garch_expanding_forecast(returns, min_window, refit_freq, horizon) -> pd.Series:
    """Refit every 500 steps. Convert: sqrt(variance)/100 for original scale.
    PITFALL: Must divide by 100!"""

# === XGBOOST ===
def xgb_optuna_tune(train_X, train_y, val_X, val_y, n_trials) -> dict:
    """Optuna TPE sampler. Returns best params."""

def xgb_expanding_predict(features, targets, params, min_window, refit_freq) -> pd.Series:
    """Walk-forward, refit every 500 steps. No shuffling."""

# === LSTM ===
class LuongAttention(nn.Module):
    """Dot-product attention over LSTM outputs."""

class VolatilityLSTM(nn.Module):
    """LSTM(input, 128, layers=2, dropout=0.3) -> [Attention] -> Linear(128,1)
    Without attention: lstm_out[:, -1, :]
    With attention: weighted context vector."""

# === PYTORCH DATA ===
class VolatilityDataset(Dataset):
    """Sliding window. x=[seq_len, n_feat], y=scalar.
    x = features[idx:idx+seq_len], y = targets[idx+seq_len]."""

def normalize_features(train, val, test):
    """Z-score with TRAINING stats only. Returns normed arrays + params."""

def create_dataloaders(features, targets, ratios, seq_len, batch_size):
    """Chrono split -> normalize -> Dataset -> DataLoader(shuffle=False)."""
```

### 3.4 `src/training.py` -- Training & Prediction

```python
def set_all_seeds(seed=42):
    """random, numpy, torch, cuda seeds. deterministic=True."""

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    """Forward -> MSE -> backward -> clip_grad_norm_(1.0) -> step."""

def validate(model, loader, criterion, device) -> float:
    """No-grad eval. Returns avg loss."""

def train_lstm(model, train_loader, val_loader, config) -> dict:
    """Adam(weight_decay), ReduceLROnPlateau(patience=5), early stop(patience=10).
    TensorBoard logging. Checkpoint best model.
    Returns {model, train_losses, val_losses, best_epoch}."""

def predict_lstm(model, loader, device) -> np.ndarray:
    """Inference. Returns predictions."""

def inverse_transform(preds, log_transformed=True) -> np.ndarray:
    """exp(pred) - 1e-10. All models must be on same original scale."""

def collect_predictions(garch, xgb, lstm, actual, timestamps) -> pd.DataFrame:
    """Align all to common test timestamps."""
```

### 3.5 `src/evaluation.py` -- Metrics, Hypothesis Test, Plots

```python
# === METRICS ===
def mse(a, p): ...
def rmse(a, p): ...
def mae(a, p): ...
def qlike(a, p):
    """Patton 2011. PITFALL: needs variance (square vol first). pred > 0."""
def r_squared(a, p): ...
def directional_accuracy(a, p):
    """Fraction of correct direction-of-change."""
def all_metrics(a, p) -> dict: ...
def metrics_table(results_dict) -> pd.DataFrame:
    """Formatted table + .to_latex()"""

# === HYPOTHESIS TEST ===
def block_bootstrap_test(baseline_err, lstm_err, block_size, n_reps, conf) -> dict:
    """d_t = baseline_err^2 - lstm_err^2. MovingBlockBootstrap on d_t.
    Returns {observed_delta, ci_lower, ci_upper, p_value, reject_null, deltas}."""

# === EDA PLOTS ===
def plot_price_volume(df): ...
def plot_return_dist(returns): ...        # histogram + KDE + normal overlay
def plot_qq(returns): ...                 # fat tails
def plot_acf_squared(returns): ...        # volatility clustering
def plot_rv_timeseries(rv): ...           # target variable
def plot_correlation(features): ...       # heatmap
def plot_rolling_stats(df): ...

# === RESULTS PLOTS ===
def plot_predictions(pred_df): ...        # all models vs actual
def plot_residuals(actual, pred, name): ... # 4-panel
def plot_training_curves(train_l, val_l): ...
def plot_bootstrap_dist(deltas, delta_obs, ci): ...
def plot_metrics_bar(table): ...          # grouped bar chart
def plot_cumulative_error(pred_df): ...
def plot_attention_weights(weights): ...  # if using attention
```

---

## 4. Execution Pipeline

```
Step 1: Data Collection + Cleaning + DuckDB
  fetch_klines() -> clean_raw_data() -> init_duckdb() -> ingest_candles()

Step 2: Feature Engineering
  build_features(df) -> realized_volatility() -> save parquet

Step 3: EDA
  All EDA plots on raw data and features

Step 4: GARCH Baseline
  test_stationarity() -> garch_expanding_forecast()

Step 5: XGBoost Baseline
  xgb_optuna_tune() -> xgb_expanding_predict()

Step 6: LSTM Training (A100)
  create_dataloaders() -> train_lstm() -> predict_lstm() -> inverse_transform()

Step 7: Evaluation
  collect_predictions() -> metrics_table() -> block_bootstrap_test()
  -> all results plots -> export LaTeX table

Step 8: Consolidate into final_notebook.ipynb
  Copy all code inline, add markdown headers, ensure self-contained
```

---

## 5. Key Technical Decisions & Pitfalls

1. **Bid-ask proxy**: Use `taker_buy_base_volume / volume` (klines lack spread data)
2. **GARCH refit every 500 steps** (not every step = 10+ hours)
3. **GARCH scale bug**: returns scaled 100x -> `predicted_vol = sqrt(variance) / 100`
4. **Student-t for GARCH**: `dist='t'` always for crypto fat tails
5. **LSTM target**: log(rv + 1e-10), invert as exp(pred) - 1e-10 before metrics
6. **Leakage prevention**: every rolling/lag uses `.shift(1)`. Validate with corruption test
7. **Same feature matrix** for XGBoost (flat) and LSTM (60-step window)
8. **ddof=1** for realized volatility (sample std, 10 observations)
9. **QLIKE needs variance**: square volatilities before passing
10. **No shuffling** anywhere in data pipeline

---

## 6. A100 Cluster

```bash
#!/bin/bash
#SBATCH --job-name=crypto-vol-lstm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module load python/3.11 cuda/12.1
source venv/bin/activate
python -c "from src.training import *; from src.models import *; ..."
```

~30 min on A100 (250 batches/epoch, early stopping ~30-50 epochs).

---

## 7. Requirements

```
python-binance>=1.0.19
duckdb>=0.10.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
optuna>=3.4.0
arch>=6.2.0
torch>=2.1.0
tensorboard>=2.15.0
pyarrow>=14.0.0
tqdm>=4.66.0
statsmodels>=0.14.0
scipy>=1.11.0
```

---

## 8. Final Notebook Consolidation Strategy

When ready to submit:
1. Create `final_notebook.ipynb`
2. Each `src/*.py` module becomes a section with markdown header
3. All `import` statements go in the first cell
4. All config constants go in the second cell
5. Helper functions defined in cells before they're used
6. Each pipeline step has markdown explanation + code cell
7. Plots render inline with `%matplotlib inline`
8. DuckDB queries shown as SQL strings (demonstrates course concept)
9. Regex patterns shown explicitly with examples (demonstrates course concept)
10. Notebook must be runnable top-to-bottom (restart kernel + run all)

---

## 9. Verification Checklist

1. **Data**: 130K rows, no gaps > 2 min, ADF confirms stationarity
2. **Features**: no leakage (corruption test), no unexpected 1.0 correlations
3. **GARCH**: reasonable summary, correct scale (divided by 100)
4. **XGBoost**: Optuna finds reasonable params, feature importance makes sense
5. **LSTM**: loss decreasing, val diverges (early stopping works), no gradient explosion
6. **Hypothesis**: bootstrap distribution ~normal, CI width reasonable
7. **End-to-end**: all predictions aligned, same scale, figures match tables
