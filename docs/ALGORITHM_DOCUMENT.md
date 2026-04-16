# CIS 5450 — Cryptocurrency Volatility Prediction: Algorithm & Pipeline Document

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Acquisition](#2-data-acquisition)
3. [Data Cleaning & Validation](#3-data-cleaning--validation)
4. [DuckDB Storage Layer](#4-duckdb-storage-layer)
5. [Feature Engineering](#5-feature-engineering)
6. [Target Variable](#6-target-variable)
7. [Data Splitting & Normalization](#7-data-splitting--normalization)
8. [Model 1: GARCH(1,1)](#8-model-1-garch11)
9. [Model 2: XGBoost](#9-model-2-xgboost)
10. [Model 3: LSTM](#10-model-3-lstm)
11. [Model 4: PatchTST Transformer](#11-model-4-patchtst-transformer)
12. [Evaluation Framework](#12-evaluation-framework)
13. [Hypothesis Testing](#13-hypothesis-testing)
14. [XGBoost Ablation Analysis](#14-xgboost-ablation-analysis)
15. [Pipeline Execution Order](#15-pipeline-execution-order)

---

## 1. Project Overview

This project builds an end-to-end machine learning pipeline to predict short-term cryptocurrency volatility. Specifically, we forecast the **10-minute forward realized volatility** of BTC/USDT using 1-minute candlestick data from Binance. We compare four model families — a classical econometric model (GARCH), a gradient-boosted tree model (XGBoost), a recurrent neural network (LSTM with optional attention), and a patch-based transformer (PatchTST) — evaluating them under identical conditions with strict chronological splitting to prevent data leakage.

The pipeline satisfies CIS 5450 course requirements for **Regex**, **DuckDB/SQL**, **Pandas**, and **Hypothesis Testing**.

---

## 2. Data Acquisition

### Source

Binance public data repository at `data.binance.vision`. This is a freely accessible archive — no API key is required and there are no regional restrictions.

### Instrument & Granularity

- **Trading pair:** BTCUSDT (Bitcoin quoted in Tether)
- **Candle interval:** 1 minute
- **Date range:** 2024-04-01 to 2025-03-31 (1 full year)
- **Expected row count:** ~525,600 candles (365 days x 1,440 minutes/day)

### Download Mechanism

Implemented in `src/data_collection.py`, function `fetch_klines()` (lines 37–104).

For each calendar day in the date range, the pipeline:

1. Constructs a URL:
   ```
   https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{YYYY-MM-DD}.zip
   ```
2. Downloads the ZIP file via HTTP GET (30-second timeout per request)
3. Extracts the single CSV file contained inside
4. Parses the CSV into a DataFrame (12 columns, no header row)
5. Appends to a running list of daily DataFrames

After all days are downloaded, the frames are concatenated and saved as a Parquet file at `data/raw/btcusdt_1m_raw.parquet` for idempotent reuse. Days that fail to download are logged but do not halt the pipeline.

### Raw Data Schema

Each row represents one 1-minute candle with 12 fields:

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `open_time` | int64 | Candle open time, Unix timestamp in milliseconds |
| 2 | `open` | float64 | Opening price in USDT |
| 3 | `high` | float64 | Highest price during the minute |
| 4 | `low` | float64 | Lowest price during the minute |
| 5 | `close` | float64 | Closing price in USDT |
| 6 | `volume` | float64 | Trade volume in BTC |
| 7 | `close_time` | int64 | Candle close time, Unix timestamp in milliseconds |
| 8 | `quote_volume` | float64 | Trade volume in USDT |
| 9 | `num_trades` | int | Number of individual trades in the minute |
| 10 | `taker_buy_base_volume` | float64 | Volume from taker (market) buy orders, in BTC |
| 11 | `taker_buy_quote_volume` | float64 | Volume from taker buy orders, in USDT |
| 12 | `ignore` | — | Unused field, dropped during cleaning |

---

## 3. Data Cleaning & Validation

Cleaning is implemented in `src/data_collection.py`, function `clean_raw_data()` (lines 194–273). It consists of six sequential steps.

### Step 1: Regex Timestamp Validation

**Function:** `validate_timestamps()` (line 143)
**Regex pattern:** `^\d{13,16}$` (defined in `src/config.py` as `RE_TIMESTAMP_MS`)

Every value in the `open_time` column is converted to a string and matched against this pattern. The pattern accepts 13-digit (millisecond-precision) and 16-digit (microsecond-precision) Unix timestamps. Rows with values that do not match (e.g., corrupted data, headers mixed into CSV rows) are dropped.

### Step 2: Regex Numeric Field Validation

**Function:** `validate_numeric_fields()` (line 167)
**Regex pattern:** `^-?\d+\.?\d*$` (defined as `RE_NUMERIC`)
**Columns validated:** `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `taker_buy_base_volume`, `taker_buy_quote_volume`

Each value in these 8 columns is matched against the numeric regex. Any row where at least one column fails is dropped. After validation, all columns are coerced to `float64` via `pd.to_numeric`.

### Step 3: Regex Trading Pair Extraction

**Function:** `extract_trading_pair()` (line 156)
**Regex pattern:** `^([A-Z]{2,10})(USDT|BTC|ETH)$` (defined as `RE_TRADING_PAIR`)

This function extracts the base and quote currencies from the symbol string (e.g., `"BTCUSDT"` → `("BTC", "USDT")`). It is called as a demonstration of regex extraction (course requirement) and logged for verification.

### Step 4: Type Casting & Timestamp Normalization

- `open_time` and `close_time` are cast to `int64`
- `num_trades` is cast to `int`
- Mixed timestamp formats are handled: values >= 1×10^15 are treated as microseconds and divided by 1000 to normalize to milliseconds
- `open_time` is converted to a UTC `datetime64` index via `pd.to_datetime(ts_ms, unit="ms", utc=True)`
- The `open_time`, `close_time`, and `ignore` columns are dropped

### Step 5: Sort & Deduplicate

The DataFrame is sorted chronologically by its datetime index. Duplicate timestamps are removed, keeping only the first occurrence.

### Step 6: Gap Handling (Forward-Fill Policy)

The data is reindexed to a complete 1-minute frequency grid covering the full time range. This reveals any missing candles.

- **Gaps ≤ 2 minutes** are forward-filled. The rationale is that short gaps are typically caused by zero-trade minutes, and the most recent candle values are a reasonable fill.
- **Gaps > 2 minutes** are not filled. These indicate genuine data unavailability (exchange maintenance, outages). Rows in these gaps remain NaN and are subsequently dropped.

Gap sizes are detected by grouping consecutive NaN values and measuring each group's length.

### Completeness Validation

Before cleaning, the function `validate_completeness()` (line 107) checks:
- Total row count
- Number of duplicate `open_time` values
- Number and locations of time gaps > 60,000 ms (1 minute)
- Null counts per column

### Output

The cleaned DataFrame is saved to `data/processed/btcusdt_1m_clean.parquet`. Typical result: ~525,600 rows with a complete 1-minute datetime index in UTC.

---

## 4. DuckDB Storage Layer

Implemented in `src/data_collection.py`, lines 280–404.

### Schema

Two tables are created in `data/processed/crypto_volatility.duckdb`:

**Table `raw_candles`:**
```sql
CREATE TABLE raw_candles (
    timestamp  TIMESTAMP WITH TIME ZONE  PRIMARY KEY,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    volume     DOUBLE,
    quote_volume           DOUBLE,
    num_trades             INTEGER,
    taker_buy_base_volume  DOUBLE,
    taker_buy_quote_volume DOUBLE
)
```

**Table `predictions`:**
```sql
CREATE TABLE predictions (
    timestamp      TIMESTAMP WITH TIME ZONE,
    model_name     VARCHAR,
    predicted_vol  DOUBLE,
    actual_vol     DOUBLE,
    PRIMARY KEY (timestamp, model_name)
)
```

### SQL Operations

1. **Bulk ingestion** (`ingest_candles`, line 316): Cleaned candle data is inserted into `raw_candles`. The table is cleared before each insertion for idempotent reloading.

2. **Time-bucket resampling** (`query_resampled`, line 332): Demonstrates DuckDB's `TIME_BUCKET` function with GROUP BY aggregation to resample 1-minute candles to arbitrary intervals (e.g., 5 minutes):
   ```sql
   SELECT
       time_bucket(INTERVAL '5 minutes', timestamp) AS bucket,
       FIRST(open ORDER BY timestamp)  AS open,
       MAX(high)                       AS high,
       MIN(low)                        AS low,
       LAST(close ORDER BY timestamp)  AS close,
       SUM(volume)                     AS volume,
       SUM(quote_volume)               AS quote_volume,
       SUM(num_trades)                 AS num_trades
   FROM raw_candles
   GROUP BY bucket
   ORDER BY bucket
   ```

3. **Multi-model comparison** (`compare_models_sql`, line 381): A 3-way JOIN across the `predictions` table comparing per-timestamp squared errors for GARCH, XGBoost, and LSTM:
   ```sql
   SELECT
       g.timestamp, g.actual_vol,
       g.predicted_vol AS garch_pred,
       x.predicted_vol AS xgboost_pred,
       l.predicted_vol AS lstm_pred,
       POWER(g.actual_vol - g.predicted_vol, 2) AS garch_se,
       POWER(x.actual_vol - x.predicted_vol, 2) AS xgboost_se,
       POWER(l.actual_vol - l.predicted_vol, 2) AS lstm_se
   FROM predictions g
   JOIN predictions x ON g.timestamp = x.timestamp AND x.model_name = 'xgboost'
   JOIN predictions l ON g.timestamp = l.timestamp AND l.model_name = 'lstm'
   WHERE g.model_name = 'garch'
   ORDER BY g.timestamp
   ```

---

## 5. Feature Engineering

Implemented in `src/feature_engineering.py`, function `build_features()` (lines 147–202).

**Critical design rule:** Every rolling or lag-based feature is `.shift(1)` so that the value at time _t_ is computed using only data from time _t−1_ or earlier. This prevents any look-ahead bias.

The pipeline constructs **52 features** organized into 7 groups.

### 5.1 Log Return (1 feature)

```python
log_return = log(close_t / close_{t-1}).shift(1)
```

The basic building block for volatility modeling. Shifted by 1 so we do not use the current minute's return to predict current-minute volatility.

### 5.2 Range-Based Volatility Estimators (8 features)

Two classical estimators, each computed at four rolling window sizes [5, 10, 30, 60]:

**Parkinson (1980) volatility** (line 29–38):
```
parkinson_w = sqrt( (1 / (4 * n * ln(2))) * sum(ln(H/L)^2) )   [rolling mean version]
```
Uses only high and low prices. Approximately 5x more efficient than a close-to-close estimator because it captures intra-period price range.

**Garman-Klass (1980) volatility** (line 41–54):
```
GK = 0.5 * (ln(H/L))^2  -  (2*ln(2) - 1) * (ln(C/O))^2       [rolling mean, then sqrt]
```
Uses open, high, low, and close. Approximately 7.4x more efficient than close-to-close. The `clip(lower=0)` ensures the square root argument is non-negative.

Both estimators are shifted by 1 to prevent leakage.

**Features produced:** `parkinson_5`, `parkinson_10`, `parkinson_30`, `parkinson_60`, `garman_klass_5`, `garman_klass_10`, `garman_klass_30`, `garman_klass_60`

### 5.3 Technical Indicators (2 features)

**Average True Range (ATR)** (line 57–71, period=14):
```
TR = max(H - L,  |H - Close_{t-1}|,  |L - Close_{t-1}|)
ATR = EWM(TR, span=14)
```
Measures average price range, capturing both gaps and intra-bar movement.

**Bollinger Band Width** (line 74–85, period=20, 2 standard deviations):
```
SMA = rolling_mean(close, 20)
STD = rolling_std(close, 20)
bollinger_width = (SMA + 2*STD - (SMA - 2*STD)) / SMA  =  4*STD / SMA
```
Measures volatility expansion and contraction relative to the moving average.

### 5.4 Volume Features (5 features)

**Volume Ratio** (line 88–92, windows [10, 30, 60]):
```
volume_ratio_w = current_volume / rolling_mean(volume, w)
```
Detects unusual volume spikes. Values > 1 indicate above-average activity.

**On-Balance Volume (OBV)** (line 95–106):
```
OBV = cumsum(sign(close_diff) * volume)
OBV_normalized = z-score over 60-minute rolling window
```
Cumulative signed volume reflecting buying vs. selling pressure. Normalized to prevent scale drift.

**Taker Ratio** (line 134–140):
```
taker_ratio = taker_buy_base_volume / (total_volume + 1e-10)
```
Proxy for bid-ask imbalance. Values > 0.5 indicate net buying pressure. Since Binance kline data does not include actual bid-ask spreads, this ratio is the best available approximation.

### 5.5 Rolling Return Statistics (24 features)

For each window in [5, 10, 30, 60], we compute 6 statistics of log returns (line 109–124):

| Feature | Formula | Captures |
|---------|---------|----------|
| `ret_mean_w` | rolling mean of returns | trend/momentum |
| `ret_std_w` | rolling std of returns | recent volatility |
| `ret_max_w` | rolling max return | extreme upward moves |
| `ret_min_w` | rolling min return | extreme downward moves |
| `ret_skew_w` | rolling skewness | asymmetry of returns |
| `ret_kurt_w` | rolling kurtosis | tail heaviness |

All 24 features are shifted by 1. These provide the model with a multi-scale view of return dynamics at different time horizons.

### 5.6 Lag Features (10 features)

```
lag_k = returns.shift(k)    for k = 1, 2, ..., 10
```

Simple autoregressive features allowing the model to directly observe the most recent 10 individual returns.

### 5.7 Cyclical Time Encoding (2 features)

```
hour = index.hour + index.minute / 60.0
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
```

Encodes time-of-day as a smooth cyclical feature. Cryptocurrency markets run 24/7, but volatility patterns still exhibit intraday seasonality (e.g., higher volatility during US/European trading hours).

### Summary Table

| Group | Count | Features |
|-------|-------|----------|
| Log Return | 1 | `log_return` |
| Volatility Estimators | 8 | `parkinson_{5,10,30,60}`, `garman_klass_{5,10,30,60}` |
| Technical Indicators | 2 | `atr`, `bollinger_width` |
| Volume Features | 5 | `volume_ratio_{10,30,60}`, `obv`, `taker_ratio` |
| Rolling Statistics | 24 | `ret_{mean,std,max,min,skew,kurt}_{5,10,30,60}` |
| Lag Features | 10 | `lag_1` through `lag_10` |
| Time Encoding | 2 | `hour_sin`, `hour_cos` |
| **Total** | **52** | |

After computing all features, warmup NaN rows (from rolling windows requiring initial data) are dropped. The feature matrix is saved to `data/processed/features.parquet`.

---

## 6. Target Variable

Implemented in `src/feature_engineering.py`, function `realized_volatility()` (lines 209–223).

### Definition

The target is **forward-looking 10-minute realized volatility**:

```
target_t = std(log_returns[t+1], log_returns[t+2], ..., log_returns[t+10])
```

where `std` uses sample standard deviation (ddof=1).

### Implementation

```python
rv = log_rets.rolling(window=10).std()   # rolling std ending at each point
rv = rv.shift(-10)                        # shift backward to align with prediction origin
```

The `shift(-10)` ensures that the value at time _t_ equals the standard deviation of the 10 returns that follow _t_. The last 10 rows are NaN because there is no future data beyond the dataset.

### Log-Transformed Target

For LSTM training stability, a log-transformed version is also saved:
```
target_log = log(realized_volatility + 1e-10)
```

The small epsilon prevents log(0). This is inverted after prediction via `exp(pred) - 1e-10`.

### Why This Target?

Realized volatility — the actual observed standard deviation of returns over a fixed horizon — is the standard benchmark target in volatility forecasting literature. It is directly observable ex-post, making it a well-defined supervised learning target. The 10-minute horizon balances granularity (short enough to be actionable) with signal quality (long enough to be meaningfully above noise).

---

## 7. Data Splitting & Normalization

### Chronological Split

All models use the same strict time-ordered split (defined in `src/config.py` lines 50–53):

| Split | Ratio | Rows (approx.) | Purpose |
|-------|-------|-----------------|---------|
| Train | 85% | ~446,760 | Model fitting / parameter estimation |
| Validation | 5% | ~26,280 | Hyperparameter tuning, early stopping, learning rate scheduling |
| Test | 10% | ~52,560 (~36 days) | Final evaluation; no model has seen this data |

**No shuffling** is used anywhere in the pipeline. DataLoaders are created with `shuffle=False`. This is critical for time series to prevent future information from leaking into the training set.

### Feature Normalization

For LSTM and Transformer models, features are z-score normalized (function `normalize_features()` in `src/models.py` lines 458–475):

```
feature_normalized = (feature - mean_train) / std_train
```

Statistics (mean and std) are computed **only on the training set** and applied identically to validation and test sets. Columns with near-zero standard deviation (< 1e-10) are left unscaled (std set to 1.0).

GARCH and XGBoost do not require normalization.

---

## 8. Model 1: GARCH(1,1)

Implemented in `src/models.py`, lines 30–135.

### Overview

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is the canonical econometric model for volatility forecasting. Unlike the ML models, GARCH does **not** use the 52 engineered features. It operates directly on the raw log return series, modeling the conditional variance as a function of past squared innovations and past variances.

### Input / Output

| | Description |
|---|---|
| **Input** | Raw log return series: `log(close_t / close_{t-1})`, scaled by 100 |
| **Output** | Predicted 10-minute forward realized volatility (one scalar per minute, original scale) |

### Model Specification

The GARCH(1,1) with constant mean:

```
Return equation:     r_t = mu + epsilon_t
Innovation:          epsilon_t = sigma_t * z_t,     z_t ~ Student-t(nu)
Variance equation:   sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
```

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| mu | Constant mean return | — |
| omega (ω) | Variance intercept | > 0 |
| alpha (α) | ARCH coefficient — sensitivity to recent shocks | >= 0 |
| beta (β) | GARCH coefficient — persistence of past variance | >= 0 |
| nu | Student-t degrees of freedom | > 2 |
| alpha + beta | Persistence | < 1 for stationarity |

### Why Student-t Distribution?

Cryptocurrency returns exhibit **extreme fat tails** (excess kurtosis typically 50–200 for 1-minute BTC data). The Student-t distribution allows the model to assign higher probability to extreme returns than the Gaussian would, producing better-calibrated variance estimates.

### Stationarity Check

Before fitting, an Augmented Dickey-Fuller (ADF) test verifies that log returns are stationary (function `test_stationarity()`, line 30). This is a prerequisite for GARCH — the model assumes a stationary return process. For crypto returns, the null hypothesis (unit root) is always rejected at p < 0.01.

### Rolling Forecast Algorithm

Function `garch_rolling_forecast()` (lines 61–135):

```
Configuration:
    GARCH_RETURN_SCALE = 100        # scale returns for numerical convergence
    GARCH_MIN_WINDOW   = 5000       # minimum observations for first fit
    GARCH_REFIT_FREQ   = 500        # re-estimate parameters every 500 steps

Algorithm:
    scaled_returns = raw_returns * 100

    FOR t = start_pos TO end:

        # 1. Re-estimate parameters periodically
        IF t is first step OR (t - start_pos) % 500 == 0:
            window_data = scaled_returns[max(0, t-5000) : t]
            Fit GARCH(1,1) with Student-t on window_data
            Extract: omega, alpha, beta, mu
            Initialize h_t = last conditional variance from fit

        # 2. Update conditional variance with GARCH recursion (EVERY step)
        epsilon = scaled_returns[t-1] - mu
        h_t = omega + alpha * epsilon^2 + beta * h_t

        # 3. Multi-step ahead forecast (10 steps for TARGET_WINDOW=10)
        persistence = alpha + beta
        h_k = h_t
        var_forecasts = []
        FOR k = 1 TO 10:
            h_k = omega + persistence * h_k
            var_forecasts.append(h_k)

        # 4. Convert to predicted realized volatility
        predicted_rv = sqrt(mean(var_forecasts)) / SCALE

        predictions[t] = predicted_rv
```

**Key design decision:** Parameters (ω, α, β) are expensive to estimate and change slowly, so they are re-estimated only every 500 steps. However, the conditional variance h_t is updated at **every** step using the GARCH recursion. This ensures predictions react immediately to new market data (e.g., a sudden price crash) even between refits.

The multi-step forecast uses the mean of the next 10 variance forecasts because our target is the standard deviation over 10 returns — so the predicted RV is `sqrt(mean(h_{t+1}, ..., h_{t+10}))`.

---

## 9. Model 2: XGBoost

Implemented in `src/models.py`, lines 142–223.

### Overview

XGBoost (eXtreme Gradient Boosting) is a gradient-boosted decision tree ensemble. Unlike GARCH, it has no structural assumptions about the data-generating process — it learns arbitrary nonlinear mappings from features to target.

### Input / Output

| | Description |
|---|---|
| **Input** | All 52 engineered features at time _t_ (a single 52-dimensional vector) |
| **Output** | Predicted realized volatility at time _t_ (scalar, original scale) |

### Hyperparameter Tuning

Function `xgb_optuna_tune()` (lines 142–191).

We use **Optuna** with the **TPE (Tree-structured Parzen Estimator)** sampler to search over 50 trials. The objective is to minimize MSE on the validation set.

| Hyperparameter | Search Range | Scale |
|----------------|-------------|-------|
| `max_depth` | 3–8 | integer |
| `learning_rate` | 0.01–0.1 | log-uniform |
| `n_estimators` | 100–1000 | integer |
| `subsample` | 0.6–1.0 | uniform |
| `colsample_bytree` | 0.6–1.0 | uniform |
| `min_child_weight` | 1–10 | integer |
| `gamma` | 0.0–0.3 | uniform |
| `reg_alpha` (L1) | 0.0–1.0 | uniform |
| `reg_lambda` (L2) | 0.0–1.0 | uniform |

The loss function is `reg:squarederror` (standard MSE regression). The best parameter set is saved to `outputs/models/xgboost_best_params.pkl`.

### Tuning Results

After 50 Optuna TPE trials, the best hyperparameters found are:

| Hyperparameter | Best Value | Interpretation |
|----------------|-----------|----------------|
| `max_depth` | 6 | Mid-range tree depth; deep enough to capture feature interactions, shallow enough to avoid overfitting |
| `learning_rate` | 0.0682 | Moderate step size (search was 0.01–0.1 log-scale) |
| `n_estimators` | 256 | Relatively few trees (search allowed up to 1000); combined with moderate LR, prevents overfitting |
| `subsample` | 0.732 | ~73% row sampling per tree — adds stochasticity to reduce variance |
| `colsample_bytree` | 0.800 | ~80% feature sampling per tree — mild feature dropout |
| `min_child_weight` | 7 | Conservative — requires at least 7 samples per leaf, suppresses splits on noise |
| `gamma` | 0.000222 | Near-zero minimum split gain — allows most informative splits through |
| `reg_alpha` (L1) | 0.151 | Light L1 regularization — mild feature sparsity pressure |
| `reg_lambda` (L2) | 0.682 | Meaningful L2 regularization — smooths leaf weights to prevent overfitting |

The combination of moderate depth (6), few trees (256), high `min_child_weight` (7), and strong L2 regularization (0.682) indicates the tuner favored a model that resists overfitting — which is expected for noisy 1-minute financial data where most variation is unpredictable.

### Test Set Performance

| Metric | Value |
|--------|-------|
| MSE | 1.076 × 10⁻⁷ |
| RMSE | 3.280 × 10⁻⁴ |
| MAE | 1.852 × 10⁻⁴ |
| QLIKE | 0.3210 |
| R² | 0.6131 |
| DA | 0.3772 |

XGBoost achieves the **highest R² (0.613)** and the **lowest QLIKE (0.321)** among all 12 models tested, including all LSTM and Transformer variants. Its low QLIKE indicates particularly well-calibrated variance forecasts. The relatively low directional accuracy (37.7%) reflects that XGBoost tends to produce smooth predictions that track the level of volatility well but lag behind rapid directional changes.

### Feature Importance Analysis

Gain-based feature importances from the trained XGBoost model reveal which features contribute most to prediction accuracy:

**Top 10 features (accounting for 87% of total importance):**

| Rank | Feature | Importance | Cumulative | Category |
|------|---------|-----------|------------|----------|
| 1 | `ret_std_30` | 32.75% | 32.75% | Rolling volatility (30-min) |
| 2 | `ret_std_10` | 17.07% | 49.82% | Rolling volatility (10-min) |
| 3 | `ret_std_60` | 13.62% | 63.43% | Rolling volatility (60-min) |
| 4 | `parkinson_30` | 7.05% | 70.48% | Parkinson estimator (30-min) |
| 5 | `garman_klass_10` | 5.42% | 75.90% | Garman-Klass estimator (10-min) |
| 6 | `parkinson_5` | 3.69% | 79.59% | Parkinson estimator (5-min) |
| 7 | `parkinson_10` | 3.19% | 82.78% | Parkinson estimator (10-min) |
| 8 | `bollinger_width` | 2.05% | 84.83% | Technical indicator |
| 9 | `parkinson_60` | 1.33% | 86.16% | Parkinson estimator (60-min) |
| 10 | `garman_klass_5` | 1.28% | 87.44% | Garman-Klass estimator (5-min) |

**Key insight:** Volatility estimators dominate overwhelmingly. The top 3 features alone — all rolling standard deviations at different windows — account for 63% of total importance. This confirms the intuition that recent realized volatility is the strongest predictor of near-future volatility (volatility clustering).

**Zero-importance features (11 features):** `ret_kurt_30`, `ret_skew_5`, and `lag_3` through `lag_10` received zero importance, meaning the model never split on them. This suggests that autoregressive lag information beyond lag_2 is redundant when rolling statistics are available.

### Feature Group Ablation

Ablation experiments measure the impact of removing entire feature groups:

| Experiment | Removed | Kept | R² | Delta R² |
|------------|---------|------|-----|----------|
| **All features (baseline)** | 0 | 52 | **0.5698** | — |
| No ATR/Bollinger | 2 | 50 | 0.5700 | +0.04% |
| No lag features | 10 | 42 | 0.5693 | -0.09% |
| No volume features | 5 | 47 | 0.5689 | -0.16% |
| No time encoding | 2 | 50 | 0.5673 | -0.44% |
| No volatility estimators | 8 | 44 | 0.5572 | -2.21% |
| No rolling stats | 24 | 28 | 0.5565 | -2.33% |
| Only rolling stats | — | 24 | 0.5556 | -2.49% |
| Only lag features | — | 10 | 0.4622 | -18.88% |

**Findings:**
- **Rolling statistics and volatility estimators are critical** — removing either drops R² by ~2.3%. These are the backbone of the model.
- **Lag features, volume, and time encoding are nearly redundant** — removing any of these has < 0.5% impact, because the rolling statistics already encode temporal and volume information.
- **ATR/Bollinger removal actually improves R² marginally** (+0.04%), suggesting slight collinearity with existing volatility features.
- **Using only lag features collapses R² to 0.462** — a 19% drop, confirming that raw lagged returns alone are insufficient without the derived rolling statistics.

### Top-K Feature Analysis

Training with only the top-K features (ranked by importance) shows diminishing returns:

| Top-K | R² | Marginal Gain |
|-------|-----|---------------|
| 5 | 0.5362 | — |
| 10 | 0.5477 | +1.15% |
| 15 | 0.5493 | +0.16% |
| 20 | 0.5602 | +1.09% |
| 30 | 0.5683 | +0.81% |
| 40 | 0.5690 | +0.07% |
| 52 (all) | 0.5698 | +0.08% |

Just the top 5 features achieve R² = 0.536 — **94% of the full model's performance** with only 10% of the features. Performance essentially saturates at 20–30 features; the remaining 22–32 features contribute only 0.15% additional R².

### Expanding Window Prediction

Function `xgb_expanding_predict()` (lines 194–223).

To prevent data leakage during evaluation, XGBoost uses a **walk-forward expanding window** on the test set:

```
Configuration:
    min_window    = val_end      # start predicting from test set boundary
    refit_freq    = 500          # refit every 500 steps

Algorithm:
    model = None

    FOR t = min_window TO n:

        # Refit periodically on all data up to time t
        IF model is None OR (t - min_window) % 500 == 0:
            model = XGBRegressor(**best_params)
            model.fit(X[0:t], y[0:t])

        # Predict at time t using features at time t
        prediction[t] = model.predict(X[t])
```

Between refits, the same trained model is used for prediction. The expanding window means each refit has access to strictly more data than the last, capturing evolving market dynamics.

### Why XGBoost for Volatility?

- Tree models naturally handle feature interactions (e.g., high volume + high kurtosis → regime change)
- Robust to irrelevant features — built-in feature selection via split gain
- No normalization required — invariant to monotonic feature transformations
- Fast training and prediction, even with 500K+ rows

---

## 10. Model 3: LSTM

Implemented in `src/models.py` (lines 230–312, 438–540) and `src/training.py`.

### Overview

Long Short-Term Memory (LSTM) networks are recurrent neural networks designed to capture temporal dependencies. Unlike XGBoost, which sees a single feature vector at time _t_, the LSTM sees a **sequence** of feature vectors covering a lookback window, enabling it to learn temporal patterns across multiple timesteps.

### Input / Output

| | Description |
|---|---|
| **Input** | Sliding window of shape `(seq_len, 52)` — the 52 normalized features over the past `seq_len` minutes |
| **Output** | Predicted log-transformed realized volatility at time `t + seq_len` (scalar), inverse-transformed to original scale for evaluation |

### Architecture

```
Input: (batch, seq_len, 52)
    │
    ▼
LSTM(input_size=52, hidden_size=128, num_layers=2, dropout=0.3, batch_first=True)
    │
    ├── lstm_out: (batch, seq_len, 128)     ← hidden state at every timestep
    └── h_n:      (2, batch, 128)           ← final hidden state per layer
    │
    ▼
[If attention=True]:
    LuongAttention(hidden_size=128)
        scores = lstm_out @ W @ h_n[-1]^T           (batch, seq_len)
        weights = softmax(scores)                    (batch, seq_len)
        context = weighted_sum(lstm_out, weights)    (batch, 128)
[If attention=False]:
    context = lstm_out[:, -1, :]     ← last timestep's hidden state  (batch, 128)
    │
    ▼
Linear(128 → 1)
    │
    ▼
Output: (batch,)    ← squeeze to scalar
```

### Luong Attention Mechanism

The optional Luong (general) attention mechanism (`LuongAttention`, lines 230–259) allows the model to attend to any timestep in the lookback window, weighted by relevance to the prediction:

1. **Score:** Each timestep's hidden state is scored against the final hidden state via a learned weight matrix W: `score_i = h_i^T W h_final`
2. **Normalize:** Scores are passed through softmax to obtain attention weights that sum to 1
3. **Context:** The output is a weighted sum of all hidden states: `context = sum(alpha_i * h_i)`

This helps when the most informative signal is not always at the most recent timestep — for example, a volatility spike 30 minutes ago may still be predictive.

### Dataset Construction

`VolatilityDataset` (lines 438–455) creates sliding windows:

```
Sample i:
    x = normalized_features[i : i + seq_len]    → shape (seq_len, 52)
    y = log_target[i + seq_len]                  → scalar
```

No shuffling is used. The DataLoader iterates sequentially through time.

### Training Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Optimizer | Adam | Adaptive learning rate per parameter |
| Initial LR | 1×10^-3 | Starting learning rate |
| Weight decay | 1×10^-4 | L2 regularization |
| Loss function | MSELoss | Mean squared error |
| Gradient clipping | max norm 1.0 | Prevents exploding gradients |
| LR scheduler | ReduceLROnPlateau | Halves LR after 5 epochs without val improvement |
| LR factor | 0.5 | Reduction multiplier |
| Early stopping | 10 epochs patience | Stops training after 10 epochs without val improvement |
| Max epochs | 100 | Upper bound (early stopping typically triggers earlier) |
| Batch size | 512 | Tuned for GPU memory |
| Random seed | 42 | Reproducibility (numpy, torch, cuda, cuDNN) |

### Training Loop

Function `train_lstm()` in `src/training.py` (lines 91–193):

```
FOR epoch = 1 TO max_epochs:
    train_loss = train_one_epoch(model, train_loader, optimizer, MSELoss)
        # For each batch:
        #   forward pass → loss → backward pass → clip gradients → optimizer step
    val_loss = validate(model, val_loader, MSELoss)
        # No gradients, just forward pass

    scheduler.step(val_loss)     # reduce LR if val_loss plateaus

    IF val_loss < best_val_loss:
        best_val_loss = val_loss
        Save model checkpoint
        Reset patience counter
    ELSE:
        patience_counter += 1
        IF patience_counter >= 10: BREAK (early stopping)

Load best model checkpoint
```

### Experiments

Six experiments are run (`scripts/run_lstm.py` lines 163–170):

| Experiment | seq_len | Attention | Lookback |
|------------|---------|-----------|----------|
| seq60_noattn | 60 | No | 1 hour |
| seq60_attn | 60 | Yes | 1 hour |
| seq120_noattn | 120 | No | 2 hours |
| seq120_attn | 120 | Yes | 2 hours |
| seq240_noattn | 240 | No | 4 hours |
| seq240_attn | 240 | Yes | 4 hours |

The experiment with the lowest validation loss is selected as the primary LSTM model. Its predictions are saved as `lstm_predictions.pkl`.

### Inference & Inverse Transform

After training, inference runs in `eval()` mode with `torch.no_grad()`. Predictions are log-transformed values, so they are converted back to original scale:

```python
predicted_rv = exp(predicted_log_rv) - 1e-10
```

---

## 11. Model 4: PatchTST Transformer

Implemented in `src/models.py`, lines 318–431.

### Overview

PatchTST segments the input time series into patches (small subsequences), projects each patch into a learned embedding, and processes the patch sequence with a standard Transformer encoder. This reduces the effective sequence length and allows the self-attention mechanism to capture dependencies at multiple time scales.

### Architecture

```
Input: (batch, seq_len, 52)
    │
    ▼
PatchEmbedding(patch_len=12, stride=6)
    - Unfold: split seq_len into overlapping patches
    - n_patches = (seq_len - 12) / 6 + 1
    - Each patch: (12 * 52) = 624 values → Linear → d_model
    │
    ▼
+ Learnable Positional Encoding: (1, n_patches, d_model)
    │
    ▼
TransformerEncoder (num_layers × [MultiHeadSelfAttention + FeedForward])
    - activation: GELU
    - norm_first: True (pre-norm for training stability)
    │
    ▼
Global Average Pooling: mean across n_patches → (batch, d_model)
    │
    ▼
Regression Head:
    LayerNorm → Linear(d_model, d_model/2) → GELU → Dropout → Linear(d_model/2, 1)
    │
    ▼
Output: (batch,)
```

### Configurations

Two model sizes are tested:

| Config | d_model | Heads | Layers | d_ff | Dropout |
|--------|---------|-------|--------|------|---------|
| small | 128 | 4 | 2 | 256 | 0.2 |
| full | 256 | 8 | 4 | 1024 | 0.2 |

Each config is tested with `seq_len` of 120 and 240 (4 experiments total).

### Training

- Optimizer: Adam (lr=1×10^-4, weight_decay=1×10^-4)
- Schedule: 5-epoch warmup + cosine decay
- Loss: MSELoss
- Early stopping: 10 epochs patience
- Max epochs: 80

---

## 12. Evaluation Framework

Implemented in `src/evaluation.py`, lines 28–98.

All models are evaluated on the same held-out test set (~52,560 samples, ~36 days) using these metrics:

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | `mean((actual - pred)^2)` | Average squared prediction error. Lower is better. |
| **RMSE** | `sqrt(MSE)` | Prediction error in the same units as the target. Lower is better. |
| **MAE** | `mean(\|actual - pred\|)` | Average absolute error, less sensitive to outliers than MSE. Lower is better. |
| **QLIKE** | `mean(σ²_a/σ²_p - log(σ²_a/σ²_p) - 1)` | Quasi-likelihood loss from Patton (2011). Robust metric for variance forecasting that heavily penalizes underestimation. Lower is better. Predicted variance is clipped to 1×10^-20 minimum. |
| **R²** | `1 - SS_res / SS_tot` | Fraction of variance explained. Higher is better. Can be negative if the model is worse than predicting the mean. |
| **DA** | `fraction(sign(Δactual) == sign(Δpred))` | Directional accuracy — how often the model correctly predicts whether volatility will increase or decrease. Higher is better; 0.5 = random. |

### Visualization Suite

**EDA plots** (before modeling):
- Price and volume dual-axis time series
- Log return distribution with Normal overlay and excess kurtosis annotation
- Q-Q plot against Normal distribution (reveals fat tails)
- ACF of returns and squared returns (reveals volatility clustering)
- Realized volatility time series
- Feature correlation heatmap
- Rolling mean and std of returns at multiple scales

**Results plots** (after modeling):
- Predicted vs. actual realized volatility overlay (all models)
- 4-panel residual analysis per model: residuals over time, histogram, Q-Q plot, ACF
- LSTM training curves (train vs. val loss per epoch)
- Bootstrap distribution histogram with CI
- Grouped bar chart comparing all metrics across models
- Cumulative squared error over time (shows where each model struggles)
- Attention weight heatmap (for LSTM with attention)

---

## 13. Hypothesis Testing

Implemented in `src/evaluation.py`, function `block_bootstrap_test()` (lines 105–165).

### Hypothesis

```
H₀: MSE_LSTM >= MSE_baseline     (LSTM is no better than baseline)
H₁: MSE_LSTM <  MSE_baseline     (LSTM is significantly better)
```

### Method: Moving Block Bootstrap

Standard bootstrap assumes i.i.d. samples, which is invalid for time series with autocorrelated errors. The **Moving Block Bootstrap** preserves temporal dependence by resampling contiguous blocks rather than individual observations.

```
Configuration:
    block_size  = 50        # ~n^(1/3) for 130K samples, captures local dependence
    n_reps      = 1000      # bootstrap replications
    confidence  = 0.95      # 95% confidence level

Algorithm:
    1. Compute the loss differential series:
       d_t = (baseline_error_t)^2 - (lstm_error_t)^2
       Positive d_t means LSTM had lower squared error at time t.

    2. Observed test statistic:
       delta_obs = mean(d_t)

    3. For b = 1 to 1000:
       Draw a Moving Block Bootstrap sample d_t* from d_t
       Compute delta_b* = mean(d_t*)

    4. 95% confidence interval:
       CI = [percentile(deltas*, 2.5%), percentile(deltas*, 97.5%)]

    5. p-value:
       p = fraction of delta_b* <= 0

    6. Decision:
       Reject H₀ if p < 0.05
       (equivalently, if the entire 95% CI is above 0)
```

If the null is rejected, we conclude that the LSTM's MSE improvement over the baseline is statistically significant at the 5% level, even after accounting for temporal dependence in the error series.

---

## 14. XGBoost Ablation Analysis

Implemented in `scripts/run_xgb_ablation.py`.

### 14.1 Feature Importance

After training XGBoost on the full feature set, gain-based feature importances are extracted. Each feature's importance is the total reduction in loss (gain) from all splits that use it, normalized to sum to 1.

### 14.2 Feature Group Ablation

Nine ablation experiments are run, each removing one group of features and retraining:

| Experiment | Features Removed | Features Kept |
|------------|-----------------|---------------|
| All features (baseline) | 0 | 52 |
| No volatility estimators | 8 (parkinson_*, garman_klass_*) | 44 |
| No rolling stats | 24 (ret_mean/std/max/min/skew/kurt_*) | 28 |
| No lag features | 10 (lag_1 through lag_10) | 42 |
| No volume features | 5 (volume_ratio_*, obv, taker_ratio) | 47 |
| No time encoding | 2 (hour_sin, hour_cos) | 50 |
| No ATR/Bollinger | 2 (atr, bollinger_width) | 50 |
| Only lag features | 42 (everything except lags) | 10 |
| Only rolling stats | 28 (everything except rolling stats) | 24 |

### 14.3 Top-K Feature Analysis

Models are trained using only the top K features (ranked by importance), for K = 5, 10, 15, 20, 30, 40, 52. This reveals the diminishing returns from adding lower-importance features.

---

## 15. Pipeline Execution Order

| Step | Script | Inputs | Outputs |
|------|--------|--------|---------|
| 1 | `scripts/run_collection.py` | Binance Vision URLs | `data/raw/btcusdt_1m_raw.parquet`, `data/processed/btcusdt_1m_clean.parquet`, `crypto_volatility.duckdb` |
| 2 | `scripts/run_features.py` | `btcusdt_1m_clean.parquet` | `features.parquet`, `target.parquet`, `target_log.parquet` |
| 3 | `scripts/run_baselines.py` | `btcusdt_1m_clean.parquet`, `features.parquet`, `target.parquet` | `garch_predictions.pkl`, `xgboost_predictions.pkl`, `xgboost_best_params.pkl` |
| 4 | `scripts/run_lstm.py` | `features.parquet`, `target.parquet` | `lstm_predictions_*.pkl`, `lstm_history_*.pkl`, `lstm_best.pt` |
| 4b | `scripts/run_transformer.py` | `features.parquet`, `target.parquet` | `tfm_predictions_*.pkl`, `tfm_history_*.pkl` |
| 5 | `scripts/run_evaluation.py` | All prediction `.pkl` files | `metrics.csv`, `metrics.tex`, all figures |
| 6 | `scripts/run_xgb_ablation.py` | `features.parquet`, `target.parquet`, `xgboost_best_params.pkl` | `xgb_feature_importance.csv`, `xgb_ablation.csv`, `xgb_topk.csv`, plots |

### File Structure

```
CIS 5450 Volatility/
├── src/
│   ├── config.py                 # All hyperparameters and paths
│   ├── data_collection.py        # Fetch, regex clean, DuckDB ingest
│   ├── feature_engineering.py    # 52 features + target computation
│   ├── models.py                 # GARCH, XGBoost, LSTM, Transformer definitions
│   ├── training.py               # Training loop, prediction, seed management
│   └── evaluation.py             # Metrics, hypothesis testing, plots
├── scripts/
│   ├── run_collection.py         # Step 1
│   ├── run_features.py           # Step 2
│   ├── run_baselines.py          # Step 3
│   ├── run_lstm.py               # Step 4
│   ├── run_transformer.py        # Step 4b
│   ├── run_evaluation.py         # Step 5
│   └── run_xgb_ablation.py       # Step 6
├── data/
│   ├── raw/                      # Downloaded parquet files
│   └── processed/                # Cleaned data, features, targets, DuckDB
├── outputs/
│   ├── models/                   # Predictions, checkpoints, params
│   ├── figures/                  # All generated plots
│   └── tables/                   # CSV and LaTeX metric tables
└── requirements.txt
```
