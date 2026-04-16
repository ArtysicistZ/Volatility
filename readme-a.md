# Predicting BTC/USDT Short-Term Volatility

**CIS 5450 — Big Data Analytics Final Project**

We predict the **realized volatility of Bitcoin over the next 10 minutes** from 1-minute OHLCV candles, and compare a classical econometric baseline (GARCH), a gradient-boosted baseline (XGBoost), and deep-learning models (LSTM, Transformer) on an identical out-of-sample test set.

> **The problem:** Given the last hour of Bitcoin prices, how "jumpy" will the next 10 minutes be? Volatility isn't the *direction* of price, but it's the *amplitude of wiggle*. Traders care because options prices, stop-loss sizing, and risk budgets all depend on it.

---

## 0. Project Layout

```
Volatility/
├── src/                      # library code (no entry points)
│   ├── config.py             # every hyperparameter / path lives here
│   ├── data_collection.py    # Binance API + regex + DuckDB
│   ├── feature_engineering.py
│   ├── models.py             # GARCH / XGBoost / LSTM / Transformer
│   ├── training.py           # LSTM training loop
│   └── evaluation.py         # metrics + block bootstrap + plots
├── scripts/                  # thin entry points — one per pipeline stage
│   ├── run_collection.py     # Stage 1
│   ├── run_features.py       # Stage 2
│   ├── run_baselines.py      # Stage 3
│   ├── run_lstm.py           # Stage 4a
│   ├── run_transformer.py    # Stage 4b
│   ├── run_xgb_ablation.py   # Stage 4c (feature-group ablation)
│   └── run_evaluation.py     # Stage 5
├── docs/                     # design + findings writeups
└── outputs/                  # figures, tables, model artifacts
```

Pipeline is strictly sequential — each stage writes parquet/pickle artifacts that the next stage reads:

```
Binance → clean candles → features + target → predictions → metrics + plots
```

---

## 1. Data Collection & Cleaning

**Source:** [src/data_collection.py](src/data_collection.py) · **Driver:** [scripts/run_collection.py](scripts/run_collection.py)

> **In a nutshell.** Our goal in Stage 1 is to turn raw Binance CSV files, which contain mixed-precision timestamps, occasional non-numeric junk, duplicates, and missing minutes into a **clean, gap-free, strictly-typed minute-indexed time series** stored both as a parquet file (for downstream ML) and a DuckDB database (for SQL queries). We do this in four passes: (1) regex-validate every field before trusting it, (2) cast to proper types and normalize timestamps to a single unit, (3) reindex to a complete 1-minute grid so lag features stay aligned, (4) ingest into DuckDB. Conceptually: **quarantine bad rows early, then guarantee a perfectly regular time axis.**

### 1.1 The raw data

We download daily ZIP files from `data.binance.vision` (Binance's public archive — no API key needed). Each ZIP contains one CSV of that day's 1-minute candles. A raw row looks like:

```csv
open_time,    open,     high,     low,      close,    volume,   close_time,    quote_volume, num_trades, taker_buy_base_volume, taker_buy_quote_volume, ignore
1711929600000,70500.10, 70520.50, 70480.20, 70505.30, 12.456,   1711929659999, 878654.32,    1203,       6.234,                 439567.12,              0
1711929660000,70505.30, 70560.00, 70499.90, 70545.80, 18.372,   1711929719999, 1296012.40,   1584,       10.102,                712654.88,              0
```

One year of BTCUSDT at 1-minute resolution ≈ **525,000 rows × 12 columns**. Plenty of chances for messy data.

But the table looks very confusing. 

### 1.2 What can go wrong

Real-world API data has the usual pathologies:

| Problem | Example | Why it matters |
|---|---|---|
| Mixed timestamp precision | `1711929600000` (ms) vs `1711929600000000` (µs) | Parsing as datetime breaks silently |
| Non-numeric fields | `"N/A"` in `close` | Math ops propagate NaN through every feature |
| Duplicate timestamps | Same minute appears twice | Rolling windows double-count |
| Missing minutes | Exchange downtime, API hiccup | Gaps in the time axis misalign lag features |

### 1.3 How we clean it — step by step

Implemented in [`clean_raw_data()`](src/data_collection.py#L194). **This is where the course's regex and SQL requirements land.**

**Step 1 — Regex validation (course requirement: regex).**
Before we trust any value, we test it against a regular expression. See [config.py:112-114](src/config.py#L112-L114):

```python
RE_TIMESTAMP_MS = r"^\d{13,16}$"                     # 13-digit ms or 16-digit µs
RE_TRADING_PAIR = r"^([A-Z]{2,10})(USDT|BTC|ETH)$"  # e.g. BTCUSDT → (BTC, USDT)
RE_NUMERIC      = r"^-?\d+\.?\d*$"                   # valid decimal number
```

- [`validate_timestamps()`](src/data_collection.py#L143) checks every `open_time` is a 13–16 digit integer; invalid rows are logged and dropped.
- [`validate_numeric_fields()`](src/data_collection.py#L167) runs `RE_NUMERIC` over every OHLCV column; a single bad cell kicks the row out.
- [`extract_trading_pair()`](src/data_collection.py#L156) uses the capture groups in `RE_TRADING_PAIR` to split `"BTCUSDT"` into `("BTC", "USDT")`.

The core of the validator is a one-liner applied columnwise:

```python
# src/data_collection.py
def validate_numeric_fields(df, columns):
    pattern = re.compile(RE_NUMERIC)
    invalid_rows = pd.Series(False, index=df.index)
    for col in columns:
        mask = df[col].astype(str).apply(lambda x: bool(pattern.fullmatch(x)))
        invalid_rows |= ~mask                 # union of "bad" masks across columns
    df = df[~invalid_rows]                    # drop any row with any bad cell
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
```

> **Analogy.** Think of regex as a bouncer at a club door. It has a strict rule ("must be all digits, must be 13 to 16 of them") and turns away anyone who doesn't match — *before* they get on the dance floor and cause trouble downstream.

**Step 2 — Type coercion & timestamp normalization.**
After the bouncer does its job, we cast to proper dtypes (`float64` for prices/volumes, `int64` for timestamps). Microsecond timestamps (`> 1e15`) are floor-divided by 1000 to unify everything in milliseconds, then converted to a UTC `DatetimeIndex`:

```python
ts_numeric = df["open_time"].astype(np.int64)
ts_ms = ts_numeric.where(ts_numeric < 1e15, ts_numeric // 1000)  # normalize us→ms
df["timestamp"] = pd.to_datetime(ts_ms, unit="ms", utc=True)
df = df.set_index("timestamp").sort_index()
df = df[~df.index.duplicated(keep="first")]
```

**Step 3 — Sort, deduplicate, handle gaps.**
We reindex to a complete 1-minute grid (`pd.date_range(..., freq="1min")`), then:
- **Gaps ≤ 2 minutes** → forward-fill (short hiccup, last-known-price is reasonable)
- **Gaps > 2 minutes** → leave as NaN and drop (better to lose data than fabricate it)

**Step 4 — DuckDB ingest (course requirement: SQL).**
Cleaned candles go into a DuckDB database ([`init_duckdb()`](src/data_collection.py#L280)) with two tables: `raw_candles` (one row per minute) and `predictions` (one row per model×timestamp). We then demo real SQL:

```sql
-- From query_resampled() — course requirement: GROUP BY + aggregations
SELECT
    time_bucket(INTERVAL '5 minutes', timestamp) AS bucket,
    FIRST(open ORDER BY timestamp) AS open,
    MAX(high)                      AS high,
    MIN(low)                       AS low,
    LAST(close ORDER BY timestamp) AS close,
    SUM(volume)                    AS volume
FROM raw_candles
GROUP BY bucket
ORDER BY bucket;
```

```sql
-- From compare_models_sql() — course requirement: JOIN
SELECT g.timestamp,
       g.actual_vol,
       g.predicted_vol AS garch_pred,
       x.predicted_vol AS xgboost_pred,
       l.predicted_vol AS lstm_pred,
       POWER(g.actual_vol - g.predicted_vol, 2) AS garch_se,
       ...
FROM predictions g
JOIN predictions x ON g.timestamp = x.timestamp AND x.model_name = 'xgboost'
JOIN predictions l ON g.timestamp = l.timestamp AND l.model_name = 'lstm';
```

**Output of Stage 1:** `data/processed/btcusdt_1m_clean.parquet` and `crypto_volatility.duckdb`.

---

## 2. Feature Engineering

**Source:** [src/feature_engineering.py](src/feature_engineering.py) · **Driver:** [scripts/run_features.py](scripts/run_features.py)

> **In a nutshell.** Our goal in Stage 2 is to turn each minute's single OHLCV row into a **~45-dimensional feature vector that summarizes everything the market did *before* that minute**, plus compute the target (the std of returns in the *next* 10 minutes). Conceptually, we're answering two parallel questions for every timestamp *t*: "what does the recent past look like?" (features) and "what will the immediate future look like?" (target). The entire design revolves around one rule — **no feature may peek at time ≥ *t*** — enforced by a `.shift(1)` at the end of every computation. We build feature *families* (volatility estimators, technical indicators, volume, lags, rolling stats, time-of-day) rather than hand-picking individual features, and let XGBoost ablation tell us which families earn their keep.

### 2.1 What we're trying to predict (the target)

Realized volatility over the **next** 10 minutes:

$$\text{RV}_t = \text{std}\big(\, r_{t+1},\, r_{t+2},\, \ldots,\, r_{t+10} \,\big), \quad r_t = \ln(P_t / P_{t-1})$$

Implemented as `rolling(10).std().shift(-10)` in [`realized_volatility()`](src/feature_engineering.py#L209):

```python
def realized_volatility(log_rets, window=10):
    rv = log_rets.rolling(window=window).std()   # std over window ENDING at t
    rv = rv.shift(-window)                       # shift back: value at t = std of [t+1 … t+10]
    return rv
```

> **Why log returns, not raw prices?** Log returns are roughly normally distributed, symmetric, and additive across time. Price levels wander; returns are *stationary* — which is exactly what every model below assumes.

### 2.2 The golden rule: no look-ahead

Every single feature ends with `.shift(1)`. If we're predicting at minute *t*, the feature for minute *t* must only use information from minute *t−1* and earlier.

> **Why this matters.** If even one feature accidentally uses the current bar's close price, the model achieves fantastic test scores — and then fails catastrophically in live trading. This is *the* #1 bug in financial ML.

### 2.3 Feature groups (~45 columns total)

We didn't hand-pick a "best" set; we built families motivated by well-known properties of financial returns, then let the XGBoost ablation ([scripts/run_xgb_ablation.py](scripts/run_xgb_ablation.py)) tell us which groups actually matter.

| Group | What it captures | Implementation |
|---|---|---|
| **Log returns** | Simple past direction | `log_returns()` |
| **Parkinson vol** | Range-based volatility from high-low | `parkinson_vol()` — $\sqrt{\frac{1}{4n\ln 2}\sum (\ln(H/L))^2}$ |
| **Garman-Klass vol** | Range-based using OHLC, ~7× more efficient than close-to-close | `garman_klass_vol()` |
| **ATR (Average True Range)** | Typical daily movement (smoothed) | EWM of true range |
| **Bollinger width** | Is price currently expanding or contracting? | `(upper − lower) / middle` band |
| **Volume ratios** | Is this minute unusually active? | `volume / rolling_mean(volume)` at windows 10/30/60 |
| **OBV (On-Balance Volume)** | Cumulative signed volume, z-scored | `sign(Δclose) * volume`, cumsum, normalized |
| **Taker ratio** | Aggressiveness of buyers (bid-ask pressure proxy) | `taker_buy_volume / total_volume` |
| **Rolling return stats** | Past shape: mean/std/max/min/skew/kurt | Across 5/10/30/60-min windows |
| **Lag features** | Raw past returns `r_{t-1}`…`r_{t-10}` | `returns.shift(lag)` |
| **Time-of-day** | Asia/EU/US trading sessions | `sin(2πh/24), cos(2πh/24)` |

> **Why so many volatility estimators?** Close-to-close volatility only uses one price per bar and throws away high/low information. Parkinson and Garman-Klass squeeze *the same information* out of every candle — like measuring a wave by its peak and trough instead of just one point on the surface.

> **Why cyclical time encoding?** Encoding "hour = 23" vs "hour = 0" as raw numbers tells the model they're far apart when really they're adjacent. `sin/cos` places them next to each other on a circle — the model learns "late night" as a continuous thing, not a jump.

A representative feature — Garman-Klass volatility — shows the `.shift(1)` discipline in action:

```python
def garman_klass_vol(open_, high, low, close, window):
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    result = np.sqrt(gk.rolling(window=window).mean().clip(lower=0))
    return result.shift(1)   # ← the crucial anti-leakage step
```

### 2.4 Feature selection via ablation (not manual pruning)

Rather than guess which features matter, [`run_xgb_ablation.py`](scripts/run_xgb_ablation.py) trains XGBoost with each feature group *removed* in turn. The MSE increase when a group is dropped = that group's marginal value. Outputs: `outputs/tables/xgb_ablation.csv` and `outputs/figures/xgb_ablation.png`.

**Output of Stage 2:** `features.parquet` (~45 columns), `target.parquet`, `target_log.parquet`.

---

## 3. Baseline Models

**Source:** [src/models.py](src/models.py) · **Driver:** [scripts/run_baselines.py](scripts/run_baselines.py)

> **In a nutshell.** Before reaching for deep learning, we establish **two strong, very different baselines** — so that any deep-model win is meaningful. GARCH is a *parametric econometric* model that explicitly encodes one fact about markets ("volatility clusters") with just three learnable numbers. XGBoost is a *non-parametric machine-learning* model that ignores theory and just learns flexible nonlinear functions from our ~45 engineered features. Both are evaluated with the same **walk-forward protocol**: refit periodically on all data up to time *t*, predict the next chunk, slide forward. Conceptually: GARCH tests whether a single structural assumption is enough; XGBoost tests how far brute-force pattern-matching on tabular features can go.

### 3.1 GARCH(1,1) — the econometric classic

GARCH = *Generalized Autoregressive Conditional Heteroskedasticity*. Ugly name; simple idea.

> **The intuition.** Volatility *clusters*: calm days follow calm days, and once the market gets shaky it tends to *stay* shaky for a while. GARCH is the simplest formal way to model that "today's volatility is partly yesterday's volatility plus a surprise."

**The model.** Let $r_t$ = log return, $\epsilon_t = r_t - \mu$ = innovation, $h_t$ = conditional variance. Then GARCH(1,1) says:

$$h_t = \omega + \alpha \cdot \epsilon_{t-1}^2 + \beta \cdot h_{t-1}$$

- $\omega$ = long-run average variance (a floor)
- $\alpha$ = how much a *shock* yesterday spikes today's variance
- $\beta$ = how much yesterday's *variance* itself persists (memory)

If $\alpha + \beta \approx 1$, volatility has long memory — which is exactly what you see in crypto.

**What we do:** [`garch_rolling_forecast()`](src/models.py#L61) uses a rolling 5000-minute window and refits $(\omega, \alpha, \beta)$ every 500 steps (refitting every minute is ~100× slower and barely changes the fit). But the key trick: **$h_t$ is recursively updated every single step** using the latest $\epsilon_{t-1}^2$, so we never let a stale variance estimate drift. We use a Student-t innovation distribution (not Gaussian) because crypto returns have fat tails — extreme moves are more common than a normal distribution would predict.

Forecasting 10 minutes ahead: iterate the GARCH recursion forward 10 steps, average the forecasted variances, take the square root → predicted RV. The inner loop of `garch_rolling_forecast()` is literally this:

```python
# Per-step variance update (runs every minute)
eps = scaled.iloc[t - 1] - mu            # yesterday's innovation
h_t = omega + alpha * eps**2 + beta * h_t  # GARCH(1,1) recursion

# 10-step-ahead forecast from the current h_t
h_k, var_forecasts = h_t, []
for k in range(horizon):                  # horizon = 10
    h_k = omega + (alpha + beta) * h_k    # mean-reverting projection
    var_forecasts.append(h_k)
pred_vol = np.sqrt(np.mean(var_forecasts)) / scale
```

### 3.2 XGBoost — the gradient boosting workhorse

> **The intuition.** Imagine you're trying to estimate someone's age from a photo. A single decision tree might split on hair color, then on wrinkles, then on clothes — and give a rough guess. XGBoost builds *hundreds* of such trees, where **each new tree focuses on correcting the mistakes of all trees before it**. The final prediction is the sum of every tree's contribution. That "boosting" step is what makes it so much more accurate than a single tree or even a random forest.

**Ingredients:**
- **Input:** all ~45 engineered features above as one flat vector per minute.
- **Output:** predicted realized volatility.
- **Hyperparameter search:** [`xgb_optuna_tune()`](src/models.py#L142) uses Optuna's TPE sampler (a smarter-than-grid-search Bayesian optimizer) for 50 trials over max_depth, learning rate, number of trees, subsampling ratios, and regularization strengths. It picks the combo with the lowest validation MSE.
- **Training protocol:** [`xgb_expanding_predict()`](src/models.py#L194) uses a **walk-forward expanding window** — refit every 500 minutes on all past data, then predict the next 500 steps. This mimics how you'd deploy the model in production: you always train on yesterday-or-earlier and predict tomorrow.

> **Why walk-forward rather than one big train/test split?** Because markets change. A model fit once on 2024 data degrades as 2025 rolls in. Walk-forward is the gold standard honest evaluation for time series.

```python
# Core of xgb_expanding_predict — refit every 500 steps, predict the next one
for t in range(min_window, n):
    if model is None or (t - min_window) % refit_freq == 0:
        model = xgb.XGBRegressor(**params)
        model.fit(X[:t], y[:t], verbose=False)    # train on all history up to t
    predictions.iloc[t] = model.predict(X[t:t+1])[0]
```

---

## 4. Deep Models

**Source:** [src/models.py](src/models.py) (classes) + [src/training.py](src/training.py) (loop) · **Drivers:** [scripts/run_lstm.py](scripts/run_lstm.py), [scripts/run_transformer.py](scripts/run_transformer.py)

> **In a nutshell.** Our goal in Stage 4 is to **exploit the sequence structure that XGBoost throws away**: XGBoost sees each minute as an independent row of features, but minute *t* is obviously a continuation of minutes *t−1*, *t−2*, …, *t−k*. We try two architectures that natively consume sequences: the **LSTM** walks through the window step-by-step maintaining a gated memory, and the **Transformer** looks at every timestep at once via self-attention. Both share identical data plumbing (chronological split, training-set normalization, sliding-window dataset) so the only thing that changes between models is the architecture. Conceptually: we hand the model the *raw history* — a 60–240 minute tensor of features — and let it figure out which parts of that history are predictive of the next 10 minutes' volatility.

### 4.1 Why not just use XGBoost?

XGBoost sees each minute as an independent row. It cannot natively model the fact that **minute *t* is a sequence continuation of minutes *t−1*, *t−2*, …** We engineered lag features to partly compensate, but a model that *natively* consumes sequences should do better. Hence LSTM and Transformer.

### 4.2 LSTM — [`VolatilityLSTM`](src/models.py#L262)

> **The intuition.** An LSTM (Long Short-Term Memory network) is a neural network that reads a sequence one step at a time and maintains a "running memory" that it decides to update, keep, or forget at each step. Think of it like reading a novel: your understanding at chapter 20 depends on chapters 1–19, but you don't remember every word — you remember the *gist*.

**Architecture:**
```
Input:  (batch, seq_len=60, ~45 features)
  │
  ▼
2-layer LSTM (hidden=128, dropout=0.3 between layers)
  │
  ▼
[Optional] Luong attention over all 60 hidden states
  │                    (learns which minutes in the window mattered most)
  ▼
Linear(128 → 1)  →  predicted log(RV)
```

**Why attention is optional.** A plain LSTM only passes its *last* hidden state to the output layer — so a signal from 50 minutes ago has to survive 50 updates to still be useful. [`LuongAttention`](src/models.py#L230) gives the model a second look at **every** timestep and learns a weighted average. We sweep `seq_len ∈ {60, 120, 240}` × `{attention, no attention}` in [`run_lstm.py`](scripts/run_lstm.py) to see when attention actually helps.

**Training specifics** ([`train_lstm()`](src/training.py#L91)):
- **Adam** optimizer, learning rate 1e-3, weight decay 1e-4 (L2 regularization).
- **Gradient clipping** at max-norm 1.0 — LSTMs can explode without it.
- **ReduceLROnPlateau** — if val loss plateaus for 5 epochs, halve the LR.
- **Early stopping** — if val loss hasn't improved in 10 epochs, stop and restore the best checkpoint.
- **Target is log-transformed** (`log(RV + ε)`) before training so MSE treats a 10× miss the same at low and high volatilities. We inverse-transform at evaluation so every model is compared on the original RV scale.

The forward pass is only a few lines — everything above is support structure around it:

```python
class VolatilityLSTM(nn.Module):
    def forward(self, x):                      # x: (batch, seq_len, n_features)
        lstm_out, (h_n, _) = self.lstm(x)      # lstm_out: (batch, seq_len, hidden)
        if self.attention is not None:
            context, _ = self.attention(lstm_out, h_n[-1])   # weighted sum over time
            out = self.fc(context)
        else:
            out = self.fc(lstm_out[:, -1, :])  # use only the last timestep
        return out.squeeze(-1)                  # → predicted log(RV)
```

### 4.3 Transformer (PatchTST) — [`VolatilityTransformer`](src/models.py#L350)

> **The intuition.** Transformers swap the LSTM's step-by-step memory for **self-attention**: every position in the sequence can *directly* look at every other position in one shot. PatchTST adds a clever preprocessing step — chop the time series into short "patches" (like dividing a paragraph into phrases instead of reading letter-by-letter). Fewer tokens → shorter attention sequences → faster training and, empirically, better accuracy on long horizons.

**Architecture:**
```
Input:  (batch, seq_len=240, ~45 features)
  │
  ▼
PatchEmbedding           # unfold into patches of length 12, stride 6
                         # → (batch, n_patches=39, patch_len * n_features)
  │                      # → Linear project to d_model=128
  ▼
+ Learnable positional encoding
  │
  ▼
TransformerEncoder       # 2 layers × [Multi-head self-attention (4 heads) + FFN]
                         # Pre-norm + GELU (stable training)
  │
  ▼
Mean-pool over patches   → (batch, d_model)
  │
  ▼
MLP head (128 → 64 → 1)  → predicted log(RV)
```

### 4.4 Shared data plumbing — [`create_dataloaders()`](src/models.py#L478)

Both deep models share identical data handling to ensure a fair fight:

1. **Chronological split.** 85% train / 5% validation / 10% test. **No shuffling** — ever. `DataLoader(shuffle=False)`.
2. **Z-score normalization using training-set statistics only.** Val and test are normalized with the *train* mean and std. This prevents leakage — future information never informs the preprocessing.
3. **Sliding-window dataset** ([`VolatilityDataset`](src/models.py#L438)): sample *i* = (features\[i : i+seq_len], target\[i+seq_len]).

---

## 5. Evaluation & Results

**Source:** [src/evaluation.py](src/evaluation.py) · **Driver:** [scripts/run_evaluation.py](scripts/run_evaluation.py)

> **In a nutshell.** Our goal in Stage 5 is to **deliver a fair, statistically defensible comparison of all four models** on the same out-of-sample test window. Point estimates ("LSTM's MSE is 4% lower") aren't enough — one time window could easily be lucky. So we pair six complementary metrics (covering error magnitude, asymmetry, and directionality) with a **Moving Block Bootstrap hypothesis test** that accounts for autocorrelation in the loss series. Conceptually: we want to say not just *"LSTM won"* but *"LSTM beat GARCH by Δ MSE, with a 95% confidence interval that excludes zero, at p < 0.05"* — the kind of claim that holds up in a paper.

### 5.1 Metrics

All six are computed for every model on the same test set ([`all_metrics()`](src/evaluation.py#L78)):

| Metric | Formula | What it tells you |
|---|---|---|
| **MSE** | $\text{mean}((y - \hat{y})^2)$ | Primary loss. Penalizes large errors heavily. |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as volatility → easier to interpret. |
| **MAE** | $\text{mean}(|y - \hat{y}|)$ | Robust to outliers. |
| **QLIKE** | $\text{mean}(\tfrac{y^2}{\hat{y}^2} - \ln\tfrac{y^2}{\hat{y}^2} - 1)$ | Volatility-specific loss (Patton 2011). Asymmetric — punishes *under*-forecasting more. |
| **R²** | $1 - \text{SSR}/\text{SST}$ | Fraction of variance explained. |
| **Directional Accuracy** | Fraction of correct $\text{sign}(\Delta)$ | Does the model at least predict up-vs-down correctly? |

### 5.2 The hypothesis test — why a simple MSE comparison isn't enough

If LSTM beats GARCH on MSE by some margin, is that a *real* improvement or just luck on this particular test window?

> **The analogy.** If team A beats team B 3–2 in one game, is A actually the better team? You'd need many games to tell. For time series, we can't replay history — so we use a **block bootstrap** to simulate it.

[`block_bootstrap_test()`](src/evaluation.py#L105) implements the Moving Block Bootstrap:

1. Compute per-timestamp loss differentials $d_t = e_t^{\text{baseline},2} - e_t^{\text{LSTM},2}$.
2. Draw **1000 bootstrap samples**, each formed by concatenating random contiguous **blocks of length 50** from $\{d_t\}$. (Blocks preserve local autocorrelation — a fatal flaw of naive i.i.d. bootstrap on time series.)
3. Compute the mean of each bootstrap sample → a distribution of 1000 $\bar{d}^*$ values.
4. **95% confidence interval** = 2.5th–97.5th percentiles.
5. **p-value** = fraction of bootstrap samples with $\bar{d}^* \leq 0$.
6. **Reject $H_0$: MSE_LSTM ≥ MSE_baseline** if p < 0.05 and 0 lies outside the CI.

```python
# Core of block_bootstrap_test — preserves autocorrelation via block resampling
d = baseline_errors**2 - lstm_errors**2          # per-timestep loss differential
bs = MovingBlockBootstrap(block_size=50, d, seed=rng)
boot_deltas = np.array([np.mean(data[0]) for data, in bs.bootstrap(n_reps=1000)])

ci_lower, ci_upper = np.percentile(boot_deltas, [2.5, 97.5])
p_value = np.mean(boot_deltas <= 0)              # one-sided test
reject_null = p_value < 0.05
```

### 5.3 Figures generated

[`generate_eda_plots()`](src/evaluation.py#L471) and [`generate_result_plots()`](src/evaluation.py#L489) emit, into `outputs/figures/`:

| File | What it shows |
|---|---|
| `01_price_volume.png` | BTC price + volume over the full year |
| `02_return_distribution.png` | Return histogram vs. Normal (fat tails visible) |
| `03_qq_plot.png` | Q-Q plot confirming non-normality |
| `04_acf_squared.png` | ACF of squared returns → **volatility clustering evidence** |
| `05_realized_volatility.png` | The target variable over time |
| `06_correlation.png` | 45×45 feature correlation heatmap |
| `07_rolling_stats.png` | Rolling mean/std at multiple windows |
| `08_predictions.png` | All models' forecasts overlaid on actuals |
| `09_residuals_*.png` | 4-panel residual diagnostics per model |
| `10_training_curves.png` | LSTM train/val loss with best-epoch marker |
| `11_bootstrap.png` | Bootstrap distribution of MSE differentials |
| `12_metrics_comparison.png` | Grouped bar chart across all 6 metrics |
| `13_cumulative_error.png` | Cumulative squared error over time per model |

Tables (MSE / RMSE / MAE / QLIKE / R² / DA per model) land in `outputs/tables/metrics.csv` and `metrics.tex` (LaTeX-ready for the report).

---

## 6. How to Run

```bash
conda activate ctestenv   # or: source .venv/bin/activate

# Stage 1: download + clean + DuckDB ingest
python scripts/run_collection.py

# Stage 2: build features + target
python scripts/run_features.py

# Stage 3: baselines
python scripts/run_baselines.py
# Or just one: python scripts/run_baselines.py --garch-only / --xgb-only

# Stage 4a: LSTM experiment grid (needs GPU for reasonable runtime)
python scripts/run_lstm.py
# Or a single config: python scripts/run_lstm.py --seq-len 60 --no-attention

# Stage 4b: Transformer
python scripts/run_transformer.py

# Stage 4c: XGBoost feature-group ablation
python scripts/run_xgb_ablation.py

# Stage 5: metrics + bootstrap test + every figure
python scripts/run_evaluation.py
```

Each stage is idempotent and picks up from cached parquet/pickle outputs — you can rerun any single stage without redoing the rest.

---

## 7. Course Requirement Map

| Requirement | Where it shows up |
|---|---|
| **Regex** | [`data_collection.py:143-191`](src/data_collection.py#L143-L191) — timestamp / numeric / trading-pair validation |
| **SQL (DuckDB)** | [`init_duckdb`](src/data_collection.py#L280), [`query_resampled`](src/data_collection.py#L332) (GROUP BY + aggregates), [`compare_models_sql`](src/data_collection.py#L381) (JOIN) |
| **Pandas** | Entire feature engineering + alignment pipeline |
| **Big Data / scale** | ~525K 1-minute rows, walk-forward retraining with 500-step refit frequency |
| **ML classical** | XGBoost + Optuna Bayesian HPO |
| **Deep learning** | LSTM (± attention) and PatchTST Transformer in PyTorch |
| **Statistical testing** | Moving Block Bootstrap for paired-loss inference |
| **EDA / visualization** | 13 figures covering distribution, autocorrelation, training curves, residuals, and results |
