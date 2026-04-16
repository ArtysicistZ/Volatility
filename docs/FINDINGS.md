# Experimental Findings
## Sequential Deep Learning for High-Frequency Crypto Volatility Prediction

---

## 1. Data

- **Source**: Binance public data repository (data.binance.vision)
- **Instrument**: BTC/USDT, 1-minute candles
- **Small dataset**: 525,600 rows (Apr 2024 - Mar 2025, 1 year)
- **Full dataset**: 4,008,720 rows (Aug 2017 - Mar 2025, 7.6 years)
- **Features**: 52 engineered features (volatility estimators, rolling stats, lag features, volume, time encoding)
- **Target**: Realized volatility over next 10 minutes = std(log_returns[t+1:t+10]), ddof=1
- **Split**: 85% train / 5% val / 10% test, strict chronological

---

## 2. Model Comparison (525K dataset, 52K test samples)

| Rank | Model | R² | MSE | RMSE | QLIKE | DA |
|------|-------|-----|-----|------|-------|-----|
| 1 | **LSTM seq120 + attention** | **0.617** | 1.07e-07 | 0.000326 | 0.335 | 49.4% |
| 2 | XGBoost (Optuna-tuned) | 0.613 | 1.08e-07 | 0.000328 | **0.321** | 37.7% |
| 3 | LSTM seq60 + attention | 0.600 | 1.11e-07 | 0.000334 | 0.345 | 49.0% |
| 4 | LSTM seq60 no attention | 0.597 | 1.12e-07 | 0.000335 | 0.342 | 49.2% |
| 5 | LSTM seq120 no attention | 0.582 | 1.16e-07 | 0.000341 | 0.328 | 49.3% |
| 6 | LSTM seq240 + attention | 0.572 | 1.19e-07 | 0.000345 | 0.370 | 49.4% |
| 7 | LSTM seq240 no attention | 0.568 | 1.20e-07 | 0.000347 | 0.345 | 49.4% |
| 8 | Transformer small seq120 | 0.543 | 1.27e-07 | 0.000357 | 0.347 | 49.5% |
| 9 | Transformer full seq120 | 0.536 | 1.29e-07 | 0.000359 | 0.402 | 49.3% |
| 10 | Transformer small seq240 | 0.493 | 1.41e-07 | 0.000376 | 0.404 | 49.2% |
| 11 | GARCH(1,1) Student-t | -14.49 | 4.31e-06 | 0.002075 | 1.058 | **50.3%** |

---

## 3. Scaling Experiment: 4M Dataset (400K test samples)

Trained LSTM and Transformer on 7.6 years of data to test if more data helps Transformers.

| Model | R² (525K data) | R² (4M data) | Val loss gap to LSTM |
|-------|---------------|-------------|---------------------|
| LSTM seq120 + attention | 0.617 | **0.540** | — |
| Transformer small seq120 | 0.543 | 0.530 | 4.5% (was 13.6%) |
| Transformer small seq240 | 0.493 | 0.529 | 4.5% (was 22.9%) |
| Transformer full seq120 | 0.536 | 0.521 | 5.3% (was 15.0%) |
| Transformer full seq240 | — | 0.483 | 11.7% |

**Key finding**: 8x more data reduced the LSTM-Transformer gap from ~25% to ~4.5% in val loss. R² dropped for all models (0.617 → 0.540) because the test set now spans 9 months of diverse market conditions vs 36 days before — a harder but more realistic evaluation.

---

## 4. LSTM Ablation: Sequence Length and Attention

| seq_len | Effective lookback | No attention R² | + Attention R² | Attention gain |
|---------|-------------------|-----------------|----------------|---------------|
| 60 (1 hour) | ~2 hours | 0.597 | 0.600 | +0.5% |
| 120 (2 hours) | ~3 hours | 0.582 | **0.617** | **+6.0%** |
| 240 (4 hours) | ~5 hours | 0.568 | 0.572 | +0.7% |

- **seq120 is the sweet spot** — seq60 too short, seq240 overfits
- **Attention helps most at seq120** — the model benefits from selectively attending to relevant timesteps in a medium-length window
- **seq240 overfits** despite best val loss (0.133) — test R² is worst among LSTM configs

---

## 5. Transformer Architecture Analysis

### Architecture
- **PatchTST-style**: patches of 12 timesteps with stride 6
- **Small config**: d_model=128, 4 heads, 2 layers (~500K params)
- **Full config**: d_model=256, 8 heads, 4 layers (~3.2M params)

### Why Transformer Underperforms LSTM

1. **Early convergence**: Transformers stopped at epoch 11-19 (out of 80) on 525K data vs LSTM's 23-42 — classic overfitting signal
2. **More params = worse**: Full config (3.2M) performed worse than small (500K) — model memorizes noise
3. **With 4M data**: Transformers trained much longer (33-79 epochs), gap narrowed to 4.5% — confirms data hunger
4. **LSTM's inductive bias wins**: Sequential gating and forget mechanism act as built-in regularizers for noisy financial data

### Estimated Data Requirements
- LSTM (225K params): works well with 446K+ samples
- Transformer small (500K params): needs 3M+ samples to compete
- Transformer full (3.2M params): likely needs 10M+ samples

---

## 6. XGBoost Feature Importance

### Top 10 Features (by gain)
| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | ret_std_30 | **32.7%** | Rolling volatility |
| 2 | ret_std_10 | 17.1% | Rolling volatility |
| 3 | ret_std_60 | 13.6% | Rolling volatility |
| 4 | parkinson_30 | 7.0% | Volatility estimator |
| 5 | garman_klass_10 | 5.4% | Volatility estimator |
| 6 | parkinson_5 | 3.7% | Volatility estimator |
| 7 | parkinson_10 | 3.2% | Volatility estimator |
| 8 | bollinger_width | 2.0% | Technical indicator |
| 9 | parkinson_60 | 1.3% | Volatility estimator |
| 10 | garman_klass_5 | 1.3% | Volatility estimator |

**Top 3 features alone account for 63.4% of total importance.** All are rolling standard deviations of returns at different windows — confirming that past volatility is overwhelmingly the best predictor of future volatility.

### Feature Group Ablation
| Configuration | R² | Drop from baseline |
|---------------|-----|-------------------|
| All 52 features | 0.570 | — |
| No ATR/Bollinger | 0.570 | 0.0% |
| No lag features (10 removed) | 0.569 | -0.1% |
| No volume features (5 removed) | 0.569 | -0.1% |
| No time encoding (2 removed) | 0.567 | -0.3% |
| No rolling stats (24 removed) | 0.556 | **-1.4%** |
| No vol estimators (8 removed) | 0.557 | **-1.3%** |
| Only rolling stats (24 kept) | 0.556 | Keeps 97.5% of R² |
| Only lag features (10 kept) | 0.462 | **Loses 10.8%** |

**Key findings:**
- Rolling stats and volatility estimators are the only features that meaningfully contribute
- Lag features, volume, time encoding, ATR/Bollinger are nearly useless
- **24 rolling stats features alone achieve 97.5% of full R²**

### Top-K Feature Analysis
| Features used | R² | % of full model |
|---------------|-----|-----------------|
| Top 5 | 0.536 | 94.1% |
| Top 10 | 0.548 | 96.1% |
| Top 20 | 0.560 | 98.3% |
| Top 30 | 0.568 | 99.7% |
| All 52 | 0.570 | 100% |

Top 5 features capture 94% of predictive power. Diminishing returns beyond 20 features.

---

## 7. GARCH Analysis

- **GARCH(1,1) with Student-t distribution** — appropriate for crypto fat tails
- **Rolling window of 5000 observations**, refit every 500 steps
- **Critical bug found and fixed**: conditional variance h_t must be updated at every timestep using the GARCH recursion, not just at refit points. Without this fix, 500 consecutive predictions were identical, causing R² = -19.8. After fix: R² = -14.5.
- **R² is negative** because GARCH predictions are systematically biased (pred mean 0.00132 vs actual mean 0.00059). GARCH overestimates volatility.
- **DA = 50.3%** (best of all models) — GARCH captures volatility direction at random, but doesn't anti-correlate like XGBoost.

---

## 8. Key Insights

### Why XGBoost matches LSTM
The nonlinearity of volatility amplitude is low. The dominant pattern — "past volatility predicts future volatility" (ARCH effect) — is a monotonic relationship that tree splits capture efficiently. Our engineered features (rolling_std, Parkinson, Garman-Klass) pre-compute the temporal patterns that LSTM learns from raw sequences. LSTM's marginal contribution is discovering subtle ordering effects in the 120-timestep window.

### Why Transformer needs more data
Transformers lack LSTM's sequential inductive bias (gating, forget mechanism). They must learn temporal structure purely from data. With 446K samples and 500K-3.2M parameters, they overfit before discovering generalizable patterns. With 4M samples, the gap narrows from 25% to 4.5%, suggesting ~10M+ samples may be needed for Transformers to match or beat LSTM on this task.

### Directional accuracy is ~50% for all models
Volatility is mean-reverting — it oscillates rapidly around its local mean. Predicting whether next-10-minute volatility goes up or down is essentially a coin flip, even when level prediction is good. This is consistent with financial theory (Ornstein-Uhlenbeck process). DA is not a useful metric for volatility prediction.

### Walk-forward adaptation adds ~4% R²
XGBoost with expanding window (refit every 500 steps) achieves R² = 0.613 vs single-fit R² = 0.570. The model benefits from seeing recent data during the test period.

---

## 9. Hypothesis Test (Pending)

- **H₀**: MSE_LSTM ≥ MSE_baseline
- **H₁**: MSE_LSTM < MSE_baseline
- **Method**: Moving Block Bootstrap (block_size=50, 1000 replications)
- **Status**: To be run after all models are finalized on the same test set

---

## 10. Files and Artifacts

### Data
- `data/raw/btcusdt_1m_full.parquet` — 4M rows raw data
- `data/processed/btcusdt_1m_clean.parquet` — cleaned data
- `data/processed/features.parquet` — 52 engineered features
- `data/processed/target.parquet` — realized volatility target

### Model Predictions
- `outputs/models/garch_predictions.pkl`
- `outputs/models/xgboost_predictions.pkl`
- `outputs/models/lstm_predictions_seq*.pkl` — all LSTM variants
- `outputs/models/lstm_predictions_tfm_*.pkl` — all Transformer variants

### Tables
- `outputs/tables/metrics.csv` — full model comparison
- `outputs/tables/xgb_feature_importance.csv`
- `outputs/tables/xgb_ablation.csv`
- `outputs/tables/xgb_topk.csv`

### Figures
- `outputs/figures/residuals_*.png` — residual analysis per model
- `outputs/figures/xgb_feature_importance.png`
- `outputs/figures/xgb_ablation.png`
- `outputs/figures/xgb_topk.png`
