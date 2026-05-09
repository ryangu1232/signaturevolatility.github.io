# Models Overview (`src/models.py`)

All models use **`TimeSeriesSplit`** for out-of-sample evaluation — no data leakage. Predictions are always clipped to `≥ 1e-12` since realized volatility is non-negative.

---

## Shared evaluation protocol

Every model is evaluated using `rolling_oos_predictions_*` wrappers that return `(y_true, y_pred, indices)`. The outer loop is either:

- **`TimeSeriesSplit(n_splits=5)`** — splits the dataset into 5 expanding train/test folds, refitting once per fold.
- **Walk-forward expanding window** — refits every `refit_every` steps from `min_train_size` onward (more frequent, more adaptive).

---

## 1. `rolling_oos_predictions_linear`

**Purpose**: OLS baseline. Used for HAR-RV and range-based HAR.

**Pipeline**:
```
X → StandardScaler → LinearRegression → ŷ
```

No regularization, no kernel. The model learns one coefficient per feature — directly interpretable as weights on the rolling RV inputs.

**Tuning**: None.

---

## 2. `rolling_oos_predictions_lasso`

**Purpose**: Penalized linear model on signature features. Intermediate baseline — adds sparsity over OLS but no non-linearity.

**Pipeline**:
```
X → StandardScaler → LassoCV (100 alphas, cv=5) wrapped in TransformedTargetRegressor → ŷ
```

`TransformedTargetRegressor` standardizes `y` before fitting so the Lasso duality-gap tolerance stays at a sensible magnitude relative to the target scale (important because RV values are small numbers like `0.003`).

**Tuning**: `LassoCV` cross-validates over 100 regularization strengths `α` using inner `cv=5` time-series splits to pick the sparsest predictive model.

---

## 3. `rolling_oos_predictions_lasso_krr`

**Purpose**: Main signature model. Two-stage pipeline: Lasso for feature selection, then Kernel Ridge Regression for non-linear fitting on the selected features.

**Pipeline**:
```
Stage 1 — Feature selection:
  X → StandardScaler → LassoCV (100 alphas, cv=5, TransformedTargetRegressor on y)
  → retain features where |coef| > 1e-10

Stage 2 — Non-linear fitting (if any features survive):
  X_selected → StandardScaler(y) → KernelRidge(kernel="rbf")
  with GridSearchCV over:
    alpha ∈ {1e-4, 1e-2, 1.0, 10.0}
    gamma ∈ {None, 0.1, 1.0, 10.0}
  using inner TimeSeriesSplit(n_splits=3)

Fallback: if Lasso zeroes all features → use Lasso prediction directly
```

The **RBF kernel on signature features** acts as a signature kernel — it measures path similarity in the truncated-signature feature space. KRR's dual form `ŷ(x) = Σᵢ αᵢ K(x, xᵢ)` weights each training path by its signature-space distance to the query path.

**Tuning**: Outer `TimeSeriesSplit(5)` for OOS evaluation; inner `TimeSeriesSplit(3)` + grid search for `(alpha, gamma)` per fold.

---

## 4. `rolling_oos_predictions_lasso_krr_walkforward`

**Purpose**: Walk-forward variant of model 3. Same two-stage LassoCV → KRR pipeline but refitted much more frequently to track regime changes.

**Pipeline**: Identical to model 3 (LassoCV → KRR with RBF kernel).

**Fitting schedule**:
```
t = min_train_size (default 200)
while t < n:
    train on X[0:t], y[0:t]         ← expanding window
    predict X[t : t+refit_every]
    t += refit_every                 ← refit_every = max(sig_windows), e.g. 60 at daily
```

Each batch refit runs the full LassoCV + inner grid search from scratch on all available history. This means hyperparameters (the optimal α for Lasso and α/γ for KRR) are re-selected every `refit_every` observations rather than once per large fold.

**Difference from model 3**: Model 3 has ~5 refits covering large test chunks. Model 4 has many more refits covering small batches — better at adapting to slowly changing volatility regimes.

---

## 5. `rolling_oos_predictions_ridge_krr`

**Purpose**: KRR directly on all signature features — no Lasso pre-selection step. Faster than model 3 and avoids the risk that Lasso drops genuinely useful but correlated features.

**Pipeline**:
```
X → StandardScaler → StandardScaler(y) → KernelRidge(kernel="rbf")
with GridSearchCV over:
  alpha ∈ {1e-4, 1e-2, 1.0, 10.0}
  gamma ∈ {None, 0.1, 1.0, 10.0}
using inner TimeSeriesSplit(n_splits=3)
```

No sparsity constraint — all signature features enter the kernel. The RBF kernel handles dimensionality implicitly through the kernel trick.

**Tuning**: Outer `TimeSeriesSplit(5)`; inner `TimeSeriesSplit(3)` + grid search per fold.

**Note**: This model is defined in `models.py` but is not currently active in the main pipeline.

---

## 6. `rolling_oos_predictions_elasticnet_krr`

**Purpose**: ElasticNet → KRR variant. ElasticNet combines L1 (sparsity) and L2 (grouping of correlated features), which suits signature tensors whose terms are structurally correlated (e.g., level-2 terms share increments with level-1 terms).

**Pipeline**:
```
Stage 1:
  X → StandardScaler → ElasticNetCV (100 alphas, cv=5, TransformedTargetRegressor on y)
  → retain features where |coef| > 1e-10

Stage 2: identical to model 3 (KRR with RBF, inner TSS(3) grid search)
```

**Tuning**: Same structure as model 3, with ElasticNet replacing Lasso in Stage 1.

**Note**: Defined in `models.py` but not currently active in the main pipeline.

---

## 7. `rolling_oos_predictions_xgboost`

**Purpose**: Nonlinear ML benchmark on HAR features. Uses the same three rolling-RV inputs as `har_rv_linear` so the only difference is model class (trees vs OLS). Directly answers: does nonlinearity help over linear HAR?

**Pipeline**:
```
X (HAR features) → StandardScaler → StandardScaler(y) → XGBRegressor
  n_estimators=200, max_depth=4, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, min_child_weight=5
```

**Tuning**: Fixed hyperparameters — conservative settings (shallow trees, low learning rate, column subsampling) suited to small RV datasets.

**Requires**: `pip install xgboost`

---

## 8. `rolling_oos_predictions_lstm`

**Purpose**: Deep learning benchmark on raw log-return windows — no manual feature engineering. Directly answers: can an LSTM learn from raw returns without signature or HAR preprocessing?

**Input**: Raw log-return windows of shape `(n, lookback_bars)` from `build_sequence_dataset`, reshaped to `(n, lookback_bars, 1)` for the LSTM.

**Architecture**:
```
Input: (batch, lookback_bars, 1)  ← standardised raw return sequence
LSTM(hidden_size=32, num_layers=1, batch_first=True)
Linear(32 → 1)
Output inverse-standardised → ŷ (clipped to ≥ 1e-12)
```

**Training**: Adam optimizer, MSE loss, 50 epochs, mini-batch size 32 (mini-batches shuffled — valid since each sample is a fixed-length window, not a streaming sequence).

**Requires**: `pip install torch`

---

## 9. `rolling_oos_predictions_garch` (GARCH-X)

**Purpose**: GARCH(1,1)-X baseline using realized variance as the exogenous input. Provides a classical econometric comparison.

**Model**:
```
h_t = ω + α · rv_{t-1} + β · h_{t-1}

where rv_t = Σ r²_{t-rv_window+1 : t}  (realized variance, squared-return units)
```

Parameters `(ω, α, β)` are estimated by quasi-MLE (L-BFGS-B) on each training fold. The terminal `h` from training is then rolled forward through the test period using observed `rv` values.

**Tuning**: No cross-validation; parameters fitted by numerical optimization (MLE) once per fold.

**Note**: Active in the pipeline as `garch_x`. Wrapped in try/except so failures (e.g. optimizer non-convergence) are skipped gracefully.

---

## Active models in the pipeline

| Model name | Function | Notes |
|---|---|---|
| `har_rv_linear` | `rolling_oos_predictions_linear` | OLS on 3 scalar rolling RV features |
| `sig_lasso_krr_{mode}` | `rolling_oos_predictions_lasso_krr` | Single-scale sig → Lasso → KRR, 5-fold |
| `sig_lasso_krr_wf_{mode}` | `rolling_oos_predictions_lasso_krr_walkforward` | Single-scale sig, rolling walk-forward refit every `max(windows)` bars |
| `xgboost_har` | `rolling_oos_predictions_xgboost` | XGBoost on same 3 HAR features as `har_rv_linear`; optional dep (`pip install xgboost`) |
| `garch_x` | `rolling_oos_predictions_garch` | GARCH-X(1,1) econometric benchmark |

## Inactive models (defined but not called in `main.py`)

| Function | Reason not active |
|---|---|
| `rolling_oos_predictions_lasso` | Superseded by Lasso → KRR |
| `rolling_oos_predictions_ridge_krr` | Replaced by Lasso → KRR after testing |
| `rolling_oos_predictions_elasticnet_krr` | Added for comparison, not included in final runs |
