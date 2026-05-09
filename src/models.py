from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rolling_oos_predictions_lasso(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    y_true_all = []
    y_pred_all = []
    idx_all = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # TransformedTargetRegressor scales y before fitting, which keeps the
        # Lasso duality-gap tolerance (tol * ||y||^2) at a sensible magnitude.
        # Without this, tiny RV values shrink the tolerance to ~1e-7 and trigger
        # ConvergenceWarning even when the solution is effectively converged.
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lasso", TransformedTargetRegressor(
                    regressor=LassoCV(
                        cv=5,
                        random_state=random_state,
                        n_alphas=100,
                        max_iter=50000,
                        selection="random",
                    ),
                    transformer=StandardScaler(),
                )),
            ]
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.maximum(preds, 1e-12)

        y_true_all.append(y_test)
        y_pred_all.append(preds)
        idx_all.append(test_idx)

    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(idx_all)



def rolling_oos_predictions_lasso_krr(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5-fold TimeSeriesSplit Lasso→KRR pipeline (used by multi_horizon)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_true_all, y_pred_all, idx_all = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        x_scaler = StandardScaler()
        X_train_s = x_scaler.fit_transform(X_train)
        X_test_s = x_scaler.transform(X_test)

        lasso_ttr = TransformedTargetRegressor(
            regressor=LassoCV(
                cv=5,
                random_state=random_state,
                n_alphas=100,
                max_iter=100000,
                tol=1e-3,
                selection="random",
            ),
            transformer=StandardScaler(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_ttr.fit(X_train_s, y_train)
        selected = np.abs(lasso_ttr.regressor_.coef_) > 1e-10

        if selected.sum() == 0:
            preds = np.maximum(lasso_ttr.predict(X_test_s), 1e-12)
        else:
            X_tr_sel = X_train[:, selected]
            X_te_sel = X_test[:, selected]

            y_scaler = StandardScaler()
            y_tr_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

            inner_cv = TimeSeriesSplit(n_splits=3)
            krr = GridSearchCV(
                KernelRidge(kernel="linear"),
                {"alpha": [1e-4, 1e-2, 1.0, 10.0, 100.0]},
                cv=inner_cv,
                scoring="neg_mean_squared_error",
                refit=True,
            )
            krr.fit(X_tr_sel, y_tr_s)
            preds_s = krr.predict(X_te_sel)
            preds = y_scaler.inverse_transform(preds_s.reshape(-1, 1)).ravel()
            preds = np.maximum(preds, 1e-12)

        y_true_all.append(y[test_idx])
        y_pred_all.append(preds)
        idx_all.append(test_idx)

    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(idx_all)


def rolling_oos_predictions_lasso_krr_walkforward(
    X: np.ndarray,
    y: np.ndarray,
    min_train_size: int = 200,
    refit_every: int = 60,
    window_size: int = 500,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward Lasso→KRR with rolling training window.

    Refits the full pipeline (LassoCV + KRR) every `refit_every` steps using
    only the most recent `window_size` observations. This bounds the model's
    memory so that stale regime data is dropped rather than accumulated.

    The first refit uses min(min_train_size, window_size) samples. Each
    subsequent refit slides the window forward by `refit_every` bars.
    """
    n = len(X)
    y_true_all, y_pred_all, idx_all = [], [], []

    t = min_train_size
    while t < n:
        batch_end = min(t + refit_every, n)

        train_start = max(0, t - window_size)
        X_train, y_train = X[train_start:t], y[train_start:t]
        X_test = X[t:batch_end]

        # --- Stage 1: Lasso on scaled features, scaled target ---
        x_scaler = StandardScaler()
        X_train_s = x_scaler.fit_transform(X_train)
        X_test_s = x_scaler.transform(X_test)

        lasso_ttr = TransformedTargetRegressor(
            regressor=LassoCV(
                cv=5,
                random_state=random_state,
                n_alphas=100,
                max_iter=100000,
                tol=1e-3,
                selection="random",
            ),
            transformer=StandardScaler(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_ttr.fit(X_train_s, y_train)
        selected = np.abs(lasso_ttr.regressor_.coef_) > 1e-10

        if selected.sum() == 0:
            preds = np.maximum(lasso_ttr.predict(X_test_s), 1e-12)
        else:
            # --- Stage 2: KRR with truncated signature kernel (linear) ---
            # Use RAW (unscaled) selected features to preserve the true
            # signature kernel inner product in the tensor algebra.
            X_tr_sel = X_train[:, selected]
            X_te_sel = X_test[:, selected]

            y_scaler = StandardScaler()
            y_tr_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

            inner_cv = TimeSeriesSplit(n_splits=3)
            param_grid = {
                "alpha": [1e-4, 1e-2, 1.0, 10.0, 100.0],
            }
            krr = GridSearchCV(
                KernelRidge(kernel="linear"),
                param_grid,
                cv=inner_cv,
                scoring="neg_mean_squared_error",
                refit=True,
            )
            krr.fit(X_tr_sel, y_tr_s)

            preds_s = krr.predict(X_te_sel)
            preds = y_scaler.inverse_transform(preds_s.reshape(-1, 1)).ravel()
            preds = np.maximum(preds, 1e-12)

        y_true_all.append(y[t:batch_end])
        y_pred_all.append(preds)
        idx_all.append(np.arange(t, batch_end))

        t = batch_end

    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(idx_all)


def rolling_oos_predictions_linear(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    y_true_all = []
    y_pred_all = []
    idx_all = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("linear", LinearRegression()),
            ]
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.maximum(preds, 1e-12)

        y_true_all.append(y_test)
        y_pred_all.append(preds)
        idx_all.append(test_idx)

    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(idx_all)


def rolling_oos_predictions_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XGBoost on signature features with TimeSeriesSplit OOS evaluation.

    Requires: pip install xgboost
    """
    try:
        from xgboost import XGBRegressor  # type: ignore
    except ImportError as e:
        raise ImportError("xgboost is not installed. Install with: pip install xgboost") from e

    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_true_all, y_pred_all, idx_all = [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        x_scaler = StandardScaler()
        X_train_s = x_scaler.fit_transform(X_train)
        X_test_s = x_scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=random_state,
            verbosity=0,
        )
        model.fit(X_train_s, y_train_s)

        preds_s = model.predict(X_test_s)
        preds = y_scaler.inverse_transform(preds_s.reshape(-1, 1)).ravel()
        preds = np.maximum(preds, 1e-12)

        y_true_all.append(y_test)
        y_pred_all.append(preds)
        idx_all.append(test_idx)

    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(idx_all)


def _compute_garch_rv_h(returns: np.ndarray, rv: np.ndarray,
                         omega: float, alpha: float, beta: float) -> np.ndarray:
    """Variance recursion: h_t = ω + α·rv_{t-1} + β·h_{t-1}."""
    n = len(returns)
    h = np.empty(n)
    h[0] = float(np.var(returns)) if n > 1 else 1e-8
    for t in range(1, n):
        h[t] = omega + alpha * rv[t - 1] + beta * h[t - 1]
        if h[t] <= 0:
            h[t] = 1e-16
    return h


def _fit_garch_rv_params(returns: np.ndarray, rv: np.ndarray):
    """Fit GARCH-X(1,1) h_t = ω + α·rv_{t-1} + β·h_{t-1} by quasi-MLE.

    rv should be in variance units (squared).  Returns (ω, α, β) or a
    fallback (ω₀, 0.05, 0.90) if the optimizer does not converge.
    """
    n = len(returns)
    h0 = float(np.var(returns)) if n > 1 else 1e-8
    r2 = returns ** 2

    def neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e15
        h = _compute_garch_rv_h(returns, rv, omega, alpha, beta)
        h = np.maximum(h, 1e-16)
        return 0.5 * float(np.sum(np.log(h) + r2 / h))

    x0 = [h0 * 0.05, 0.05, 0.90]
    bounds = [(1e-15, None), (0.0, None), (0.0, 0.9999)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000, "ftol": 1e-10})
        omega, alpha, beta = res.x
        if omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1:
            return float(omega), float(alpha), float(beta)
    except Exception:
        pass
    return h0 * 0.05, 0.05, 0.90


def rolling_oos_predictions_garch(
    returns: np.ndarray,
    y_target: np.ndarray,
    n_splits: int = 5,
    rv_window: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling OOS RV-GARCH(1,1): h_t = ω + α·rv_{t-1} + β·h_{t-1}.

    Parameters are fitted once per TimeSeriesSplit fold by quasi-MLE; the
    variance recursion then rolls through the test period using actual
    observed rv values.  rv_t = Σ_{i=0}^{rv_window-1} r²_{t-i} (realized
    variance in squared-return units).

    For multi-bar horizons h the one-step conditional variance forecast is
    scaled by horizon_scale so predictions match the target RV magnitude.
    """
    returns = np.asarray(returns, dtype=float)
    y_target = np.asarray(y_target, dtype=float)

    rv = np.array([
        np.sum(returns[max(0, t - rv_window + 1): t + 1] ** 2)
        for t in range(len(returns))
    ])

    one_step_scale = np.nanmean(np.abs(returns))
    target_scale = np.nanmean(y_target)
    horizon_scale = max((target_scale / max(one_step_scale, 1e-12)) ** 2, 1.0)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_true_all, y_pred_all, idx_all = [], [], []

    for train_idx, test_idx in tscv.split(returns):
        r_train = returns[train_idx]
        rv_train = rv[train_idx]

        omega, alpha, beta = _fit_garch_rv_params(r_train, rv_train)

        # Run recursion through training to obtain terminal h
        h_seq = _compute_garch_rv_h(r_train, rv_train, omega, alpha, beta)
        h = float(h_seq[-1])

        preds = []
        for t in test_idx:
            h_next = omega + alpha * rv[t] + beta * h
            h_next = max(h_next, 1e-16)
            preds.append(float(np.sqrt(h_next * horizon_scale)))
            h = h_next

        preds = np.maximum(np.asarray(preds, dtype=float), 1e-12)
        y_true_all.append(y_target[test_idx])
        y_pred_all.append(preds)
        idx_all.append(test_idx)

    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(idx_all)
