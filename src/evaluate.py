from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    yt = np.maximum(y_true, eps)
    yp = np.maximum(y_pred, eps)
    return float(np.mean(np.log(yp) + yt / yp))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ql = qlike(y_true, y_pred)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "QLIKE": float(ql),
    }
