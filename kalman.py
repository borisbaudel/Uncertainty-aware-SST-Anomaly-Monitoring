from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LocalLevelParams:
    """
    Local level model:
      x_k = x_{k-1} + w_k,      w_k ~ N(0, q)
      y_k = x_k + v_k,          v_k ~ N(0, r)
    """
    q: float  # process noise variance
    r: float  # measurement noise variance


@dataclass
class KalmanResult:
    x_filt: np.ndarray
    P_filt: np.ndarray
    x_pred: np.ndarray
    P_pred: np.ndarray
    x_smooth: np.ndarray
    P_smooth: np.ndarray
    innovations: np.ndarray
    S: np.ndarray  # innovation variance


def kalman_filter_local_level(y: np.ndarray, params: LocalLevelParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1D Kalman filter for the local level model with missing data support (NaNs).
    Returns x_pred, P_pred, x_filt, P_filt, innovations, S.
    """
    n = len(y)
    x_pred = np.zeros(n)
    P_pred = np.zeros(n)
    x_filt = np.zeros(n)
    P_filt = np.zeros(n)
    innov = np.full(n, np.nan)
    S = np.full(n, np.nan)

    # Initialize (diffuse-ish)
    x = 0.0
    P = 10.0  # broad prior

    for k in range(n):
        # predict
        x = x  # F=1
        P = P + params.q
        x_pred[k] = x
        P_pred[k] = P

        if np.isfinite(y[k]):
            # update
            S_k = P + params.r
            K = P / S_k
            innov_k = y[k] - x
            x = x + K * innov_k
            P = (1 - K) * P

            innov[k] = innov_k
            S[k] = S_k

        x_filt[k] = x
        P_filt[k] = P

    return x_pred, P_pred, x_filt, P_filt, innov, S


def rts_smoother_local_level(
    x_pred: np.ndarray, P_pred: np.ndarray,
    x_filt: np.ndarray, P_filt: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rauch–Tung–Striebel smoother for local level model (F=1).
    """
    n = len(x_filt)
    x_s = x_filt.copy()
    P_s = P_filt.copy()

    for k in range(n - 2, -1, -1):
        # For F=1, predicted covariance at k+1 is P_pred[k+1] = P_filt[k] + q
        C = P_filt[k] / P_pred[k + 1]
        x_s[k] = x_filt[k] + C * (x_s[k + 1] - x_pred[k + 1])
        P_s[k] = P_filt[k] + C**2 * (P_s[k + 1] - P_pred[k + 1])

    return x_s, P_s


def kalman_smooth_local_level(y: np.ndarray, params: LocalLevelParams) -> KalmanResult:
    x_pred, P_pred, x_filt, P_filt, innov, S = kalman_filter_local_level(y, params)
    x_s, P_s = rts_smoother_local_level(x_pred, P_pred, x_filt, P_filt, params.q)
    return KalmanResult(
        x_filt=x_filt, P_filt=P_filt,
        x_pred=x_pred, P_pred=P_pred,
        x_smooth=x_s, P_smooth=P_s,
        innovations=innov, S=S
    )


def robust_variance_estimates(y: np.ndarray) -> tuple[float, float]:
    """
    Quick heuristic to set (q, r) from data scale.
    - r: measurement noise variance ~ small fraction of series variance
    - q: process noise variance ~ variance of differences / 10
    You can tune these in CLI.
    """
    y0 = y[np.isfinite(y)]
    if len(y0) < 10:
        return 0.01, 0.05
    var_y = float(np.var(y0))
    dy = np.diff(y0)
    var_dy = float(np.var(dy)) if len(dy) > 5 else var_y
    r = max(1e-6, 0.05 * var_y)
    q = max(1e-6, 0.10 * var_dy)
    return q, r