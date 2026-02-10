from __future__ import annotations

import numpy as np
import pandas as pd


def linear_trend_per_decade(t: pd.DatetimeIndex, y: np.ndarray) -> tuple[float, float]:
    """
    OLS trend slope in units of y per decade.
    Returns (slope_per_decade, intercept).
    Ignores NaNs.
    """
    mask = np.isfinite(y)
    t0 = t[mask]
    y0 = y[mask]
    if len(y0) < 10:
        return np.nan, np.nan

    # Convert time to years sin
    # 
    # 
    # ce start
    x = (t0 - t0[0]).days.values / 365.25
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y0, rcond=None)[0]
    return float(slope * 10.0), float(intercept)  # per decade


def extremes_metrics(y: np.ndarray, q: float = 0.99) -> dict:
    """
    Basic extremes / variability metrics.
    """
    yy = y[np.isfinite(y)]
    if len(yy) == 0:
        return {"mean": np.nan, "std": np.nan, "p99": np.nan, "p01": np.nan, "max": np.nan, "min": np.nan}

    return {
        "mean": float(np.mean(yy)),
        "std": float(np.std(yy)),
        "p99": float(np.quantile(yy, q)),
        "p01": float(np.quantile(yy, 1 - q)),
        "max": float(np.max(yy)),
        "min": float(np.min(yy)),
    }