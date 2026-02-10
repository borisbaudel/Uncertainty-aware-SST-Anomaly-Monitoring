from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_timeseries_with_threshold(t, y, y_s, y_roll30, thr, title, out_png):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(t, y, label="Anomaly (raw)")
    plt.plot(t, y_s, label="Kalman RTS smooth")
    plt.plot(t, y_roll30, label="Rolling mean (30d)")
    plt.axhline(thr, linestyle="--", label="P90 threshold")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("SST anomaly (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()



def plot_timeseries_with_uncertainty(
    t: pd.DatetimeIndex,
    y: np.ndarray,
    y_smooth: np.ndarray,
    y_smooth_std: np.ndarray,
    title: str,
    out_png: str,
) -> None:
    plt.figure()
    plt.plot(t, y, linewidth=1, label="Anomaly (raw)")
    plt.plot(t, y_smooth, linewidth=2, label="Kalman RTS smooth")

    lo = y_smooth - 2 * y_smooth_std
    hi = y_smooth + 2 * y_smooth_std
    plt.fill_between(t, lo, hi, alpha=0.2, label="±2σ (smooth)")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("SST anomaly (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_histogram(y: np.ndarray, title: str, out_png: str) -> None:
    yy = y[np.isfinite(y)]
    plt.figure()
    plt.hist(yy, bins=60)
    plt.title(title)
    plt.xlabel("SST anomaly (°C)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()