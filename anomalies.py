from __future__ import annotations
import xarray as xr


def daily_climatology(series: xr.DataArray, baseline_start: str, baseline_end: str) -> xr.DataArray:
    """
    Compute day-of-year climatology over a baseline window.
    series: time-indexed DataArray (daily)
    Returns: climatology indexed by dayofyear (1..366)
    """
    base = series.sel(time=slice(baseline_start, baseline_end))
    # Group by day of year (handles leap years by including 366)
    clim = base.groupby("time.dayofyear").mean("time", skipna=True)
    clim.name = "climatology"
    return clim


def anomalies_from_climatology(series: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """
    Subtract day-of-year climatology to get anomalies.
    """
    anom = series.groupby("time.dayofyear") - clim
    anom.name = "sst_anomaly"
    anom.attrs["units"] = series.attrs.get("units", "degC")
    return anom