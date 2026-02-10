from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from iosst_io import Region, open_oisst_regional_mean_years
from anomalies import daily_climatology, anomalies_from_climatology
from kalman import LocalLevelParams, kalman_smooth_local_level, robust_variance_estimates
from diagnostics import linear_trend_per_decade, extremes_metrics
from plots import plot_timeseries_with_uncertainty, plot_histogram, plot_timeseries_with_threshold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uncertainty-aware SST Anomaly Monitor (NOAA OISST) + Kalman + diagnostics"
    )

    p.add_argument("--lon-min", type=float, required=True)
    p.add_argument("--lon-max", type=float, required=True)
    p.add_argument("--lat-min", type=float, required=True)
    p.add_argument("--lat-max", type=float, required=True)

    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")

    p.add_argument("--baseline-start", type=str, default="1991-01-01")
    p.add_argument("--baseline-end", type=str, default="2020-12-31")

    p.add_argument("--q", type=float, default=None, help="process noise variance (Kalman)")
    p.add_argument("--r", type=float, default=None, help="measurement noise variance (Kalman)")

    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--cache-dir", type=str, default="oisst_cache", help="Local cache directory for OISST NetCDF files.")
    p.add_argument("--offline", action="store_true", help="Offline strict mode (never download).")

    p.add_argument(
        "--climatology-file",
        type=str,
        default=None,
        help="Path to precomputed daily climatology NetCDF (e.g. oisst_cache/sst.day.ltm.1991-2020.nc). "
             "If set, baseline-start/end are ignored for climatology computation (offline).",
    )

    return p.parse_args()


def years_between(a: str, b: str) -> list[int]:
    y0 = int(a[:4])
    y1 = int(b[:4])
    return list(range(y0, y1 + 1))


def _subset_lon_wrap_360(ds: xr.Dataset | xr.DataArray, lon_min: float, lon_max: float) -> xr.Dataset | xr.DataArray:
    lon_min = lon_min % 360
    lon_max = lon_max % 360

    if lon_min <= lon_max:
        return ds.sel(lon=slice(lon_min, lon_max))

    a = ds.sel(lon=slice(lon_min, 360))
    b = ds.sel(lon=slice(0, lon_max))
    out = xr.concat([a, b], dim="lon").sortby("lon")
    _, idx = np.unique(out["lon"].values, return_index=True)
    out = out.isel(lon=np.sort(idx))
    return out


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    if args.offline and not cache_dir.exists():
        raise FileNotFoundError(f"Offline mode: cache dir not found: {cache_dir}")

    print(f"[diag] cache_dir={cache_dir.resolve()} offline={args.offline}")

    region = Region(
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
    )

    # ----------------------------
    # 1) Observations
    # ----------------------------
    years_obs = years_between(args.start, args.end)
    print(f"Opening OISST years (obs): {years_obs[0]}..{years_obs[-1]} (count={len(years_obs)})")

    # Si offline: vérifier que tous les fichiers annuels nécessaires existent
    if args.offline:
        for y in years_obs:
            f = cache_dir / f"sst.day.mean.{y}.nc"
            if not f.exists():
                raise FileNotFoundError(f"Offline mode: missing local file {f}")

    sst_obs = open_oisst_regional_mean_years(
        years_obs,
        region=region,
        time_start=args.start,
        time_end=args.end,
        cache_dir=str(cache_dir),
        offline=args.offline,
    )

    # ----------------------------
    # 2) Climatology (prefer file)
    # ----------------------------
    baseline_label: str | None = None

    if args.climatology_file is not None:
        clim_path = Path(args.climatology_file)
        if not clim_path.exists():
            raise FileNotFoundError(f"Climatology file not found: {clim_path}")

        print(f"Loading climatology file: {clim_path}")
        ds_clim = xr.open_dataset(clim_path, engine="netcdf4", decode_times=True, mask_and_scale=True)

        if "sst" not in ds_clim.data_vars:
            raise KeyError(f"'sst' not found in climatology file variables: {list(ds_clim.data_vars)}")

        ds_clim_reg = ds_clim.sel(lat=slice(region.lat_min, region.lat_max))
        ds_clim_reg = _subset_lon_wrap_360(ds_clim_reg, region.lon_min, region.lon_max)

        w = np.cos(np.deg2rad(ds_clim_reg["lat"]))
        w = w / w.mean()

        clim_reg = ds_clim_reg["sst"].weighted(w).mean(dim=("lat", "lon"), skipna=True)

        n = int(clim_reg.sizes["time"])
        doy = xr.DataArray(np.arange(1, n + 1), dims=("time",), name="dayofyear")

        clim_doy = clim_reg.assign_coords(dayofyear=doy).swap_dims({"time": "dayofyear"}).drop_vars("time")

        if 366 not in clim_doy["dayofyear"].values:
            clim_doy = xr.concat(
                [clim_doy, clim_doy.sel(dayofyear=365).assign_coords(dayofyear=366)],
                dim="dayofyear",
            ).sortby("dayofyear")

        ds_clim.close()

        doy_obs = xr.DataArray(
            pd.DatetimeIndex(sst_obs["time"].values).dayofyear,
            dims=("time",),
            coords={"time": sst_obs["time"]},
            name="dayofyear",
        )

        clim_on_obs = clim_doy.sel(dayofyear=doy_obs)
        anom = (sst_obs - clim_on_obs).load()

        baseline_label = f"climatology_file:{clim_path.as_posix()}"

        print("SST obs stats  :", float(sst_obs.min()), float(sst_obs.max()), float(sst_obs.mean()))
        print("CLIM stats     :", float(clim_on_obs.min()), float(clim_on_obs.max()), float(clim_on_obs.mean()))
        print("ANOM stats     :", float(anom.min()), float(anom.max()), float(anom.mean()))

    else:
        years_all = sorted(set(years_between(args.baseline_start, args.baseline_end) + years_obs))
        print(f"Opening OISST years (baseline+obs): {years_all[0]}..{years_all[-1]} (count={len(years_all)})")

        if args.offline:
            for y in years_all:
                f = cache_dir / f"sst.day.mean.{y}.nc"
                if not f.exists():
                    raise FileNotFoundError(f"Offline mode: missing local file {f}")

        sst_reg_all = open_oisst_regional_mean_years(
            years_all,
            region=region,
            time_start=args.baseline_start,
            time_end=args.end,
            cache_dir=str(cache_dir),
            offline=args.offline,
        )

        clim = daily_climatology(sst_reg_all, args.baseline_start, args.baseline_end)
        anom = anomalies_from_climatology(sst_reg_all, clim).sel(time=slice(args.start, args.end)).load()
        baseline_label = f"baseline_window:{args.baseline_start}..{args.baseline_end}"

        print("ANOM stats (fallback):", float(anom.min()), float(anom.max()), float(anom.mean()))

    # ----------------------------
    # 4) Kalman + diagnostics
    # ----------------------------
    t = pd.DatetimeIndex(anom["time"].values)
    y = anom.to_numpy().astype(float)

    thr = np.nanpercentile(y, 90)
    k = 3

    above = y > thr
    events = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= k:
                events.append({
                    "start": str(t[i].date()),
                    "end": str(t[j - 1].date()),
                    "duration_days": int(j - i),
                    "threshold_degC": float(thr),
                    "peak_anomaly_degC": float(np.nanmax(y[i:j])),
                    "mean_anomaly_degC": float(np.nanmean(y[i:j])),
                })
            i = j
        else:
            i += 1

    if args.q is None or args.r is None:
        q0, r0 = robust_variance_estimates(y)
        q = q0 if args.q is None else args.q
        r = r0 if args.r is None else args.r
    else:
        q, r = args.q, args.r

    res = kalman_smooth_local_level(y, LocalLevelParams(q=q, r=r))
    y_s = res.x_smooth
    y_s_std = np.sqrt(np.clip(res.P_smooth, 0.0, np.inf))

    trend_raw, _ = linear_trend_per_decade(t, y)
    trend_s, _ = linear_trend_per_decade(t, y_s)
    metrics_raw = extremes_metrics(y)
    metrics_s = extremes_metrics(y_s)
     

     # 30-day rolling mean for readability
    y_roll30 = pd.Series(y, index=t).rolling(30, center=True, min_periods=10).mean().to_numpy()


     # Save event catalogue
    pd.DataFrame(events).to_csv(outdir / "events.csv", index=False)

# Save a short text summary (easy to read/share)
    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Region: lon[{region.lon_min},{region.lon_max}] lat[{region.lat_min},{region.lat_max}]\n")
        f.write(f"Window: {args.start} .. {args.end}\n")
        f.write(f"Baseline: {baseline_label}\n")
        f.write(f"Kalman: q={q:.6g}, r={r:.6g}\n")
        f.write(f"Trend raw (°C/dec): {trend_raw:.4f}\n")
        f.write(f"Trend smooth (°C/dec): {trend_s:.4f}\n")
        f.write(f"Events detected: {len(events)} (P90, >= {k} days)\n")











    summary = {
        "region": {"lon_min": region.lon_min, "lon_max": region.lon_max, "lat_min": region.lat_min, "lat_max": region.lat_max},
        "window": {"start": args.start, "end": args.end},
        "baseline": baseline_label,
        "kalman": {"q": float(q), "r": float(r)},
        "trend_anomaly_per_decade_raw_degC": trend_raw,
        "trend_anomaly_per_decade_smooth_degC": trend_s,
        "metrics_raw": metrics_raw,
        "metrics_smooth": metrics_s,
        "event_detection": {
            "method": "P90_consecutive_days",
            "threshold_degC": float(thr),
            "min_duration_days": k,
            "events": events,
        },
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote:", outdir / "summary.json")

    title = (
        f"SST anomaly (OISST) | region "
        f"{region.lon_min},{region.lon_max},{region.lat_min},{region.lat_max} | "
        f"{args.start}..{args.end}"
    )

    plot_timeseries_with_uncertainty(
        t, y, y_s, y_s_std,
        title=title,
        out_png=str(outdir / "timeseries_kalman.png"),
    )
    plot_histogram(y, title="Anomaly distribution (raw)", out_png=str(outdir / "hist_raw.png"))
    plot_histogram(y_s, title="Anomaly distribution (Kalman smooth)", out_png=str(outdir / "hist_smooth.png"))
    plot_timeseries_with_threshold(
    t, y, y_s, y_roll30, thr,
    title=title + " | Rolling mean + P90",
    out_png=str(outdir / "timeseries_threshold.png"),
    )

    print("Wrote plots in:", outdir)


if __name__ == "__main__":
    main()
