# iosst_io.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import os
import time
import urllib.request

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Region:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


def oisst_year_url(year: int) -> str:
    return (
        "https://psl.noaa.gov/thredds/dodsC/"
        f"Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
    )


def oisst_year_http_url(year: int) -> str:
    return (
        "https://psl.noaa.gov/thredds/fileServer/"
        f"Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
    )


def _safe_open_dataset(
    path_or_url: str | Path,
    *,
    prefer_h5netcdf: bool = True,
    chunks: dict | None = None,
    keep_vars: tuple[str, ...] = ("sst",),
) -> xr.Dataset:
    """
    Robust opener:
    - prefers h5netcdf (Windows-friendly) if installed
    - falls back to netcdf4
    - drops extra variables to avoid slow formatting / accidental reads
    - uses chunks to avoid loading everything
    - disables xarray's internal file cache for safety
    """
    # Avoid xarray keeping open file handles too aggressively on Windows

    # Make sure we pass a string path
    p = str(path_or_url)

    # Only keep desired vars (if they exist). We'll open first, then subset.
    def _subset_vars(ds: xr.Dataset) -> xr.Dataset:
        existing = [v for v in keep_vars if v in ds.data_vars]
        if existing:
            ds = ds[existing]
        return ds

    # Try engines
    engines: list[str] = []
    if prefer_h5netcdf:
        engines.append("h5netcdf")
    engines.append("netcdf4")

    last_err: Exception | None = None
    for eng in engines:
        try:
            ds = xr.open_dataset(
                p,
                engine=eng,
                decode_times=True,
                mask_and_scale=True,
                chunks=chunks,
            )
            ds = _subset_vars(ds)
            return ds
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to open dataset {p} with engines={engines}: {last_err}")


def subset_region_time(ds: xr.Dataset, region: Region, time_start: str, time_end: str) -> xr.Dataset:
    # Keep only SST to reduce overhead
    if "sst" in ds.data_vars:
        ds = ds[["sst"]]

    ds = ds.sel(time=slice(time_start, time_end))

    lon = ds["lon"]
    # Convert 0..360 -> -180..180 if needed
    if float(lon.min()) >= 0.0 and float(lon.max()) > 180.0:
        lon180 = ((lon + 180) % 360) - 180
        ds = ds.assign_coords(lon=lon180).sortby("lon")

    return ds.sel(
        lon=slice(region.lon_min, region.lon_max),
        lat=slice(region.lat_min, region.lat_max),
    )


def area_weights_lat(lat: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def regional_mean_sst(ds: xr.Dataset) -> xr.DataArray:
    if "sst" not in ds.data_vars:
        raise KeyError(f"'sst' not found. Available vars: {list(ds.data_vars)}")

    sst = ds["sst"]
    w = area_weights_lat(ds["lat"])
    reg = sst.weighted(w).mean(dim=("lat", "lon"), skipna=True)
    reg.name = "sst_regional_mean"
    reg.attrs["units"] = sst.attrs.get("units", "degC")
    return reg


def _download_year_to_cache(year: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"sst.day.mean.{year}.nc"
    if out.exists():
        return out

    url = oisst_year_http_url(year)
    print(f"[download] {year} -> {url} -> {out}")
    urllib.request.urlretrieve(url, out)
    return out


def open_oisst_regional_mean_years(
    years: Iterable[int],
    region: Region,
    time_start: str,
    time_end: str,
    cache_dir: str | None = None,
    offline: bool = False,
    max_retries: int = 6,
) -> xr.DataArray:
    series: list[xr.DataArray] = []
    cache_path = Path(cache_dir) if cache_dir is not None else None

    # Chunks: lecture stable sans avaler le fichier annuel entier
    # (OISST est time x lat x lon : chunker principalement sur time)
    chunks = {"time": 31}  # ~1 mois

    for y in years:
        ys = max(time_start, f"{y}-01-01")
        ye = min(time_end, f"{y}-12-31")
        if ys > ye:
            continue

        local_file: Path | None = None
        if cache_path is not None:
            candidate = cache_path / f"sst.day.mean.{y}.nc"
            if candidate.exists():
                local_file = candidate
            elif offline:
                raise FileNotFoundError(f"Offline mode: missing local file {candidate}")

        last_err: Exception | None = None
        for k in range(max_retries):
            try:
                if local_file is not None:
                    print(f"[open-local] {y} -> {local_file}")
                    ds = _safe_open_dataset(local_file, chunks=chunks, keep_vars=("sst",))
                else:
                    if (cache_path is not None) and (not offline):
                        local_file = _download_year_to_cache(y, cache_path)
                        print(f"[open-local] {y} -> {local_file}")
                        ds = _safe_open_dataset(local_file, chunks=chunks, keep_vars=("sst",))
                    else:
                        url = oisst_year_url(y)
                        print(f"[open-dap] {y} -> {url}")
                        # OPeNDAP: netcdf4 est souvent le plus compatible
                        ds = _safe_open_dataset(url, prefer_h5netcdf=False, chunks=chunks, keep_vars=("sst",))

                print(f"[subset] year={y} time={ys}..{ye}")
                ds_sub = subset_region_time(ds, region, ys, ye)

                print(f"[mean] year={y} computing regional mean")
                reg = regional_mean_sst(ds_sub).load()

                print(f"[done] year={y} loaded {reg.sizes.get('time', 'NA')} steps")
                ds.close()

                series.append(reg)
                last_err = None
                break

            except Exception as e:
                last_err = e
                wait = 2**k
                print(f"[retry] year={y} attempt={k+1}/{max_retries} failed: {e} -> sleep {wait}s")
                time.sleep(wait)

        if last_err is not None:
            raise RuntimeError(f"Failed too many times for year={y}: {last_err}")

    if not series:
        raise ValueError("No data returned for requested window/years.")

    out = xr.concat(series, dim="time").sortby("time")
    out.name = "sst_regional_mean"
    return out
