# 🌍 Uncertainty-aware SST Anomaly Monitoring (NOAA OISST)

This repository implements an **end-to-end, uncertainty-aware pipeline** for monitoring **regional sea surface temperature (SST) anomalies** using **NOAA OISST daily satellite observations**.

The methodology combines:
- daily climatology baselines (1991–2020),
- regional aggregation with physical weighting,
- **Kalman filtering and Rauch–Tung–Striebel (RTS) smoothing**,
- trend and extreme-event diagnostics,
- fully reproducible visual and machine-readable outputs.


<img width="2022" height="1002" alt="image" src="https://github.com/user-attachments/assets/be2a2601-044e-4d54-a646-ebd656d17fad" />


<img width="1338" height="990" alt="image" src="https://github.com/user-attachments/assets/54638033-a645-475e-9b52-e814272eea90" />


<img width="1346" height="980" alt="image" src="https://github.com/user-attachments/assets/6efaaf84-eecf-49b0-8e44-14755b5bd501" />

<img width="1316" height="994" alt="image" src="https://github.com/user-attachments/assets/19906f57-292e-4808-8b7e-8911b1c82c40" />

## 📌 Region and Period

- **Region (example)**  
  Longitude: \\([-6^\circ, 6^\circ]\\)  
  Latitude: \\([34^\circ, 44^\circ]\\)

- **Analysis window**  
  2019-01-01 → 2020-12-31

## 📡 Data

The pipeline uses **NOAA Optimum Interpolation Sea Surface Temperature (OISST v2.x)** daily fields:

- **Daily observations**  
  `sst.day.mean.YYYY.nc`

- **Daily climatology (baseline)**  
  `sst.day.ltm.1991-2020.nc`

Spatial resolution: \\(0.25^\circ \\times 0.25^\circ\\)  
Temporal resolution: daily


## 🌐 Regional Mean SST

For each day \\(t\\), the regional mean SST is computed as a latitude-weighted average:

$$
\overline{T}(t)
= \frac{\sum_{i,j} T(t,\phi_i,\lambda_j)\,\cos(\phi_i)} {\sum_{i,j} \cos(\phi_i)}
$$

where \\(\phi\\) denotes latitude and \\(\lambda\\) longitude.

## 📆 Daily Climatology and Anomalies

Let \\(d(t)\\) be the day-of-year index.

The daily climatology \\(T_{\mathrm{clim}}(d)\\) is defined as the mean SST for each calendar day over the 1991–2020 baseline.

The SST anomaly time series is then:

$$
A(t) = T(t) - T_{\mathrm{clim}}(d(t))
$$

This removes the seasonal cycle and isolates interannual and intraseasonal variability.

## 📐 State-Space Model (Local Level)

Anomalies are modeled using a **local-level state-space model**:

$$
x_t = x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, q)
$$

$$
y_t = x_t + v_t, \quad v_t \sim \mathcal{N}(0, r)
$$

where:
- \\(y_t\\) is the observed SST anomaly,
- \\(x_t\\) is the latent climate signal,
- \\(q\\) is the process noise variance,
- \\(r\\) is the observation noise variance.

## 🔁 Kalman Filter (Forward Pass)

**Prediction**
$$
\hat{x}_{t|t-1} = \hat{x}_{t-1|t-1}
$$

$$
P_{t|t-1} = P_{t-1|t-1} + q
$$

**Update**
$$
K_t = \frac{P_{t|t-1}}{P_{t|t-1} + r}
$$

$$
\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \left(y_t - \hat{x}_{t|t-1}\right)
$$

$$
P_{t|t} = (1 - K_t)\,P_{t|t-1}
$$

## 🔄 Riccati Equation (Scalar Form)

The covariance recursion follows a discrete Riccati equation:

$$
P_{t|t} =
\frac{r \,(P_{t-1|t-1} + q)}
     {(P_{t-1|t-1} + q) + r}
$$

### Steady-State Solution

At convergence:

$$
P^2 + qP - rq = 0
$$

yielding the positive root:

$$
P =
\frac{-q + \sqrt{q^2 + 4rq}}{2}
$$

## ⏪ RTS Smoother (Backward Pass)

To exploit all observations \\(y_{1:T}\\), a **Rauch–Tung–Striebel smoother** is applied.

**Smoother gain**
$$G_t = \frac{P_{t|t}}{P_{t|t} + q} $$

**Backward recursion**

$$ \hat{x}_{t|T} = \hat{x}_{t|t} + G_t\left(\hat{x}_{t+1|T} - \hat{x}_{t+1|t}\right) $$

$$
P_{t|T} = P_{t|t} + G_t^2\left(P_{t+1|T} - P_{t+1|t}\right)
$$

The smoothed uncertainty is given by \\(\sigma_t = \sqrt{P_{t|T}}\\).

## 📈 Trend and Extreme Event Diagnostics

### Linear Trend
A linear trend is fitted as:

$$ A(t) = \alpha + \beta t + \varepsilon(t) $$

Reported in \\(^\circ\mathrm{C}\\) per decade.

### Extreme Events
- Threshold: 90th percentile (P90) of \\(A(t)\\)
- Minimum duration: ≥ 3 consecutive days

Each event is characterized by duration, peak anomaly, and mean anomaly.

## 📊 Outputs

The pipeline produces:
- `summary.json` — trends, parameters, diagnostics
- `events.csv` — catalog of extreme warm events
- `timeseries_kalman.png` — anomalies + RTS smoothing + uncertainty
- `timeseries_threshold.png` — rolling mean + P90 threshold
- histograms of anomaly distributions
  

## ▶️ Example Usage (Offline)

```bash
python diag.py \
  --offline \
  --cache-dir ./oisst_cache \
  --lon-min -6 --lon-max 6 \
  --lat-min 34 --lat-max 44 \
  --start 2019-01-01 --end 2020-12-31 \
  --climatology-file ./oisst_cache/sst.day.ltm.1991-2020.nc \
  --outdir ./outputs_2019_2020

## 📦 Data Availability (External Sources)

Due to their size (≈ 450–500 MB per year), the raw satellite datasets used in this project
are **not included in this GitHub repository**.

All data are **publicly available** from the NOAA Physical Sciences Laboratory (PSL)
and can be downloaded directly from the official THREDDS file server.

### 🌊 Daily SST Observations (NOAA OISST)

The analysis relies on daily mean Sea Surface Temperature (SST) fields from
**NOAA OISST v2.x**.

For each year \\(Y\\), the required file is:

**Direct download links (HTTP):**

- **2019**  
  https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres/sst.day.mean.2019.nc

- **2020**  
  https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres/sst.day.mean.2020.nc

- **2022 (example test year)**  
  https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres/sst.day.mean.2022.nc

Each file contains daily SST fields on a \\(0.25^\circ \\times 0.25^\circ\\) global grid.

---

## 📆 Daily Climatology (Baseline 1991–2020)

To remove the seasonal cycle, SST anomalies are computed relative to a daily climatology
constructed over the 1991–2020 baseline.

The corresponding file is:

$$
\texttt{sst.day.ltm.1991--2020.nc}
$$

**Direct download link:**

https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres/sst.day.ltm.1991-2020.nc

---

### 📁 Expected Local Data Structure

After manual download, files should be placed in a local cache directory, for example:

