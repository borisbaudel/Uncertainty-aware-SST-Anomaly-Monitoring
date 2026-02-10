# 🌍 Uncertainty-aware SST Anomaly Monitoring (NOAA OISST)

This repository implements an **end-to-end, uncertainty-aware pipeline** for monitoring **regional sea surface temperature (SST) anomalies** using **NOAA OISST daily satellite observations**.

The methodology combines:
- daily climatology baselines (1991–2020),
- regional aggregation with physical weighting,
- **Kalman filtering and Rauch–Tung–Striebel (RTS) smoothing**,
- trend and extreme-event diagnostics,
- fully reproducible visual and machine-readable outputs.

---

## 📌 Region and Period

- **Region (example)**  
  Longitude: \\([-6^\circ, 6^\circ]\\)  
  Latitude: \\([34^\circ, 44^\circ]\\)

- **Analysis window**  
  2019-01-01 → 2020-12-31

---

## 📡 Data

The pipeline uses **NOAA Optimum Interpolation Sea Surface Temperature (OISST v2.x)** daily fields:

- **Daily observations**  
  `sst.day.mean.YYYY.nc`

- **Daily climatology (baseline)**  
  `sst.day.ltm.1991-2020.nc`

Spatial resolution: \\(0.25^\circ \\times 0.25^\circ\\)  
Temporal resolution: daily

---

## 🌐 Regional Mean SST

For each day \\(t\\), the regional mean SST is computed as a latitude-weighted average:

$$
\overline{T}(t)
=
\frac{\sum_{i,j} T(t,\phi_i,\lambda_j)\,\cos(\phi_i)}
     {\sum_{i,j} \cos(\phi_i)}
$$

where \\(\phi\\) denotes latitude and \\(\lambda\\) longitude.

---

## 📆 Daily Climatology and Anomalies

Let \\(d(t)\\) be the day-of-year index.

The daily climatology \\(T_{\mathrm{clim}}(d)\\) is defined as the mean SST for each calendar day over the 1991–2020 baseline.

The SST anomaly time series is then:

$$
A(t) = \overline{T}(t) - T_{\mathrm{clim}}(d(t))
$$

This removes the seasonal cycle and isolates interannual and intraseasonal variability.

---

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

---

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

---

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

---

## ⏪ RTS Smoother (Backward Pass)

To exploit all observations \\(y_{1:T}\\), a **Rauch–Tung–Striebel smoother** is applied.

**Smoother gain**
$$
G_t = \frac{P_{t|t}}{P_{t|t} + q}
$$

**Backward recursion**
$$
\hat{x}_{t|T}
=
\hat{x}_{t|t}
+
G_t\left(\hat{x}_{t+1|T} - \hat{x}_{t+1|t}\right)
$$

$$
P_{t|T}
=
P_{t|t}
+
G_t^2\left(P_{t+1|T} - P_{t+1|t}\right)
$$

The smoothed uncertainty is given by \\(\sigma_t = \sqrt{P_{t|T}}\\).

---

## 📈 Trend and Extreme Event Diagnostics

### Linear Trend
A linear trend is fitted as:
$$
A(t) = \alpha + \beta t + \varepsilon(t)
$$

Reported in \\(^\circ\mathrm{C}\\) per decade.

### Extreme Events
- Threshold: 90th percentile (P90) of \\(A(t)\\)
- Minimum duration: ≥ 3 consecutive days

Each event is characterized by duration, peak anomaly, and mean anomaly.

---

## 📊 Outputs

The pipeline produces:
- `summary.json` — trends, parameters, diagnostics
- `events.csv` — catalog of extreme warm events
- `timeseries_kalman.png` — anomalies + RTS smoothing + uncertainty
- `timeseries_threshold.png` — rolling mean + P90 threshold
- histograms of anomaly distributions

---

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
