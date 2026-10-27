# Results

Location: 11.85N, 37.45E (NASA POWER grid cell). Record: 1990-2025.
Training data before 2016-01; 120 held-out months.

## 1. Rainfall forecast error (held-out months)

| Model             | 1-month RMSE (mm) | 2-month RMSE (mm) | 3-month RMSE (mm) |
| ----------------- | ----------------- | ----------------- | ----------------- |
| Seasonal naive    | 82.43             | 82.78             | 83.13             |
| Climatology       | 65.95             | 66.23             | 66.51             |
| SARIMA            | 65.62             | 64.14             | 65.51             |
| Gradient boosting | 71.38             | 69.91             | 72.09             |

Skill against climatology at 1 month: SARIMA +0.5%, boosting -8.2%, seasonal naive -25.0%. Positive means better than the long-run monthly average.

## 2. Reservoir release rules (held-out months)

Capacity 1,900 Mm3, firm demand 253 Mm3/month, max release 380 Mm3/month.

| Release rule      | Shortfall months | Unmet (Mm3) | Spill (Mm3) | Spill months | Energy (GWh) |
| ----------------- | ---------------- | ----------- | ----------- | ------------ | ------------ |
| No forecast       | 7.0              | 703.4       | 4,908.3     | 18.0         | 7,275.7      |
| Climatology       | 8.0              | 867.0       | 1,841.5     | 8.0          | 8,027.0      |
| SARIMA forecast   | 8.0              | 800.6       | 1,944.6     | 10.0         | 8,001.8      |
| Boosting forecast | 8.0              | 788.1       | 1,972.9     | 9.0          | 7,994.9      |
| Perfect foresight | 7.0              | 703.4       | 1,466.0     | 8.0          | 8,119.1      |

## 3. Solar and wind under dry and weak-rain conditions

| Condition                 | solar_kwh_m2_day | wind_ms_50m | rain_mm_month | months |
| ------------------------- | ---------------- | ----------- | ------------- | ------ |
| Dry season (Dec-Feb)      | 6.20             | 4.13        | 6.93          | 108.00 |
| Rainy season, weak years  | 5.55             | 3.99        | 202.92        | 36.00  |
| Rainy season, other years | 5.53             | 3.94        | 328.60        | 108.00 |

## 4. Covering the hydropower gap

Hydropower shortfall months in the 1990-2025 simulation under no-forecast operation: 21.
A 100 MW portfolio (50 MW solar, 50 MW wind) covers 36% of the shortfall energy; 200 MW covers 64%.
Worst single month needs about 478 MW of 50/50 solar and wind to cover it fully.

## 5. Battery dispatch (SYNTHETIC load, read the caveats)

A 40 MWp solar plant with a 50 MWh / 12.5 MW battery is reserved for the evening peak of a synthetic feeder
(mean 10 MW). Each night the operator decides how much to top the battery up from hydro. Cost = 2 x evening peak
energy hydro must supply + 1 x overnight top-up energy. Daily solar is real NASA POWER data; the load profile is invented and the
hourly PV shape is a clear-sky bell scaled to the real daily total.

| Day-ahead solar forecast | RMSE (kWh/m2/day) | MAE   | Bias  | Skill vs climatology |
| ------------------------ | ----------------- | ----- | ----- | -------------------- |
| Climatology              | 0.820             | 0.613 | 0.057 | 0.000                |
| Persistence              | 0.807             | 0.558 | 0.000 | 0.016                |
| AR(1)                    | 0.704             | 0.509 | 0.029 | 0.141                |

| Policy                | Evening peak left to hydro (MWh) | Overnight top-up (MWh) | Weighted cost | Curtailed PV (MWh) | Saving vs best static (%) |
| --------------------- | -------------------------------- | ---------------------- | ------------- | ------------------ | ------------------------- |
| Solar only            | 91,601.7                         | 0.0                    | 183,203.3     | 133,214.3          | 0.0                       |
| Always top up         | 84,756.3                         | 198,532.6              | 368,045.2     | 323,659.3          | -100.9                    |
| Top up Jun-Sep        | 85,975.2                         | 66,304.3               | 238,254.8     | 192,871.1          | -30.0                     |
| Forecast: climatology | 91,601.7                         | 0.0                    | 183,203.3     | 133,214.3          | 0.0                       |
| Forecast: persistence | 89,096.6                         | 8,088.3                | 186,281.4     | 138,342.9          | -1.7                      |
| Forecast: AR(1)       | 90,479.5                         | 2,373.5                | 183,332.4     | 134,261.9          | -0.1                      |
| Perfect foresight     | 84,756.3                         | 8,088.3                | 177,600.9     | 133,215.0          | 3.1                       |

## Assumptions to state when you present this

Inflow is modelled as lagged rainfall, scaled to an assumed mean annual inflow. Reservoir size, demand and the plant are illustrative.
The point of the experiment is the comparison between release rules, not the absolute volumes.
