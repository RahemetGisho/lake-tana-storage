"""Monthly rainfall forecasting with a leak-free rolling-origin evaluation.

Every forecast row is made at an *origin* month o using only data up to and including o,
and predicts months o+1 ... o+H. Models with parameters (SARIMA, gradient boosting) are
refitted once per calendar year on data strictly before that year, then updated month by
month with new observations but without refitting. Nothing from the future is visible.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODEL_ORDER = ["seasonal_naive", "climatology", "sarima", "gbm"]
MODEL_LABEL = {
    "seasonal_naive": "Seasonal naive",
    "climatology": "Climatology",
    "sarima": "SARIMA",
    "gbm": "Gradient boosting",
}


def _cal_month(o_month: int, h: int) -> int:
    """Calendar month (1-12) of the target h months after an origin in month o_month."""
    return (o_month - 1 + h) % 12 + 1


def rolling_forecasts(y: pd.Series, test_start: str, horizon: int) -> dict[str, np.ndarray]:
    """Return {model: array (n_test, horizon)}. Row i is the forecast made *before* test month i
    (origin = month i-1) for test months i, i+1, ..., i+horizon-1."""
    vals = y.values.astype(float)
    months = y.index.month.values
    n = len(vals)
    t0 = y.index.get_loc(pd.Timestamp(test_start))
    test_pos = np.arange(t0, n)
    out = {m: np.full((len(test_pos), horizon), np.nan) for m in MODEL_ORDER}

    # ---- seasonal naive and expanding climatology (no parameters, no refitting) ----
    for i, t in enumerate(test_pos):
        o = t - 1
        for h in range(1, horizon + 1):
            p = o + h
            out["seasonal_naive"][i, h - 1] = vals[p - 12]
            cm = _cal_month(months[o], h)
            hist = vals[: o + 1][months[: o + 1] == cm]
            out["climatology"][i, h - 1] = hist.mean()

    # ---- parametric models, refitted once per calendar year ----
    years = y.index.year.values[test_pos]
    for yr in np.unique(years):
        rows = np.where(years == yr)[0]
        t_first = test_pos[rows[0]]            # first test month of this year
        _fit_sarima_block(y, vals, out["sarima"], rows, test_pos, t_first, horizon)
        _fit_gbm_block(vals, months, out["gbm"], rows, test_pos, t_first, horizon)
    return out


def _fit_sarima_block(y, vals, target, rows, test_pos, t_first, horizon):
    z = pd.Series(np.sqrt(np.clip(vals, 0, None)), index=y.index)
    train = z.iloc[:t_first]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False, maxiter=200)
        for i in rows:
            o = test_pos[i] - 1
            r = res if o == t_first - 1 else res.append(z.iloc[t_first: o + 1], refit=False)
            fc = np.asarray(r.forecast(horizon))
            target[i, :] = np.clip(fc, 0, None) ** 2


def _gbm_features(a, o, h, cm_target, clim_target):
    lags = [a[o - k] for k in range(12)]
    ang = 2 * np.pi * cm_target / 12
    return lags + [np.sin(ang), np.cos(ang), clim_target, float(np.mean(lags[:3]))]


def _fit_gbm_block(vals, months, target, rows, test_pos, t_first, horizon):
    train_end = t_first - 1                       # last observed month available for training
    clim = np.array([vals[: train_end + 1][months[: train_end + 1] == m].mean() for m in range(1, 13)])
    a = vals - clim[months - 1]                   # anomalies relative to training climatology
    for h in range(1, horizon + 1):
        X, Y = [], []
        for o in range(11, train_end - h + 1):    # target o+h must be <= train_end
            cm = _cal_month(months[o], h)
            X.append(_gbm_features(a, o, h, cm, clim[cm - 1]))
            Y.append(a[o + h])
        model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=0)
        model.fit(np.array(X), np.array(Y))
        for i in rows:
            o = test_pos[i] - 1
            cm = _cal_month(months[o], h)
            x = np.array([_gbm_features(a, o, h, cm, clim[cm - 1])])
            target[i, h - 1] = max(0.0, clim[cm - 1] + float(model.predict(x)[0]))


def score_forecasts(fc: dict[str, np.ndarray], y: pd.Series, test_start: str) -> pd.DataFrame:
    """Per-model, per-horizon error. Wet season = June-September, which carries most of the water."""
    vals = y.values.astype(float)
    months = y.index.month.values
    n = len(vals)
    t0 = y.index.get_loc(pd.Timestamp(test_start))
    test_pos = np.arange(t0, n)
    rec = []
    for model, arr in fc.items():
        for h in range(1, arr.shape[1] + 1):
            pos = test_pos + h - 1
            ok = pos < n
            obs, pred, mo = vals[pos[ok]], arr[ok, h - 1], months[pos[ok]]
            err = pred - obs
            wet = (mo >= 6) & (mo <= 9)
            rec.append({
                "model": model, "horizon": h,
                "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
                "mae_mm": float(np.mean(np.abs(err))),
                "rmse_wet_mm": float(np.sqrt(np.mean(err[wet] ** 2))),
                "bias_mm": float(np.mean(err)),
                "n": int(ok.sum()),
            })
    df = pd.DataFrame(rec)
    base = df[df.model == "climatology"].set_index("horizon")["rmse_mm"]
    df["skill_vs_climatology"] = 1 - df["rmse_mm"] / df["horizon"].map(base)
    return df
