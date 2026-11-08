"""02_phone_forecast.py
Daily PHONE demand forecast, 28 days ahead, for call-center staffing.

sql/03_channel_mix.sql Q5 shows 55% of phone requests arrive in office hours,
with strong weekly and yearly seasonality. Staffing rotas are set about four
weeks out, so the horizon is 28 days.

Method
  * Direct multi-step forecast: every lag is >= 28 days, so each day of the
    horizon uses only data known on the forecast date.
  * Detrended target: log1p(n) - L, where L is the mean log level of the 28 days
    ending 28 days earlier. Phone volume keeps falling, and a tree model cannot
    extrapolate to levels it has never seen, so it learns the deviation from the
    recent level instead.
  * Features: lags 28/35/42/364 relative to L, day of week, month, day of year,
    US federal holidays.
  * Baselines: seasonal naive (same weekday 4 weeks earlier) and yearly naive (364 days).
  * Rolling-origin backtest: last 6 x 28 days, each fold trained only on earlier data.
  * Prediction interval: empirical 10%-90% quantiles of backtest actual/forecast ratios.

Input : mart.daily_channel_volume
Output: outputs/tables/Phone-Forecast-Backtest.csv, Phone-Forecast-Next-28-Days.csv, figure Phone-Demand-Forecast
"""

from __future__ import annotations

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar

from common import PAL, TAB_DIR, engine, fig_path, style_axes

H = 28
N_FOLDS = 6
DAY = pd.to_timedelta(1, unit="D")
PARAMS = dict(objective="regression_l1", learning_rate=0.03, num_leaves=31, min_child_samples=20,
              feature_fraction=0.9, verbose=-1, seed=42, deterministic=True, force_row_wise=True)
NUM_ROUNDS = 600
FEATURES = ["rel_lag_28", "rel_lag_35", "rel_lag_42", "rel_lag_364", "dow", "month", "doy",
            "is_holiday", "near_holiday"]

daily = (pd.read_sql("SELECT day, n FROM mart.daily_channel_volume WHERE channel = 'PHONE' ORDER BY day",
                     engine(), parse_dates=["day"])
         .set_index("day")["n"].asfreq("D", fill_value=0))
print(f"Series: {daily.index.min().date()} to {daily.index.max().date()} ({len(daily)} days)")
dow_names = dict(enumerate("Mon Tue Wed Thu Fri Sat Sun".split()))
print("Mean daily phone requests by weekday, 2026:")
print(daily["2026"].groupby(daily["2026"].index.dayofweek).mean().round(0).rename(index=dow_names).to_string())

holidays = USFederalHolidayCalendar().holidays(daily.index.min(), daily.index.max() + H * DAY)


def make_features(idx: pd.DatetimeIndex, y: pd.Series) -> pd.DataFrame:
    """Features for the days in idx; y is only read at dates at least 28 days earlier."""
    ly = np.log1p(y)
    f = pd.DataFrame(index=idx)
    f["level"] = ly.rolling(28).mean().reindex(idx - 28 * DAY).values
    for lag in (28, 35, 42, 364):
        f[f"rel_lag_{lag}"] = ly.reindex(idx - lag * DAY).values - f["level"]
    f["dow"], f["month"], f["doy"] = idx.dayofweek, idx.month, idx.dayofyear
    f["is_holiday"] = idx.isin(holidays).astype(int)
    f["near_holiday"] = (pd.Series(idx.isin(holidays), index=idx)
                         .rolling(3, center=True, min_periods=1).max().astype(int).values)
    return f


def fit(idx, y):
    X = make_features(idx, y)
    return lgb.train(PARAMS, lgb.Dataset(X[FEATURES], np.log1p(y[idx]) - X["level"]), num_boost_round=NUM_ROUNDS)


def forecast(model, idx, y):
    X = make_features(idx, y)
    return np.expm1(model.predict(X[FEATURES]) + X["level"].values)


def train_index(y):
    return y.index[y.index >= y.index.min() + (364 + 28) * DAY]   # first day with every lag available


end = daily.index.max()
fold_starts = [end - (H * k - 1) * DAY for k in range(N_FOLDS, 0, -1)]
rows = []
for start in fold_starts:
    horizon = pd.date_range(start, periods=H, freq="D")
    hist = daily[daily.index < start]
    model = fit(train_index(hist), hist)
    rows.append(pd.DataFrame({"day": horizon, "fold_start": start.date(), "actual": daily[horizon].values,
                              "lightgbm": forecast(model, horizon, hist),
                              "seasonal_naive_28": hist.reindex(horizon - 28 * DAY).values,
                              "yearly_naive_364": hist.reindex(horizon - 364 * DAY).values}))
bt = pd.concat(rows, ignore_index=True)
bt.to_csv(TAB_DIR / "Phone-Forecast-Backtest.csv", index=False)

MODELS = ("lightgbm", "seasonal_naive_28", "yearly_naive_364")


def wape(d, col):
    return 100 * (d[col] - d["actual"]).abs().sum() / d["actual"].sum()


sc = pd.DataFrame([{"model": c, "MAE": (bt[c] - bt.actual).abs().mean(), "WAPE_%": wape(bt, c),
                    "bias_%": 100 * (bt[c] - bt.actual).sum() / bt.actual.sum()} for c in MODELS]).round(2)
print(f"\n=== Rolling-origin backtest: {N_FOLDS} folds x {H} days ({bt.day.min().date()} to {bt.day.max().date()}) ===")
print(sc.to_string(index=False))
print("\nWAPE (%) by fold:")
print(bt.groupby("fold_start").apply(lambda d: pd.Series({c: wape(d, c) for c in MODELS[:2]}),
                                     include_groups=False).round(1).to_string())

ratio = bt["actual"] / bt["lightgbm"]
q10, q90 = ratio.quantile([0.10, 0.90])
cover = ((bt.actual >= bt.lightgbm * q10) & (bt.actual <= bt.lightgbm * q90)).mean()
print(f"\n80% interval multipliers: [{q10:.3f}, {q90:.3f}]  (backtest coverage: {cover:.1%})")

future = pd.date_range(end + DAY, periods=H, freq="D")
fc = pd.DataFrame({"day": future, "forecast": forecast(fit(train_index(daily), daily), future, daily)})
fc["low_80"], fc["high_80"] = fc.forecast * q10, fc.forecast * q90
fc.round(0).to_csv(TAB_DIR / "Phone-Forecast-Next-28-Days.csv", index=False)
weekly = (fc.assign(week=fc.day.dt.to_period("W-SUN").astype(str))
            .groupby("week")[["forecast", "low_80", "high_80"]].sum().round(0).astype(int))
print(f"\n=== Next 28 days ({future[0].date()} to {future[-1].date()}): weekly PHONE requests ===")
print(weekly.to_string())

# ---- figure: actual phone demand in the PHONE color, model output in the model color
fig, ax = plt.subplots(2, 1, figsize=(12, 8.5), facecolor=PAL["surface"],
                       gridspec_kw={"height_ratios": [2.2, 1]})
recent = daily["2025-10-01":]
ax[0].plot(recent.index, recent.values, color=PAL["channel"]["PHONE"], lw=1.1, label="Actual phone requests")
ax[0].plot(bt.day, bt.lightgbm, color=PAL["model"]["lightgbm"], lw=1.6, label="LightGBM backtest (28 days ahead)")
for s in fold_starts:
    ax[0].axvline(s, color=PAL["reference"], alpha=0.5, ls=":", lw=1)
ax[0].fill_between(fc.day, fc.low_80, fc.high_80, color=PAL["model"]["lightgbm"], alpha=0.18, label="80% interval")
ax[0].plot(fc.day, fc.forecast, color=PAL["model"]["lightgbm"], lw=2.2, label="Forecast, next 28 days")
ax[0].set(title="Daily phone requests: backtest folds (dotted lines) and 28-day forecast", ylabel="Requests per day")
ax[0].legend(fontsize=9, ncol=2, frameon=False, loc="upper right")

bar_colors = [PAL["model"]["lightgbm"], PAL["model"]["baseline"], PAL["reference_light"]]
ax[1].bar(["LightGBM", "Seasonal naive (28 d)", "Yearly naive (364 d)"], sc["WAPE_%"], color=bar_colors,
          width=0.6, edgecolor=PAL["surface"], linewidth=2)
for i, v in enumerate(sc["WAPE_%"]):
    ax[1].text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=10, color=PAL["text_primary"])
ax[1].set(title=f"Backtest error (WAPE, {N_FOLDS} folds x {H} days), lower is better", ylabel="WAPE (%)")
for a in ax:
    style_axes(a)
ax[1].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(fig_path("Phone-Demand-Forecast"), dpi=150); plt.close(fig)
print("\nSaved: Phone-Demand-Forecast figure, Phone-Forecast-Backtest.csv, Phone-Forecast-Next-28-Days.csv")
