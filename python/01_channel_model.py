"""01_channel_model.py
Channel model: predicts whether a request will arrive digitally (ONLINE/MOBILE)
or by phone.

Use case: find the segments most likely to call in (complaint type, address
history, hour) and target them with digital-shift outreach such as SMS links,
QR codes or an online option inside the phone menu.

Design choices
  * Time-based split: train 2020-2023, validation 2024 (early stopping),
    test 2025-01 to 2026-09. The model will always predict the future.
  * No `yr` feature: trees cannot extrapolate to unseen years; the trend is
    handled by monitoring and recalibration (03, 04).
  * Address history features look only backwards (LAG / 1 PRECEDING in SQL).
  * Compared: complaint-type rate baseline, logistic regression (sklearn
    Pipeline), LightGBM, and LightGBM without history features (ablation).

Input : mart.ml_requests (deterministic 10% sample, ~2.04M rows)
Output: model channel_lgbm_v1 in the registry, outputs/tables/Channel-Model-Test-Metrics.csv,
        figures ROC-and-Calibration, Feature-Importance, Cumulative-Gain
"""

from __future__ import annotations

import time

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (CATEGORICAL, FEATURES, HISTORY, LGB_PARAMS, ML_COLUMNS, NUMERIC, PAL, TAB_DIR,
                    TARGET, engine, fig_path, load_registry, prepare_features, register, style_axes)

t0 = time.time()
df = pd.read_sql(f"SELECT {ML_COLUMNS} FROM mart.ml_requests ORDER BY unique_key", engine())
print(f"Rows read: {len(df):,}  ({time.time() - t0:.0f} s)")

train, valid, test = df[df.yr <= 2023], df[df.yr == 2024], df[df.yr >= 2025]
print(f"train={len(train):,}  valid={len(valid):,}  test={len(test):,}")
print("Digital rate  train=%.3f  valid=%.3f  test=%.3f" % (train[TARGET].mean(), valid[TARGET].mean(), test[TARGET].mean()))

X_tr, cats = prepare_features(train)
X_va, _ = prepare_features(valid, cats)
X_te, _ = prepare_features(test, cats)
y_tr, y_va, y_te = train[TARGET].values, valid[TARGET].values, test[TARGET].values


def metrics(name, y, p):
    return {"model": name, "roc_auc": roc_auc_score(y, p),
            "pr_auc_phone": average_precision_score(1 - y, 1 - p),   # minority class = phone
            "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)}


results, preds = [], {}

# ---- baseline: digital rate of the complaint type in the training years
rate = train.groupby("complaint_type")[TARGET].mean()
preds["baseline"] = test["complaint_type"].map(rate).fillna(train[TARGET].mean()).clip(1e-4, 1 - 1e-4).values
results.append(metrics("Baseline: complaint type rate", y_te, preds["baseline"]))

# ---- logistic regression pipeline
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), CATEGORICAL),
    ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", StandardScaler())]), NUMERIC)])
logreg = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=300))])
as_str = {c: str for c in CATEGORICAL}
logreg.fit(X_tr.astype(as_str), y_tr)
preds["logistic"] = logreg.predict_proba(X_te.astype(as_str))[:, 1]
results.append(metrics("Logistic regression (one-hot)", y_te, preds["logistic"]))


# ---- LightGBM, with and without address history
def fit_lgb(cols):
    t = time.time()
    dtr = lgb.Dataset(X_tr[cols], y_tr, categorical_feature=[c for c in cols if c in CATEGORICAL])
    dva = lgb.Dataset(X_va[cols], y_va, reference=dtr)
    m = lgb.train(LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"LightGBM ({len(cols)} features) best_iter={m.best_iteration} ({time.time() - t:.0f} s)")
    return m


gbm = fit_lgb(FEATURES)
preds["lightgbm"] = gbm.predict(X_te[FEATURES], num_iteration=gbm.best_iteration)
results.append(metrics("LightGBM (all features)", y_te, preds["lightgbm"]))

no_hist = [c for c in FEATURES if c not in HISTORY]
gbm_nh = fit_lgb(no_hist)
preds["ablation"] = gbm_nh.predict(X_te[no_hist], num_iteration=gbm_nh.best_iteration)
results.append(metrics("LightGBM without address history", y_te, preds["ablation"]))

res = pd.DataFrame(results).round(4)
print("\n=== Test set (2025-01 to 2026-09) ===")
print(res.to_string(index=False))
res.to_csv(TAB_DIR / "Channel-Model-Test-Metrics.csv", index=False)

print("\n=== Performance over time (drift check) ===")
yr = test.yr.values
print(pd.DataFrame([{"year": y, "n": int((yr == y).sum()),
                     "actual_digital": y_te[yr == y].mean(),
                     "mean_prediction": preds["lightgbm"][yr == y].mean(),
                     "roc_auc": roc_auc_score(y_te[yr == y], preds["lightgbm"][yr == y])}
                    for y in sorted(set(yr))]).round(4).to_string(index=False))

# ---- business metric: how many phone requests does the riskiest slice capture?
order = np.argsort(preds["lightgbm"])            # lowest P(digital) = highest P(phone)
phone = (1 - y_te)[order]
cum_phone = np.cumsum(phone) / phone.sum()
frac = np.arange(1, len(phone) + 1) / len(phone)
gain = {k: float(cum_phone[int(len(phone) * k / 100) - 1]) for k in (10, 20, 30)}
print("\n=== Cumulative gain: share of phone requests caught by the top-k% riskiest requests ===")
print({f"top {k}%": round(v, 3) for k, v in gain.items()}, f"(random: 0.10 / 0.20 / 0.30; test phone rate = {1 - y_te.mean():.3f})")

# ---- figures (colors from config/palette.json)
labels = {"baseline": "Baseline", "logistic": "Logistic regression",
          "lightgbm": "LightGBM", "ablation": "LightGBM without history"}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=PAL["surface"])
for key in ("lightgbm", "logistic", "ablation", "baseline"):
    fpr, tpr, _ = roc_curve(y_te, preds[key])
    ax[0].plot(fpr, tpr, lw=1.8, color=PAL["model"][key],
               label=f"{labels[key]} (AUC {roc_auc_score(y_te, preds[key]):.3f})")
ax[0].plot([0, 1], [0, 1], "--", color=PAL["reference"], lw=1)
ax[0].set(title="ROC curves, test 2025-01 to 2026-09", xlabel="False positive rate", ylabel="True positive rate")
ax[0].legend(fontsize=9, loc="lower right", frameon=False)
for key in ("lightgbm", "logistic"):
    frac_pos, mean_pred = calibration_curve(y_te, preds[key], n_bins=15, strategy="quantile")
    ax[1].plot(mean_pred, frac_pos, "o-", ms=5, lw=1.8, color=PAL["model"][key], label=labels[key])
ax[1].plot([0, 1], [0, 1], "--", color=PAL["reference"], lw=1, label="Perfect calibration")
ax[1].set(title="Calibration (15 quantile bins)", xlabel="Predicted P(digital)", ylabel="Observed digital rate")
ax[1].legend(fontsize=9, frameon=False)
for a in ax:
    style_axes(a)
fig.tight_layout(); fig.savefig(fig_path("ROC-and-Calibration"), dpi=150); plt.close(fig)

imp = pd.Series(gbm.feature_importance("gain"), index=FEATURES)
imp = (imp / imp.sum()).sort_values()
print("\n=== Feature importance (share of total gain) ===")
print(imp.sort_values(ascending=False).round(3).to_string())
fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=PAL["surface"])
ax.barh(imp.index, imp.values, height=0.7,
        color=[PAL["model"]["lightgbm"] if c in HISTORY else PAL["reference"] for c in imp.index])
for i, v in enumerate(imp.values):
    ax.text(v + 0.004, i, f"{v:.0%}", va="center", fontsize=9, color=PAL["text_primary"])
ax.set(title="LightGBM feature importance (share of gain)\nviolet = address history features", xlabel="Share of total gain")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
style_axes(ax); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(fig_path("Feature-Importance"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=PAL["surface"])
step = max(1, len(frac) // 2000)
ax.plot(frac[::step], cum_phone[::step], lw=2, color=PAL["model"]["lightgbm"], label="LightGBM")
ax.plot([0, 1], [0, 1], "--", color=PAL["reference"], lw=1, label="Random targeting")
for k, v in gain.items():
    ax.plot(k / 100, v, "o", ms=7, color=PAL["model"]["lightgbm"])
    ax.annotate(f"top {k}% → {v:.0%} of phone requests", (k / 100, v), textcoords="offset points",
                xytext=(10, -12), fontsize=9, color=PAL["text_primary"])
ax.set(title="Cumulative gain: catching phone requests in advance",
       xlabel="Requests ranked by P(phone)", ylabel="Share of phone requests caught")
ax.legend(frameon=False, loc="lower right"); style_axes(ax)
fig.tight_layout(); fig.savefig(fig_path("Cumulative-Gain"), dpi=150); plt.close(fig)

# ---- register v1. It only becomes active when no other version is active yet,
# so re-running this script never demotes a model promoted by 04_retrain.py.
bundle = {"model": gbm, "features": FEATURES, "categories": cats, "best_iteration": gbm.best_iteration,
          "version": "channel_lgbm_v1", "calibrator": None}
card = {"trained_on": "2020-2023, early stopping on 2024", "best_iteration": gbm.best_iteration,
        "test_2025_2026": res.iloc[2, 1:].to_dict()}
register(bundle, card, activate=False)
print(f"\nRegistered channel_lgbm_v1; active model: {load_registry()['active']}  (total {time.time() - t0:.0f} s)")
