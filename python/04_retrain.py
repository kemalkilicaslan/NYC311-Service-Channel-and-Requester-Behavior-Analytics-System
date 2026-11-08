"""04_retrain.py
Champion / challenger retraining, triggered by the monitoring alert of
03_batch_scoring.py (2026-09 calibration gap -5.7 pp with PSI 0.047: the score
distribution barely moved, the relationship between features and outcome did).

Candidates, all compared on the same unseen holdout (2026-01 to 2026-09)
  A  champion v1                 : trained 2020-2023
  B  champion + static Platt     : v1 recalibrated once on 2025-H2
  C  challenger                  : same features and parameters, trained 2020-01 to 2025-06,
                                   early stopping on 2025-H2
  D  challenger + rolling Platt  : for each month m, a calibrator fitted on months m-3..m-1
                                   (labels known by then) and applied to m, which is how a
                                   monthly recalibration job would run in production
Promotion rule: lowest holdout log loss AND |calibration gap| < 3 pp.
If D wins, its calibrator is refitted on the last three complete months and
stored in the bundle, so the next scoring run uses it.

Output: registry entry for channel_lgbm_v2, outputs/tables/Champion-Challenger-Comparison.csv, figure Calibration-Gap-by-Month
"""

from __future__ import annotations

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common import (CATEGORICAL, FEATURES, LGB_PARAMS, ML_COLUMNS, PAL, TAB_DIR, TARGET, engine,
                    fig_path, load_model, load_registry, logit, predict, prepare_features, register, style_axes)

GAP_LIMIT_PP = 3.0
CHALLENGER = "channel_lgbm_v2"

df = pd.read_sql(f"SELECT {ML_COLUMNS} FROM mart.ml_requests ORDER BY unique_key", engine(), parse_dates=["created_at"])
month = df.created_at.dt.to_period("M")
holdout = df[df.created_at >= "2026-01-01"]
h2_2025 = df[(df.created_at >= "2025-07-01") & (df.created_at < "2026-01-01")]
train_c = df[df.created_at < "2025-07-01"]
print(f"challenger train={len(train_c):,}  2025-H2={len(h2_2025):,}  holdout 2026-01..09={len(holdout):,}")

# The champion is the active model. If this script already promoted the challenger,
# compare against the version it replaced, so re-runs reproduce the same decision.
reg = load_registry()
champ_version = reg["active"]
if champ_version == CHALLENGER:
    champ_version = reg["models"][CHALLENGER]["replaces"]
champ = load_model(champ_version)
print(f"champion: {champ_version}")
X_all_champ, _ = prepare_features(df, champ["categories"])
p_champ = pd.Series(predict(champ, X_all_champ), index=df.index)

# ---- C: retrain on recent data (category lists are relearned as well)
X_tr, cats_c = prepare_features(train_c)
X_all_c, _ = prepare_features(df, cats_c)
dtr = lgb.Dataset(X_tr, train_c[TARGET], categorical_feature=CATEGORICAL)
dva = lgb.Dataset(X_all_c.loc[h2_2025.index], h2_2025[TARGET], reference=dtr)
chal = lgb.train(LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                 callbacks=[lgb.early_stopping(50, verbose=False)])
p_chal = pd.Series(chal.predict(X_all_c, num_iteration=chal.best_iteration), index=df.index)
print(f"challenger best_iter={chal.best_iteration}")


def platt(scores: pd.Series, mask) -> LogisticRegression:
    return LogisticRegression().fit(logit(scores[mask].values).reshape(-1, 1), df.loc[mask, TARGET])


def apply(cal: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return cal.predict_proba(logit(scores).reshape(-1, 1))[:, 1]


def rolling_platt(scores: pd.Series) -> np.ndarray:
    out = pd.Series(np.nan, index=holdout.index)
    for m in sorted(month[holdout.index].unique()):
        idx = holdout.index[month[holdout.index] == m]
        out[idx] = apply(platt(scores, month.between(m - 3, m - 1)), scores[idx].values)
    return out.values


h = holdout.index
cands = {
    "A champion v1": p_champ[h].values,
    "B champion + static Platt": apply(platt(p_champ, h2_2025.index), p_champ[h].values),
    "C challenger": p_chal[h].values,
    "D challenger + rolling Platt": rolling_platt(p_chal),
}
y = holdout[TARGET].values
res = pd.DataFrame([{"candidate": k, "roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p),
                     "brier": brier_score_loss(y, p), "calib_gap_pp": 100 * (p.mean() - y.mean())}
                    for k, p in cands.items()]).round(4)
print("\n=== Holdout 2026-01 to 2026-09 ===")
print(res.to_string(index=False))
res.to_csv(TAB_DIR / "Champion-Challenger-Comparison.csv", index=False)

hm = month[h].astype(str).values
gaps = (pd.DataFrame({"month": hm, "y": y, **cands}).groupby("month")
        .apply(lambda g: pd.Series({k: 100 * (g[k].mean() - g.y.mean()) for k in cands}), include_groups=False))
print("\n=== Monthly calibration gap (pp): mean prediction - actual ===")
print(gaps.round(2).to_string())

# ---- figure: monthly calibration gap per candidate
colors = {"A champion v1": PAL["reference"], "B champion + static Platt": PAL["reference_light"],
          "C challenger": PAL["model"]["logistic"], "D challenger + rolling Platt": PAL["model"]["lightgbm"]}
fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=PAL["surface"])
ax.axhspan(-GAP_LIMIT_PP, GAP_LIMIT_PP, color=PAL["grid"], alpha=0.6, lw=0)
ax.axhline(0, color=PAL["text_secondary"], lw=0.8)
x = np.arange(len(gaps))
for k in cands:
    ax.plot(x, gaps[k].values, "o-", ms=5, lw=1.8, color=colors[k], label=k)
# end labels, pushed apart so close values stay readable
last_vals = gaps.iloc[-1].sort_values()
label_y = np.maximum.accumulate(last_vals.values - 0.45 * np.arange(len(last_vals))) + 0.45 * np.arange(len(last_vals))
for k, yv in zip(last_vals.index, label_y):
    ax.text(x[-1] + 0.15, yv, f"{last_vals[k]:+.1f}", va="center", fontsize=9, color=PAL["text_primary"])
ax.set_xticks(x, gaps.index)
ax.set_xlim(-0.3, len(gaps) - 0.4)
ax.set_title("Only rolling recalibration keeps the model inside the ±3 pp alert band", pad=28)
ax.set(
       xlabel=None, ylabel="Calibration gap, pp (prediction - actual)")
ax.legend(fontsize=9, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
style_axes(ax)
fig.tight_layout(); fig.savefig(fig_path("Calibration-Gap-by-Month"), dpi=150); plt.close(fig)

best = res.sort_values("log_loss").iloc[0]
ok = abs(best.calib_gap_pp) < GAP_LIMIT_PP
print(f"\nLowest log loss: {best.candidate} | calibration gap {best.calib_gap_pp:+.2f} pp -> "
      f"{'promotion rule met' if ok else 'promotion rule NOT met'}")

if ok and best.candidate[0] in "CD":
    last = month.max()   # 2026-09: refit on 06..08 so scoring 2026-09 stays out of sample
    calibrator = platt(p_chal, month.between(last - 3, last - 1)) if best.candidate[0] == "D" else None
    bundle = {"model": chal, "features": FEATURES, "categories": cats_c, "best_iteration": chal.best_iteration,
              "version": CHALLENGER, "calibrator": calibrator}
    card = {"trained_on": "2020-01..2025-06, early stopping on 2025-H2",
            "calibrated_on": f"{last - 3}..{last - 1}" if calibrator is not None else None,
            "holdout_2026": best.drop("candidate").to_dict(), "replaces": champ["version"]}
    register(bundle, card, activate=True)
    print(f"Promoted {CHALLENGER} (replaces {champ['version']}); registry updated.")
else:
    print(f"Keeping {champ['version']}.")
