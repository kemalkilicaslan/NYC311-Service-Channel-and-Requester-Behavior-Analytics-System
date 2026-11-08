"""03_batch_scoring.py
Batch scoring: scores every request of one month (not a sample) with the active
model from the registry, writes the scores to PostgreSQL and reports monitoring
metrics.

    python 03_batch_scoring.py --month 2026-09 --reference-month 2025-09

Steps
  1. Read the month from mart.request_history (features are already built in SQL).
  2. common.prepare_features with the training category lists, then common.predict.
  3. Write to mart.channel_scores with COPY. Rows for the same month and model
     version are deleted first, so re-runs are idempotent.
  4. Monitor: AUC, calibration gap (mean prediction - actual), PSI of the score
     distribution against a reference month. PSI > 0.2 or |gap| > 3 pp raises an alert.
  5. Outreach list: the segments with the most expected phone requests.
"""

from __future__ import annotations

import argparse
import io
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ML_COLUMNS, TAB_DIR, engine, load_active, prepare_features, predict

PSI_LIMIT, GAP_LIMIT_PP = 0.2, 3.0


def score_month(month: str, bundle: dict, eng) -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-01")
    df = pd.read_sql(f"SELECT {ML_COLUMNS} FROM mart.request_history "
                     "WHERE created_at >= %(start)s AND created_at < %(end)s ORDER BY unique_key",
                     eng, params={"start": start, "end": start + pd.offsets.MonthBegin(1)})
    X, _ = prepare_features(df, bundle["categories"])
    df["p_digital"] = predict(bundle, X)
    return df


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-6, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def write_scores(df: pd.DataFrame, month: str, version: str, eng) -> None:
    out = df[["unique_key", "created_at", "p_digital"]].assign(
        p_digital=lambda d: d.p_digital.round(5), score_month=month, model_version=version)
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False)
    buf.seek(0)
    raw = eng.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS mart.channel_scores (
                               unique_key BIGINT, created_at TIMESTAMP, p_digital REAL,
                               score_month TEXT, model_version TEXT, scored_at TIMESTAMP DEFAULT now())""")
            cur.execute("DELETE FROM mart.channel_scores WHERE score_month = %s AND model_version = %s",
                        (month, version))
            cur.copy_expert("COPY mart.channel_scores (unique_key, created_at, p_digital, score_month, model_version) "
                            "FROM STDIN WITH (FORMAT csv)", buf)
        raw.commit()
    finally:
        raw.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch scoring with the active channel model")
    ap.add_argument("--month", default="2026-09", help="month to score (YYYY-MM)")
    ap.add_argument("--reference-month", default="2025-09", help="reference month for PSI (YYYY-MM)")
    args = ap.parse_args()

    t0 = time.time()
    eng = engine()
    bundle = load_active()
    version = bundle["version"]
    print(f"Active model: {version}  (calibrator: {'yes' if bundle.get('calibrator') is not None else 'no'})")

    cur = score_month(args.month, bundle, eng)
    ref = score_month(args.reference_month, bundle, eng)
    write_scores(cur, args.month, version, eng)
    n_db = pd.read_sql("SELECT COUNT(*) AS n FROM mart.channel_scores WHERE score_month = %(m)s AND model_version = %(v)s",
                       eng, params={"m": args.month, "v": version})["n"].iloc[0]
    print(f"Scored: {len(cur):,} requests -> {n_db:,} rows in mart.channel_scores  ({time.time() - t0:.0f} s)")

    mon = pd.DataFrame([{"month": m, "n": len(d), "actual_digital_%": 100 * d.is_digital.mean(),
                         "mean_pred_%": 100 * d.p_digital.mean(), "roc_auc": roc_auc_score(d.is_digital, d.p_digital)}
                        for m, d in ((args.reference_month, ref), (args.month, cur))]).round(3)
    score_psi = psi(ref.p_digital.values, cur.p_digital.values)
    gap = 100 * (cur.p_digital.mean() - cur.is_digital.mean())
    print("\n=== Monitoring ===")
    print(mon.to_string(index=False))
    print(f"PSI of scores ({args.reference_month} -> {args.month}): {score_psi:.4f}")
    print(f"Calibration gap (prediction - actual): {gap:+.2f} pp")
    alerts = ([f"PSI > {PSI_LIMIT}: score distribution shifted"] if score_psi > PSI_LIMIT else []) + \
             ([f"|calibration gap| > {GAP_LIMIT_PP:.0f} pp: recalibrate or retrain"] if abs(gap) > GAP_LIMIT_PP else [])
    print("ALERT: " + " | ".join(alerts) if alerts else "Status: OK (no threshold crossed)")
    mon.assign(psi=score_psi, calib_gap_pp=gap, model_version=version).to_csv(
        TAB_DIR / f"Scoring-Monitor-{args.month}-{version}.csv", index=False)

    seg = (cur.assign(p_phone=1 - cur.p_digital)
              .groupby(["complaint_type", "borough"], dropna=False)
              .agg(n=("p_phone", "size"), mean_p_phone=("p_phone", "mean"),
                   actual_phone_rate=("is_digital", lambda s: 1 - s.mean()))
              .query("n >= 300")
              .assign(expected_phone_requests=lambda d: d.n * d.mean_p_phone)
              .sort_values("expected_phone_requests", ascending=False)
              .head(12).round(3).reset_index())
    print(f"\n=== Outreach targets ({args.month}): 12 segments with the most expected phone requests ===")
    print(seg.to_string(index=False))
    seg.to_csv(TAB_DIR / f"Outreach-Targets-{args.month}-{version}.csv", index=False)


if __name__ == "__main__":
    main()
