"""03_requester_history.py
Spark SQL port of sql/05_requester_behavior.sql: per-requester history built
with window functions over ~20M rows, then compared with PostgreSQL.

  ROW_NUMBER() - 1                         prior requests from the same address
  LAG(channel), LAG(created_at)            previous channel and days since it
  AVG(is_digital) ... 1 PRECEDING          digital share of earlier requests (no target leakage)
  COUNT(*) OVER (PARTITION BY requester)   heavy requester filter (never a model feature)

Checks
  * B1 concentration, B2 transition matrix, B3 history segments (aggregates)
  * the deterministic 10% ML sample, compared value by value on every row
"""

from __future__ import annotations

from session import Timer, compare, ctas, get_spark, pg

spark = get_spark("nyc311-requester-history", driver_memory="10g")
spark.conf.set("spark.sql.shuffle.partitions", "200")   # one big sort: smaller tasks spill less

with Timer("mart_request_history (window functions over ~20M rows)"):
    ctas(spark, "nyc311.mart_request_history", """USING parquet PARTITIONED BY (yr) AS
        SELECT unique_key, created_at, requester_key, channel, is_digital, agency, complaint_type,
               complaint_family, location_type, borough, hr, dow, is_weekend, mon,
               ROW_NUMBER() OVER w - 1 AS prior_n,
               LAG(channel) OVER w AS prev_channel,
               CAST(ROUND(CAST(unix_timestamp(created_at) - unix_timestamp(LAG(created_at) OVER w) AS DECIMAL(20, 6))
                          / 86400, 2) AS DECIMAL(12, 2)) AS days_since_prev,
               AVG(is_digital) OVER (PARTITION BY requester_key ORDER BY created_at, unique_key
                                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_digital_rate,
               COUNT(*) OVER (PARTITION BY requester_key) AS req_total,
               yr
        FROM nyc311.service_requests
        WHERE is_digital IS NOT NULL AND created_at < TIMESTAMP '2026-10-01 00:00:00'
          AND requester_key IS NOT NULL
        WINDOW w AS (PARTITION BY requester_key ORDER BY created_at, unique_key)""")
print(f"  rows: {spark.table('nyc311.mart_request_history').count():,}")

print("\n=== Parity with PostgreSQL mart.request_history ===")
# B1: NTILE over requesters. Ties in n are broken by requester_key in both engines,
# otherwise the tile boundary would be arbitrary and the shares could differ slightly.
B1 = """
WITH r AS (SELECT requester_key, MAX(req_total) AS n FROM {t} GROUP BY 1),
     t AS (SELECT n, NTILE(100) OVER (ORDER BY n DESC, requester_key) AS pctile FROM r)
SELECT 'all' AS k, COUNT(*) AS requesters, SUM(n) AS requests,
       SUM(CASE WHEN n = 1 THEN 1 ELSE 0 END) AS one_time,
       SUM(CASE WHEN pctile = 1 THEN n ELSE 0 END) AS top1pct_requests,
       SUM(CASE WHEN n > 1000 THEN 1 ELSE 0 END) AS heavy_requesters,
       SUM(CASE WHEN n > 1000 THEN n ELSE 0 END) AS heavy_requests
FROM t"""
with Timer("B1 on Spark"):
    b1 = spark.sql(B1.format(t="nyc311.mart_request_history")).toPandas()
print(b1.drop(columns="k").to_string(index=False))
compare("B1 concentration (NTILE)", b1, pg(B1.format(t="mart.request_history")), keys=["k"])

B2 = """SELECT prev_channel, channel, COUNT(*) AS n FROM {t}
        WHERE prev_channel IS NOT NULL AND req_total <= 1000 AND days_since_prev <= 365 GROUP BY 1, 2"""
b2 = spark.sql(B2.format(t="nyc311.mart_request_history")).toPandas()
compare("B2 channel transition matrix (LAG)", b2, pg(B2.format(t="mart.request_history")),
        keys=["prev_channel", "channel"])
print((b2.pivot(index="prev_channel", columns="channel", values="n")
         .pipe(lambda d: d.div(d.sum(axis=1), axis=0) * 100).round(1).to_string()))

B3 = """SELECT CASE WHEN prior_n = 0 THEN '0 first request'
                    WHEN prior_digital_rate = 0 THEN '1 always phone'
                    WHEN prior_digital_rate < 0.5 THEN '2 mostly phone'
                    WHEN prior_digital_rate < 1 THEN '3 mostly digital'
                    ELSE '4 always digital' END AS history_segment,
               COUNT(*) AS n, SUM(is_digital) AS digital
        FROM {t} WHERE req_total <= 1000 GROUP BY 1"""
compare("B3 history segments (running AVG, 1 PRECEDING)",
        spark.sql(B3.format(t="nyc311.mart_request_history")).toPandas(),
        pg(B3.format(t="mart.request_history")), keys=["history_segment"])

print("\n=== Row-level check on the 10% ML sample (unique_key % 10 = 0) ===")
COLS = "unique_key, prior_n, prev_channel, days_since_prev, prior_digital_rate, req_total"
with Timer("Spark sample -> pandas"):
    s_ml = spark.sql(f"SELECT {COLS} FROM nyc311.mart_request_history WHERE unique_key % 10 = 0").toPandas()
with Timer("PostgreSQL sample -> pandas"):
    # MOD() instead of %: psycopg2 reads a bare % in the SQL text as a parameter placeholder
    p_ml = pg(f"SELECT {COLS} FROM mart.request_history WHERE MOD(unique_key, 10) = 0")
compare("ML sample, every row and column", s_ml, p_ml, keys=["unique_key"])

spark.stop()
