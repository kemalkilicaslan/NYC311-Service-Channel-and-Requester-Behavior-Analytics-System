"""02_channel_mix.py
Spark SQL port of sql/03_channel_mix.sql: the three channel marts and the
window-function queries, each compared row by row with PostgreSQL.

  mart_daily_channel_volume  day x channel
  mart_monthly_channel       month x channel, share (window SUM) and 12-month rolling mean (ROWS frame)
  mart_segment_channel       year x borough x family x hour x weekend x channel
  Q1 yearly share + LAG,  Q4 top 3 complaint types per channel (ROW_NUMBER)

Shares are computed in DECIMAL so rounding matches PostgreSQL NUMERIC exactly.
The marts are saved as Hive tables in the nyc311 database.
"""

from __future__ import annotations

from session import Timer, compare, ctas, get_spark, pg

spark = get_spark("nyc311-channel-mix")
KNOWN = "channel IN ('ONLINE', 'MOBILE', 'PHONE') AND created_at < TIMESTAMP '2026-10-01 00:00:00'"

with Timer("mart_daily_channel_volume"):
    ctas(spark, "nyc311.mart_daily_channel_volume", f"""USING parquet AS
        SELECT to_date(created_at) AS day, channel, COUNT(*) AS n
        FROM nyc311.service_requests WHERE {KNOWN}
        GROUP BY 1, 2""")

with Timer("mart_monthly_channel"):
    ctas(spark, "nyc311.mart_monthly_channel", """USING parquet AS
        WITH m AS (
            SELECT CAST(date_trunc('MONTH', day) AS DATE) AS month_start, channel, SUM(n) AS n
            FROM nyc311.mart_daily_channel_volume GROUP BY 1, 2
        )
        SELECT month_start, channel, n,
               ROUND(CAST(n AS DECIMAL(20, 6)) * 100 / SUM(n) OVER (PARTITION BY month_start), 2) AS share_pct,
               ROUND(AVG(n) OVER (PARTITION BY channel ORDER BY month_start
                                  ROWS BETWEEN 11 PRECEDING AND CURRENT ROW))                  AS n_roll12
        FROM m""")

with Timer("mart_segment_channel"):
    ctas(spark, "nyc311.mart_segment_channel", f"""USING parquet AS
        SELECT yr, borough, complaint_family, hr, is_weekend, channel, is_digital, COUNT(*) AS n
        FROM nyc311.service_requests
        WHERE {KNOWN} AND borough IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5, 6, 7""")

print("\n=== Parity with PostgreSQL marts ===")
compare("mart_daily_channel_volume",
        spark.table("nyc311.mart_daily_channel_volume").toPandas(),
        pg("SELECT day, channel, n FROM mart.daily_channel_volume"), keys=["day", "channel"])
compare("mart_monthly_channel (window SUM share + ROWS 11 PRECEDING rolling mean)",
        spark.table("nyc311.mart_monthly_channel").toPandas(),
        pg("SELECT month_start, channel, n, share_pct, n_roll12 FROM mart.monthly_channel"),
        keys=["month_start", "channel"])
compare("mart_segment_channel",
        spark.table("nyc311.mart_segment_channel").toPandas(),
        pg("SELECT yr, borough, complaint_family, hr, is_weekend, channel, is_digital, n FROM mart.segment_channel"),
        keys=["yr", "borough", "complaint_family", "hr", "is_weekend", "channel", "is_digital"])

# Q1: yearly shares and the year-over-year change of the phone share (LAG over an aggregate)
Q1_SPARK = """
WITH y AS (SELECT year(day) AS yr, channel, SUM(n) AS n FROM nyc311.mart_daily_channel_volume GROUP BY 1, 2),
     s AS (SELECT yr, channel, n, CAST(n AS DECIMAL(20, 6)) * 100 / SUM(n) OVER (PARTITION BY yr) AS share FROM y),
     p AS (SELECT yr,
                  MAX(CASE WHEN channel = 'PHONE'  THEN share END) AS phone,
                  MAX(CASE WHEN channel = 'ONLINE' THEN share END) AS online,
                  MAX(CASE WHEN channel = 'MOBILE' THEN share END) AS mobile,
                  SUM(n) AS total
           FROM s GROUP BY yr)
SELECT yr, ROUND(phone, 1) AS phone_pct, ROUND(online, 1) AS online_pct, ROUND(mobile, 1) AS mobile_pct,
       ROUND(phone - LAG(phone) OVER (ORDER BY yr), 1) AS phone_pp_change, total
FROM p ORDER BY yr"""
Q1_PG = """
WITH y AS (SELECT EXTRACT(YEAR FROM day)::INT AS yr, channel, SUM(n) AS n FROM mart.daily_channel_volume GROUP BY 1, 2),
     s AS (SELECT yr, channel, n, 100.0 * n / SUM(n) OVER (PARTITION BY yr) AS share FROM y)
SELECT yr,
       ROUND(MAX(share) FILTER (WHERE channel = 'PHONE'), 1)  AS phone_pct,
       ROUND(MAX(share) FILTER (WHERE channel = 'ONLINE'), 1) AS online_pct,
       ROUND(MAX(share) FILTER (WHERE channel = 'MOBILE'), 1) AS mobile_pct,
       ROUND(MAX(share) FILTER (WHERE channel = 'PHONE')
             - LAG(MAX(share) FILTER (WHERE channel = 'PHONE')) OVER (ORDER BY yr), 1) AS phone_pp_change,
       SUM(n) AS total
FROM s GROUP BY yr"""
q1 = spark.sql(Q1_SPARK).toPandas()
print("\n=== Q1. Yearly channel share (Spark SQL) ===")
print(q1.to_string(index=False))
compare("Q1 yearly share + LAG", q1, pg(Q1_PG), keys=["yr"])

# Q4: top 3 complaint types per channel in 2025 (ROW_NUMBER); ties broken by name in both engines
Q4 = """
WITH c AS (SELECT channel, complaint_type, COUNT(*) AS n FROM {src}
           WHERE created_at >= {d1} AND created_at < {d2} AND channel IN ('ONLINE', 'MOBILE', 'PHONE')
           GROUP BY 1, 2),
     r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY channel ORDER BY n DESC, complaint_type) AS rn FROM c)
SELECT channel, rn, complaint_type, n FROM r WHERE rn <= 3"""
with Timer("Q4 on Spark (one partition read thanks to pruning)"):
    q4 = spark.sql(Q4.format(src="nyc311.service_requests", d1="TIMESTAMP '2025-01-01 00:00:00'",
                             d2="TIMESTAMP '2026-01-01 00:00:00'")).toPandas()
compare("Q4 top 3 per channel (ROW_NUMBER)", q4,
        pg(Q4.format(src="core.service_requests", d1="'2025-01-01'", d2="'2026-01-01'")), keys=["channel", "rn"])

print("\n=== Hive tables in nyc311 ===")
spark.sql("SHOW TABLES IN nyc311").show(truncate=False)
spark.stop()
