-- =====================================================================
-- 03_channel_mix.sql
-- How channel mix changed over time and across segments (CTEs + window functions).
--
-- Universe (from 02_data_quality.sql)
--   * channel IN (ONLINE, MOBILE, PHONE): UNKNOWN/OTHER dropped (8.6%, 81% of it from DOB + DOT)
--   * created_at < 2026-10-01: October 2026 holds only 5 days
--
-- Mart tables (R and Python read these instead of 22.7M rows)
--   mart.daily_channel_volume : day x channel                       (~7K rows)
--   mart.monthly_channel      : month x channel + rolling mean      (~240 rows)
--   mart.segment_channel      : year x borough x family x hour x weekend x channel (~57K rows)
-- =====================================================================

\timing on
SET work_mem = '256MB';

DROP TABLE IF EXISTS mart.daily_channel_volume;
CREATE TABLE mart.daily_channel_volume AS
SELECT created_at::DATE AS day, channel, COUNT(*) AS n
FROM core.service_requests
WHERE channel IN ('ONLINE', 'MOBILE', 'PHONE') AND created_at < '2026-10-01'
GROUP BY 1, 2;

DROP TABLE IF EXISTS mart.monthly_channel;
CREATE TABLE mart.monthly_channel AS
WITH m AS (
    SELECT date_trunc('month', day)::DATE AS month_start, channel, SUM(n) AS n
    FROM mart.daily_channel_volume GROUP BY 1, 2
)
SELECT month_start, channel, n,
       ROUND(100.0 * n / SUM(n) OVER (PARTITION BY month_start), 2)  AS share_pct,
       ROUND(AVG(n) OVER (PARTITION BY channel ORDER BY month_start
                          ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)) AS n_roll12
FROM m;

DROP TABLE IF EXISTS mart.segment_channel;
CREATE TABLE mart.segment_channel AS
SELECT yr, borough, complaint_family, hr, is_weekend, channel, is_digital, COUNT(*) AS n
FROM core.v_requests
WHERE is_digital IS NOT NULL AND created_at < '2026-10-01' AND borough IS NOT NULL
GROUP BY 1, 2, 3, 4, 5, 6, 7;

ANALYZE mart.daily_channel_volume;
ANALYZE mart.monthly_channel;
ANALYZE mart.segment_channel;

\echo '=== Q1. Yearly channel share and year-over-year change in phone share (pp) ==='
WITH y AS (
    SELECT EXTRACT(YEAR FROM day)::INT AS yr, channel, SUM(n) AS n
    FROM mart.daily_channel_volume GROUP BY 1, 2
), s AS (
    SELECT yr, channel, n, 100.0 * n / SUM(n) OVER (PARTITION BY yr) AS share FROM y
)
SELECT yr,
       ROUND(MAX(share) FILTER (WHERE channel = 'PHONE'), 1)  AS phone_pct,
       ROUND(MAX(share) FILTER (WHERE channel = 'ONLINE'), 1) AS online_pct,
       ROUND(MAX(share) FILTER (WHERE channel = 'MOBILE'), 1) AS mobile_pct,
       ROUND(MAX(share) FILTER (WHERE channel = 'PHONE')
             - LAG(MAX(share) FILTER (WHERE channel = 'PHONE')) OVER (ORDER BY yr), 1) AS phone_pp_change,
       SUM(n) AS total
FROM s GROUP BY yr ORDER BY yr;

\echo '=== Q2. Digital share by borough (2025) and gap to the city average ==='
WITH b AS (
    SELECT borough, SUM(n) AS n,
           100.0 * SUM(n * is_digital) / SUM(n)                       AS digital_pct,
           100.0 * SUM(n) FILTER (WHERE channel = 'MOBILE') / SUM(n)  AS mobile_pct
    FROM mart.segment_channel WHERE yr = 2025 GROUP BY borough
)
SELECT borough, n, ROUND(digital_pct, 1) AS digital_pct, ROUND(mobile_pct, 1) AS mobile_pct,
       ROUND(digital_pct - SUM(digital_pct * n) OVER () / SUM(n) OVER (), 1) AS gap_to_city_pp
FROM b ORDER BY digital_pct DESC;

\echo '=== Q3. Phone dependence by complaint family: 2020 vs 2025 ==='
WITH f AS (
    SELECT complaint_family,
           100.0 * SUM(n) FILTER (WHERE yr = 2020 AND channel = 'PHONE') / NULLIF(SUM(n) FILTER (WHERE yr = 2020), 0) AS phone_2020,
           100.0 * SUM(n) FILTER (WHERE yr = 2025 AND channel = 'PHONE') / NULLIF(SUM(n) FILTER (WHERE yr = 2025), 0) AS phone_2025,
           SUM(n) FILTER (WHERE yr = 2025) AS n_2025
    FROM mart.segment_channel GROUP BY complaint_family
)
SELECT complaint_family, n_2025,
       ROUND(phone_2020, 1)              AS phone_pct_2020,
       ROUND(phone_2025, 1)              AS phone_pct_2025,
       ROUND(phone_2025 - phone_2020, 1) AS change_pp
FROM f
WHERE n_2025 IS NOT NULL          -- Buildings (DOB) has no known channel
ORDER BY phone_2025 DESC;

\echo '=== Q4. Top 3 complaint types per channel (2025, ROW_NUMBER) ==='
WITH c AS (
    SELECT channel, complaint_type, COUNT(*) AS n
    FROM core.service_requests
    WHERE created_at >= '2025-01-01' AND created_at < '2026-01-01'
      AND channel IN ('ONLINE', 'MOBILE', 'PHONE')
    GROUP BY 1, 2
), r AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY channel ORDER BY n DESC)  AS rn,
              ROUND(100.0 * n / SUM(n) OVER (PARTITION BY channel), 1)  AS pct_of_channel
    FROM c
)
SELECT channel, rn, complaint_type, n, pct_of_channel FROM r WHERE rn <= 3 ORDER BY channel, rn;

\echo '=== Q5. Time-of-day profile: share of each channel at night (22-06), office hours (09-17), weekends ==='
SELECT channel, SUM(n) AS n,
       ROUND(100.0 * SUM(n) FILTER (WHERE hr >= 22 OR hr < 6)  / SUM(n), 1) AS night_pct,
       ROUND(100.0 * SUM(n) FILTER (WHERE hr BETWEEN 9 AND 16) / SUM(n), 1) AS office_hours_pct,
       ROUND(100.0 * SUM(n) FILTER (WHERE is_weekend)          / SUM(n), 1) AS weekend_pct
FROM mart.segment_channel GROUP BY channel ORDER BY channel;

\echo '=== Q6. Which families drove phone volume up between 2022 and 2023? ==='
WITH p AS (
    SELECT complaint_family,
           SUM(n) FILTER (WHERE yr = 2022 AND channel = 'PHONE') AS phone_2022,
           SUM(n) FILTER (WHERE yr = 2023 AND channel = 'PHONE') AS phone_2023,
           SUM(n) FILTER (WHERE yr = 2022)                       AS all_2022,
           SUM(n) FILTER (WHERE yr = 2023)                       AS all_2023
    FROM mart.segment_channel WHERE yr IN (2022, 2023) GROUP BY complaint_family
)
SELECT complaint_family, phone_2022, phone_2023,
       phone_2023 - phone_2022                                                            AS phone_diff,
       ROUND(100.0 * (phone_2023 - phone_2022) / SUM(phone_2023 - phone_2022) OVER (), 1)  AS pct_of_total_diff,
       ROUND(100.0 * phone_2022 / all_2022, 1)                                            AS phone_pct_2022,
       ROUND(100.0 * phone_2023 / all_2023, 1)                                            AS phone_pct_2023
FROM p WHERE phone_2022 IS NOT NULL
ORDER BY phone_diff DESC LIMIT 5;

\echo '=== Mart sizes ==='
SELECT 'daily_channel_volume' AS mart, COUNT(*) AS n_rows FROM mart.daily_channel_volume
UNION ALL SELECT 'monthly_channel', COUNT(*) FROM mart.monthly_channel
UNION ALL SELECT 'segment_channel', COUNT(*) FROM mart.segment_channel;
