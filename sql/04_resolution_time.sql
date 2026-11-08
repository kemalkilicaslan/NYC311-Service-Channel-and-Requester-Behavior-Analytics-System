-- =====================================================================
-- 04_resolution_time.sql
-- Does the channel a request arrives through change how fast it gets closed?
--
-- Raw medians make mobile look far faster than phone. Channels carry very
-- different complaint mixes, though, so the comparison is repeated inside each
-- complaint type (stratification) and then re-weighted with one common set of
-- type weights (direct standardization).
--
-- Universe: 2025, known channel, valid duration.
-- Mart: mart.heat_survival, HEAT/HOT WATER requests 2024-2025 for the R survival
--       analysis. Requests that are still open are kept as censored (event = 0).
-- =====================================================================

\timing on
SET work_mem = '512MB';

\echo '=== R1. Crude comparison: resolution time by channel (2025) ==='
SELECT channel, COUNT(*) AS n,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_hours)::NUMERIC, 1) AS median_h,
       ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY resolution_hours)::NUMERIC, 1) AS p90_h,
       ROUND(100.0 * COUNT(*) FILTER (WHERE resolution_hours <= 24)  / COUNT(*), 1)    AS within_24h_pct,
       ROUND(100.0 * COUNT(*) FILTER (WHERE resolution_hours <= 168) / COUNT(*), 1)    AS within_7d_pct
FROM core.service_requests
WHERE created_at >= '2025-01-01' AND created_at < '2026-01-01'
  AND channel IN ('ONLINE', 'MOBILE', 'PHONE') AND is_closed_valid
GROUP BY channel ORDER BY median_h;

\echo '=== R2. Within the 8 largest complaint types: median hours by channel ==='
WITH c AS (
    SELECT complaint_type, channel, COUNT(*) AS n,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_hours) AS med
    FROM core.service_requests
    WHERE created_at >= '2025-01-01' AND created_at < '2026-01-01'
      AND channel IN ('ONLINE', 'MOBILE', 'PHONE') AND is_closed_valid
    GROUP BY 1, 2
), t AS (
    SELECT complaint_type, SUM(n) AS n_type, DENSE_RANK() OVER (ORDER BY SUM(n) DESC) AS rk
    FROM c GROUP BY 1
)
SELECT t.complaint_type, t.n_type,
       ROUND(MAX(med) FILTER (WHERE channel = 'MOBILE')::NUMERIC, 1) AS mobile_med,
       ROUND(MAX(med) FILTER (WHERE channel = 'ONLINE')::NUMERIC, 1) AS online_med,
       ROUND(MAX(med) FILTER (WHERE channel = 'PHONE')::NUMERIC, 1)  AS phone_med
FROM c JOIN t USING (complaint_type)
WHERE t.rk <= 8
GROUP BY t.complaint_type, t.n_type
ORDER BY t.n_type DESC;

\echo '=== R3. Direct standardization: share closed within 7 days with the complaint mix held fixed ==='
-- Every channel gets the SAME weights (total volume of each type), so any gap
-- that remains belongs to the channel, not to its complaint mix.
WITH c AS (
    SELECT complaint_type, channel, COUNT(*) AS n, AVG((resolution_hours <= 168)::INT) AS p7
    FROM core.service_requests
    WHERE created_at >= '2025-01-01' AND created_at < '2026-01-01'
      AND channel IN ('ONLINE', 'MOBILE', 'PHONE') AND is_closed_valid
    GROUP BY 1, 2
), common AS (               -- types with at least 500 requests in all three channels
    SELECT complaint_type, SUM(n) AS w FROM c GROUP BY 1 HAVING COUNT(*) = 3 AND MIN(n) >= 500
)
SELECT c.channel, COUNT(*) AS n_types,
       ROUND(100 * SUM(c.p7 * c.n) / SUM(c.n), 1) AS crude_within_7d_pct,
       ROUND(100 * SUM(c.p7 * m.w) / SUM(m.w), 1) AS standardized_within_7d_pct
FROM c JOIN common m USING (complaint_type)
GROUP BY c.channel ORDER BY c.channel;

DROP TABLE IF EXISTS mart.heat_survival;
CREATE TABLE mart.heat_survival AS
WITH snap AS (SELECT MAX(created_at) AS cutoff FROM core.service_requests)
SELECT r.unique_key, r.created_at, r.channel, r.borough,
       EXTRACT(HOUR FROM r.created_at)::INT        AS hr,
       EXTRACT(ISODOW FROM r.created_at) IN (6, 7) AS is_weekend,
       EXTRACT(YEAR FROM r.created_at)::INT        AS yr,
       CASE WHEN r.is_closed_valid THEN r.resolution_hours
            ELSE ROUND(EXTRACT(EPOCH FROM (s.cutoff - r.created_at)) / 3600.0, 2) END AS time_h,
       r.is_closed_valid::INT                      AS event
FROM core.service_requests r CROSS JOIN snap s
WHERE r.complaint_type = 'HEAT/HOT WATER'
  AND r.created_at >= '2024-01-01' AND r.created_at < '2026-01-01'
  AND r.channel IN ('ONLINE', 'MOBILE', 'PHONE')
  AND r.borough IS NOT NULL
  AND (r.is_closed_valid OR r.closed_at IS NULL);   -- negative durations dropped

\echo '=== mart.heat_survival summary ==='
SELECT channel, COUNT(*) AS n, SUM(event) AS events, COUNT(*) - SUM(event) AS censored,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY time_h)::NUMERIC, 1) AS median_time_h
FROM mart.heat_survival GROUP BY channel ORDER BY channel;
