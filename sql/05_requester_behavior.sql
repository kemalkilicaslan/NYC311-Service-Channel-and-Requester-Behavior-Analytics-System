-- =====================================================================
-- 05_requester_behavior.sql
-- Requester behavior: request history per address, channel loyalty and
-- channel switching.
--
-- Assumption: the dataset has no person ID. requester_key is a building (BBL)
-- or zip + address, so results describe buildings, not households.
--
-- The window functions (ROW_NUMBER, LAG, running mean) need one sort of
-- ~20M rows. They are computed once into an UNLOGGED table and every analysis
-- below, plus the ML sample, reads from it.
--
-- Marts
--   mart.request_history (UNLOGGED, ~20.4M rows): history features per request
--   mart.ml_requests     (deterministic 10% sample): input to the Python model
-- =====================================================================

\timing on
SET work_mem = '1GB';
SET maintenance_work_mem = '2GB';

DROP TABLE IF EXISTS mart.request_history;
CREATE UNLOGGED TABLE mart.request_history AS
SELECT unique_key, created_at, requester_key, channel, is_digital,
       agency, complaint_type, complaint_family, location_type, borough,
       hr, dow, is_weekend, yr, mon,
       ROW_NUMBER() OVER w - 1                                                   AS prior_n,
       LAG(channel) OVER w                                                       AS prev_channel,
       ROUND((EXTRACT(EPOCH FROM created_at - LAG(created_at) OVER w) / 86400.0)::NUMERIC, 2) AS days_since_prev,
       -- "1 PRECEDING" excludes the current request, so the target never leaks into the feature
       AVG(is_digital) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_digital_rate,
       -- req_total looks into the future: used only to filter heavy requesters, never as a feature
       COUNT(*) OVER (PARTITION BY requester_key)                                AS req_total
FROM core.v_requests
WHERE is_digital IS NOT NULL AND created_at < '2026-10-01' AND requester_key IS NOT NULL
WINDOW w AS (PARTITION BY requester_key ORDER BY created_at, unique_key);

ANALYZE mart.request_history;

\echo '=== B1. Concentration: requesters, one-time share, Pareto shares ==='
WITH r AS (
    SELECT requester_key, MAX(req_total) AS n FROM mart.request_history GROUP BY 1
), t AS (
    SELECT n, NTILE(100) OVER (ORDER BY n DESC) AS pctile FROM r
)
SELECT COUNT(*)                                                      AS requesters,
       SUM(n)                                                        AS requests,
       ROUND(100.0 * COUNT(*) FILTER (WHERE n = 1) / COUNT(*), 1)    AS one_time_pct,
       ROUND(100.0 * SUM(n) FILTER (WHERE pctile = 1)   / SUM(n), 1) AS top1pct_share,
       ROUND(100.0 * SUM(n) FILTER (WHERE pctile <= 10) / SUM(n), 1) AS top10pct_share,
       COUNT(*) FILTER (WHERE n > 1000)                              AS heavy_requesters,
       ROUND(100.0 * SUM(n) FILTER (WHERE n > 1000) / SUM(n), 1)     AS heavy_share_pct
FROM t;

\echo '=== B2. Channel transition matrix: previous channel -> next channel (row %) ==='
\echo '    (heavy requesters with >1000 requests excluded, gap <= 365 days)'
WITH tr AS (
    SELECT prev_channel, channel, COUNT(*) AS n
    FROM mart.request_history
    WHERE prev_channel IS NOT NULL AND req_total <= 1000 AND days_since_prev <= 365
    GROUP BY 1, 2
)
SELECT prev_channel, SUM(n) AS n_from,
       ROUND(100.0 * SUM(n) FILTER (WHERE channel = 'PHONE')  / SUM(n), 1) AS to_phone,
       ROUND(100.0 * SUM(n) FILTER (WHERE channel = 'ONLINE') / SUM(n), 1) AS to_online,
       ROUND(100.0 * SUM(n) FILTER (WHERE channel = 'MOBILE') / SUM(n), 1) AS to_mobile
FROM tr GROUP BY prev_channel ORDER BY prev_channel;

\echo '=== B3. Digital share of a request given the address history ==='
SELECT CASE WHEN prior_n = 0              THEN '0 first request'
            WHEN prior_digital_rate = 0   THEN '1 always phone'
            WHEN prior_digital_rate < 0.5 THEN '2 mostly phone'
            WHEN prior_digital_rate < 1   THEN '3 mostly digital'
            ELSE                               '4 always digital' END AS history_segment,
       COUNT(*)                                                       AS n,
       ROUND(100.0 * AVG(is_digital), 1)                              AS digital_pct
FROM mart.request_history
WHERE req_total <= 1000
GROUP BY 1 ORDER BY 1;

\echo '=== B4. Cohort: addresses whose first request came in 2020, by first channel ==='
WITH first_req AS (
    SELECT requester_key, channel AS first_channel
    FROM mart.request_history
    WHERE prior_n = 0 AND yr = 2020 AND req_total <= 1000
)
SELECT f.first_channel, h.yr,
       COUNT(DISTINCT h.requester_key)     AS active_requesters,
       COUNT(*)                            AS requests,
       ROUND(100.0 * AVG(h.is_digital), 1) AS digital_pct
FROM first_req f JOIN mart.request_history h USING (requester_key)
WHERE h.yr IN (2020, 2022, 2024, 2025)
GROUP BY 1, 2 ORDER BY 1, 2;

-- ML sample: unique_key % 10 = 0 gives a reproducible 10% sample.
-- Target: is_digital (1 = ONLINE/MOBILE, 0 = PHONE)
DROP TABLE IF EXISTS mart.ml_requests;
CREATE TABLE mart.ml_requests AS
SELECT unique_key, created_at, is_digital, channel,
       agency, complaint_type, complaint_family, location_type, borough,
       hr, dow, is_weekend, yr, mon,
       prior_n, prev_channel, days_since_prev, prior_digital_rate
FROM mart.request_history
WHERE unique_key % 10 = 0;

\echo '=== mart.ml_requests by year ==='
SELECT yr, COUNT(*) AS n, ROUND(100.0 * AVG(is_digital), 1) AS digital_pct
FROM mart.ml_requests GROUP BY yr ORDER BY yr;
