-- =====================================================================
-- 02_data_quality.sql
-- 1) core.v_requests: derived fields shared by every later script
-- 2) Data quality report: missingness, invalid durations, channel coverage,
--    partial periods, heavy requesters, category spelling variants, duplicates
--
-- Derived fields live in a view instead of new columns: computing them at
-- query time is cheaper than an ALTER + UPDATE over 22.7M rows, and the heavy
-- aggregates are materialized in the mart layer anyway.
-- =====================================================================

\timing on
SET work_mem = '256MB';

CREATE OR REPLACE VIEW core.v_requests AS
SELECT r.*,
       EXTRACT(YEAR   FROM created_at)::INT              AS yr,
       EXTRACT(MONTH  FROM created_at)::INT              AS mon,
       EXTRACT(HOUR   FROM created_at)::INT              AS hr,
       EXTRACT(ISODOW FROM created_at)::INT              AS dow,          -- 1 = Mon ... 7 = Sun
       EXTRACT(ISODOW FROM created_at) IN (6, 7)         AS is_weekend,
       CASE WHEN channel IN ('ONLINE', 'MOBILE') THEN 1
            WHEN channel = 'PHONE'               THEN 0 END AS is_digital, -- UNKNOWN/OTHER -> NULL
       CASE WHEN complaint_type ILIKE 'noise%'                       THEN 'Noise'
            WHEN complaint_type IN ('Illegal Parking', 'Blocked Driveway',
                                    'Derelict Vehicles', 'Abandoned Vehicle') THEN 'Parking & Vehicles'
            WHEN complaint_type = 'HEAT/HOT WATER'                   THEN 'Heat/Hot Water'
            WHEN agency = 'HPD'   THEN 'Housing Conditions'
            WHEN agency = 'DSNY'  THEN 'Sanitation'
            WHEN agency = 'DOT'   THEN 'Streets & Traffic'
            WHEN agency = 'DEP'   THEN 'Water & Environment'
            WHEN agency = 'DPR'   THEN 'Parks & Trees'
            WHEN agency = 'DOB'   THEN 'Buildings'
            WHEN agency = 'DOHMH' THEN 'Health'
            WHEN agency = 'DHS'   THEN 'Homeless Services'
            WHEN agency = 'NYPD'  THEN 'Police - Other'
            ELSE 'Other' END                             AS complaint_family,
       -- Requester proxy: the dataset has no person ID, so a building (BBL) or
       -- zip + address stands in for the requester.
       COALESCE(NULLIF(bbl, '0000000000'),
                incident_zip || '|' || upper(incident_address)) AS requester_key
FROM core.service_requests r;

\echo '=== 1. Missing values in key columns (%) ==='
SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE closed_at IS NULL)     / COUNT(*), 2) AS closed_at,
       ROUND(100.0 * COUNT(*) FILTER (WHERE borough IS NULL)       / COUNT(*), 2) AS borough,
       ROUND(100.0 * COUNT(*) FILTER (WHERE incident_zip IS NULL)  / COUNT(*), 2) AS incident_zip,
       ROUND(100.0 * COUNT(*) FILTER (WHERE latitude IS NULL)      / COUNT(*), 2) AS lat_lon,
       ROUND(100.0 * COUNT(*) FILTER (WHERE location_type IS NULL) / COUNT(*), 2) AS location_type,
       ROUND(100.0 * COUNT(*) FILTER (WHERE bbl IS NULL)           / COUNT(*), 2) AS bbl,
       ROUND(100.0 * COUNT(*) FILTER (WHERE channel IN ('UNKNOWN', 'OTHER')) / COUNT(*), 2) AS channel_unknown
FROM core.service_requests;

\echo '=== 2. Resolution time validity ==='
SELECT COUNT(*) FILTER (WHERE closed_at IS NULL)                       AS not_closed,
       COUNT(*) FILTER (WHERE closed_at < created_at)                  AS closed_before_created,
       COUNT(*) FILTER (WHERE closed_at > now())                       AS closed_in_future,
       COUNT(*) FILTER (WHERE resolution_hours = 0)                    AS zero_duration,
       COUNT(*) FILTER (WHERE resolution_hours > 24 * 365)             AS over_1_year,
       COUNT(*) FILTER (WHERE status = 'Closed' AND closed_at IS NULL) AS closed_status_no_date
FROM core.service_requests;

\echo '=== 3. Where do UNKNOWN channels come from? ==='
SELECT agency,
       COUNT(*)                                                                 AS n,
       COUNT(*) FILTER (WHERE channel = 'UNKNOWN')                              AS n_unknown,
       ROUND(100.0 * COUNT(*) FILTER (WHERE channel = 'UNKNOWN') / COUNT(*), 1) AS pct_of_agency,
       ROUND(100.0 * COUNT(*) FILTER (WHERE channel = 'UNKNOWN')
             / SUM(COUNT(*) FILTER (WHERE channel = 'UNKNOWN')) OVER (), 1)     AS pct_of_all_unknown
FROM core.service_requests
GROUP BY agency
HAVING COUNT(*) FILTER (WHERE channel = 'UNKNOWN') > 1000
ORDER BY n_unknown DESC;

\echo '=== 4. Partial period check: daily volume of the latest months ==='
SELECT date_trunc('month', created_at)::DATE                               AS month_start,
       COUNT(*)                                                            AS n,
       COUNT(DISTINCT created_at::DATE)                                    AS days_observed,
       ROUND(COUNT(*)::NUMERIC / COUNT(DISTINCT created_at::DATE))         AS per_day
FROM core.service_requests
WHERE created_at >= '2026-07-01'
GROUP BY 1 ORDER BY 1;

\echo '=== 5. Heaviest requesters (top 10 addresses) ==='
SELECT requester_key,
       MIN(borough)                                       AS borough,
       COUNT(*)                                           AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3) AS pct_of_all,
       MODE() WITHIN GROUP (ORDER BY complaint_type)      AS top_complaint,
       MODE() WITHIN GROUP (ORDER BY channel)             AS top_channel
FROM core.v_requests
WHERE requester_key IS NOT NULL
GROUP BY requester_key
ORDER BY n DESC
LIMIT 10;

\echo '=== 6. Category spelling variants in location_type ==='
SELECT lower(regexp_replace(location_type, '[^a-zA-Z]', '', 'g')) AS normalized,
       COUNT(DISTINCT location_type)                             AS n_variants,
       string_agg(DISTINCT location_type, ' | ')                 AS variants,
       COUNT(*)                                                  AS n
FROM core.service_requests
WHERE location_type IS NOT NULL
GROUP BY 1
HAVING COUNT(DISTINCT location_type) > 1
ORDER BY n DESC
LIMIT 5;

\echo '=== 7. Duplicate reports flagged in resolution_description (raw) ==='
SELECT agency,
       COUNT(*) FILTER (WHERE resolution_description ILIKE '%duplicate%')                               AS n_duplicate,
       ROUND(100.0 * COUNT(*) FILTER (WHERE resolution_description ILIKE '%duplicate%') / COUNT(*), 2)  AS pct
FROM raw.service_requests
GROUP BY agency
HAVING COUNT(*) FILTER (WHERE resolution_description ILIKE '%duplicate%') > 10000
ORDER BY n_duplicate DESC;

\echo '=== 8. Analysis universe ==='
SELECT COUNT(*)                                                                    AS all_rows,
       COUNT(*) FILTER (WHERE is_digital IS NOT NULL)                              AS known_channel,
       COUNT(*) FILTER (WHERE is_digital IS NOT NULL AND created_at < '2026-10-01') AS known_channel_full_months,
       COUNT(*) FILTER (WHERE is_closed_valid)                                     AS valid_duration
FROM core.v_requests;
