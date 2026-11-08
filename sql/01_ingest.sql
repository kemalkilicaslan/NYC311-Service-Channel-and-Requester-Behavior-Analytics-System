-- =====================================================================
-- 01_ingest.sql
-- Creates the database, loads the 16 GB CSV into a raw layer and builds a
-- typed, cleaned core table partitioned by year. Safe to re-run: the load and
-- the transform are skipped when their target tables already hold data.
--
-- Layers
--   raw  : 1:1 copy of the CSV, every column TEXT
--   core : typed and cleaned, RANGE-partitioned on created_at (one partition per year)
--   mart : small analysis tables consumed by R and Python
--
-- Run from the project root:  ./run_sql.sh sql/01_ingest.sql
-- =====================================================================

\timing on

SELECT 'CREATE DATABASE nyc311 ENCODING ''UTF8'' TEMPLATE template0'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'nyc311')
\gexec

\connect nyc311
SET client_encoding = 'UTF8';
SET maintenance_work_mem = '2GB';
SET work_mem = '256MB';

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS mart;

-- ---------------------------------------------------------------------
-- raw: UNLOGGED skips the write-ahead log, which roughly halves load time.
-- The layer can always be rebuilt from the CSV, so durability is not needed.
-- ---------------------------------------------------------------------
CREATE UNLOGGED TABLE IF NOT EXISTS raw.service_requests (
    unique_key TEXT, created_date TEXT, closed_date TEXT, agency TEXT, agency_name TEXT,
    complaint_type TEXT, descriptor TEXT, descriptor_2 TEXT, location_type TEXT,
    incident_zip TEXT, incident_address TEXT, street_name TEXT, cross_street_1 TEXT,
    cross_street_2 TEXT, intersection_street_1 TEXT, intersection_street_2 TEXT,
    address_type TEXT, city TEXT, landmark TEXT, facility_type TEXT, status TEXT,
    due_date TEXT, resolution_description TEXT, resolution_action_updated_date TEXT,
    community_board TEXT, council_district TEXT, police_precinct TEXT, bbl TEXT,
    borough TEXT, x_coordinate_state_plane TEXT, y_coordinate_state_plane TEXT,
    open_data_channel_type TEXT, park_facility_name TEXT, park_borough TEXT,
    vehicle_type TEXT, taxi_company_borough TEXT, taxi_pick_up_location TEXT,
    bridge_highway_name TEXT, bridge_highway_direction TEXT, road_ramp TEXT,
    bridge_highway_segment TEXT, latitude TEXT, longitude TEXT, location TEXT
);

SELECT NOT EXISTS (SELECT 1 FROM raw.service_requests) AS need_load \gset
\if :need_load
    \echo 'raw.service_requests is empty: loading CSV...'
    \copy raw.service_requests FROM 'data/csv/nyc311_service_requests.csv' WITH (FORMAT csv, HEADER true)
    ANALYZE raw.service_requests;
\else
    \echo 'raw.service_requests already loaded: skipping CSV load.'
\endif

-- ---------------------------------------------------------------------
-- core: yearly partitions let date-filtered queries scan only the years they
-- touch (partition pruning), the same idea as PARTITIONED BY (year) in Hive.
--
-- Cleaning rules (measured in 02_data_quality.sql)
--   * closed before created, or closed in the future -> is_closed_valid = FALSE,
--     resolution_hours = NULL
--   * zip codes that are not 5 digits ('0', 'NA', 'N/A', ...) or are '00000' -> NULL
--   * borough 'Unspecified' -> NULL
--   * coordinates outside the NYC bounding box -> NULL
--   * duplicated or derivable columns (park_borough, location, state-plane x/y,
--     cross streets, the 97%+ empty bridge and taxi columns) are dropped
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.service_requests (
    unique_key            BIGINT    NOT NULL,
    created_at            TIMESTAMP NOT NULL,
    closed_at             TIMESTAMP,
    due_at                TIMESTAMP,
    resolution_updated_at TIMESTAMP,
    agency                TEXT,
    agency_name           TEXT,
    complaint_type        TEXT,
    descriptor            TEXT,
    location_type         TEXT,
    incident_zip          CHAR(5),
    incident_address      TEXT,
    city                  TEXT,
    borough               TEXT,
    community_board       TEXT,
    council_district      SMALLINT,
    police_precinct       TEXT,
    bbl                   TEXT,
    status                TEXT,
    channel               TEXT,
    latitude              DOUBLE PRECISION,
    longitude             DOUBLE PRECISION,
    resolution_hours      NUMERIC(12,2),
    is_closed_valid       BOOLEAN
) PARTITION BY RANGE (created_at);   -- primary key added after the bulk insert (below)

DO $$
BEGIN
    FOR y IN 2020..2027 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS core.service_requests_%s PARTITION OF core.service_requests
               FOR VALUES FROM (%L) TO (%L)', y, make_date(y, 1, 1), make_date(y + 1, 1, 1));
    END LOOP;
END $$;
CREATE TABLE IF NOT EXISTS core.service_requests_default PARTITION OF core.service_requests DEFAULT;

SELECT NOT EXISTS (SELECT 1 FROM core.service_requests) AS need_build \gset
\if :need_build
\echo 'core.service_requests is empty: transforming raw -> core...'
INSERT INTO core.service_requests
WITH typed AS (
    SELECT unique_key::BIGINT                                      AS unique_key,
           created_date::TIMESTAMP                                 AS created_at,
           NULLIF(closed_date, '')::TIMESTAMP                      AS closed_at,
           NULLIF(due_date, '')::TIMESTAMP                         AS due_at,
           NULLIF(resolution_action_updated_date, '')::TIMESTAMP   AS resolution_updated_at,
           agency, agency_name, complaint_type, descriptor, location_type,
           CASE WHEN incident_zip ~ '^[0-9]{5}$' AND incident_zip <> '00000' THEN incident_zip END AS incident_zip,
           incident_address, city,
           NULLIF(borough, 'Unspecified')                          AS borough,
           community_board,
           CASE WHEN council_district ~ '^[0-9]+$' THEN council_district::SMALLINT END AS council_district,
           police_precinct, bbl, status,
           open_data_channel_type                                  AS channel,
           latitude::FLOAT8  AS lat,
           longitude::FLOAT8 AS lon
    FROM raw.service_requests
), located AS (
    SELECT t.*, (lat BETWEEN 40.45 AND 40.95 AND lon BETWEEN -74.30 AND -73.65) AS in_nyc
    FROM typed t
)
SELECT unique_key, created_at, closed_at, due_at, resolution_updated_at, agency, agency_name,
       complaint_type, descriptor, location_type, incident_zip, incident_address, city, borough,
       community_board, council_district, police_precinct, bbl, status, channel,
       CASE WHEN in_nyc THEN lat END,
       CASE WHEN in_nyc THEN lon END,
       CASE WHEN ok THEN ROUND(EXTRACT(EPOCH FROM (closed_at - created_at)) / 3600.0, 2) END,
       ok
FROM located
CROSS JOIN LATERAL (SELECT COALESCE(closed_at >= created_at AND closed_at <= now(), FALSE) AS ok) v;

-- VACUUM right after the bulk insert writes the hint bits and the visibility map
-- once. Without it the first index build pays for those writes, and index-only
-- scans stay off until autovacuum runs. Keys and indexes are then built once over
-- the loaded data instead of being updated row by row during the insert, with
-- parallel workers building each partition's btree.
VACUUM (ANALYZE) core.service_requests;
SET max_parallel_maintenance_workers = 4;
ALTER TABLE core.service_requests ADD PRIMARY KEY (unique_key, created_at);
CREATE INDEX IF NOT EXISTS service_requests_channel_created_at_idx ON core.service_requests (channel, created_at);
CREATE INDEX IF NOT EXISTS service_requests_complaint_type_idx     ON core.service_requests (complaint_type);
CREATE INDEX IF NOT EXISTS service_requests_bbl_created_at_idx     ON core.service_requests (bbl, created_at);
\else
\echo 'core.service_requests already built: skipping transform.'
\endif

\echo '=== Row counts: raw vs core (must match) ==='
SELECT (SELECT COUNT(*) FROM raw.service_requests)  AS raw_rows,
       (SELECT COUNT(*) FROM core.service_requests) AS core_rows,
       pg_size_pretty(pg_total_relation_size('raw.service_requests')) AS raw_size;

\echo '=== Partitions ==='
SELECT c.relname AS partition, c.reltuples::BIGINT AS est_rows,
       pg_size_pretty(pg_total_relation_size(c.oid)) AS size
FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid
WHERE i.inhparent = 'core.service_requests'::regclass
ORDER BY 1;

\echo '=== Partition pruning: a 2025 filter must touch only the 2025 partition ==='
EXPLAIN (COSTS OFF)
SELECT COUNT(*) FROM core.service_requests
WHERE created_at >= '2025-01-01' AND created_at < '2026-01-01';
