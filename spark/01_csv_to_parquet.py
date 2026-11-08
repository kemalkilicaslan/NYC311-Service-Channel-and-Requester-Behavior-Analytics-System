"""01_csv_to_parquet.py
CSV -> Hive table stored as Parquet and partitioned by year, with the same
cleaning rules as sql/01_ingest.sql and the derived fields of core.v_requests
(sql/02_data_quality.sql).

In a columnar format the derived fields cost almost nothing to store, so they
are materialized here instead of living in a view.

Checks against PostgreSQL
  * rows per year
  * cleaning results: valid durations, total resolution hours, NULL counts
  * derived fields: channel x complaint family counts
Partition pruning is shown with EXPLAIN on a one-year filter.

Output: Hive table nyc311.service_requests (data/spark-warehouse)
"""

from __future__ import annotations

from session import CSV_PATH, Timer, compare, get_spark, pg

spark = get_spark("nyc311-csv-to-parquet")

COLUMNS = """unique_key created_date closed_date agency agency_name complaint_type descriptor descriptor_2
location_type incident_zip incident_address street_name cross_street_1 cross_street_2 intersection_street_1
intersection_street_2 address_type city landmark facility_type status due_date resolution_description
resolution_action_updated_date community_board council_district police_precinct bbl borough
x_coordinate_state_plane y_coordinate_state_plane open_data_channel_type park_facility_name park_borough
vehicle_type taxi_company_borough taxi_pick_up_location bridge_highway_name bridge_highway_direction road_ramp
bridge_highway_segment latitude longitude location""".split()

# Every column read as STRING, like the raw layer in PostgreSQL. escape='"' handles
# the RFC 4180 doubled quotes inside quoted fields.
schema = ", ".join(f"{c} STRING" for c in COLUMNS)
raw = spark.read.csv(str(CSV_PATH), schema=schema, header=True, escape='"', mode="FAILFAST")
raw.createOrReplaceTempView("raw_requests")

TS = "yyyy-MM-dd'T'HH:mm:ss.SSS"
core = spark.sql(f"""
WITH typed AS (
    SELECT CAST(unique_key AS BIGINT)                                AS unique_key,
           to_timestamp(created_date, "{TS}")                         AS created_at,
           to_timestamp(NULLIF(closed_date, ''), "{TS}")              AS closed_at,
           agency, complaint_type, descriptor, location_type,
           CASE WHEN incident_zip RLIKE '^[0-9]{{5}}$' AND incident_zip <> '00000' THEN incident_zip END AS incident_zip,
           incident_address,
           NULLIF(borough, 'Unspecified')                             AS borough,
           CASE WHEN council_district RLIKE '^[0-9]+$' THEN CAST(council_district AS SMALLINT) END AS council_district,
           bbl, status,
           open_data_channel_type                                     AS channel,
           CAST(latitude AS DOUBLE)                                   AS lat,
           CAST(longitude AS DOUBLE)                                  AS lon
    FROM raw_requests
), cleaned AS (
    SELECT *,
           COALESCE(closed_at >= created_at AND closed_at <= current_timestamp(), FALSE) AS is_closed_valid,
           (lat BETWEEN 40.45 AND 40.95 AND lon BETWEEN -74.30 AND -73.65)              AS in_nyc
    FROM typed
)
SELECT unique_key, created_at, closed_at, agency, complaint_type, descriptor, location_type,
       incident_zip, incident_address, borough, council_district, bbl, status, channel,
       CASE WHEN in_nyc THEN lat END AS latitude,
       CASE WHEN in_nyc THEN lon END AS longitude,
       -- decimal arithmetic, so rounding matches PostgreSQL NUMERIC(12,2) exactly
       CASE WHEN is_closed_valid THEN
            CAST(ROUND(CAST(unix_timestamp(closed_at) - unix_timestamp(created_at) AS DECIMAL(20, 6)) / 3600, 2)
                 AS DECIMAL(12, 2)) END                              AS resolution_hours,
       is_closed_valid,
       month(created_at)                                             AS mon,
       hour(created_at)                                              AS hr,
       extract(DAYOFWEEK_ISO FROM created_at)                        AS dow,
       extract(DAYOFWEEK_ISO FROM created_at) IN (6, 7)              AS is_weekend,
       CASE WHEN channel IN ('ONLINE', 'MOBILE') THEN 1 WHEN channel = 'PHONE' THEN 0 END AS is_digital,
       CASE WHEN lower(complaint_type) LIKE 'noise%'                 THEN 'Noise'
            WHEN complaint_type IN ('Illegal Parking', 'Blocked Driveway',
                                    'Derelict Vehicles', 'Abandoned Vehicle') THEN 'Parking & Vehicles'
            WHEN complaint_type = 'HEAT/HOT WATER' THEN 'Heat/Hot Water'
            WHEN agency = 'HPD'   THEN 'Housing Conditions'
            WHEN agency = 'DSNY'  THEN 'Sanitation'
            WHEN agency = 'DOT'   THEN 'Streets & Traffic'
            WHEN agency = 'DEP'   THEN 'Water & Environment'
            WHEN agency = 'DPR'   THEN 'Parks & Trees'
            WHEN agency = 'DOB'   THEN 'Buildings'
            WHEN agency = 'DOHMH' THEN 'Health'
            WHEN agency = 'DHS'   THEN 'Homeless Services'
            WHEN agency = 'NYPD'  THEN 'Police - Other'
            ELSE 'Other' END                                         AS complaint_family,
       -- ASCII-only upper case: PostgreSQL's C collation leaves non-ASCII letters alone,
       -- while Spark's upper() would turn mis-encoded 'â' into 'Â' and split requester keys
       COALESCE(NULLIF(bbl, '0000000000'),
                incident_zip || '|' || translate(incident_address, 'abcdefghijklmnopqrstuvwxyz',
                                                 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')) AS requester_key,
       year(created_at)                                              AS yr
FROM cleaned
""")

spark.sql("CREATE DATABASE IF NOT EXISTS nyc311")
print("\nWriting nyc311.service_requests (Parquet, PARTITIONED BY yr)...")
with Timer("CSV -> Parquet"):
    (core.write.mode("overwrite").format("parquet").partitionBy("yr")
         .option("compression", "snappy").saveAsTable("nyc311.service_requests"))

t = spark.table("nyc311.service_requests")
print("\n=== Hive table ===")
spark.sql("SHOW PARTITIONS nyc311.service_requests").show(truncate=False)
files = spark.sql("SELECT input_file_name() AS f FROM nyc311.service_requests").distinct().count()
size_gb = sum(p.stat().st_size for p in (CSV_PATH.parents[1] / "spark-warehouse").rglob("*.parquet")) / 1e9
print(f"Parquet files: {files:,} | size on disk: {size_gb:.2f} GB (CSV: {CSV_PATH.stat().st_size / 1e9:.2f} GB)")

print("\n=== Parity with PostgreSQL ===")
with Timer("Spark rows per year"):
    s_year = spark.sql("SELECT yr, COUNT(*) AS n FROM nyc311.service_requests GROUP BY yr").toPandas()
compare("rows per year", s_year,
        pg("SELECT EXTRACT(YEAR FROM created_at)::int AS yr, COUNT(*) AS n FROM core.service_requests GROUP BY 1"),
        keys=["yr"])

checks = """COUNT(*) AS n,
            SUM(CASE WHEN is_closed_valid THEN 1 ELSE 0 END)  AS valid_duration,
            SUM(resolution_hours)                              AS total_resolution_hours,
            SUM(CASE WHEN closed_at IS NULL THEN 1 ELSE 0 END) AS closed_null,
            SUM(CASE WHEN borough IS NULL THEN 1 ELSE 0 END)   AS borough_null,
            SUM(CASE WHEN incident_zip IS NULL THEN 1 ELSE 0 END) AS zip_null,
            SUM(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END)  AS lat_null,
            SUM(CASE WHEN council_district IS NULL THEN 1 ELSE 0 END) AS council_null"""
s_chk = spark.sql(f"SELECT 'all' AS k, {checks} FROM nyc311.service_requests").toPandas()
p_chk = pg(f"SELECT 'all' AS k, {checks} FROM core.service_requests")
print(s_chk.drop(columns="k").T.rename(columns={0: "spark"}).join(p_chk.drop(columns="k").T.rename(columns={0: "postgres"})).to_string())
compare("cleaning rules", s_chk, p_chk, keys=["k"])

s_fam = spark.sql("""SELECT channel, complaint_family, COUNT(*) AS n,
                            COUNT(DISTINCT requester_key) AS requesters
                     FROM nyc311.service_requests GROUP BY 1, 2""").toPandas()
p_fam = pg("""SELECT channel, complaint_family, COUNT(*) AS n, COUNT(DISTINCT requester_key) AS requesters
              FROM core.v_requests GROUP BY 1, 2""")
compare("derived fields (channel x family, distinct requesters)", s_fam, p_fam, keys=["channel", "complaint_family"])

print("\n=== Partition pruning: a 2025 filter reads only yr=2025 ===")
plan = spark.sql("SELECT COUNT(*) FROM nyc311.service_requests WHERE yr = 2025")._jdf.queryExecution().executedPlan().toString()
print([line.strip() for line in plan.splitlines() if "PartitionFilters" in line or "FileScan" in line][0][:400])
with Timer("COUNT(*) WHERE yr = 2025"):
    print(f"  rows: {t.where('yr = 2025').count():,}")

spark.stop()
