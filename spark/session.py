"""Shared Spark setup and PostgreSQL parity checks for the spark/ scripts.

Spark runs in local mode (local[*]) on one machine. The same code runs on a
cluster by changing only the master URL and the warehouse path.

The Hive metastore is an embedded Derby database under data/, and the
warehouse (Parquet files) lives in data/spark-warehouse. Both are rebuildable
from the CSV and are git-ignored with the rest of data/.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CSV_PATH = DATA / "csv" / "nyc311_service_requests.csv"
WAREHOUSE = DATA / "spark-warehouse"
os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")

from pyspark.sql import SparkSession  # noqa: E402  (JAVA_HOME must be set first)


def get_spark(app: str, driver_memory: str = "8g") -> SparkSession:
    spark = (SparkSession.builder
             .appName(app)
             .master("local[*]")
             .config("spark.driver.memory", driver_memory)
             .config("spark.sql.warehouse.dir", str(WAREHOUSE))
             .config("spark.hadoop.javax.jdo.option.ConnectionURL",
                     f"jdbc:derby:;databaseName={DATA / 'metastore_db'};create=true")
             .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={DATA}")
             # TIMESTAMP in PostgreSQL has no time zone; UTC here gives Spark the same
             # wall-clock arithmetic (no daylight-saving jumps in durations)
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.ui.showConsoleProgress", "false")
             .enableHiveSupport()
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark {spark.version} | {spark.sparkContext.defaultParallelism} cores | "
          f"catalog: {spark.conf.get('spark.sql.catalogImplementation')}")
    return spark


def ctas(spark: SparkSession, table: str, body: str) -> None:
    """Rebuild a Hive table from a query. The Hive session catalog has no
    CREATE OR REPLACE TABLE ... AS SELECT, so this is DROP + CREATE TABLE AS."""
    spark.sql(f"DROP TABLE IF EXISTS {table}")
    spark.sql(f"CREATE TABLE {table} {body}")


_engine = None


def pg(sql: str) -> pd.DataFrame:
    """Run a query against the PostgreSQL nyc311 database (password from ~/.pgpass)."""
    global _engine
    if _engine is None:
        _engine = create_engine(f"postgresql+psycopg2://{os.getenv('PGUSER', 'postgres')}@"
                                f"{os.getenv('PGHOST', 'localhost')}:5432/nyc311")
    return pd.read_sql(sql, _engine)


def compare(name: str, spark_df: pd.DataFrame, pg_df: pd.DataFrame, keys: list[str], tol: float = 1e-6) -> bool:
    """Full outer join of two result sets on keys; prints missing rows and value mismatches."""
    a = spark_df.copy()
    b = pg_df.copy()
    for d in (a, b):
        for k in keys:
            d[k] = d[k].astype(str)
    m = a.merge(b, on=keys, how="outer", suffixes=("_spark", "_pg"), indicator=True)
    only = m[m["_merge"] != "both"]
    bad = {}
    for col in [c for c in a.columns if c not in keys]:
        x = pd.to_numeric(m[f"{col}_spark"], errors="coerce")
        y = pd.to_numeric(m[f"{col}_pg"], errors="coerce")
        if x.notna().any() or y.notna().any():
            diff = ~(((x - y).abs() <= tol * y.abs().clip(lower=1)) | (x.isna() & y.isna()))
        else:
            diff = m[f"{col}_spark"].astype(str) != m[f"{col}_pg"].astype(str)
        diff &= m["_merge"] == "both"
        if diff.any():
            bad[col] = int(diff.sum())
    ok = only.empty and not bad
    print(f"[{'MATCH' if ok else 'DIFF '}] {name}: {len(a):,} Spark rows vs {len(b):,} PostgreSQL rows"
          + ("" if ok else f" | only in one side: {len(only):,} | mismatched values: {bad}"))
    return ok


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t
        print(f"  {self.label}: {self.seconds:.1f} s")
