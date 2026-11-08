#!/usr/bin/env bash
# Usage: ./run_all.sh            run all 17 steps on the existing database
#        ./run_all.sh --fresh    drop the nyc311 database, the Spark warehouse and every
#                                output first, then rebuild everything from the CSV
#
# Each step writes its log to outputs/logs/ and its run time to
# outputs/tables/Pipeline-Run-Times.csv. The run stops at the first failing step.
# Python is taken from $PYTHON (default: python3) and must have requirements.txt installed.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/Library/PostgreSQL/18/bin:$PATH"
PY="${PYTHON:-python3}"
FIG_PREFIX="NYC311-Service-Channel-and-Requester-Behavior-Analytics-"

if [[ "${1:-}" == "--fresh" ]]; then
    echo "Fresh run: dropping database nyc311, Spark warehouse, metastore and outputs"
    psql -h "${PGHOST:-localhost}" -U "${PGUSER:-postgres}" -d postgres -w -v ON_ERROR_STOP=1 \
         -c "DROP DATABASE IF EXISTS nyc311 WITH (FORCE)"
    rm -rf data/metastore_db data/spark-warehouse data/derby.log \
           outputs/tables outputs/logs outputs/models outputs/figures "${FIG_PREFIX}"*.png
fi
mkdir -p outputs/logs outputs/tables
TIMES=outputs/tables/Pipeline-Run-Times.csv
echo "step,script,seconds" > "$TIMES"

n=0
step() {   # step <log name> <script> <command...>
    local log="outputs/logs/$1.log" script="$2"; shift 2
    local out="$log"
    [[ "$log" == outputs/logs/sql_* ]] && out=/dev/null   # run_sql.sh already tees to $log
    n=$((n + 1))
    printf '%2d  %-34s ' "$n" "$script"
    local t0=$SECONDS
    if ! "$@" > "$out" 2>&1; then
        echo "FAILED, see $log"; tail -20 "$log"; exit 1
    fi
    local dt=$((SECONDS - t0))
    echo "${dt} s"
    echo "$n,$script,$dt" >> "$TIMES"
}

for f in sql/0[1-5]_*.sql; do
    step "sql_$(basename "${f%.sql}")" "$f" ./run_sql.sh "$f"
done
for f in R/0[1-4]_*.R; do
    step "R_$(basename "${f%.R}")" "$f" Rscript "$f"
done
py() { (cd python && "$PY" -u "$@"); }
sp() { (cd spark && "$PY" -u "$@"); }
step python_01_channel_model    python/01_channel_model.py   py 01_channel_model.py
step python_02_phone_forecast   python/02_phone_forecast.py  py 02_phone_forecast.py
step python_03_batch_scoring_v1 python/03_batch_scoring.py   py 03_batch_scoring.py --month 2026-09 --reference-month 2025-09
step python_04_retrain          python/04_retrain.py         py 04_retrain.py
step python_03_batch_scoring_v2 python/03_batch_scoring.py   py 03_batch_scoring.py --month 2026-09 --reference-month 2025-09
step spark_01_csv_to_parquet    spark/01_csv_to_parquet.py   sp 01_csv_to_parquet.py
step spark_02_channel_mix       spark/02_channel_mix.py      sp 02_channel_mix.py
step spark_03_requester_history spark/03_requester_history.py sp 03_requester_history.py

echo "Done in $SECONDS s. Step times: $TIMES"
