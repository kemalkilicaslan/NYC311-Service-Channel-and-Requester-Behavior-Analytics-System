#!/usr/bin/env bash
# Usage: ./run_sql.sh sql/02_data_quality.sql
# Runs a script with psql and tees the output to outputs/logs/sql_<script>.log.
# The password comes from ~/.pgpass, so it never appears in code.
set -euo pipefail
export PATH="/Library/PostgreSQL/18/bin:$PATH"
script="$1"
db="${PGDATABASE:-nyc311}"
[[ "$(basename "$script")" == 01_* ]] && db="postgres"   # 01 creates the database itself
mkdir -p outputs/logs
log="outputs/logs/sql_$(basename "${script%.sql}").log"
psql -h "${PGHOST:-localhost}" -U "${PGUSER:-postgres}" -d "$db" -w \
     -v ON_ERROR_STOP=1 -P pager=off -f "$script" 2>&1 | tee "$log"
