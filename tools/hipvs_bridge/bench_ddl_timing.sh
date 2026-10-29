#!/usr/bin/env bash
# End-to-end timing of the graded DDL window on one seekdb instance:
#   CREATE VECTOR INDEX ... lib=cuvs  ->  first query returns
#
# Splits the wall clock into the stages the bridge can account for:
#   ddl_ms      whole statement as the client sees it
#   build_ms    cuVS CAGRA build (bridge trace, SEEKDB_CUVS_TRACE=1)
#   snapshot_ms cuVS snapshot write (bridge trace)
#   other_ms    ddl_ms - build_ms - snapshot_ms: SQL layer, task scheduling,
#               meta/DDL persistence, stream copy into the snapshot file
#   q1_ms       first query after DDL (cold: adapter rebuild, cuVS resources,
#               hashmap init)  vs  q2_ms steady state
#
# Usage:
#   ./bench_ddl_timing.sh --observer PATH --bridge PATH --base-dir DIR --port PORT
#                         [--table t --dim 128 --runs 5] [--data FILE] [--load]
#
#   --load   create the table and LOAD --data (raw float32, n x dim) first;
#            without it the table must already exist and be populated.
set -Eeuo pipefail

usage() {
  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

OBSERVER=""
BRIDGE=""
BASE_DIR=""
PORT=""
TABLE="t_bench"
DIM=128
RUNS=5
DATA=""
LOAD=0
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
IDX="idx_cagra"

while (($# > 0)); do
  case "$1" in
    --observer) OBSERVER=$2; shift 2 ;;
    --bridge) BRIDGE=$2; shift 2 ;;
    --base-dir) BASE_DIR=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --table) TABLE=$2; shift 2 ;;
    --dim) DIM=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --data) DATA=$2; shift 2 ;;
    --load) LOAD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "$OBSERVER" ]] || { echo "observer is not executable: $OBSERVER" >&2; exit 2; }
[[ -f "$BRIDGE" ]] || { echo "bridge does not exist: $BRIDGE" >&2; exit 2; }
[[ -n "$BASE_DIR" && "$BASE_DIR" != "/" ]] || { echo "unsafe base-dir" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "invalid port: $PORT" >&2; exit 2; }
[[ "$RUNS" =~ ^[0-9]+$ ]] || { echo "invalid --runs" >&2; exit 2; }
command -v mysql >/dev/null || { echo "mysql client not found" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 2; }

EVIDENCE=$BASE_DIR/evidence
RUN_DIR=$BASE_DIR/run
LOG=$EVIDENCE/observer.log
TRACE=$EVIDENCE/obvsag-trace.log
ROWS=$EVIDENCE/runs.tsv
mkdir -p "$EVIDENCE" "$RUN_DIR"
rm -f "$LOG" "$TRACE" "$ROWS"
: > "$ROWS"

OBSERVER_PID=""
cleanup() {
  local rc=$?
  if [[ -n "$OBSERVER_PID" ]] && [[ -d "/proc/$OBSERVER_PID" ]]; then
    kill "$OBSERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      [[ -d "/proc/$OBSERVER_PID" ]] || break
      sleep 1
    done
    kill -9 "$OBSERVER_PID" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

mysql_cmd() {
  mysql --connect-timeout=5 --protocol=TCP -h127.0.0.1 -P"$PORT" -uroot -A -N -B "$@"
}

echo "== start observer on port $PORT =="
ulimit -n 65535 || true
env -u OB_VSAG_USE_CUVS \
  "LD_LIBRARY_PATH=$(dirname "$BRIDGE"):$HIPVS_ROOT/lib:$ROCM_ROOT/lib:${LD_LIBRARY_PATH:-}" \
  OB_VSAG_TRACE=1 \
  OB_VSAG_TRACE_FILE="$TRACE" \
  SEEKDB_CUVS_TRACE=1 \
  "$OBSERVER" \
  --nodaemon \
  --port "$PORT" \
  --data-dir "$RUN_DIR/store" \
  --base-dir "$RUN_DIR" \
  --parameter __min_full_resource_pool_memory=2147483648 \
  --parameter memory_budget=2G \
  --parameter datafile_size=20G \
  --parameter cpu_count=24 \
  --parameter local_ip=127.0.0.1 \
  >>"$LOG" 2>&1 &
OBSERVER_PID=$!

ready=0
for _ in $(seq 1 180); do
  [[ -d "/proc/$OBSERVER_PID" ]] || { echo "observer exited, see $LOG" >&2; exit 1; }
  if mysql_cmd -e 'select 1' >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
[[ "$ready" == "1" ]] || { echo "observer did not become ready, see $LOG" >&2; exit 1; }

mysql_cmd -e 'create database if not exists cagra_bench' >/dev/null

if [[ "$LOAD" == "1" ]]; then
  [[ -f "$DATA" ]] || { echo "--load needs an existing --data file" >&2; exit 2; }
  echo "== load $DATA into cagra_bench.$TABLE (dim=$DIM) =="
  mysql_cmd -e "use cagra_bench; drop table if exists $TABLE;
                create table $TABLE(id int primary key, v vector($DIM));" >/dev/null
  python3 - "$DATA" "$DIM" "$TABLE" <<'PY' | mysql_cmd
import array, os, sys

path, dim, table = sys.argv[1], int(sys.argv[2]), sys.argv[3]
data = array.array('f')
with open(path, 'rb') as handle:
    data.fromfile(handle, os.path.getsize(path) // 4)
rows = len(data) // dim
batch = 200
print('use cagra_bench;')
for start in range(0, rows, batch):
    values = []
    for row in range(start, min(start + batch, rows)):
        vector = data[row * dim:(row + 1) * dim]
        values.append("(%d,'[%s]')" % (row, ','.join('%.6g' % value for value in vector)))
    print('insert into %s(id,v) values %s;' % (table, ','.join(values)))
print('select count(*) from %s;' % table)
PY
else
  count=$(mysql_cmd -e "select count(*) from cagra_bench.$TABLE" 2>/dev/null || true)
  [[ -n "$count" && "$count" != "0" ]] || { echo "table cagra_bench.$TABLE is empty or missing; use --load --data" >&2; exit 2; }
  echo "== reuse existing table cagra_bench.$TABLE ($count rows) =="
fi

QUERY=$(mysql_cmd -e "select v from cagra_bench.$TABLE order by id limit 1" 2>/dev/null || true)
[[ -n "$QUERY" ]] || { echo "cannot build a probe query" >&2; exit 2; }

DDL="alter table cagra_bench.$TABLE add vector index $IDX(v) with (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64)"
Q_SQL="select id from cagra_bench.$TABLE order by l2_distance(v, '$QUERY') approximate limit 10"

now_ns() { date +%s%N; }
log_lines() { if [[ -f "$LOG" ]]; then wc -l < "$LOG" | tr -d ' '; else echo 0; fi; }

extract_us() { # $1 = stage marker ("build n=" / "serialize n="), $2 = log offset
  tail -n +"$2" "$LOG" 2>/dev/null \
    | sed -n "s/.*\[seekdb_cuvs\] $1.*us=\([0-9][0-9]*\).*/\1/p" \
    | tail -1
}

ms() { echo $(( $1 / 1000 )); } # us -> ms

echo "== $RUNS timed DDL runs =="
for run in $(seq 1 "$RUNS"); do
  mysql_cmd -e "alter table cagra_bench.$TABLE drop index $IDX" >/dev/null 2>&1 || true
  sleep 1
  offset=$(log_lines)
  start=$(now_ns)
  if ! mysql_cmd -e "$DDL" >>"$EVIDENCE/ddl-$run.log" 2>&1; then
    echo "DDL failed on run $run, see $EVIDENCE/ddl-$run.log" >&2
    exit 1
  fi
  stop=$(now_ns)

  q1_start=$(now_ns)
  mysql_cmd -e "$Q_SQL" >"$EVIDENCE/query-$run.txt" 2>&1 || true
  q1_stop=$(now_ns)

  q2_start=$(now_ns)
  mysql_cmd -e "$Q_SQL" >/dev/null 2>&1 || true
  q2_stop=$(now_ns)

  build_us=$(extract_us 'build n=' "$offset")
  snap_us=$(extract_us 'serialize n=' "$offset")
  build_ms=$(ms "${build_us:-0}")
  snap_ms=$(ms "${snap_us:-0}")
  ddl_ms=$(ms $(( stop - start )))
  other_ms=$(( ddl_ms - build_ms - snap_ms ))
  q1_ms=$(ms $(( q1_stop - q1_start )))
  q2_ms=$(ms $(( q2_stop - q2_start )))

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run" "$ddl_ms" "$build_ms" "$snap_ms" "$other_ms" "$q1_ms" "$q2_ms" >> "$ROWS"
  printf 'run=%s ddl=%sms build=%sms snapshot=%sms other=%sms q1=%sms q2=%sms\n' \
    "$run" "$ddl_ms" "$build_ms" "$snap_ms" "$other_ms" "$q1_ms" "$q2_ms"
done

echo
echo "== summary (ms) =="
awk -F'\t' '
  BEGIN {
    printf "%-4s %9s %9s %9s %9s %9s %9s\n", "run", "ddl", "build", "snapshot", "other", "q1", "q2"
  }
  {
    printf "%-4s %9s %9s %9s %9s %9s %9s\n", $1, $2, $3, $4, $5, $6, $7
    for (i = 2; i <= 7; i++) { sum[i] += $i; }
  }
  END {
    printf "%-4s %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f\n", "mean", \
      sum[2] / NR, sum[3] / NR, sum[4] / NR, sum[5] / NR, sum[6] / NR, sum[7] / NR
  }
' "$ROWS"

echo
echo "== cuVS stage lines seen in the observer log =="
grep -h '\[seekdb_cuvs\]' "$LOG" 2>/dev/null | tail -20 || true
echo
echo "raw: $ROWS"
echo "log: $LOG"
echo "trace: $TRACE"
