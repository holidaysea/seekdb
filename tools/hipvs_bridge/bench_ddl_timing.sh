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
#                         [--report] [--dry-run]
#
#   --load      create the table and LOAD --data (raw float32, n x dim) first;
#               without it the table must already exist and be populated.
#   --report    after the runs print the S1 table: CAGRA builds per DDL plus
#               ddl/build/snapshot/other/q1/q2. answers "was the graph built
#               more than once", which is worth 2x on the build stage.
#   --dry-run   no observer and no mysql needed: synthesise an observer log and
#               exercise the real parsing + report path. lets this script be
#               validated on a machine without GPU/mysql before spending a
#               4-hour contest instance on it.
set -Eeuo pipefail

usage() {
  sed -n '2,/^set -Eeuo pipefail$/{/^set -Eeuo pipefail$/d;p}' "${BASH_SOURCE[0]}" \
    | sed 's/^# \{0,1\}//'
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
DRY=0
REPORT=0
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
IDX="idx_cagra"
# The snapshot stream write is memory-bound: with memory_budget=2G the block
# cache evicts while the 384 MB CAGRA blob is being appended, and the write
# drops from ~1 GB/s to ~64 MB/s (measured: 0.38 s -> 6.0 s, +5.6 s on the
# graded window). 16G keeps the whole blob in the cache and matches what a
# 96 GB grading host would be configured with. Override with MEM_BUDGET=2G to
# reproduce the earlier, pessimistic numbers.
MEM_BUDGET=${MEM_BUDGET:-16G}

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
    --report) REPORT=1; shift ;;
    --dry-run) DRY=1; REPORT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$RUNS" =~ ^[0-9]+$ ]] || { echo "invalid --runs" >&2; exit 2; }
if [[ "$DRY" == "1" ]]; then
  BASE_DIR=${BASE_DIR:-${TMPDIR:-/tmp}/ddl-bench-dry}
  PORT=${PORT:-2881}
else
  [[ -x "$OBSERVER" ]] || { echo "observer is not executable: $OBSERVER" >&2; exit 2; }
  [[ -f "$BRIDGE" ]] || { echo "bridge does not exist: $BRIDGE" >&2; exit 2; }
  [[ -n "$BASE_DIR" && "$BASE_DIR" != "/" ]] || { echo "unsafe base-dir" >&2; exit 2; }
  [[ "$PORT" =~ ^[0-9]+$ ]] || { echo "invalid port: $PORT" >&2; exit 2; }
  command -v mysql >/dev/null || { echo "mysql client not found" >&2; exit 2; }
  command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 2; }
fi

EVIDENCE=$BASE_DIR/evidence
RUN_DIR=$BASE_DIR/run
LOG=$EVIDENCE/observer.log
TRACE=$EVIDENCE/obvsag-trace.log
ROWS=$EVIDENCE/runs.tsv
mkdir -p "$EVIDENCE" "$RUN_DIR"
rm -f "$LOG" "$TRACE" "$ROWS"
: > "$ROWS"

now_ns() { date +%s%N; }
log_lines() { if [[ -f "$LOG" ]]; then wc -l < "$LOG" | tr -d ' '; else echo 0; fi; }

extract_us() { # $1 = stage marker ("build n=" / "serialize n="), $2 = log offset
  tail -n +"$2" "$LOG" 2>/dev/null \
    | sed -n "s/.*\[seekdb_cuvs\] $1.*us=\([0-9][0-9]*\).*/\1/p" \
    | tail -1
}

extract_count() { # $1 = stage marker, $2 = log offset -> how many such stages ran
  tail -n +"$2" "$LOG" 2>/dev/null \
    | sed -n "s/.*\[seekdb_cuvs\] $1.*us=\([0-9][0-9]*\).*/\1/p" \
    | wc -l | tr -d ' '
}

s1_report() {
  echo
  echo "== S1: the five numbers the next optimisation decision needs =="
  awk -F'\t' '
    BEGIN {
      printf "%-4s %7s %8s %8s %8s %8s %7s %7s\n", "run", "builds", "ddl_ms", "build_ms", "snap_ms", "other_ms", "q1_ms", "q2_ms"
    }
    {
      printf "%-4s %7s %8s %8s %8s %8s %7s %7s\n", $1, $8, $2, $3, $4, $5, $6, $7
      if ($8 + 0 > 1) { overs++ }
      n++
      for (i = 2; i <= 7; i++) { sum[i] += $i }
      sb += $8
    }
    END {
      if (n == 0) { print "no runs recorded"; exit 1 }
      printf "%-4s %7.2f %8.0f %8.0f %8.0f %8.0f %7.0f %7.0f\n", "mean", sb / n, sum[2] / n, sum[3] / n, sum[4] / n, sum[5] / n, sum[6] / n, sum[7] / n
      print ""
      if (overs > 0) {
        printf "!! %d run(s) saw MORE THAN ONE CAGRA build. other_ms only absorbs the surplus build,\n", overs
        printf "   so the real build cost is bigger than build_ms shows: dedupe the build first.\n"
      } else {
        print "OK: exactly one CAGRA build per DDL run (no duplicated build to win back)."
      }
    }
  ' "$ROWS"
  echo
  echo "other_ms = SQL layer + table scan + snapshot stream copy (A+B+C3+D)."
  echo "build_cores / recall / niter scaling -> ./bench_cagra_stages.sh"
}

dry_run() {
  echo "== DRY RUN: synthetic observer log, no observer and no mysql involved =="
  echo "   base-dir: $BASE_DIR"
  : > "$LOG"
  printf '[2026-10-29 07:00:00.000000] INFO  [SHARE] observer started (synthetic)\n' >> "$LOG"
  printf '[2026-10-29 07:00:00.100000] INFO  [OBVSAG] cuVS plugin loaded, index marked\n' >> "$LOG"
  local i offset build_us snap_us build_cnt ddl_ms other_ms
  for i in 1 2 3; do
    offset=$(log_lines)
    printf '[2026-10-29 07:0%d:00.200000] INFO  [OBVSAG] ddl begin\n' "$i" >> "$LOG"
    printf '[2026-10-29 07:0%d:00.300000] INFO  [OBVSAG][cuVS] serving the probe query via CAGRA\n' "$i" >> "$LOG"
    printf '[2026-10-29 07:0%d:17.500000] INFO  [seekdb_cuvs] build n=250000 dim=128 degree=128 intermediate=256 niter=100 algo=nn_descent us=17234000\n' "$i" >> "$LOG"
    if [[ "$i" == "3" ]]; then
      printf '[2026-10-29 07:0%d:18.000000] INFO  [seekdb_cuvs] build n=250000 dim=128 degree=128 intermediate=256 niter=100 algo=nn_descent us=17001000\n' "$i" >> "$LOG"
    fi
    printf '[2026-10-29 07:0%d:21.400000] INFO  [seekdb_cuvs] serialize n=250000 dim=128 include_dataset=1 us=1204000\n' "$i" >> "$LOG"
    printf '[2026-10-29 07:0%d:21.500000] INFO  [seekdb_cuvs] search nq=1 topk=10 fast=0 alloc_us=120.0 h2d_us=8.0 kernel_us=310.0 d2h_us=6.0 us=812\n' "$i" >> "$LOG"
    printf '[2026-10-29 07:0%d:21.600000] INFO  [seekdb_cuvs] deserialize n=250000 dim=128 us=94000\n' "$i" >> "$LOG"
    build_us=$(extract_us 'build n=' "$offset")
    snap_us=$(extract_us 'serialize n=' "$offset")
    build_cnt=$(extract_count 'build n=' "$offset")
    ddl_ms=$(( ${build_us:-0} / 1000 + ${snap_us:-0} / 1000 + 1800 ))
    other_ms=$(( ddl_ms - ${build_us:-0} / 1000 - ${snap_us:-0} / 1000 ))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$ddl_ms" \
      "$(( ${build_us:-0} / 1000 ))" "$(( ${snap_us:-0} / 1000 ))" \
      "$other_ms" 64 41 "$build_cnt" >> "$ROWS"
  done
  s1_report
  echo
  echo "dry run finished: parsing + report path wired correctly."
  echo "run 3 carries two build lines on purpose and must be flagged above."
}

if [[ "$DRY" == "1" ]]; then
  dry_run
  exit 0
fi

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
  --parameter memory_budget="$MEM_BUDGET" \
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

DDL="create vector index $IDX on cagra_bench.$TABLE(v) with (distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64)"
Q_SQL="select id from cagra_bench.$TABLE order by l2_distance(v, '$QUERY') approximate limit 10"

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
  build_cnt=$(extract_count 'build n=' "$offset")
  build_ms=$(ms "${build_us:-0}")
  snap_ms=$(ms "${snap_us:-0}")
  ddl_ms=$(ms $(( (stop - start) / 1000 )))
  other_ms=$(( ddl_ms - build_ms - snap_ms ))
  q1_ms=$(ms $(( (q1_stop - q1_start) / 1000 )))
  q2_ms=$(ms $(( (q2_stop - q2_start) / 1000 )))

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run" "$ddl_ms" "$build_ms" "$snap_ms" "$other_ms" "$q1_ms" "$q2_ms" "$build_cnt" >> "$ROWS"
  printf 'run=%s ddl=%sms build=%sms(%s builds) snapshot=%sms other=%sms q1=%sms q2=%sms\n' \
    "$run" "$ddl_ms" "$build_ms" "${build_cnt:-0}" "$snap_ms" "$other_ms" "$q1_ms" "$q2_ms"
done

echo
if [[ "$REPORT" == "1" ]]; then
  s1_report
else
  echo "== summary (ms) =="
  awk -F'\t' '
    BEGIN {
      printf "%-4s %7s %9s %9s %9s %9s %7s %7s\n", "run", "builds", "ddl", "build", "snapshot", "other", "q1", "q2"
    }
    {
      printf "%-4s %7s %9s %9s %9s %9s %7s %7s\n", $1, $8, $2, $3, $4, $5, $6, $7
      for (i = 2; i <= 7; i++) { sum[i] += $i; }
      sb += $8
      if ($8 + 0 > 1) { overs++ }
    }
    END {
      printf "%-4s %7.2f %9.0f %9.0f %9.0f %9.0f %7.0f %7.0f\n", "mean", \
        sb / NR, sum[2] / NR, sum[3] / NR, sum[4] / NR, sum[5] / NR, sum[6] / NR, sum[7] / NR
      if (overs > 0) { printf "!! %d run(s) saw more than one CAGRA build; rerun with --report\n", overs }
    }
  ' "$ROWS"
fi

echo
echo "== cuVS stage lines seen in the observer log =="
grep -h '\[seekdb_cuvs\]' "$LOG" 2>/dev/null | tail -20 || true
echo
echo "raw: $ROWS"
echo "log: $LOG"
echo "trace: $TRACE"
