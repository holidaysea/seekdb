#!/usr/bin/env bash
# bench_frontier.sh - one-shot Pareto search for the graded DDL window.
#
# Why this exists
# ---------------
# The graded number is the wall clock from `create vector index ... lib=cuvs`
# until the first fixed query returns, on BOTH datasets, weighted
# SIFT^0.4 * Hidden^0.6.  Build time and snapshot bytes are the two things we
# control, and recall is a hard gate (must stay within 0.005 of the frozen
# baseline measured on the SAME machine).  So the search is:
#
#   1. measure the V5 defaults on this machine (the baseline we are compared
#      against, so we never have to trust the published 23/30 s numbers),
#   2. walk a small grid of (graph_degree, intermediate_graph_degree, niter)
#      that earlier sessions showed to be the only live knobs,
#   3. keep every configuration whose recall is not worse than baseline,
#   4. print the best one as ready-to-paste #defines.
#
# Everything runs through bench_ddl_timing.sh, i.e. through the real SQL path:
# no synthetic shortcut, no cached graph, no prebuilt index.
#
# Usage
#   tools/hipvs_bridge/bench_frontier.sh --observer <seekdb> --bridge <so> \
#       --base-dir /root/work/frontier [--data-dir <dir with *.f32>] [--quick]
#
# Output
#   <base-dir>/frontier.tsv and a summary table on stdout.

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBSERVER=""
BRIDGE=""
BASE_DIR=""
DATA_DIR=""
RUNS_PER_CFG=2
QUICK=0
PORT="${PORT:-2881}"
DIM_SMALL="${DIM_SMALL:-128}"
DIM_LARGE="${DIM_LARGE:-256}"
N_ROWS="${N_ROWS:-250000}"

usage() {
  sed -n '2,/^set -Eeuo pipefail$/{/^set -Eeuo pipefail$/d;s/^# \{0,1\}//;p}' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --observer) OBSERVER="$2"; shift 2 ;;
    --bridge)   BRIDGE="$2"; shift 2 ;;
    --base-dir) BASE_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --runs)     RUNS_PER_CFG="$2"; shift 2 ;;
    --port)     PORT="$2"; shift 2 ;;
    --quick)    QUICK=1; RUNS_PER_CFG=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$OBSERVER" || -z "$BRIDGE" || -z "$BASE_DIR" ]]; then
  echo "--observer, --bridge and --base-dir are required" >&2
  usage >&2
  exit 1
fi
[[ -x "$OBSERVER" ]] || { echo "not executable: $OBSERVER" >&2; exit 1; }
[[ -f "$BRIDGE" ]] || { echo "no such bridge: $BRIDGE" >&2; exit 1; }

mkdir -p "$BASE_DIR"
RESULT="$BASE_DIR/frontier.tsv"
: > "$RESULT"
printf 'dim\tgraph_degree\tintermediate\tniter\tddl_ms\tbuild_ms\tsnapshot_ms\tother_ms\n' >> "$RESULT"

# ---------------------------------------------------------------- utilities
now_ms() { date +%s%3N; }

# run_one <dim> <graph_degree> <intermediate> <niter> <runs> <label>
run_one() {
  local dim="$1" gd="$2" ig="$3" niter="$4" runs="$5" label="$6"
  local out="$BASE_DIR/$label"
  local data=""
  if [[ -n "$DATA_DIR" ]]; then
    if [[ "$dim" == "$DIM_SMALL" ]]; then data="$DATA_DIR/sift250k128.f32"; else data="$DATA_DIR/hidden250k256.f32"; fi
    [[ -f "$data" ]] || data=""
  fi
  local load_args=()
  [[ -n "$data" ]] && load_args=(--load --data "$data")

  local t0 t1
  t0=$(now_ms)
  SEEKDB_CUVS_GRAPH_DEGREE="$gd" \
  SEEKDB_CUVS_INTERMEDIATE_DEGREE="$ig" \
  SEEKDB_CUVS_NITER="$niter" \
  SEEKDB_CUVS_TRACE=1 \
  MEM_BUDGET="${MEM_BUDGET:-16G}" \
    "$HERE/bench_ddl_timing.sh" \
      --observer "$OBSERVER" --bridge "$BRIDGE" \
      --base-dir "$out" --port "$PORT" --dim "$dim" \
      --table "t_f_$label" --runs "$runs" "${load_args[@]}" \
      > "$out.log" 2>&1 || {
        echo "  !! $label failed, see $out.log" >&2
        return 1
      }
  t1=$(now_ms)

  # bench_ddl_timing.sh writes the per-run rows to $out/runs.tsv
  local rows="$out/runs.tsv"
  if [[ ! -f "$rows" ]]; then
    echo "  !! $label produced no runs.tsv" >&2
    return 1
  fi
  awk -F'\t' -v OFS='\t' -v dim="$dim" -v gd="$gd" -v ig="$ig" -v n="$niter" '
    NR > 1 {
      ddl += $4; bld += $5; snap += $6; oth += $7; cnt++
    }
    END {
      if (cnt == 0) { print "  !! no rows" > "/dev/stderr"; exit 1 }
      printf "%s\t%s\t%s\t%s\t%.0f\t%.0f\t%.0f\t%.0f\n", dim, gd, ig, n, ddl/cnt, bld/cnt, snap/cnt, oth/cnt
    }' "$rows" >> "$RESULT"
  printf '  %-22s dim=%-3s gd=%-4s ig=%-4s n=%-3s  %6.0f ms  (%5.0f ms wall)\n' \
    "$label" "$dim" "$gd" "$ig" "$niter" \
    "$(awk -F'\t' -v d="$dim" -v g="$gd" -v i="$ig" -v n="$niter" '$1==d && $2==g && $3==i && $4==n {print $5; exit}' "$RESULT")" \
    "$(awk "BEGIN{print $t1-$t0}")"
}

# score <sift_ms> <hidden_ms>  -> official build score, assuming the published baseline
score_of() {
  awk -v s="$1" -v h="$2" 'BEGIN{
    bs = 23.0 / s; bh = 30.0 / h;
    if (bs < 1) bs = 1; if (bh < 1) bh = 1;
    acc = exp(0.4 * log(bs) + 0.6 * log(bh));
    pt = 14.0 * log(acc) / log(3.0);
    if (pt < 0) pt = 0; if (pt > 14) pt = 14;
    printf "sift %.2fx hidden %.2fx composite %.2fx build %.2f total %.2f", bs, bh, acc, pt, pt + 4
  }'
}

echo "== bench_frontier =="
echo "observer : $OBSERVER"
echo "bridge   : $BRIDGE"
echo "base-dir : $BASE_DIR"
echo "runs/cfg : $RUNS_PER_CFG   quick=$QUICK"
echo

# ------------------------------------------------------------------ phase 1
# The baseline is the V5 default triple.  We emulate it through the env knobs
# of our own bridge so the comparison is on one binary, one machine, one disk.
echo "-- phase 1: V5 defaults on this machine (the real denominator)"
run_one "$DIM_SMALL" 128 256 100 1 baseline_small || true
run_one "$DIM_LARGE" 128 256 100 1 baseline_large || true

# ------------------------------------------------------------------ phase 2
# Grid.  Every point here survived the earlier sweeps on at least one side;
# the point of the run is to find the one that survives on BOTH dimensions.
echo
echo "-- phase 2: candidate grid"
GRID_SMALL=(
  "192 192 40"
  "192 192 45"
  "176 192 45"
  "192 160 45"
  "160 192 60"
)
GRID_LARGE=(
  "192 192 40"
  "192 192 45"
  "176 192 45"
  "192 192 50"
  "160 192 60"
)

idx=0
for cfg in "${GRID_SMALL[@]}"; do
  read -r gd ig n <<< "$cfg"
  idx=$((idx + 1))
  run_one "$DIM_SMALL" "$gd" "$ig" "$n" "$RUNS_PER_CFG" "s${idx}_${gd}_${ig}_${n}" || true
done

idx=0
for cfg in "${GRID_LARGE[@]}"; do
  read -r gd ig n <<< "$cfg"
  idx=$((idx + 1))
  run_one "$DIM_LARGE" "$gd" "$ig" "$n" "$RUNS_PER_CFG" "h${idx}_${gd}_${ig}_${n}" || true
done

# ------------------------------------------------------------------ phase 3
echo
echo "-- phase 3: summary (mean of $RUNS_PER_CFG run(s) per point)"
column -t -s$'\t' "$RESULT" || cat "$RESULT"

echo
echo "-- best point that appears on both dimensions"
awk -F'\t' 'NR>1 && $5>0 {
    key = $2" "$3" "$4;
    if ($1 == '"$DIM_SMALL"') s[key] = $5;
    if ($1 == '"$DIM_LARGE"') h[key] = $5;
  }
  END {
    best = ""; bestpts = -1;
    for (k in s) if (k in h) {
      acc_s = 23.0 / s[k]; acc_h = 30.0 / h[k];
      if (acc_s < 1) acc_s = 1; if (acc_h < 1) acc_h = 1;
      acc = exp(0.4 * log(acc_s) + 0.6 * log(acc_h));
      pts = 14.0 * log(acc) / log(3.0);
      printf "  gd/ig/n = %-14s sift %6.0f ms  hidden %6.0f ms  -> %.2fx/%.2fx  %.2f pts\n", k, s[k], h[k], acc_s, acc_h, pts;
      if (pts > bestpts) { bestpts = pts; best = k }
    }
    if (best != "") printf "\n  BEST: %s  (%.2f pts)\n", best, bestpts;
  }' "$RESULT"

echo
echo "-- paste-ready defaults (GRAPH_DEGREE / INTERMEDIATE_DEGREE / NITER)"
echo "   #define SK_DEF_GRAPH_DEGREE       <gd>"
echo "   #define SK_DEF_INTERMEDIATE_DEGREE <ig>"
echo "   #define SK_DEF_NITER              <n>"
echo
echo "NOTE: scores above assume the published 23/30 s baseline; the phase-1 rows"
echo "      give the same-machine denominator, use those for the real ratio."
echo "      Recall must be checked separately (bench_cagra_stages --csv) before"
echo "      adopting a point: the gate is baseline_recall - 0.005, not more."
