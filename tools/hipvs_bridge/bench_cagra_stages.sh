#!/usr/bin/env bash
# Parameter sweep for the seekdb <-> hipVS CAGRA build, without building seekdb.
#
# Everything the graded window pays for is measured here: CAGRA build, snapshot
# write (full vs graph-only), snapshot reload, single-query latency and recall
# against exact CPU kNN. Build knobs go through the bridge environment
# variables, so one compiled binary covers the whole matrix.
#
# Usage:
#   ./bench_cagra_stages.sh [--out DIR] [--sift FILE] [--hidden FILE] [--quick]
#
#   --sift FILE    raw little-endian float32 file, 250000 x 128 rows
#   --hidden FILE  raw little-endian float32 file, 250000 x 256 rows
#   --quick        skip the exact-kNN recall pass (saves ~1 min per case)
#
# Without --sift/--hidden the datasets are synthetic (clustered), which is good
# enough to rank parameter choices but NOT to certify recall.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${TMPDIR:-/tmp}/cagra_bench_$(date +%Y%m%d_%H%M%S)
SIFT_DATA=""
HIDDEN_DATA=""
QUICK=0

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while (($# > 0)); do
  case "$1" in
    --out) OUT_DIR=$2; shift 2 ;;
    --sift) SIFT_DATA=$2; shift 2 ;;
    --hidden) HIDDEN_DATA=$2; shift 2 ;;
    --quick) QUICK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

CC=${CC:-gcc}
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
N=${N:-250000}

command -v "$CC" >/dev/null || { echo "compiler not found: $CC" >&2; exit 2; }
[[ -r "$HIPVS_ROOT/include/cuvs/core/c_api.h" ]] || {
  echo "hipVS headers are missing under $HIPVS_ROOT/include (set HIPVS_ROOT)" >&2; exit 2; }

mkdir -p "$OUT_DIR"
CSV=$OUT_DIR/stages.csv
rm -f "$CSV"

echo "== environment ==" | tee "$OUT_DIR/env.txt"
{
  date -Is
  uname -a
  echo "HIPVS_ROOT=$HIPVS_ROOT ROCM_ROOT=$ROCM_ROOT"
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname 2>&1 | head -40
  else
    echo "rocm-smi: not installed"
  fi
  ls -l /dev/kfd /dev/dri/render* 2>&1 || true
} | tee -a "$OUT_DIR/env.txt"

echo "== build bridge =="
HIPVS_ROOT=$HIPVS_ROOT ROCM_ROOT=$ROCM_ROOT CC="$CC" \
  "$SCRIPT_DIR/build.sh" "$OUT_DIR/libseekdb_cuvs_bridge.so"

echo "== build bench =="
"$CC" -O2 -std=gnu11 -fopenmp -Wall -Wextra \
  -I"$SCRIPT_DIR" \
  "$SCRIPT_DIR/bench_cagra_stages.c" \
  -o "$OUT_DIR/bench_cagra_stages" \
  -L"$OUT_DIR" -lseekdb_cuvs_bridge -Wl,-rpath,"$OUT_DIR"

run_case() {
  local tag=$1 n=$2 dim=$3 data=$4 degree=$5 inter=$6 niter=$7 algo=${8:-nn_descent} fast=${9:-0}
  local -a args=(--n "$n" --dim "$dim" --tag "$tag" --csv "$CSV" --workdir "$OUT_DIR")
  if [[ -n "$data" ]]; then args+=(--data "$data"); fi
  if [[ "$QUICK" == "1" ]]; then args+=(--no-recall); fi
  echo "== $tag n=$n dim=$dim algo=$algo degree=$degree intermediate=$inter niter=$niter fast=$fast =="
  SEEKDB_CUVS_BUILD_ALGO=$algo \
  SEEKDB_CUVS_SEARCH_FAST=$fast \
  SEEKDB_CUVS_GRAPH_DEGREE=$degree \
  SEEKDB_CUVS_INTERMEDIATE_DEGREE=$inter \
  SEEKDB_CUVS_NITER=$niter \
  SEEKDB_CUVS_TRACE=1 \
  LD_LIBRARY_PATH="$OUT_DIR:$HIPVS_ROOT/lib:$ROCM_ROOT/lib:${LD_LIBRARY_PATH:-}" \
    "$OUT_DIR/bench_cagra_stages" "${args[@]}" 2>&1 | tee "$OUT_DIR/$tag.log"
}

# Baseline column is the frozen V5 configuration (128 / 256 / 100).
run_case "sift_base_128_256_100"   "$N" 128 "$SIFT_DATA"   128 256 100
run_case "sift_128_192_100"        "$N" 128 "$SIFT_DATA"   128 192 100
run_case "sift_128_128_100"        "$N" 128 "$SIFT_DATA"   128 128 100
run_case "sift_128_256_20"         "$N" 128 "$SIFT_DATA"   128 256  20
run_case "sift_64_128_100"         "$N" 128 "$SIFT_DATA"    64 128 100

# L0 lever: swap the builder itself.  V5 forces NN_DESCENT, whose per-iteration
# graph maintenance is host-side; IVF_PQ (the cuVS default) builds the candidate
# graph through GPU searches instead.  This is the only candidate for >2x, so it
# is priced first and at several degrees.
run_case "sift_ivfpq_64_128_100"   "$N" 128 "$SIFT_DATA"    64 128 100 ivf_pq
run_case "sift_ivfpq_128_192_100"  "$N" 128 "$SIFT_DATA"   128 192 100 ivf_pq
run_case "sift_auto_128_192_100"   "$N" 128 "$SIFT_DATA"   128 192 100 auto
run_case "hidden_ivfpq_64_128_100" "$N" 256 "$HIDDEN_DATA"  64 128 100 ivf_pq
run_case "hidden_ivfpq_128_192_100" "$N" 256 "$HIDDEN_DATA" 128 192 100 ivf_pq

run_case "hidden_base_128_256_100" "$N" 256 "$HIDDEN_DATA" 128 256 100
run_case "hidden_128_192_100"      "$N" 256 "$HIDDEN_DATA" 128 192 100
run_case "hidden_128_256_20"       "$N" 256 "$HIDDEN_DATA" 128 256  20
# Hidden carries 0.6 of the weighted score, so the L1 lever has to be priced here too.
run_case "hidden_64_192_100"       "$N" 256 "$HIDDEN_DATA"  64 192 100
run_case "hidden_64_128_100"       "$N" 256 "$HIDDEN_DATA"  64 128 100

# L5 query fast path: identical graph, only the per-query device scratch and the
# search-params object are reused.  The pair below isolates the win (and proves
# it is not a recall trade, since the index is the same).
echo
echo "== L5 query fast path (same graph, scratch+params reuse) =="
run_case "sift_l5_off"             "$N" 128 "$SIFT_DATA"   128 192 100 nn_descent 0
run_case "sift_l5_on"              "$N" 128 "$SIFT_DATA"   128 192 100 nn_descent 1
run_case "hidden_l5_off"           "$N" 256 "$HIDDEN_DATA" 128 192 100 nn_descent 0
run_case "hidden_l5_on"            "$N" 256 "$HIDDEN_DATA" 128 192 100 nn_descent 1

echo
echo "== stage table =="
if command -v column >/dev/null 2>&1; then
  column -t -s, "$CSV"
else
  cat "$CSV"
fi
echo
echo "csv: $CSV"
echo "logs: $OUT_DIR/*.log"
cat <<'EOF'

How to read it:
  build_ms       CAGRA build (nn-descent + prune) -- the dominant stage
  build_cores    average cores busy during build; >2 means the stage is CPU
                 bound (host graph maintenance), which is what limits this path
  recall         recall@topk vs exact CPU kNN; the contest gate compares to the
                 baseline recall within 0.005, so compare rows to *_base_*
  snapshot_ms    production snapshot (graph + dataset + duplicated vectors)
  snapshot_slim  same graph without the dataset: the size the snapshot could
                 shrink to once the dataset is re-attached after load
EOF
