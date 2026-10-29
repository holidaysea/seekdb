#!/usr/bin/env bash
# Fast pre-flight: syntax/type check the bridge and the bench harness without
# linking, so a typo does not cost a full hipcc build. Same include paths and
# the same -D__HIP_PLATFORM_AMD__ as build.sh.
#
# Usage: ./syntax_check.sh
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
CC=${CC:-gcc}
WARN=${WARN:--Wall -Wextra}

command -v "$CC" >/dev/null || { echo "compiler not found: $CC" >&2; exit 2; }
[[ -r "$HIPVS_ROOT/include/cuvs/core/c_api.h" ]] || {
  echo "hipVS headers are missing under $HIPVS_ROOT/include" >&2; exit 2; }

echo "checking seekdb_cuvs_bridge.c"
# shellcheck disable=SC2086
"$CC" -fsyntax-only -std=gnu11 $WARN \
  -D__HIP_PLATFORM_AMD__ \
  -I"$SCRIPT_DIR" -I"$HIPVS_ROOT/include" -I"$ROCM_ROOT/include" \
  "$SCRIPT_DIR/seekdb_cuvs_bridge.c"

echo "checking bench_cagra_stages.c"
# shellcheck disable=SC2086
"$CC" -fsyntax-only -std=gnu11 -fopenmp $WARN \
  -I"$SCRIPT_DIR" \
  "$SCRIPT_DIR/bench_cagra_stages.c"

echo "ok"
