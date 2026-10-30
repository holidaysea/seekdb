#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# provision_gpu.sh -- bring a fresh GPU instance from "repo copied" to "cuVS
# observer binary ready", in one command.
#
# The three things that silently cost us a full evening each time:
#   1. deps/3rd is not in the tarball; without the symlink CMake dies with
#      "can't find suitable compiler" because clang-17 is not on PATH.
#   2. OB_BUILD_CUVS defaults to OFF and build.sh never turns it on, so the
#      binary you get is CPU-only and every DDL silently runs the VSAG path.
#   3. the bridge has to be compiled with system gcc: the devtools ld is too
#      old to link against this glibc (it cannot read .relr.dyn and then claims
#      it "cannot find /lib/x86_64-linux-gnu/libc.so.6").
#
# usage: bash tools/hipvs_bridge/provision_gpu.sh [/path/to/repo] [jobs]
# ---------------------------------------------------------------------------
set -euo pipefail

REPO=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
JOBS=${2:-16}
DEPS_CACHE=${DEPS_CACHE:-/opt/seekdb-deps-cache/ac032efc6b707fc0273e6b553db53011/3rd}
BRIDGE_SO=${BRIDGE_SO:-$REPO/build_bridge/libseekdb_cuvs_bridge.so}

cd "$REPO"

echo "== 1/5 deps/3rd =="
if [[ ! -e deps/3rd ]]; then
  mkdir -p deps
  ln -sfn "$DEPS_CACHE" deps/3rd
fi
export PATH="$REPO/deps/3rd/usr/local/oceanbase/devtools/bin:$PATH"
command -v clang-17 >/dev/null || { echo "clang-17 still not on PATH" >&2; exit 2; }
command -v cmake >/dev/null || { echo "cmake still not on PATH" >&2; exit 2; }

echo "== 2/5 bridge .so =="
mkdir -p "$(dirname "$BRIDGE_SO")"
gcc -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ \
  -I/opt/hipvs/include -I/opt/hipvs/include/cuvs -I/opt/rocm/include \
  -L/opt/hipvs/lib -Wl,-rpath,/opt/hipvs/lib \
  -o "$BRIDGE_SO" tools/hipvs_bridge/seekdb_cuvs_bridge.c -lcuvs_c -lpthread
echo "   $BRIDGE_SO $(stat -c %s "$BRIDGE_SO") B, $(nm -D --defined-only "$BRIDGE_SO" | grep -c seekdb_cuvs) symbols"

echo "== 3/5 cmake (cuVS ON) =="
cmake -S "$REPO" -B build_release -DOB_BUILD_CUVS=ON -DCUVS_BRIDGE_LIB="$BRIDGE_SO"
grep -E "OB_BUILD_CUVS|CUVS_BRIDGE_LIB" build_release/CMakeCache.txt

echo "== 4/5 build (jobs=$JOBS) =="
cmake --build build_release --target seekdb -j"$JOBS"

echo "== 5/5 verify =="
BIN=build_release/src/observer/seekdb
ls -l "$BIN"
markers=$(strings -a "$BIN" | grep -c "OBVSAG\]\[cuVS" || true)
echo "   cuVS log markers in binary: $markers (0 means the GPU path is compiled out)"
[[ "$markers" -gt 0 ]]
echo "READY $BIN $BRIDGE_SO"
