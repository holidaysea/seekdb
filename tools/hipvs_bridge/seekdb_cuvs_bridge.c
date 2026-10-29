/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------------- *
 * Build/search knobs.
 *
 * The defaults below reproduce the frozen V5 behaviour exactly (128 / 256 /
 * 100). They are overridable from the environment so that a parameter sweep
 * needs no rebuild: the contest harness can A/B the same binary in one run.
 * Every override is range checked and logged; an out-of-range value is ignored
 * with a warning instead of silently degrading the graph.
 * ------------------------------------------------------------------------- */
#define SK_DEF_GRAPH_DEGREE        128
#define SK_DEF_INTERMEDIATE_DEGREE 256
#define SK_DEF_NITER               100

static int sk_trace_enabled(void)
{
  static int cached = -1;
  if (cached < 0) { cached = getenv("SEEKDB_CUVS_TRACE") != NULL ? 1 : 0; }
  return cached;
}

static int sk_search_trace_enabled(void)
{
  static int cached = -1;
  if (cached < 0) { cached = getenv("SEEKDB_CUVS_TRACE_SEARCH") != NULL ? 1 : 0; }
  return cached;
}

static double sk_now_us(void)
{
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) { return 0.0; }
  return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static long sk_env_long(const char* name, long fallback, long lo, long hi)
{
  const char* raw = getenv(name);
  char* end = NULL;
  long parsed = 0;
  if (raw == NULL || *raw == '\0') { return fallback; }
  parsed = strtol(raw, &end, 10);
  if (end == raw || parsed < lo || parsed > hi) {
    fprintf(stderr, "[seekdb_cuvs] ignoring %s=%s (expected %ld..%ld)\n", name, raw, lo, hi);
    return fallback;
  }
  return parsed;
}

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  /* L5 fast path (SEEKDB_CUVS_SEARCH_FAST=1): the three device scratch buffers
   * and the search-params object are kept across calls instead of being created
   * and destroyed per query.  Guarded by a spin flag because OB only serialises
   * on g_ob_cuvs_mu for the single-query entry point. */
  int    fast;
  volatile int busy;
  void*  qd;  size_t qd_bytes;
  void*  nd;  size_t nd_bytes;
  void*  dd;  size_t dd_bytes;
  cuvsCagraSearchParams_t sp;
  size_t sp_topk;
} sk_cuvs_index;

/* Build-algorithm selector.  V5 hard-codes NN_DESCENT, whose per-iteration graph
 * maintenance runs on the host (see detail/nn_descent.cuh: graph_.update_graph /
 * add_reverse_edges under OpenMP).  IVF_PQ is the cuVS *default* and builds the
 * intermediate graph through GPU searches instead, so it is the only knob that
 * can plausibly move the 23-30 s window by more than 2x.  Default stays
 * nn_descent: not setting this variable is byte-identical to V5. */
static const char* sk_algo_name(enum cuvsCagraGraphBuildAlgo algo)
{
  switch (algo) {
    case IVF_PQ:      return "ivf_pq";
    case AUTO_SELECT: return "auto";
    case NN_DESCENT:  return "nn_descent";
    default:          return "other";
  }
}

static enum cuvsCagraGraphBuildAlgo sk_env_build_algo(void)
{
  const char* raw = getenv("SEEKDB_CUVS_BUILD_ALGO");
  if (raw == NULL || *raw == '\0') { return NN_DESCENT; }
  if (strcmp(raw, "nn_descent") == 0 || strcmp(raw, "nnd") == 0) { return NN_DESCENT; }
  if (strcmp(raw, "ivf_pq") == 0 || strcmp(raw, "ivfpq") == 0) { return IVF_PQ; }
  if (strcmp(raw, "auto") == 0) { return AUTO_SELECT; }
  fprintf(stderr, "[seekdb_cuvs] ignoring SEEKDB_CUVS_BUILD_ALGO=%s "
                  "(expected nn_descent|ivf_pq|auto)\n", raw);
  return NN_DESCENT;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  long graph_degree = sk_env_long("SEEKDB_CUVS_GRAPH_DEGREE", SK_DEF_GRAPH_DEGREE, 16, 512);
  long intermediate = sk_env_long("SEEKDB_CUVS_INTERMEDIATE_DEGREE",
                                  SK_DEF_INTERMEDIATE_DEGREE, 16, 1024);
  long niter = sk_env_long("SEEKDB_CUVS_NITER", SK_DEF_NITER, 1, 1000);
  const enum cuvsCagraGraphBuildAlgo algo = sk_env_build_algo();
  /* cuVS recommends intermediate >= 1.5 * graph_degree; a smaller value only
   * starves the pruning pass, so clamp instead of trusting a typo. */
  if (intermediate < graph_degree) { intermediate = graph_degree * 3 / 2; }
  h->ip->build_algo = algo;
  h->ip->graph_degree = (size_t)graph_degree;
  h->ip->intermediate_graph_degree = (size_t)intermediate;
  h->ip->nn_descent_niter = (size_t)niter;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  {
    const double build_t0 = sk_now_us();
    if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
    if (sk_trace_enabled()) {
      fprintf(stderr, "[seekdb_cuvs] build n=%ld dim=%ld algo=%s degree=%ld intermediate=%ld "
                      "niter=%ld us=%.0f\n",
              n, dim, sk_algo_name(algo), graph_degree, intermediate, niter,
              sk_now_us() - build_t0);
    }
  }
  h->n = n; h->dim = dim;
  h->fast = (int)sk_env_long("SEEKDB_CUVS_SEARCH_FAST", 0, 0, 1);
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
  int rc = 0;
  const double search_t0 = sk_now_us();
  const int fast = h->fast && nq > 0 && topk > 0;
  const size_t need_q = sizeof(float) * (size_t)nq * (size_t)h->dim;
  const size_t need_o = sizeof(unsigned) * (size_t)nq * (size_t)topk;
  float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  double t_alloc_start = search_t0, t_alloc = 0, t_h2d = 0, t_kernel = 0, t_d2h = 0;

  if (fast) {
    /* Optimistic spin: OB serialises queries on g_ob_cuvs_mu, so contention is
     * the exception, not the rule; the batch entry point is the only caller that
     * can overlap. */
    while (__sync_lock_test_and_set(&h->busy, 1)) { }
    if (h->qd_bytes < need_q) {
      if (h->qd) { cuvsRMMFree(h->res, h->qd, h->qd_bytes); h->qd = 0; h->qd_bytes = 0; }
      if (cuvsRMMAlloc(h->res, &h->qd, need_q) != CUVS_SUCCESS) { rc = 10; goto unlock; }
      h->qd_bytes = need_q;
    }
    if (h->nd_bytes < need_o) {
      if (h->nd) { cuvsRMMFree(h->res, h->nd, h->nd_bytes); h->nd = 0; h->nd_bytes = 0; }
      if (cuvsRMMAlloc(h->res, (void**)&h->nd, need_o) != CUVS_SUCCESS) { rc = 11; goto unlock; }
      h->nd_bytes = need_o;
    }
    if (h->dd_bytes < need_o) {
      if (h->dd) { cuvsRMMFree(h->res, h->dd, h->dd_bytes); h->dd = 0; h->dd_bytes = 0; }
      if (cuvsRMMAlloc(h->res, (void**)&h->dd, need_o) != CUVS_SUCCESS) { rc = 12; goto unlock; }
      h->dd_bytes = need_o;
    }
    /* itopk_size depends on topk only, so the params object outlives the call
     * unless topk changes (it does not, in practice: always 10/100). */
    if (h->sp == 0 || h->sp_topk != (size_t)topk) {
      if (h->sp) { cuvsCagraSearchParamsDestroy(h->sp); h->sp = 0; }
      if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { rc = 14; goto unlock; }
      {
        size_t required_itopk = (size_t)topk * 4;
        if (required_itopk < 512) required_itopk = 512;
        if (h->sp->itopk_size < required_itopk) h->sp->itopk_size = required_itopk;
        h->sp->search_width = 4;
      }
      h->sp_topk = (size_t)topk;
    }
    qd = (float*)h->qd; nd = h->nd; dd = (float*)h->dd; sp = h->sp;
  } else {
    if (cuvsRMMAlloc(h->res, (void**)&qd, need_q) != CUVS_SUCCESS) { rc = 10; goto done; }
    if (cuvsRMMAlloc(h->res, (void**)&nd, need_o) != CUVS_SUCCESS) { rc = 11; goto done; }
    if (cuvsRMMAlloc(h->res, (void**)&dd, need_o) != CUVS_SUCCESS) { rc = 12; goto done; }
  }
  t_alloc = sk_now_us() - t_alloc_start;

  if (cudaMemcpy(qd, query, need_q, cudaMemcpyDefault) != cudaSuccess) { rc = 13; goto unlock; }
  t_h2d = sk_now_us() - t_alloc_start - t_alloc;
  {
    DLManagedTensor qt, nt, dt; long qs[2]; long nsh[2]; long dsh[2];
    qs[0] = nq; qs[1] = h->dim; nsh[0] = nq; nsh[1] = topk; dsh[0] = nq; dsh[1] = topk;
    qt.dl_tensor.data=qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    if (sp == 0) {
      if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc = 14; goto unlock; }
      {
        size_t required_itopk = (size_t)topk * 4;
        if (required_itopk < 512) required_itopk = 512;
        if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
        sp->search_width = 4;
      }
    }
    {
      cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
      const double k0 = sk_now_us();
      if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 15; goto unlock; }
      t_kernel = sk_now_us() - k0;
    }
    {
      const double d0 = sk_now_us();
      if (cudaMemcpy(out_ids, nd, sizeof(unsigned)*(size_t)nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc = 16; goto unlock; }
      if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float)*(size_t)nq*topk, cudaMemcpyDefault);
      t_d2h = sk_now_us() - d0;
    }
  }

unlock:
  if (fast) {
    __sync_lock_release(&h->busy);
  }
done:
  if (!fast) {
    if (sp) cuvsCagraSearchParamsDestroy(sp);
    if (dd) cuvsRMMFree(h->res, dd, need_o);
    if (nd) cuvsRMMFree(h->res, nd, need_o);
    if (qd) cuvsRMMFree(h->res, qd, need_q);
  }
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  else if (sk_search_trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] search nq=%ld topk=%ld fast=%d alloc_us=%.1f "
                    "h2d_us=%.1f kernel_us=%.1f d2h_us=%.1f us=%.0f\n",
            nq, topk, fast, t_alloc, t_h2d, t_kernel, t_d2h, sk_now_us() - search_t0);
  }
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  if (h->dd) cuvsRMMFree(h->res, h->dd, h->dd_bytes);
  if (h->nd) cuvsRMMFree(h->res, h->nd, h->nd_bytes);
  if (h->qd) cuvsRMMFree(h->res, h->qd, h->qd_bytes);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  const double t0 = sk_now_us();
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  if (sk_trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] serialize n=%ld dim=%ld include_dataset=1 us=%.0f\n",
            index->n, index->dim, sk_now_us() - t0);
  }
  return 0;
}

/* Benchmark-only entry point: same call as seekdb_cuvs_serialize but with the
 * include_dataset flag exposed, so a sweep can price the dataset copy that the
 * snapshot currently writes twice (see docs: snapshot slimming).
 * The snapshot written with include_dataset=0 is NOT loadable by
 * seekdb_cuvs_deserialize until the dataset is re-attached, so production code
 * must keep using seekdb_cuvs_serialize. */
int seekdb_cuvs_bench_serialize(void* handle, const char* filename, int include_dataset)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  const double t0 = sk_now_us();
  if (cuvsCagraSerialize(index->res, filename, index->idx, include_dataset != 0) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] bench serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  if (sk_trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] bench_serialize n=%ld dim=%ld include_dataset=%d us=%.0f\n",
            index->n, index->dim, include_dataset, sk_now_us() - t0);
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  const double load_t0 = sk_now_us();
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  if (sk_trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] deserialize n=%ld dim=%ld us=%.0f\n",
            n, dim, sk_now_us() - load_t0);
  }
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
