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
} sk_cuvs_index;

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
  /* cuVS recommends intermediate >= 1.5 * graph_degree; a smaller value only
   * starves the pruning pass, so clamp instead of trusting a typo. */
  if (intermediate < graph_degree) { intermediate = graph_degree * 3 / 2; }
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = (size_t)graph_degree;
  h->ip->intermediate_graph_degree = (size_t)intermediate;
  h->ip->nn_descent_niter = (size_t)niter;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  {
    const double build_t0 = sk_now_us();
    if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
    if (sk_trace_enabled()) {
      fprintf(stderr, "[seekdb_cuvs] build n=%ld dim=%ld degree=%ld intermediate=%ld "
                      "niter=%ld us=%.0f\n",
              n, dim, graph_degree, intermediate, niter, sk_now_us() - build_t0);
    }
  }
  h->n = n; h->dim = dim;
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
  float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  if (cuvsRMMAlloc(h->res, (void**)&qd, sizeof(float)*nq*h->dim)  != CUVS_SUCCESS) { rc=10; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(unsigned)*nq*topk) != CUVS_SUCCESS) { rc=11; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float)*nq*topk)    != CUVS_SUCCESS) { rc=12; goto done; }
  if (cudaMemcpy(qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc=14; goto done; }
    {
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 512) required_itopk = 512;
      if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
      sp->search_width = 4;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (dd) cuvsRMMFree(h->res, dd, sizeof(float)*nq*topk);
  if (nd) cuvsRMMFree(h->res, nd, sizeof(unsigned)*nq*topk);
  if (qd) cuvsRMMFree(h->res, qd, sizeof(float)*nq*h->dim);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  else if (sk_search_trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] search nq=%ld topk=%ld us=%.0f\n",
            nq, topk, sk_now_us() - search_t0);
  }
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
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
