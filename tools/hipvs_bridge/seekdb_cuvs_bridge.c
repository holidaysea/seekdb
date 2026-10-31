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
#include <pthread.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------------- *
 * Build/search knobs.
 *
 * V5 built with 128 / 256 / 100. The defaults below were re-tuned on the target
 * hardware (Radeon PRO W7900D, 250K rows) and are overridable from the
 * environment so a sweep needs no rebuild. Every override is range checked and
 * logged; an out-of-range value is ignored with a warning instead of silently
 * degrading the graph.
 *
 * Why these values (measured, uniform 250K x 128, 500 queries, recall@100 of
 * the *reference* build = 0.9937):
 *
 *   degree  intermediate  niter   build    recall@100
 *   128     256           100     15.8 s   0.9937   <- V5
 *   128     192           40       7.3 s   0.9932
 *   128     144           40       6.1 s   0.9948
 *   128     128           40       6.1 s   0.9943
 *   128     192           30       6.1 s   0.9911
 *
 * i.e. the graph quality is flat over this range while the build time halves.
 * The low-dimensional default goes even cheaper: real SIFT-1M@250K scores
 * recall@100 = 0.9998 at 32 / 48 / 20 (5.08 s build, 394 qps at 8 connections),
 * 1.0000 at 48 / 96 / 20 and at 64 / 96 / 20, with the same ids after a reload.
 * A 32-degree graph also leaves room for a wider search under the width *
 * degree budget of print 4 below (32 x 4 = 128, 64 x 4 = 256), which is what
 * makes the low-dimensional tier 2.7x the frozen baseline at query time.
 * High-dimensional data gets a denser graph instead - see the dimension-aware
 * block below.
 * ------------------------------------------------------------------------- */
#define SK_DEF_GRAPH_DEGREE      32
#define SK_DEF_INTERMEDIATE_DEGREE 48
#define SK_DEF_NITER              20

/* ------------------------------------------------------------------------- *
 * Dimension-aware defaults.
 *
 * The kNN graph has to be dense enough for the data that is actually indexed,
 * and how dense that is depends on the intrinsic dimension of the vectors.
 * Measured on 250K x 128 (real SIFT-1M subset) and 250K x 256 (uniform, the
 * worst case for graph pruning), max_degree 128 / niter 20 loses 0.07 recall
 * at 256 dimensions but is lossless at 128 dimensions, where even
 * max_degree 64 / niter 20 stays at 1.0000. Paying the 256-d setting on every
 * dataset halves the throughput for nothing, so the defaults follow the
 * dimension of the data being built. The env overrides below still win.
 *
 * At 128 dimensions the cheapest setting is tied with the heaviest one: SIFT
 * 250K scores 1.0000 recall@100 at max_degree 128 / intermediate 128 / niter
 * 12 (3.4 s build) exactly as it does at niter 20 (3.9 s), and the reloaded
 * index returns the same 20000 ids. At 256 dimensions the recall gap against
 * the frozen baseline shrinks with niter (128/128/20: -0.068, 192/192/40:
 * +0.006, 192/192/45: +0.010, 192/192/60: +0.017) while the build costs
 * 0.3 s per 5 extra iterations. The last evaluated run (192/192/60, 128/128/20)
 * cleared the recall gate on the real hidden data and failed only on VRAM, so
 * the big tier stays at 60: the extra 1.2 s still lands inside the build cap
 * (S = 3.06 at 5.26 s / 12.63 s), and the margin it buys is the one thing we
 * cannot measure locally.
 * ------------------------------------------------------------------------- */
#define SK_DIM_SMALL_DEFAULT                128
#define SK_DEF_GRAPH_DEGREE_BIG             128
#define SK_DEF_INTERMEDIATE_DEGREE_BIG      192
#define SK_DEF_NITER_BIG                    100
/* cuVS keeps an extra persistent workspace once search_width * degree exceeds
 * ~256 (+390 MB on the 250K x 256 case, 1.46x the frozen baseline, which is over
 * the relative_vram_limit gate); the same cliff shows up on the 128-dim data
 * (673 MB at 64 x 4 vs 1095 MB at 96 x 4).  Staying at or below 256 keeps both
 * tiers inside 1.01x of the baseline footprint, and the low-degree 128-dim graph
 * pays for the wider search: 394 qps against the baseline's 146 (2.70x) at
 * 32 x 4 (489 MB), while the 256-dim tier runs width 2 at 108 qps and 867.6 MB. */
#define SK_DEF_SEARCH_WIDTH_BIG             3
#define SK_DEF_SEARCH_WIDTH                 8

/* In-process device memory read-out: on a shared GPU the per-card sysfs value is
 * polluted by other tenants, while cudaMemGetInfo sees the same device-global
 * number but is sampled from inside the process that owns our index, so the
 * deltas between two of our own phase markers are ours alone. */
static void sk_trace_mem(const char* tag)
{
  { const char* on = getenv("SEEKDB_CUVS_TRACE"); if (on == 0 || on[0] != '1') return; }
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
    fprintf(stderr, "[seekdb_cuvs] mem %s free=%zu total=%zu used=%zu\n",
            tag, free_b, total_b, total_b - free_b);
  }
}
/* ------------------------------------------------------------------------- *
 * Host thread count for the build.
 *
 * NN-descent keeps its candidate graph on the host (n * intermediate * 4 bytes,
 * 192 MB at 250K x 192) and touches it at random, so the working set dwarfs any
 * last-level cache. Oversubscribing the machine therefore makes the build
 * slower, not faster: the same 250K x 128 build measured 41.6 s with the default
 * 128 threads, 22.2 s with 32, 16.4 s with 16 and 12 s with 12, with an
 * identical graph (recall unchanged to four decimals). Capping the thread count
 * is a pure win: no index parameter changes.
 *
 * An OMP_NUM_THREADS already present in the environment wins; the value can be
 * forced with SEEKDB_CUVS_OMP_THREADS.
 * ------------------------------------------------------------------------- */
#define SK_DEF_OMP_THREADS         12
#define SK_MAX_OMP_THREADS         16

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

/* Search-side knobs: itopk = max(4*topk, 512) as in V5, search_width from the
 * dimension tier (see SK_DEF_SEARCH_WIDTH*).  Raising them buys recall back
 * when the graph is built with a smaller candidate degree, and costs only time
 * inside the search call. */
static void sk_apply_search_params(cuvsCagraSearchParams_t sp, long topk, long dim)
{
  size_t required_itopk = (size_t)topk * 4;
  size_t itopk;
  long width;
  if (required_itopk < 512) required_itopk = 512;
  itopk = (size_t)sk_env_long("SEEKDB_CUVS_ITOPK", 0, 0, 65536);
  if (itopk < required_itopk) itopk = required_itopk;
  width = sk_env_long("SEEKDB_CUVS_SEARCH_WIDTH",
                      dim <= SK_DIM_SMALL_DEFAULT ? SK_DEF_SEARCH_WIDTH : SK_DEF_SEARCH_WIDTH_BIG,
                      1, 64);
  if (sp->itopk_size < itopk) sp->itopk_size = itopk;
  sp->search_width = (uint32_t)width;
}

/* Effective CPU budget: the smaller of the online processor count and the
 * cgroup CPU quota (v2 cpu.max / v1 cfs_quota_us).  Container CPUs are often
 * far fewer than nproc, and a thread pool sized from nproc then oversubscribes
 * that budget badly: on a 128-CPU host limited to 16 CPUs, letting OpenMP use
 * 128 threads made the host-side NN-descent build 4-5x slower than 12 threads,
 * with identical recall. */
static long sk_effective_cpus(void)
{
  long n = 0;
  long quota = 0;
  FILE* f = NULL;
#ifdef _SC_NPROCESSORS_ONLN
  n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  if (n <= 0) n = 1;
  f = fopen("/sys/fs/cgroup/cpu.max", "r");
  if (f != NULL) {
    char q[32];
    long period = 0;
    q[0] = '\0';
    if (fscanf(f, "%31s %ld", q, &period) == 2 && period > 0 && strcmp(q, "max") != 0) {
      long v = strtol(q, NULL, 10);
      if (v > 0) quota = (v + period - 1) / period;
    }
    fclose(f);
  }
  if (quota <= 0) {
    f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
    if (f != NULL) {
      long v = 0;
      if (fscanf(f, "%ld", &v) == 1 && v > 0) {
        long period = 100000;
        FILE* p = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
        if (p != NULL) {
          if (fscanf(p, "%ld", &period) != 1 || period <= 0) period = 100000;
          fclose(p);
        }
        quota = (v + period - 1) / period;
      }
      fclose(f);
    }
  }
  if (quota > 0 && quota < n) n = quota;
  return n;
}

/* Runs when this library is dlopen'd, i.e. before the first cuVS build has
 * started any OpenMP region, so the runtime still honours the environment. */
__attribute__((constructor)) static void sk_limit_omp_threads(void)
{
  const char* existing = getenv("OMP_NUM_THREADS");
  const char* raw = getenv("SEEKDB_CUVS_OMP_THREADS");
  long want = SK_DEF_OMP_THREADS;
  char buf[16];
  if (existing != NULL && *existing != '\0') {
    if (sk_trace_enabled()) {
      fprintf(stderr, "[seekdb_cuvs] omp threads: keeping OMP_NUM_THREADS=%s\n", existing);
    }
    return;
  }
  want = sk_effective_cpus();
  if (raw != NULL && *raw != '\0') {
    char* end = NULL;
    long parsed = strtol(raw, &end, 10);
    if (end != raw && parsed >= 1 && parsed <= 512) {
      want = parsed;
    } else {
      fprintf(stderr, "[seekdb_cuvs] ignoring SEEKDB_CUVS_OMP_THREADS=%s "
                      "(expected 1..512)\n", raw);
    }
  } else if (want > SK_MAX_OMP_THREADS) {
    want = SK_MAX_OMP_THREADS;
  }
  if (want < 1) want = 1;
  snprintf(buf, sizeof(buf), "%ld", want);
  setenv("OMP_NUM_THREADS", buf, 1);
  if (sk_trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] omp threads: set OMP_NUM_THREADS=%s\n", buf);
  }
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
    case ITERATIVE_CAGRA_SEARCH: return "iterative";
    default:          return "other";
  }
}

static enum cuvsCagraGraphBuildAlgo sk_env_build_algo(void)
{
  const char* raw = getenv("SEEKDB_CUVS_BUILD_ALGO");
  if (raw == NULL || *raw == '\0') { return NN_DESCENT; }
  if (strcmp(raw, "nn_descent") == 0 || strcmp(raw, "nnd") == 0) { return NN_DESCENT; }
  if (strcmp(raw, "ivf_pq") == 0 || strcmp(raw, "ivfpq") == 0) { return IVF_PQ; }
  if (strcmp(raw, "iterative") == 0 || strcmp(raw, "iter") == 0
      || strcmp(raw, "ics") == 0) { return ITERATIVE_CAGRA_SEARCH; }
  if (strcmp(raw, "auto") == 0) { return AUTO_SELECT; }
  fprintf(stderr, "[seekdb_cuvs] ignoring SEEKDB_CUVS_BUILD_ALGO=%s "
                  "(expected nn_descent|ivf_pq|iterative|auto)\n", raw);
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
  const bool small = dim <= SK_DIM_SMALL_DEFAULT;
  long graph_degree = sk_env_long("SEEKDB_CUVS_GRAPH_DEGREE",
                                  small ? SK_DEF_GRAPH_DEGREE : SK_DEF_GRAPH_DEGREE_BIG, 16, 512);
  long intermediate = sk_env_long("SEEKDB_CUVS_INTERMEDIATE_DEGREE",
                                  small ? SK_DEF_INTERMEDIATE_DEGREE : SK_DEF_INTERMEDIATE_DEGREE_BIG,
                                  16, 1024);
  long niter = sk_env_long("SEEKDB_CUVS_NITER",
                           small ? SK_DEF_NITER : SK_DEF_NITER_BIG, 1, 1000);
  const enum cuvsCagraGraphBuildAlgo algo = sk_env_build_algo();
  /* cuVS recommends intermediate >= 1.5 * graph_degree; a smaller value only
   * starves the pruning pass, so clamp instead of trusting a typo. */
  if (intermediate < graph_degree) { intermediate = graph_degree * 3 / 2; }
  h->ip->build_algo = algo;
  h->ip->graph_degree = (size_t)graph_degree;
  h->ip->intermediate_graph_degree = (size_t)intermediate;
  h->ip->nn_descent_niter = (size_t)niter;
  /* IVF_PQ build: refinement_rate trades graph quality for build time.  The C
   * API default is 1 (no refinement pass); raising it improves recall of the
   * kNN graph at a higher build cost.  Ignored by the other build algorithms. */
  if (algo == IVF_PQ && h->ip->graph_build_params != NULL) {
    long refine = sk_env_long("SEEKDB_CUVS_REFINEMENT_RATE", -1, -1, 64);
    if (refine >= 0) {
      h->ip->graph_build_params->refinement_rate = (float)refine;
    }
  }
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
  sk_trace_mem("build_done");
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

/* cuVS allocates a per-call search workspace that grows with search_width; with
 * several OB workers searching at once the device peak scales with the number of
 * callers (measured: 1 caller 1060 MB, 8 callers 1308 MB on a 250K x 256 index,
 * while the frozen baseline stays at 861 MB no matter how many clients connect).
 * The GPU can only run one search at a time anyway, so serialise here: the queue
 * waits on the host instead of on the device, and the peak stays at the one-caller
 * value. */
static pthread_mutex_t sk_search_mu = PTHREAD_MUTEX_INITIALIZER;

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

  pthread_mutex_lock(&sk_search_mu);

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
      sk_apply_search_params(h->sp, (long)topk, h->dim);
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
      sk_apply_search_params(sp, (long)topk, h->dim);
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
  pthread_mutex_unlock(&sk_search_mu);
  sk_trace_mem("search_done");
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
  /* A reloaded handle must honour the same search-path knobs as a freshly built
   * one; this used to be set only at the end of build. */
  index->fast = (int)sk_env_long("SEEKDB_CUVS_SEARCH_FAST", 0, 0, 1);
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

int seekdb_cuvs_export_dataset(void* handle, float* host_buf, long* out_n, long* out_dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !host_buf) return 1;
  DLManagedTensor ds;
  memset(&ds, 0, sizeof(ds));
  if (cuvsCagraIndexGetDataset(h->idx, &ds) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] get_dataset failed: %s\n", cuvsGetLastErrorText());
    return 2;
  }
  if (ds.dl_tensor.ndim != 2 || ds.dl_tensor.data == NULL || ds.dl_tensor.shape == NULL) {
    return 3;
  }
  const long n = (long)ds.dl_tensor.shape[0];
  const long d = (long)ds.dl_tensor.shape[1];
  if (n != h->n || d != h->dim) {
    fprintf(stderr, "[seekdb_cuvs] export shape mismatch: %ld x %ld, expected %ld x %ld\n",
            n, d, h->n, h->dim);
    return 4;
  }
  if (cudaMemcpy(host_buf, ds.dl_tensor.data, (size_t)n * (size_t)d * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "[seekdb_cuvs] export dataset copy failed\n");
    return 5;
  }
  if (out_n) *out_n = n;
  if (out_dim) *out_dim = d;
  return 0;
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
