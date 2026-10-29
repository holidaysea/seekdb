/* Stage benchmark for the seekdb <-> hipVS CAGRA path.
 *
 * Purpose: on a GPU node, price every stage that sits inside the graded
 * "CREATE VECTOR INDEX .. lib=cuvs -> first query returns" window, WITHOUT
 * building seekdb. It links against the very same libseekdb_cuvs_bridge.so the
 * database loads, so the numbers transfer.
 *
 * Build knobs are read by the bridge from the environment, which lets one
 * binary sweep the parameter matrix:
 *   SEEKDB_CUVS_GRAPH_DEGREE / SEEKDB_CUVS_INTERMEDIATE_DEGREE / SEEKDB_CUVS_NITER
 *   SEEKDB_CUVS_TRACE=1        stage timings on stderr
 *   SEEKDB_CUVS_TRACE_SEARCH=1 per-search timings on stderr
 *
 * Usage:
 *   bench_cagra_stages --n 250000 --dim 128 --topk 10 --workdir /tmp/cagra \
 *                      [--data base.f32] [--clusters 256] [--queries 100]
 *                      [--latency-iters 200] [--no-recall] [--skip-slim] [--csv out.csv]
 *
 * --data takes a raw little-endian float32 file of n*dim rows; without it the
 * dataset is synthetic (clustered by default, which is closer to SIFT than
 * uniform noise for recall purposes).
 */
#include "seekdb_cuvs_bridge.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
  long n, dim, queries, topk, latency_iters, clusters;
  const char *data_path, *workdir, *tag, *csv_path;
  int recall, slim;
} Options;

static double now_us(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* user+sys CPU microseconds of this process: delta/wall gives the average
 * number of cores busy, which tells CPU-bound stages (host graph maintenance,
 * snapshot writes) from GPU-bound ones. */
static double cpu_us(void)
{
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) != 0) { return 0.0; }
  return (double)ru.ru_utime.tv_sec * 1e6 + (double)ru.ru_utime.tv_usec +
         (double)ru.ru_stime.tv_sec * 1e6 + (double)ru.ru_stime.tv_usec;
}

static void die(const char *msg)
{
  fprintf(stderr, "bench_cagra_stages: %s\n", msg);
  exit(2);
}

static void *xmalloc(size_t bytes)
{
  void *p = malloc(bytes ? bytes : 1);
  if (p == NULL) { die("out of memory"); }
  return p;
}

static long file_bytes(const char *path)
{
  struct stat st;
  if (path == NULL || stat(path, &st) != 0) { return -1; }
  return (long)st.st_size;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

static double rng_unit(void) /* [0,1) */
{
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)(rng_state >> 11) / 9007199254740992.0;
}

static void gen_dataset(float *base, long n, long dim, long clusters, uint64_t seed)
{
  long i, j, c;
  float *centers = NULL;
  rng_state = seed;
  if (clusters > 0) {
    centers = (float *)xmalloc((size_t)clusters * (size_t)dim * sizeof(float));
    for (c = 0; c < clusters * dim; ++c) { centers[c] = (float)(rng_unit() * 2.0 - 1.0); }
  }
  for (i = 0; i < n; ++i) {
    const float *center = NULL;
    if (clusters > 0) { center = centers + (long)(rng_unit() * (double)clusters) * dim; }
    for (j = 0; j < dim; ++j) {
      const double noise = (rng_unit() * 2.0 - 1.0) * (clusters > 0 ? 0.12 : 1.0);
      base[i * dim + j] = (float)((center ? (double)center[j] : 0.0) + noise);
    }
  }
  free(centers);
}

static int load_dataset(const char *path, float **out, long n, long dim)
{
  FILE *fp = fopen(path, "rb");
  long bytes = file_bytes(path);
  if (fp == NULL) { return -1; }
  if (bytes != n * dim * (long)sizeof(float)) {
    fprintf(stderr, "bench_cagra_stages: %s is %ld bytes, expected %ld (n=%ld dim=%ld f32)\n",
            path, bytes, n * dim * (long)sizeof(float), n, dim);
    fclose(fp);
    return -1;
  }
  *out = (float *)xmalloc((size_t)n * (size_t)dim * sizeof(float));
  if (fread(*out, sizeof(float), (size_t)n * (size_t)dim, fp) != (size_t)n * (size_t)dim) {
    fclose(fp);
    return -1;
  }
  fclose(fp);
  return 0;
}

/* Exact top-k by CPU brute force. Used to score recall of the GPU graph; with
 * nq sampled queries over 250K rows this is a couple of seconds on any server. */
static void exact_knn(const float *base, long n, long dim, const float *queries,
                      long nq, long k, unsigned int *out_ids, float *out_dist)
{
  long q;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (q = 0; q < nq; ++q) {
    const float *query = queries + q * dim;
    unsigned int *ids = out_ids + q * k;
    float *dst = out_dist + q * k;
    long i, m;
    for (m = 0; m < k; ++m) { ids[m] = 0xFFFFFFFFu; dst[m] = 3.4e38f; }
    for (i = 0; i < n; ++i) {
      const float *row = base + i * dim;
      double acc = 0.0;
      long j;
      float d;
      for (j = 0; j < dim; ++j) {
        const double diff = (double)row[j] - (double)query[j];
        acc += diff * diff;
      }
      d = (float)acc;
      if (d < dst[k - 1]) {
        long pos = k - 1;
        while (pos > 0 && dst[pos - 1] > d) { dst[pos] = dst[pos - 1]; ids[pos] = ids[pos - 1]; --pos; }
        dst[pos] = d;
        ids[pos] = (unsigned int)i;
      }
    }
  }
}

static double recall_at_k(const unsigned int *got, const unsigned int *want, long nq, long k)
{
  long q, i, j, hits = 0;
  for (q = 0; q < nq; ++q) {
    for (i = 0; i < k; ++i) {
      const unsigned int id = got[q * k + i];
      if (id == 0xFFFFFFFFu) { continue; }
      for (j = 0; j < k; ++j) {
        if (want[q * k + j] == id) { ++hits; break; }
      }
    }
  }
  return nq > 0 && k > 0 ? (double)hits / (double)(nq * k) : 0.0;
}

static int cmp_double(const void *a, const void *b)
{
  const double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : (x > y ? 1 : 0);
}

static const char *env_or(const char *name, const char *fallback)
{
  const char *v = getenv(name);
  return (v != NULL && *v != '\0') ? v : fallback;
}

static void usage(void)
{
  fprintf(stderr,
          "usage: bench_cagra_stages [--n 250000] [--dim 128] [--topk 10]\n"
          "       [--queries 100] [--latency-iters 200] [--clusters 256]\n"
          "       [--data base.f32] [--workdir /tmp] [--tag name] [--csv out.csv]\n"
          "       [--no-recall] [--skip-slim]\n");
}

static void parse_args(int argc, char **argv, Options *o)
{
  int i;
  o->n = 250000; o->dim = 128; o->queries = 100; o->topk = 10;
  o->latency_iters = 200; o->clusters = 256;
  o->data_path = NULL; o->workdir = "/tmp"; o->tag = "cagra"; o->csv_path = NULL;
  o->recall = 1; o->slim = 1;
  for (i = 1; i < argc; ++i) {
    const char *a = argv[i];
    const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
    if (strcmp(a, "--n") == 0 && v) { o->n = atol(v); ++i; }
    else if (strcmp(a, "--dim") == 0 && v) { o->dim = atol(v); ++i; }
    else if (strcmp(a, "--queries") == 0 && v) { o->queries = atol(v); ++i; }
    else if (strcmp(a, "--topk") == 0 && v) { o->topk = atol(v); ++i; }
    else if (strcmp(a, "--latency-iters") == 0 && v) { o->latency_iters = atol(v); ++i; }
    else if (strcmp(a, "--clusters") == 0 && v) { o->clusters = atol(v); ++i; }
    else if (strcmp(a, "--data") == 0 && v) { o->data_path = v; ++i; }
    else if (strcmp(a, "--workdir") == 0 && v) { o->workdir = v; ++i; }
    else if (strcmp(a, "--tag") == 0 && v) { o->tag = v; ++i; }
    else if (strcmp(a, "--csv") == 0 && v) { o->csv_path = v; ++i; }
    else if (strcmp(a, "--no-recall") == 0) { o->recall = 0; }
    else if (strcmp(a, "--skip-slim") == 0) { o->slim = 0; }
    else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) { usage(); exit(0); }
    else { usage(); die("unknown or incomplete argument"); }
  }
  if (o->n < 1000 || o->dim < 8 || o->topk < 1 || o->queries < 1) { die("bad numeric argument"); }
}

int main(int argc, char **argv)
{
  Options opt;
  float *base = NULL, *queries = NULL;
  unsigned int *gpu_ids = NULL, *gpu_ids2 = NULL, *exact_ids = NULL;
  float *gpu_dist = NULL, *exact_dist = NULL;
  void *handle = NULL, *reloaded = NULL;
  char full_path[1024], slim_path[1024];
  double t0, wall_build, cores_build, wall_full, wall_slim, wall_load;
  long full_bytes, slim_bytes;
  long q, i;
  double recall = -1.0, recall_reload = -1.0;
  double *lat = NULL, lat_sum = 0.0, lat_p50 = 0.0, lat_p95 = 0.0;
  int rc;

  parse_args(argc, argv, &opt);
  snprintf(full_path, sizeof(full_path), "%s/%s.full.snapshot", opt.workdir, opt.tag);
  snprintf(slim_path, sizeof(slim_path), "%s/%s.slim.snapshot", opt.workdir, opt.tag);

  if (opt.data_path != NULL) {
    if (load_dataset(opt.data_path, &base, opt.n, opt.dim) != 0) { die("cannot load --data"); }
  } else {
    base = (float *)xmalloc((size_t)opt.n * (size_t)opt.dim * sizeof(float));
    gen_dataset(base, opt.n, opt.dim, opt.clusters, 0xC0FFEEULL);
  }
  /* Queries are rows of the base set (perturbed only by a tiny offset) so that
   * recall has a well defined ground truth even for synthetic data. */
  queries = (float *)xmalloc((size_t)opt.queries * (size_t)opt.dim * sizeof(float));
  for (q = 0; q < opt.queries; ++q) {
    const long row = (long)(rng_unit() * (double)opt.n) % opt.n;
    for (i = 0; i < opt.dim; ++i) {
      queries[q * opt.dim + i] = base[row * opt.dim + i] + (float)((rng_unit() - 0.5) * 0.002);
    }
  }
  fprintf(stderr, "bench: n=%ld dim=%ld queries=%ld topk=%ld data=%s clusters=%ld\n",
          opt.n, opt.dim, opt.queries, opt.topk,
          opt.data_path ? opt.data_path : "synthetic", opt.clusters);

  gpu_ids = (unsigned int *)xmalloc((size_t)opt.queries * (size_t)opt.topk * sizeof(unsigned int));
  gpu_ids2 = (unsigned int *)xmalloc((size_t)opt.queries * (size_t)opt.topk * sizeof(unsigned int));
  gpu_dist = (float *)xmalloc((size_t)opt.queries * (size_t)opt.topk * sizeof(float));

  /* ---- stage 1: build ---- */
  t0 = now_us();
  {
    const double c0 = cpu_us();
    handle = seekdb_cuvs_build(base, opt.n, opt.dim);
    if (handle == NULL) {
      fprintf(stderr, "bench: seekdb_cuvs_build failed\n");
      return 3;
    }
    wall_build = now_us() - t0;
    cores_build = wall_build > 0.0 ? (cpu_us() - c0) / wall_build : 0.0;
  }
  fprintf(stderr, "bench: build wall=%.1f ms cpu=%.2f cores (=> %s bound)\n",
          wall_build / 1000.0, cores_build, cores_build > 2.0 ? "CPU" : "GPU/IO");

  /* ---- stage 2: single-query latency ---- */
  lat = (double *)xmalloc((size_t)opt.latency_iters * sizeof(double));
  for (q = 0; q < opt.latency_iters; ++q) {
    const long qi = q % opt.queries;
    const double lt0 = now_us();
    rc = seekdb_cuvs_search(handle, queries + qi * opt.dim, 1, opt.topk, gpu_ids, gpu_dist);
    lat[q] = now_us() - lt0;
    if (rc != 0) { fprintf(stderr, "bench: single search rc=%d\n", rc); return 4; }
  }
  for (q = 0; q < opt.latency_iters; ++q) { lat_sum += lat[q]; }
  qsort(lat, (size_t)opt.latency_iters, sizeof(double), cmp_double);
  lat_p50 = lat[opt.latency_iters / 2];
  lat_p95 = lat[(opt.latency_iters * 95) / 100];

  /* ---- stage 3: batch QPS (one call, all queries) ---- */
  {
    const double b0 = now_us();
    const int repeats = 5;
    double batch_us;
    rc = seekdb_cuvs_search(handle, queries, opt.queries, opt.topk, gpu_ids, gpu_dist);
    if (rc != 0) { fprintf(stderr, "bench: batch search rc=%d\n", rc); return 5; }
    batch_us = now_us() - b0;
    fprintf(stderr, "bench: batch nq=%ld topk=%ld %.2f ms (%.0f qps, 1st call)\n",
            opt.queries, opt.topk, batch_us / 1000.0,
            batch_us > 0.0 ? (double)opt.queries * 1e6 / batch_us : 0.0);
    {
      const double r0 = now_us();
      int r;
      for (r = 0; r < repeats; ++r) {
        if (seekdb_cuvs_search(handle, queries, opt.queries, opt.topk, gpu_ids, gpu_dist) != 0) {
          fprintf(stderr, "bench: repeat batch search failed\n");
          return 5;
        }
      }
      batch_us = (now_us() - r0) / repeats;
    }
    fprintf(stderr, "bench: batch nq=%ld topk=%ld %.2f ms (%.0f qps, steady)\n",
            opt.queries, opt.topk, batch_us / 1000.0,
            batch_us > 0.0 ? (double)opt.queries * 1e6 / batch_us : 0.0);
  }

  /* ---- stage 4: recall vs exact ---- */
  if (opt.recall) {
    exact_ids = (unsigned int *)xmalloc((size_t)opt.queries * (size_t)opt.topk * sizeof(unsigned int));
    exact_dist = (float *)xmalloc((size_t)opt.queries * (size_t)opt.topk * sizeof(float));
    exact_knn(base, opt.n, opt.dim, queries, opt.queries, opt.topk, exact_ids, exact_dist);
    recall = recall_at_k(gpu_ids, exact_ids, opt.queries, opt.topk);
    fprintf(stderr, "bench: recall@%ld = %.4f\n", opt.topk, recall);
  }

  /* ---- stage 5: snapshot write (production format + graph-only) ---- */
  t0 = now_us();
  if (seekdb_cuvs_serialize(handle, full_path) != 0) {
    fprintf(stderr, "bench: seekdb_cuvs_serialize failed\n");
    return 6;
  }
  wall_full = now_us() - t0;
  full_bytes = file_bytes(full_path);
  if (opt.slim) {
    t0 = now_us();
    if (seekdb_cuvs_bench_serialize(handle, slim_path, 0) != 0) {
      fprintf(stderr, "bench: bench_serialize(graph-only) failed\n");
      return 7;
    }
    wall_slim = now_us() - t0;
    slim_bytes = file_bytes(slim_path);
  } else {
    wall_slim = -1.0;
    slim_bytes = -1;
  }
  fprintf(stderr, "bench: snapshot full=%.1f ms %ld B (%.0f MB/s) | graph-only=%.1f ms %ld B\n",
          wall_full / 1000.0, full_bytes,
          wall_full > 0.0 ? (double)full_bytes / 1048576.0 / (wall_full / 1e6) : 0.0,
          wall_slim / 1000.0, slim_bytes);
  fprintf(stderr, "bench: graph_degree=%s intermediate=%s niter=%s\n",
          env_or("SEEKDB_CUVS_GRAPH_DEGREE", "128"),
          env_or("SEEKDB_CUVS_INTERMEDIATE_DEGREE", "256"),
          env_or("SEEKDB_CUVS_NITER", "100"));

  /* ---- stage 6: snapshot reload + consistency ---- */
  seekdb_cuvs_free(handle);
  handle = NULL;
  t0 = now_us();
  reloaded = seekdb_cuvs_deserialize(full_path, opt.n, opt.dim);
  wall_load = now_us() - t0;
  if (reloaded == NULL) {
    fprintf(stderr, "bench: deserialize failed\n");
    return 8;
  }
  rc = seekdb_cuvs_search(reloaded, queries, opt.queries, opt.topk, gpu_ids2, gpu_dist);
  if (rc != 0) { fprintf(stderr, "bench: reload search rc=%d\n", rc); return 9; }
  if (opt.recall && exact_ids != NULL) {
    recall_reload = recall_at_k(gpu_ids2, exact_ids, opt.queries, opt.topk);
  }
  {
    long mismatched = 0;
    for (q = 0; q < opt.queries * opt.topk; ++q) {
      if (gpu_ids[q] != gpu_ids2[q]) { ++mismatched; }
    }
    fprintf(stderr, "bench: reload=%.1f ms (%.0f MB/s) id-mismatch=%ld/%ld recall@%ld=%.4f\n",
            wall_load / 1000.0, wall_load > 0.0 ? (double)full_bytes / 1048576.0 / (wall_load / 1e6) : 0.0,
            mismatched, opt.queries * opt.topk, opt.topk, recall_reload);
  }
  seekdb_cuvs_free(reloaded);

  /* ---- machine readable row ---- */
  {
    const long total_bytes = full_bytes > 0 ? full_bytes : 0;
    FILE *csv = (opt.csv_path != NULL) ? fopen(opt.csv_path, "a") : NULL;
    FILE *sink = csv != NULL ? csv : stdout;
    if (csv != NULL && file_bytes(opt.csv_path) <= 0) {
      fprintf(csv, "tag,n,dim,topk,queries,graph_degree,intermediate,niter,"
                   "build_ms,build_cores,lat_p50_us,lat_p95_us,recall,"
                   "snapshot_ms,snapshot_bytes,snapshot_slim_ms,snapshot_slim_bytes,"
                   "reload_ms,reload_recall\n");
    }
    fprintf(sink, "%s,%ld,%ld,%ld,%ld,%s,%s,%s,%.1f,%.2f,%.1f,%.1f,%.4f,%.1f,%ld,%.1f,%ld,%.1f,%.4f\n",
            opt.tag, opt.n, opt.dim, opt.topk, opt.queries,
            env_or("SEEKDB_CUVS_GRAPH_DEGREE", "128"),
            env_or("SEEKDB_CUVS_INTERMEDIATE_DEGREE", "256"),
            env_or("SEEKDB_CUVS_NITER", "100"),
            wall_build / 1000.0, cores_build, lat_p50, lat_p95, recall,
            wall_full / 1000.0, total_bytes, wall_slim / 1000.0, slim_bytes,
            wall_load / 1000.0, recall_reload);
    if (csv != NULL) { fclose(csv); }
  }

  free(exact_dist);
  free(exact_ids);
  free(gpu_dist);
  free(gpu_ids2);
  free(gpu_ids);
  free(lat);
  free(queries);
  free(base);


  return 0;
}
