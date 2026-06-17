# SISAP2026 Indexing Challenge — Task 3 (Sparse Indexing) with kANNolo

How to use `sisap_task3_build` and `sisap_task3_search` to build an HNSW index over the SPLADE-v3 sparse embeddings and produce result files in the format required by the challenge. See `SISAP_cfp.md` / `SISAP_README.md` for the full task description and dataset layout.

## Prerequisites

- `RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli` — the `sisap` feature is part of `default` and produces `target/release/sisap_task3_build`/`sisap_task3_search`; `--features cli` additionally builds the `hnsw_build`/`hnsw_search` bins (useful as a baseline/comparison). `target-cpu=native` is important for search performance (SIMD distance kernels).
  Requires `libhdf5-dev` (the `hdf5` crate links against it).
- Data lives under `/data3/silvio/sisap2026/`:
  - `datasets/<name>/<name>.h5` (e.g. `datasets/fiqa-dev/fiqa-dev.h5`, `datasets/nq/nq.h5`) — each contains:
    - `train`: document collection as a CSR sparse matrix (`data`/`indices`/`indptr` + `shape` attribute).
    - `otest/queries`: query set, same CSR layout.
    - `otest/knns` / `otest/dists`: gold-standard neighbors (1-based ids) and distances, used for development-time recall checks.
  - `indexes/` — where built indexes are saved/loaded.

## `sisap_task3_build`

Reads the `train` CSR group directly from HDF5 (no intermediate conversion) and builds an HNSW index.

```bash
./target/release/sisap_task3_build \
  --h5-file /data3/silvio/sisap2026/datasets/fiqa-dev/fiqa-dev.h5 \
  --group train \
  --output-file /data3/silvio/sisap2026/indexes/fiqa-dev.hnsw \
  --m 16 --ef-construction 150
```

| Flag | Default | Meaning |
|---|---|---|
| `--h5-file` | required | Dataset HDF5 file. |
| `--group` | `train` | HDF5 group containing the CSR matrix to index. |
| `--output-file`, `-o` | required | Path for the serialized index. |
| `--m` | 16 | HNSW `M`. |
| `--ef-construction` | 150 | HNSW `ef_construction`. |
| `--reorder-egb` | off | After building, reorder the ground level + dataset via EGB (recursive graph bisection) for better cache locality. ~2-3% faster search at the same recall, for a small one-time cost (~1 minute on NQ). |

**Produces:**
- `<output-file>` — serialized HNSW index (bincode, `PlainSparseDataset<u16, f16, DotProduct>` — component ids narrowed to `u16`, values to `f16`).
- `<output-file>.buildtime` — sidecar text file with load+build wall-clock time (seconds, `f64`), consumed by `sisap_task3_search` for the `buildtime` attr.
- `<output-file>.permutation` — only if `--reorder-egb` was used: binary sidecar (`n` little-endian `u64`s) mapping reordered ids back to original doc ids. `sisap_task3_search` loads it automatically (if present) and remaps result ids before writing output, so this is transparent to callers.

## `sisap_task3_search`

Loads the index, reads the query CSR group, runs a parallel batched search for each `ef_search` value, and writes one result HDF5 file per value.

```bash
./target/release/sisap_task3_search \
  --h5-file /data3/silvio/sisap2026/datasets/fiqa-dev/fiqa-dev.h5 \
  --query-group otest/queries \
  --index-file /data3/silvio/sisap2026/indexes/fiqa-dev.hnsw \
  -k 30 --ef-search 50,100,200 \
  --algo-name kannolo-hnsw --output-dir results/task3 \
  --m 16 --ef-construction 150
```

| Flag | Default | Meaning |
|---|---|---|
| `--h5-file` | required | Dataset HDF5 file (used for queries). |
| `--query-group` | `otest/queries` | HDF5 group with CSR query vectors. |
| `--index-file`, `-i` | required | Index produced by `sisap_task3_build`. |
| `-k` | 30 | Neighbors per query (challenge uses `k=30`). |
| `--ef-search` | `100` | Comma-separated list — one result file per value, for sweeping the (≤15) allowed search configs. |
| `--algo-name` | `kannolo-hnsw` | Used in `algo` attr and output filenames. |
| `--output-dir` | `results/task3` | Output directory (created if missing). |
| `--m`, `--ef-construction` | 16, 150 | Must match the build — only used to fill the `params` attr/filename. |

**Produces**, for each `ef_search` value `E`: `<output-dir>/<algo-name>_M<m>_efC<ef-construction>_efS<E>.h5`, containing:

- `knns`: `n_queries x k`, `int64`, **1-based** ids (row-major).
- `dists`: `n_queries x k`, `float32` (dot product, higher = better — matches the gold standard's semantics, no sign flip).
- Root attrs: `algo`, `task="task3"`, `buildtime` (from the sidecar, or `0.0`), `querytime` (wall-clock search time — what task 3 scores), `params` (e.g. `"M=16,efConstruction=150,efSearch=100"`).

Missing results (shouldn't normally happen) are padded with id `-1` / `+inf`.

**Console output** per `ef_search`:
```
ef_search=<E>: avg_query_time=<us> us, recall@<kk>=<r>, querytime=<s>s -> <path>
```
- `avg_query_time` = `querytime / n_queries` (parallel-batch wall clock, so a relative indicator rather than a strict sequential per-query time).
- `recall@<kk>` is computed against the gold standard at `<parent of --query-group>/knns` (`kk = min(k, k_gold)`). If absent (e.g. on the real evaluation dataset), only `avg_query_time`/`querytime` are printed.

## Workflow

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli

./target/release/sisap_task3_build \
  --h5-file /data3/silvio/sisap2026/datasets/<name>/<name>.h5 \
  --group train \
  --output-file /data3/silvio/sisap2026/indexes/<name>.hnsw \
  --m <M> --ef-construction <efC> --reorder-egb

./target/release/sisap_task3_search \
  --h5-file /data3/silvio/sisap2026/datasets/<name>/<name>.h5 \
  --query-group otest/queries \
  --index-file /data3/silvio/sisap2026/indexes/<name>.hnsw \
  -k 30 --ef-search <comma-separated values, up to 15> \
  --algo-name kannolo-hnsw --output-dir results/task3 \
  --m <M> --ef-construction <efC>
```

Use `<name> = fiqa-dev` for development (tune `--m`/`--ef-construction` and the `--ef-search` sweep against `recall@30`/`querytime`), then `<name> = nq` for the real submission.

Constraints (`SISAP_cfp.md`): 8 vCPUs, 24 GB RAM, 8h wall-clock total (build included, but **only `querytime` is scored**); single index + up to 15 search-parameter configs; submit the `results/task3/*.h5` files.

## Docker submission

`Dockerfile` + `entrypoint.sh` implement the `docker run ... sisap-baseline --task task3 --dataset <name>` interface from `SISAP_cfp.md` (8 vCPUs, 24GB RAM, `/app/data:ro`, `/app/results:rw`).

**Key design point — compile at container *start*, not at image build time:**
`-C target-cpu=native` must target the CPU that will actually *run* the search (the challenge's AMD EPYC 7F72, Zen 2 — no AVX-512), which may differ from whatever machine runs `docker build`. So:

- **Image build** (`Dockerfile`): installs the pinned nightly toolchain (`rust-toolchain.toml`) and `libhdf5-dev`, then runs `cargo fetch --locked` to pre-download every dependency (including the git `rgb` crate) into the image. No compilation happens here.
- **Container start** (`entrypoint.sh`): runs `RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli,multivec --offline` (fully offline thanks to the pre-fetch — fast, ~tens of seconds to a few minutes, negligible against the 8h budget), then builds a two-stage rerank index (`sisap_task3_rerank_build`, L1 fraction 0.75, M=32, efC=1000) and runs the 15 tuned `(kC, ef_search, lambda)` configs via `sisap_task3_rerank_search`, writing to `/app/results/task3/`.

This guarantees `native` always matches the actual evaluation hardware, with no risk of an illegal-instruction (SIGILL) crash from CPU-feature mismatches, regardless of where the image is built.

**Build and test:**
```bash
docker build -t kannolo-sisap .

docker run --rm \
  --cpus=8 --memory=24g --memory-swap=24g --memory-swappiness 0 \
  -v /data3/silvio/sisap2026/datasets:/app/data:ro \
  -v /path/to/results:/app/results:rw \
  kannolo-sisap --task task3 --dataset fiqa-dev   # or --dataset nq
```

Verified end-to-end on `fiqa-dev` under these exact resource limits: all 15 result files written with correct `algo`/`buildtime`/`params` attrs, and the in-container compile correctly detects the host's SIMD features (confirmed via `objdump` — AVX-512 present on this dev machine, would be AVX2-only on the EPYC 7F72).

**Submission approach — two-stage rerank index:**
`sisap_task3_rerank_build` builds an HNSW over L1-fraction-pruned (75%) sparse vectors as a fast first stage, plus saves the full-precision dataset for reranking. At search time, `sisap_task3_rerank_search` retrieves candidates from the pruned first stage and reranks with exact dot products. The 15 configs sweep `(kC, ef_search, lambda)` covering target recalls 0.895–0.970, measured 16–95% faster than the EGB+ET baseline in sequential benchmarks.

Full-scale (`nq`) build timing was verified at 8 cores/24GB (via `systemd-run --scope -p CPUQuota=800% -p MemoryMax=24G` cgroup): rerank index build ~34 min (peak RSS ~11GB), well within the 8h budget.
