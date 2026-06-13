---
license: mit
---

# SISAP2026 Indexing Challenge - Development Datasets

This repository contains the development data files used in the SISAP2026 indexing challenge.

Datasets for previous editions:
- <https://huggingface.co/datasets/sadit/SISAP2025>

## Repository Structure

The datasets are organized into subdirectories by dataset type:

```
SISAP2026/
├── wikipedia/              # Large Wikipedia dataset
│   ├── benchmark-dev-wikipedia-bge-m3.h5
│   └── config.json
├── wikipedia-small/        # Small Wikipedia dataset for testing
│   ├── benchmark-dev-wikipedia-bge-m3-small.h5
│   └── config.json
├── llama-dev/              # Llama embeddings
│   ├── llama-dev.h5
│   └── config.json
├── nq/                     # Natural Questions (sparse)
│   ├── nq.h5
│   └── config.json
├── fiqa-dev/               # Financial QA (sparse)
│   ├── fiqa-dev.h5
│   └── config.json
├── task-1-spot-check/      # Task 1 validation dataset
│   ├── benchmark-dev-gooaq-small.h5
│   └── config.json
├── task-2-spot-check/      # Task 2 validation dataset
│   ├── benchmark-dev-llama-small.h5
│   └── config.json
└── task-3-spot-check/      # Task 3 validation dataset
    ├── benchmark-dev-fiqa-small.h5
    └── config.json
```

## Downloading Files

### Download a specific file

```python
from huggingface_hub import hf_hub_download

# Download a specific dataset
file_path = hf_hub_download(
    repo_id="SISAP-Challenges/SISAP2026",
    filename="wikipedia/benchmark-dev-wikipedia-bge-m3.h5",
    repo_type="dataset"
)

# Download a config file
config_path = hf_hub_download(
    repo_id="SISAP-Challenges/SISAP2026",
    filename="wikipedia/config.json",
    repo_type="dataset"
)
```

### Download entire subdirectory

```python
from huggingface_hub import snapshot_download

# Download all files from a specific dataset
local_dir = snapshot_download(
    repo_id="SISAP-Challenges/SISAP2026",
    repo_type="dataset",
    allow_patterns="wikipedia/*"
)
```

## Datasets

### WIKIPEDIA (English articles)

**Location:** `wikipedia/benchmark-dev-wikipedia-bge-m3.h5`

- **Source repo:** <https://huggingface.co/datasets/wikimedia/wikipedia>
- **Model:** BGE-m3 - <https://huggingface.co/BAAI/bge-m3>
- **Similarity:** Cosine / dot product
- **Content of the h5 file:**
  - dataset `train`: a 6.35 million vector database, i.e., a matrix of size $1024 \times 6350000$ (f16)
  - group `itrain`: collection of data related to in-distribution queries (articles removed from the English Wikipedia corpus):
     - `itest/queries`: a 10'000 vector database, i.e., a matrix of size $1024 \times 10000$ (f16)
     - `itest/knns`: the gold-standard identifiers for the 1000 nearest neighbors of `itest/queries` in `train`, i.e., a matrix $1000 \times 10000$ (i32).
     - `itest/dists`: the gold-standard distances (1-dot) for the 1000 nearest neighbors of `itest/queries` in `train`, i.e., a matrix $1000 \times 10000$ (f32).
  - group `otrain`: collection of data related to out-of-distribution queries (same model in random articles from the Spanish Wikipedia, i.e., cross-lingual retrieval):
     - `otest/queries`: a 10'000 vector database, i.e., a matrix of size $1024 \times 10000$ (f16)
     - `otest/knns`: the gold-standard identifiers for the 1000 nearest neighbors of `itest/queries` in `train`, i.e., a matrix $1000 \times 10000$ (i32).
     - `otest/dists`: the gold-standard distances (1-dot) for the 1000 nearest neighbors of `itest/queries` in `train`, i.e., a matrix $1000 \times 10000$ (f32).
  - group `allknn`:
     - `allknn/knns`: the gold-standard identifiers for the all-knn graph of `train` i.e., a matrix $32 \times 6350000$ (i32).
     - `allknn/dists`: the gold-standard distances (1-dot) for the all-knn graph of `train` i.e., a matrix $32 \times 6350000$ (f32).

### WIKIPEDIA Small (English articles)

**Location:** `wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5`

- This is small version of WIKIPEDIA database for testing and developing purposes
- The `train` dataset is a 200k vector database
- Same structure as the full WIKIPEDIA dataset

### LLAMA (Llama-3-8B-262k)

**Location:** `llama-dev/llama-dev.h5`

- **Source repo:** <https://huggingface.co/datasets/vector-index-bench/vibe>
- **Model:** Llama-3.2-8B
- **Similarity:** Dot product (vectors are not normalized)
- **Content of the h5 file:**
  - dataset `train`: a 256k vector database, i.e., a matrix of size $128 \times 256921$ (f32)
  - group `test`: collection of development queries:
     - `test/queries`: a 1'000 vector database, i.e., a matrix of size $128 \times 1000$ (f32)
     - `test/knns`: the gold-standard identifiers for the 100 nearest neighbors of `test/queries` in `train`, i.e., a matrix $100 \times 1000$ (i64).
     - `test/dists`: the gold-standard distances (dot product) for the 100 nearest neighbors of `test/queries` in `train`, i.e., a matrix $100 \times 1000$ (f64).

### NQ (Natural Questions)

**Location:** `nq/nq.h5`

- **Source repo:** <https://github.com/beir-cellar/beir>
- **Model:** SPLADE-v3 (sparse embeddings)
- **Similarity:** Dot product (vectors are not normalized)
- **Content of the h5 file:**
  - group `train`: a 2.68 million sparse vector database, i.e., a sparse matrix (CSR) of size $30522 \times 2681468$ (f32). It contains `data`, `indices`, `indptr` datasets and a `shape` attribute.
  - group `otest`: collection of development queries:
     - `otest/queries`: 3452 query embeddings, i.e., a sparse matrix (CSR) of size $30522 \times 3452$ (f32). It contains `data`, `indices`, `indptr` datasets and a `shape` attribute.
     - `otest/knns`: the gold-standard identifiers for the 100 nearest neighbors of `otest/queries` in `train`, i.e., a matrix $100 \times 3452$ (i32).
     - `otest/dists`: the gold-standard distances (dot product) for the 100 nearest neighbors of `otest/queries` in `train`, i.e., a matrix $100 \times 3452$ (f32).
  - See example below to know how to work with the file

### FIQA (Financial Question Answering)

**Location:** `fiqa-dev/fiqa-dev.h5`

- **Source repo:** <https://github.com/beir-cellar/beir>
- **Model:** SPLADE-v3 (sparse embeddings)
- **Similarity:** Dot product (vectors are not normalized)
- **Content of the h5 file:**
  - group `train`: a 57k sparse vector database, i.e., a sparse matrix (CSR) of size $30522 \times 57638$ (f32). It contains `data`, `indices`, `indptr` datasets and a `shape` attribute.
  - group `otest`: collection of development queries:
     - `otest/queries`: 6648 query embeddings, i.e., a sparse matrix (CSR) of size $30522 \times 6648$ (f32). It contains `data`, `indices`, `indptr` datasets and a `shape` attribute.
     - `otest/knns`: the gold-standard identifiers for the 100 nearest neighbors of `otest/queries` in `train`, i.e., a matrix $100 \times 6648$ (i32).
     - `otest/dists`: the gold-standard distances (dot product) for the 100 nearest neighbors of `otest/queries` in `train`, i.e., a matrix $100 \times 6648$ (f32).
  - See example below to know how to work with the file

### Spot-Check Datasets

**Locations:**
- `task-1-spot-check/benchmark-dev-gooaq-small.h5`
- `task-2-spot-check/benchmark-dev-llama-small.h5`
- `task-3-spot-check/benchmark-dev-fiqa-small.h5`

These are smaller validation datasets used for spot-checking implementations before running on the full datasets.

## Configuration Files

Each dataset directory contains a `config.json` file with metadata about the dataset:

```json
{
    "task": "task1",
    "data": "train",
    "gt_I": ["allknn", "knns"],
    "k": 15,
    "dataset_name": "wikipedia",
    "filename": "benchmark-dev-wikipedia-bge-m3.h5"
}
```

## Python Examples

### Loading Datasets

```python
import h5py
from huggingface_hub import hf_hub_download

# Download and load a dataset
file_path = hf_hub_download(
    repo_id="SISAP-Challenges/SISAP2026",
    filename="wikipedia/benchmark-dev-wikipedia-bge-m3.h5",
    repo_type="dataset"
)

with h5py.File(file_path, 'r') as f:
    train = f['train'][:]
    print(f"Train shape: {train.shape}")
```

### Loading Sparse Matrices

Here is a small example of how to load the sparse matrices from `nq/nq.h5` and `fiqa-dev/fiqa-dev.h5` using `scipy`:

```python
import h5py
from scipy.sparse import csr_matrix
from huggingface_hub import hf_hub_download

def load_sparse_matrix(h5_group):
    indptr = h5_group['indptr'][:]
    indices = h5_group['indices'][:]
    data = h5_group['data'][:]
    shape = tuple(h5_group.attrs['shape'])
    return csr_matrix((data, indices, indptr), shape=shape)

# Download the file
file_path = hf_hub_download(
    repo_id="SISAP-Challenges/SISAP2026",
    filename="nq/nq.h5",
    repo_type="dataset"
)

with h5py.File(file_path, 'r') as f:
    train_matrix = load_sparse_matrix(f['train'])
    query_matrix = load_sparse_matrix(f['otest']['queries'])
    
    print(f"Train shape: {train_matrix.shape}")
    print(f"Query shape: {query_matrix.shape}")
```

### Loading Configuration

```python
import json
from huggingface_hub import hf_hub_download

config_path = hf_hub_download(
    repo_id="SISAP-Challenges/SISAP2026",
    filename="wikipedia/config.json",
    repo_type="dataset"
)

with open(config_path, 'r') as f:
    config = json.load(f)
    print(config)
```

## Notes

- h5py/HDF5.jl packages read matrices in the expected platform order, so be careful since it could permute dimensions w.r.t what is here explained, however, the final order is what is expected anyway for fast implementations.
- All large `.h5` files are stored using Git LFS (Large File Storage)
- Config files provide metadata and parameters for each dataset