#!/usr/bin/env bash
set -euo pipefail

# Accepts TIRA-style arguments:
#   --input <path-to-h5-file>
#   --task-description <path-to-config.json>
#   --output <output-directory>
# Also accepts the legacy interface used for local testing:
#   --task task3
#   --dataset <name>

H5_FILE=""
OUTPUT_DIR=""
TASK="task3"
DATASET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)            H5_FILE="$2"; shift 2 ;;
        --task-description) CONFIG_FILE="$2"; shift 2 ;;
        --output)           OUTPUT_DIR="$2"; shift 2 ;;
        --task)             TASK="$2"; shift 2 ;;
        --dataset)          DATASET="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; shift ;;
    esac
done

# If --input was given, resolve glob and derive dataset name from filename
if [[ -n "$H5_FILE" ]]; then
    H5_FILE=$(echo $H5_FILE)  # expand glob if needed
    DATASET=$(basename "$H5_FILE" .h5)
fi

# If config.json was provided, read task from it (no python3 needed — simple grep)
if [[ -n "${CONFIG_FILE:-}" && -f "$CONFIG_FILE" ]]; then
    _TASK=$(grep -oE '"task"\s*:\s*"[^"]+"' "$CONFIG_FILE" | grep -oE '[^"]+$' | tr -d '"')
    [[ -n "$_TASK" ]] && TASK="$_TASK"
fi

# Fallback: if no H5_FILE set, construct from dataset name (legacy mode)
if [[ -z "$H5_FILE" ]]; then
    H5_FILE="/app/data/${DATASET}/${DATASET}.h5"
fi

# Output dir: TIRA provides --output, legacy uses /app/results
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="/app/results"
fi

if [[ "$TASK" != "task3" ]]; then
    echo "Only task3 is supported by this image (got task=$TASK)" >&2
    exit 1
fi

if [[ -z "$DATASET" ]]; then
    echo "Could not determine dataset name" >&2
    exit 1
fi

cd /app

echo "Task: $TASK  Dataset: $DATASET"
echo "Input: $H5_FILE"
echo "Output: $OUTPUT_DIR"

echo "Compiling kANNolo for the host CPU..."
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli,multivec --offline

INDEX_FILE="/tmp/${DATASET}_rerank.hnsw"
M=32
EFC=1000
L1=0.75

echo "Building rerank index for ${DATASET} (L1 fraction=${L1})..."
./target/release/sisap_task3_rerank_build \
    --h5-file "$H5_FILE" --group train \
    --output-file "$INDEX_FILE" \
    --m "$M" --ef-construction "$EFC" \
    --l1-fraction "$L1"

# 15 configs (kC, ef_search, lambda) — recalls 0.895..0.970
CONFIGS=(
    "50  50 0.01"
    "50  54 0.01"
    "50  54 0.02"
    "50  54 0.03"
    "60  60 0.02"
    "60  64 0.02"
    "75  75 0.01"
    "60  60 0.04"
    "60  60 0.05"
    "75  79 0.03"
    "75  79 0.04"
    "75  87 0.04"
    "75  87 0.05"
    "75  83 0.07"
    "75  83 0.09"
)

for CFG in "${CONFIGS[@]}"; do
    read -r KC EF_SEARCH LAMBDA <<< "$CFG"
    echo "--- kC=$KC ef_search=$EF_SEARCH lambda=$LAMBDA ---"
    ./target/release/sisap_task3_rerank_search \
        --h5-file "$H5_FILE" --query-group otest/queries \
        --index-file "$INDEX_FILE" -k 30 \
        --k-candidates "$KC" --ef-search "$EF_SEARCH" \
        --lambda "$LAMBDA" --alpha "0.25" --query-top-h "9999" \
        --algo-name kannolo-hnsw-rerank --output-dir "${OUTPUT_DIR}/task3" \
        --m "$M" --ef-construction "$EFC" --l1-fraction "$L1"
done

echo "Done."
