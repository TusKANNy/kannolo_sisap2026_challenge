#!/usr/bin/env bash
set -euo pipefail

TASK="task3"
DATASET="fiqa-dev"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; shift ;;
    esac
done

if [[ "$TASK" != "task3" ]]; then
    echo "Only task3 is supported by this image (got --task $TASK)" >&2
    exit 1
fi

cd /app

# Compile for the host CPU. Dependencies were pre-fetched at image build time,
# so this runs fully offline. Doing this at container start (rather than at
# image build time) ensures "-C target-cpu=native" targets the machine that
# will actually run the search, not the (possibly different) machine that
# built the image.
echo "Compiling kANNolo for the host CPU..."
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli,multivec --offline

H5_FILE="/app/data/${DATASET}/${DATASET}.h5"
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

# 15 configs (kC, ef_search, lambda) selected from sequential grid search on NQ,
# covering target recalls 0.895..0.970 in steps of ~0.005.
# All use full query (h=9999) and alpha=0.25.
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
        --algo-name kannolo-hnsw-rerank --output-dir /app/results/task3 \
        --m "$M" --ef-construction "$EFC" --l1-fraction "$L1"
done

echo "Done."
