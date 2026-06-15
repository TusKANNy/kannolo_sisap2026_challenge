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
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli --offline

H5_FILE="/app/data/${DATASET}/${DATASET}.h5"
INDEX_FILE="/tmp/${DATASET}.hnsw"

M=32
EFC=1000

echo "Building index for ${DATASET}..."
./target/release/sisap_task3_build \
    --h5-file "$H5_FILE" --group train \
    --output-file "$INDEX_FILE" \
    --m "$M" --ef-construction "$EFC" --reorder-egb

# (ef_search, lambda) pairs selected from results/task3/grids/best_nq_M32_efC1000.tsv
# covering target accuracies 0.875 .. 0.945 in steps of 0.005.
CONFIGS=(
    "30 0.04"
    "31 0.04"
    "32 0.04"
    "30 0.05"
    "31 0.05"
    "30 0.06"
    "35 0.05"
    "30 0.07"
    "31 0.07"
    "30 0.08"
    "32 0.08"
    "30 0.09"
    "37 0.08"
    "32 0.10"
    "36 0.10"
)

for CFG in "${CONFIGS[@]}"; do
    read -r EF_SEARCH LAMBDA <<< "$CFG"
    echo "--- ef_search=$EF_SEARCH lambda=$LAMBDA ---"
    ./target/release/sisap_task3_search \
        --h5-file "$H5_FILE" --query-group otest/queries \
        --index-file "$INDEX_FILE" -k 30 --ef-search "$EF_SEARCH" \
        --early-termination distance-adaptive --lambda "$LAMBDA" \
        --algo-name kannolo-hnsw --output-dir /app/results/task3 \
        --m "$M" --ef-construction "$EFC"
done

echo "Done."
