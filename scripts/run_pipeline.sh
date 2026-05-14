#!/usr/bin/env bash
# End-to-end pipeline for one photo directory.
#
# Usage:
#   scripts/run_pipeline.sh <photos_dir>
#
# Reads from .env (or shell env):
#   VISION_API_KEY   (required)   OpenAI-compatible API key
#   VISION_BASE_URL  (optional)   API base URL
#   VISION_MODEL     (optional)   Model name
#   SAMPLE_RATIO     (optional)   Random sampling ratio for Smart_Sampler (default: 0.5)
#   CONCURRENCY      (optional)   Parallel labeling workers (default: 5)
#   BATCH_SIZE       (optional)   CLIP indexing batch size (default: 4)
#   NUM_WORKERS      (optional)   CLIP DataLoader workers (default: 2)

set -euo pipefail

PHOTOS_DIR="${1:?Usage: $0 <photos_dir>}"
[ -d "$PHOTOS_DIR" ] || { echo "Not a directory: $PHOTOS_DIR" >&2; exit 1; }

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

: "${VISION_API_KEY:?VISION_API_KEY must be set (in .env or environment)}"

SAMPLE_RATIO="${SAMPLE_RATIO:-0.5}"
CONCURRENCY="${CONCURRENCY:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"

NAME="$(basename "$PHOTOS_DIR")"
METADATA_DIR="${METADATA_DIR:-$PROJECT_ROOT/data}"
EMB="$METADATA_DIR/embeddings/${NAME}.pkl"
SAMPLES="$METADATA_DIR/samples/${NAME}.json"
LABELS="$METADATA_DIR/labels/${NAME}.json"

mkdir -p "$METADATA_DIR/embeddings" "$METADATA_DIR/samples" "$METADATA_DIR/labels"

echo "=========================================="
echo "Pipeline: $NAME"
echo "  photos      = $PHOTOS_DIR"
echo "  sample      = $SAMPLE_RATIO"
echo "  concurrency = $CONCURRENCY"
echo "=========================================="

echo "[1/3] Indexing (CLIP embeddings)..."
uv run python src/Local_Indexer.py "$PHOTOS_DIR" \
    -o "$EMB" \
    --incremental \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"

echo "[2/3] Sampling (random, ratio=$SAMPLE_RATIO)..."
uv run python src/Smart_Sampler.py "$EMB" \
    -o "$SAMPLES" \
    --method random \
    --sample-ratio "$SAMPLE_RATIO"

echo "[3/3] Labeling (concurrency=$CONCURRENCY)..."
uv run python src/Vision_Labeler.py "$SAMPLES" \
    -o "$LABELS" \
    --incremental \
    --concurrency "$CONCURRENCY"

echo "Done: $NAME"
