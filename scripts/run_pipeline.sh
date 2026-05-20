#!/usr/bin/env bash
# End-to-end pipeline for one photo directory.
#
# Usage:
#   scripts/run_pipeline.sh <photos_dir> [--root <user_root>]
#
# With --root, embeddings/samples/labels are written next to that root
# (<root>/.metadata/{embeddings,samples,labels}/<name>.{pkl,json}) so each
# user's per-photo metadata stays portable with their photos.
#
# Without --root: legacy behavior — writes to $METADATA_DIR/{embeddings,samples,labels}.
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

PHOTOS_DIR=""
USER_ROOT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --root) USER_ROOT="$2"; shift 2 ;;
        --root=*) USER_ROOT="${1#--root=}"; shift ;;
        --) shift; break ;;
        -*) echo "Unknown flag: $1" >&2; exit 1 ;;
        *) if [ -z "$PHOTOS_DIR" ]; then PHOTOS_DIR="$1"; else echo "Extra arg: $1" >&2; exit 1; fi; shift ;;
    esac
done
[ -n "$PHOTOS_DIR" ] || { echo "Usage: $0 <photos_dir> [--root <user_root>]" >&2; exit 1; }
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
if [ -n "$USER_ROOT" ]; then
    [ -d "$USER_ROOT" ] || { echo "Not a directory: $USER_ROOT" >&2; exit 1; }
    OUT_BASE="$USER_ROOT/.metadata"
    LAYOUT="sidecar"
else
    OUT_BASE="${METADATA_DIR:-$PROJECT_ROOT/data}"
    LAYOUT="global"
fi

EMB="$OUT_BASE/embeddings/${NAME}.pkl"
SAMPLES="$OUT_BASE/samples/${NAME}.json"
LABELS="$OUT_BASE/labels/${NAME}.json"

mkdir -p "$OUT_BASE/embeddings" "$OUT_BASE/samples" "$OUT_BASE/labels"

echo "=========================================="
echo "Pipeline: $NAME"
echo "  photos      = $PHOTOS_DIR"
echo "  layout      = $LAYOUT"
echo "  out_base    = $OUT_BASE"
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
