#!/usr/bin/env bash
# Run Face_Clusterer Phase 3 in subprocess batches to bound memory.
# Each subprocess processes BATCH_SIZE new images, exits, OS reclaims
# the leaky memory held by insightface/onnxruntime, then we loop again.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LABELS="${LABELS:-data/labels/labels_all.json}"
OUT="${OUT:-data/faces/face_clusters.json}"
BATCH_SIZE="${BATCH_SIZE:-2000}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
JSONL="data/faces/face_assignments.jsonl"

TOTAL=$(python3 -c "import json; d=json.load(open('$LABELS')); print(sum(1 for r in d.get('results',[]) if 'error' not in r))")
echo "Total images to scan: $TOTAL"

round=0
while true; do
    round=$((round + 1))
    processed=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
    echo
    echo "════════════════════════════════════════════"
    echo "Round $round — JSONL has $processed lines so far"
    echo "════════════════════════════════════════════"

    # Run subprocess; capture exit code
    set +e
    caffeinate -dis uv run python src/Face_Clusterer.py "$LABELS" -o "$OUT" \
        --from-cache --max-new "$BATCH_SIZE"
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
        echo "Subprocess exited with code $rc — stopping loop"
        exit $rc
    fi

    # Check: did this round finish all remaining? Build script ran rebuild → done.
    if [ -f "$OUT" ] && [ "$(stat -f %m "$OUT" 2>/dev/null)" -gt "$(stat -f %m "$JSONL" 2>/dev/null || echo 0)" ]; then
        echo "✅ face_clusters.json freshly written — done!"
        break
    fi

    # Brief pause so OS can finish reclaiming the freed subprocess memory
    # before the next batch fires up insightface again.
    echo "...sleeping ${SLEEP_BETWEEN}s before next batch"
    sleep "$SLEEP_BETWEEN"
done

echo
echo "=== Final stats ==="
python3 -c "
import json
d = json.load(open('$OUT'))
print(f'Clusters: {len(d[\"clusters\"])}')
print(f'Total faces: {d[\"n_faces\"]}')
print(f'Images with faces: {d[\"n_images\"]}')
print(f'Noise faces: {d[\"n_noise\"]}')
"
