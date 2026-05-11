#!/bin/bash
# Run labeler on remaining unlabeled photos for each directory
# Uses nohup-safe approach: run each dir, --incremental saves progress
cd "/Users/chun/Documents/Python/Local Photo Labeler"
export $(grep -v '^#' .env | xargs)

LABELER="uv run python Xiaomi_Labeler.py"

DIRS=(
    "samples_4_Xiaomi_Mi5_remaining.json|labels_4_Xiaomi_Mi5.json"
    "samples_6_Xiaomi_Mi11Ultra_remaining.json|labels_6_Xiaomi_Mi11Ultra.json"
    "samples_5_Xiaomi_MiMix3_remaining.json|labels_5_Xiaomi_MiMix3.json"
    "samples_8_Xiaomi_Mi13Ultra_remaining.json|labels_8_Xiaomi_Mi13Ultra.json"
)

for entry in "${DIRS[@]}"; do
    samples="${entry%%|*}"
    labels="${entry##*|}"
    echo "=== $(date): Starting $samples -> $labels ==="
    $LABELER "$samples" -o "$labels" --no-reasoning --incremental --concurrency 5
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== $(date): $labels DONE ==="
    else
        echo "=== $(date): $labels exited with code $rc (progress saved) ==="
    fi
done

echo "=== $(date): ALL DIRECTORIES PROCESSED ==="
