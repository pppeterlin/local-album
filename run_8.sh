#!/bin/bash
# 等待 6_Xiaomi_Mi11Ultra 標注完成後，處理 8_Xiaomi_Mi13Ultra
set -e

PROJECT_DIR="/Users/chun/Documents/Python/Local Photo Labeler"
PHOTOS_ROOT="/Volumes/970EvoP2T/Chun/Pictures"
SAMPLE_RATIO=0.8
CONCURRENCY=5  # 100 RPM 下建議 5-10

source "$PROJECT_DIR/.env"
export MIMO_API_KEY

DIR="8_Xiaomi_Mi13Ultra"
PHOTOS_DIR="$PHOTOS_ROOT/$DIR"
EMB_FILE="$PROJECT_DIR/embeddings_${DIR}.pkl"
SAMPLES_FILE="$PROJECT_DIR/samples_${DIR}.json"
LABELS_FILE="$PROJECT_DIR/labels_${DIR}.json"

# 等待 6_Xiaomi_Mi11Ultra 標注完成
echo "Waiting for 6_Xiaomi_Mi11Ultra labeling to complete..."
while true; do
    if [ -f "$PROJECT_DIR/labels_6_Xiaomi_Mi11Ultra.json" ]; then
        DONE=$(python3 -c "import json; d=json.load(open('$PROJECT_DIR/labels_6_Xiaomi_Mi11Ultra.json')); print(d.get('count', 0))")
        TOTAL=$(python3 -c "import json; d=json.load(open('$PROJECT_DIR/samples_6_Xiaomi_Mi11Ultra.json')); print(len(d.get('paths', [])))")
        echo "6_Xiaomi_Mi11Ultra: $DONE/$TOTAL labels"
        if [ "$DONE" -ge "$TOTAL" ]; then
            echo "6_Xiaomi_Mi11Ultra labeling complete!"
            break
        fi
    else
        echo "labels_6_Xiaomi_Mi11Ultra.json not found yet..."
    fi
    sleep 60
done

echo "=========================================="
echo "Processing: $DIR (80% sampling, concurrency=$CONCURRENCY)"
echo "=========================================="

cd "$PROJECT_DIR"

echo "[1/3] Indexing..."
uv run python Local_Indexer.py "$PHOTOS_DIR" -o "$EMB_FILE" --incremental --batch-size 4 --num-workers 2

echo "[2/3] Sampling 80%..."
uv run python Smart_Sampler.py "$EMB_FILE" -o "$SAMPLES_FILE" --method random --sample-ratio $SAMPLE_RATIO

echo "[3/3] Labeling (concurrency=$CONCURRENCY)..."
uv run python Xiaomi_Labeler.py "$SAMPLES_FILE" -o "$LABELS_FILE" --no-reasoning --incremental --concurrency $CONCURRENCY

echo "Done: $DIR"
