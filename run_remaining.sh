#!/bin/bash
# 批次處理剩餘照片目錄
set -e

PROJECT_DIR="/Users/chun/Documents/Python/Local Photo Labeler"
PHOTOS_ROOT="/Volumes/970EvoP2T/Chun/Pictures"
SAMPLE_RATIO=0.5

source "$PROJECT_DIR/.env"
export MIMO_API_KEY

DIRS=(
    "6_Xiaomi_Mi11Ultra"
    "7_Xiaomi_13Pro"
)

for DIR in "${DIRS[@]}"; do
    echo "=========================================="
    echo "Processing: $DIR"
    echo "=========================================="
    
    PHOTOS_DIR="$PHOTOS_ROOT/$DIR"
    EMB_FILE="$PROJECT_DIR/embeddings_${DIR}.pkl"
    SAMPLES_FILE="$PROJECT_DIR/samples_${DIR}.json"
    LABELS_FILE="$PROJECT_DIR/labels_${DIR}.json"
    
    cd "$PROJECT_DIR"
    
    echo "[1/3] Indexing..."
    uv run python Local_Indexer.py "$PHOTOS_DIR" -o "$EMB_FILE" --incremental --batch-size 4 --num-workers 2
    
    echo "[2/3] Sampling 50%..."
    uv run python Smart_Sampler.py "$EMB_FILE" -o "$SAMPLES_FILE" --method random --sample-ratio $SAMPLE_RATIO
    
    echo "[3/3] Labeling..."
    uv run python Xiaomi_Labeler.py "$SAMPLES_FILE" -o "$LABELS_FILE" --no-reasoning --incremental
    
    echo "Done: $DIR"
done

echo "All done!"
