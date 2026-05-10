#!/bin/bash
# 批次處理多個照片目錄（順序執行，節省 RAM）
# 流程：索引 → 抽樣 50% → 標注

set -e

PROJECT_DIR="/Users/chun/Documents/Python/Local Photo Labeler"
PHOTOS_ROOT="/Volumes/970EvoP2T/Chun/Pictures"
SAMPLE_RATIO=0.5

# 載入 API key
source "$PROJECT_DIR/.env"
export MIMO_API_KEY

# 要處理的目錄（順序執行）
DIRS=(
    "3_Xiaomi_MiNote"
    "4_Xiaomi_Mi5"
    "5_Xiaomi_MiMix3"
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
    
    # 檢查是否已經完成（增量模式，重跑安全）
    if [ -f "$LABELS_FILE" ]; then
        EXISTING=$(python3 -c "import json; d=json.load(open('$LABELS_FILE')); print(d.get('succeeded', 0))")
        echo "Existing labels: $EXISTING"
    fi
    
    # Stage 1: 索引
    echo "[1/3] Indexing..."
    cd "$PROJECT_DIR"
    uv run python Local_Indexer.py "$PHOTOS_DIR" \
        -o "$EMB_FILE" \
        --incremental \
        --batch-size 4 \
        --num-workers 2
    
    # Stage 2: 抽樣 50%
    echo "[2/3] Sampling 50%..."
    uv run python Smart_Sampler.py "$EMB_FILE" \
        -o "$SAMPLES_FILE" \
        --method random \
        --sample-ratio $SAMPLE_RATIO
    
    # Stage 3: 標注
    echo "[3/3] Labeling..."
    uv run python Xiaomi_Labeler.py "$SAMPLES_FILE" \
        -o "$LABELS_FILE" \
        --no-reasoning \
        --incremental
    
    echo "Done: $DIR"
    echo ""
done

echo "=========================================="
echo "All directories processed!"
echo "=========================================="
