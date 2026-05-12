#!/usr/bin/env python3
"""
generate_face_thumbs.py — 產生人臉群組縮圖

從 face_clusters.json 中裁切每個人臉群組的最佳人臉，存成縮圖。
bbox 坐標對應的是長邊 1280px 的圖片，需要先縮放再裁切。
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path("/Users/chun/Documents/Python/Local Photo Labeler")
FACES_FILE = PROJECT_DIR / "face_clusters.json"
THUMBS_DIR = PROJECT_DIR / "face_thumbs"


def resize_for_crop(img, max_long_edge=1280):
    """縮放圖片（與 Face_Clusterer.py 相同邏輯）。"""
    h, w = img.shape[:2]
    if max(h, w) > max_long_edge:
        scale = max_long_edge / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def main():
    THUMBS_DIR.mkdir(exist_ok=True)

    with open(FACES_FILE, "r", encoding="utf-8") as f:
        faces_data = json.load(f)

    clusters = faces_data.get("clusters", {})
    images_data = faces_data.get("images", {})

    print(f"共 {len(clusters)} 個群組")

    for face_id, info in clusters.items():
        count = info["count"]
        sample_images = info["images"][:5]

        best_face = None
        best_score = 0

        # 找出偵測分數最高的人臉
        for img_path in sample_images:
            if img_path not in images_data:
                continue

            for face in images_data[img_path]:
                if face["face_id"] == face_id and face["det_score"] > best_score:
                    best_score = face["det_score"]
                    best_face = {
                        "path": img_path,
                        "bbox": face["bbox"],
                        "score": face["det_score"],
                    }

        if not best_face:
            print(f"  {face_id}: 無法取得人臉縮圖")
            continue

        # 讀取圖片並縮放（與偵測時相同）
        img_path = best_face["path"]
        img = cv2.imread(img_path)
        if img is None:
            print(f"  {face_id}: 無法讀取 {img_path}")
            continue

        # 縮放到長邊 1280px（與 Face_Clusterer.py 相同）
        img = resize_for_crop(img, 1280)

        # bbox 坐標對應縮放後的圖片
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in best_face["bbox"]]

        # 邊界檢查
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # 擴大範圍（30%），確保臉部完整
        face_w = x2 - x1
        face_h = y2 - y1
        if face_w <= 0 or face_h <= 0:
            print(f"  {face_id}: bbox 無效 ({x1},{y1},{x2},{y2})")
            continue

        margin_x = int(face_w * 0.3)
        margin_y = int(face_h * 0.3)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        # 裁切並縮放
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            print(f"  {face_id}: 裁切失敗")
            continue

        # 縮放到 200x200
        face_resized = cv2.resize(face_crop, (200, 200), interpolation=cv2.INTER_AREA)

        # 儲存
        thumb_path = THUMBS_DIR / f"{face_id}.jpg"
        cv2.imwrite(str(thumb_path), face_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {face_id}: {thumb_path.name} (score={best_score:.3f}, {count}張)")

    print(f"\n縮圖已儲存到 {THUMBS_DIR}")
    print(f"共 {len(list(THUMBS_DIR.glob('*.jpg')))} 個縮圖")


if __name__ == "__main__":
    main()
