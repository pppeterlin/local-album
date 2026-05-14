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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import FACES_DIR  # noqa: E402

FACES_FILE = FACES_DIR / "face_clusters.json"
THUMBS_DIR = FACES_DIR / "face_thumbs"


def resize_for_crop(img, max_long_edge=1280):
    """縮放圖片（與 Face_Clusterer.py 相同邏輯）。"""
    h, w = img.shape[:2]
    if max(h, w) > max_long_edge:
        scale = max_long_edge / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def crop_face_thumb(image_path: str, bbox, out_path: Path,
                    max_long_edge: int = 1280,
                    margin: float = 0.3,
                    size: int = 200,
                    jpeg_quality: int = 90) -> bool:
    """讀圖、縮放、依 bbox 加 margin 裁切、輸出方形縮圖。
    bbox 必須來自相同 max_long_edge 縮放後的偵測。回傳是否成功。"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    img = resize_for_crop(img, max_long_edge)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    fw, fh = x2 - x1, y2 - y1
    if fw <= 0 or fh <= 0:
        return False
    mx, my = int(fw * margin), int(fh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return True


def main():
    THUMBS_DIR.mkdir(exist_ok=True)

    with open(FACES_FILE, "r", encoding="utf-8") as f:
        faces_data = json.load(f)

    clusters = faces_data.get("clusters", {})
    images_data = faces_data.get("images", {})

    print(f"共 {len(clusters)} 個群組")

    # 載入手動覆蓋清單，避免重產
    overrides_file = FACES_DIR / "face_thumb_overrides.json"
    overrides = json.loads(overrides_file.read_text(encoding="utf-8")) if overrides_file.exists() else {}
    if overrides:
        print(f"略過 {len(overrides)} 個手動指定的縮圖")

    for face_id, info in clusters.items():
        if face_id in overrides:
            continue

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
                    best_face = {"path": img_path, "bbox": face["bbox"]}

        if not best_face:
            print(f"  {face_id}: 無法取得人臉縮圖")
            continue

        thumb_path = THUMBS_DIR / f"{face_id}.jpg"
        ok = crop_face_thumb(best_face["path"], best_face["bbox"], thumb_path)
        if ok:
            print(f"  {face_id}: {thumb_path.name} (score={best_score:.3f}, {count}張)")
        else:
            print(f"  {face_id}: 裁切失敗 ({best_face['path']})")

    print(f"\n縮圖已儲存到 {THUMBS_DIR}")
    print(f"共 {len(list(THUMBS_DIR.glob('*.jpg')))} 個縮圖")


if __name__ == "__main__":
    main()
