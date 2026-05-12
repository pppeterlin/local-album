"""
Face_Clusterer.py — 人臉偵測與分群（分層抽樣優化版）

策略：
  1. 按日期分層抽樣（同一天高機率相同的人）
  2. 每個日期抽 N 張做臉偵測
  3. 用抽樣結果做分群
  4. 其餘圖片分配到最近的群組

這樣 57K 張圖可能只需要處理 3-5K 張就能建立完整人臉索引。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gc

LOGGER = logging.getLogger("FaceClusterer")


def extract_date(path: str) -> str:
    """從檔名或 EXIF 嘗試提取日期。"""
    import re
    basename = os.path.basename(path)
    # 嚴格匹配：IMG_20230815_xxx.jpg 或 2023-08-15_xxx.jpg
    m = re.match(r'.*?(\d{4})(\d{2})(\d{2})[_\-]', basename)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if 1990 <= int(y) <= 2030 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
    # 從路徑中的年份目錄提取
    parts = Path(path).parts
    for part in parts:
        m = re.match(r'^(\d{4})$', part)
        if m and 1990 <= int(m.group(1)) <= 2030:
            return m.group(1)
    return "unknown"


class FaceClusterer:
    """人臉偵測 + 編碼 + 分群（分層抽樣版）。"""

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640):
        self.model_name = model_name
        self.det_size = det_size
        self.app = None

    def _init_model(self):
        if self.app is not None:
            return
        from insightface.app import FaceAnalysis
        LOGGER.info("Loading InsightFace %s (det_size=%d) ...", self.model_name, self.det_size)
        self.app = FaceAnalysis(name=self.model_name, root="~/.insightface",
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
        LOGGER.info("InsightFace ready")

    def detect_faces(self, image_path: str) -> List[Dict]:
        import cv2
        self._init_model()
        img = cv2.imread(image_path)
        if img is None:
            return []
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        faces = self.app.get(img)
        return [{"bbox": f.bbox.tolist(), "embedding": f.embedding, "det_score": float(f.det_score)} for f in faces]

    def process_all_images(
        self,
        labels_path: str,
        output_path: str,
        eps: float = 0.4,
        min_samples: int = 2,
        per_day: int = 8,
    ) -> Dict:
        """
        分層抽樣人臉分群。

        Args:
            labels_path: labels.json 路徑
            output_path: face_clusters.json 路徑
            eps: DBSCAN 距離閾值
            min_samples: DBSCAN 最少樣本數
            per_day: 每天抽幾張（預設 8）
        """
        # 載入 labels
        LOGGER.info("Loading labels from %s", labels_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        image_paths = [r["path"] for r in labels_data.get("results", []) if "error" not in r]
        LOGGER.info("Total images: %d", len(image_paths))

        # 分層抽樣：按日期分組，每天抽 per_day 張
        date_groups = defaultdict(list)
        for path in image_paths:
            date = extract_date(path)
            date_groups[date].append(path)

        LOGGER.info("Unique dates: %d", len(date_groups))

        # 抽樣
        sampled_paths = []
        for date, paths in sorted(date_groups.items()):
            if len(paths) <= per_day:
                sampled_paths.extend(paths)
            else:
                # 隨機抽 per_day 張
                rng = np.random.RandomState(hash(date) % 2**31)
                indices = rng.choice(len(paths), size=per_day, replace=False)
                sampled_paths.extend([paths[i] for i in indices])

        LOGGER.info("Sampled: %d images (%.1f%%) from %d dates",
                     len(sampled_paths), len(sampled_paths) / len(image_paths) * 100, len(date_groups))

        # Phase 1: 偵測抽樣圖片的人臉
        emb_path = Path(output_path).with_suffix(".pkl")
        sample_data = self._detect_faces_batch(sampled_paths, emb_path, desc="Sampling")

        # Phase 2: 分群
        clusters, centroids = self._cluster(sample_data, eps, min_samples)

        # Phase 3: 把所有圖片的人臉分配到群組
        # 先檢查是否已有完整的偵測結果
        full_emb_path = Path(output_path).with_name("face_embeddings_full.pkl")
        if full_emb_path.exists():
            LOGGER.info("Loading full face embeddings from %s", full_emb_path)
            with open(full_emb_path, "rb") as f:
                all_data = pickle.load(f)
        else:
            # 偵測所有圖片
            all_data = self._detect_faces_batch(image_paths, full_emb_path, desc="Full scan")

        result = self._assign_and_save(all_data, clusters, centroids, output_path)

        # 清理
        gc.collect()
        return result

    def _detect_faces_batch(self, paths: List[str], emb_path: Path, desc: str = "") -> Dict:
        """批次偵測人臉，增量存檔。"""
        # 檢查 checkpoint
        existing = {"image_paths": [], "embeddings": [], "bboxes": [], "det_scores": []}
        if emb_path.exists():
            LOGGER.info("Loading checkpoint from %s", emb_path)
            with open(emb_path, "rb") as f:
                existing = pickle.load(f)
            existing_set = set(existing["image_paths"])
            new_paths = [p for p in paths if p not in existing_set]
            if not new_paths:
                LOGGER.info("All %d images already processed", len(paths))
                return existing
            LOGGER.info("Resuming: %d existing, %d new", len(existing["image_paths"]), len(new_paths))
        else:
            new_paths = paths

        new_data = {"image_paths": [], "embeddings": [], "bboxes": [], "det_scores": []}

        for i, path in enumerate(new_paths):
            if (i + 1) % 200 == 0:
                LOGGER.info("[%s] %d / %d (faces so far: %d)",
                           desc, i + 1, len(new_paths), len(existing["embeddings"]) + len(new_data["embeddings"]))

            faces = self.detect_faces(path)
            for face in faces:
                new_data["image_paths"].append(path)
                new_data["embeddings"].append(face["embedding"])
                new_data["bboxes"].append(face["bbox"])
                new_data["det_scores"].append(face["det_score"])

            # 每 1000 張存 checkpoint
            if (i + 1) % 1000 == 0 and new_data["embeddings"]:
                merged = self._merge(existing, new_data)
                with open(emb_path, "wb") as f:
                    pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
                LOGGER.info("[%s] Checkpoint: %d faces", desc, len(merged["embeddings"]))
                gc.collect()

        # 最終存檔
        merged = self._merge(existing, new_data)
        with open(emb_path, "wb") as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("[%s] Done: %d faces in %d images", desc, len(merged["embeddings"]), len(set(merged["image_paths"])))

        return merged

    def _merge(self, a: Dict, b: Dict) -> Dict:
        return {
            "image_paths": a["image_paths"] + b["image_paths"],
            "embeddings": a["embeddings"] + b["embeddings"],
            "bboxes": a["bboxes"] + b["bboxes"],
            "det_scores": a["det_scores"] + b["det_scores"],
        }

    def _cluster(self, data: Dict, eps: float, min_samples: int) -> Tuple:
        """DBSCAN 分群。"""
        from sklearn.cluster import DBSCAN

        total = len(data["embeddings"])
        if total == 0:
            return {}, np.array([])

        embeddings = np.array(data["embeddings"])
        LOGGER.info("Clustering %d faces with DBSCAN(eps=%.3f, min_samples=%d)", total, eps, min_samples)

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
        labels = clustering.fit_predict(embeddings)

        n_clusters = len(set(labels) - {-1})
        LOGGER.info("Clusters=%d, noise=%d", n_clusters, int((labels == -1).sum()))

        # 計算質心
        centroids = []
        for cid in range(n_clusters):
            mask = labels == cid
            c = embeddings[mask].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            centroids.append(c)

        # 建立 sample 的 cluster mapping
        clusters = {}
        for i in range(total):
            clusters[i] = int(labels[i])

        return clusters, np.array(centroids) if centroids else np.array([])

    def _assign_and_save(self, all_data: Dict, sample_clusters: Dict, centroids: np.ndarray, output_path: str) -> Dict:
        """分配所有人臉到群組並存檔。"""
        total = len(all_data["embeddings"])
        all_embeddings = np.array(all_data["embeddings"])
        image_paths = all_data["image_paths"]

        # 分配
        assignments = np.full(total, -1, dtype=int)

        if len(centroids) > 0:
            LOGGER.info("Assigning %d faces to nearest cluster...", total)
            for i in range(total):
                sims = all_embeddings[i] @ centroids.T
                best = np.argmax(sims)
                if sims[best] > 0.3:
                    assignments[i] = int(best)

        # 建立結果
        n_clusters = len(centroids) if len(centroids) > 0 else 0
        clusters_info: Dict[str, Dict] = {}
        for cid in range(n_clusters):
            mask = assignments == cid
            indices = np.where(mask)[0]
            cluster_images = list(set(image_paths[i] for i in indices))
            clusters_info[f"face_{cid}"] = {
                "id": cid,
                "count": int(mask.sum()),
                "images": cluster_images,
            }

        images_result: Dict[str, List[Dict]] = {}
        for i in range(total):
            path = image_paths[i]
            cid = int(assignments[i])
            if path not in images_result:
                images_result[path] = []
            images_result[path].append({
                "bbox": all_data["bboxes"][i],
                "det_score": all_data["det_scores"][i],
                "face_id": f"face_{cid}" if cid >= 0 else "unknown",
                "cluster": cid,
            })

        n_noise = int((assignments == -1).sum())

        result = {
            "model": f"insightface/{self.model_name}",
            "n_images": len(set(image_paths)),
            "n_faces": total,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "clusters": clusters_info,
            "images": images_result,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        LOGGER.info("Saved → %s (%d faces, %d clusters)", output_path, total, n_clusters)

        return result


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="人臉偵測與分群（分層抽樣版）")
    p.add_argument("labels", help="labels.json 路徑")
    p.add_argument("-o", "--output", default="face_clusters.json")
    p.add_argument("--model", default="buffalo_l")
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--eps", type=float, default=0.4)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--per-day", type=int, default=8, help="每天抽幾張（預設 8）")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    _setup_logging(args.log_level)

    clusterer = FaceClusterer(model_name=args.model, det_size=args.det_size)
    clusterer.process_all_images(
        labels_path=args.labels,
        output_path=args.output,
        eps=args.eps,
        min_samples=args.min_samples,
        per_day=args.per_day,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
