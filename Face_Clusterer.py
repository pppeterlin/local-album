"""
Face_Clusterer.py — 本地人臉偵測與分群

使用 InsightFace (ArcFace) 進行人臉偵測 + 編碼，再用 DBSCAN 分群。
輸出：每張圖片包含哪些人臉群組。

用法：
  python Face_Clusterer.py labels.json -o face_clusters.json
  python Face_Clusterer.py labels.json --eps 0.4 --min-samples 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("FaceClusterer")


class FaceClusterer:
    """人臉偵測 + 編碼 + 分群。"""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.det_size = det_size
        self.device = device or self._select_device()
        self.app = None

    @staticmethod
    def _select_device() -> str:
        """自動選擇裝置。"""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _init_model(self):
        """延遲載入模型。"""
        if self.app is not None:
            return

        import insightface
        from insightface.app import FaceAnalysis

        LOGGER.info("Loading InsightFace %s on %s ...", self.model_name, self.device)
        self.app = FaceAnalysis(
            name=self.model_name,
            root="~/.insightface",
            allowed_modules=["detection", "recognition"],
        )
        # InsightFace 用 CPU provider（MPS 不支援 ONNX）
        providers = ["CPUExecutionProvider"]
        self.app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
        LOGGER.info("InsightFace ready")

    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        偵測圖片中的人臉，回傳人臉資訊列表。

        Returns:
            [{"bbox": [x1,y1,x2,y2], "embedding": np.ndarray(512,), "det_score": float}, ...]
        """
        import cv2

        self._init_model()

        img = cv2.imread(image_path)
        if img is None:
            LOGGER.warning("Cannot read image: %s", image_path)
            return []

        faces = self.app.get(img)
        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.tolist(),
                "embedding": face.embedding,  # 512-dim, L2-normalized
                "det_score": float(face.det_score),
            })
        return results

    def process_all_images(
        self,
        labels_path: str,
        output_path: str,
        eps: float = 0.4,
        min_samples: int = 2,
        max_images: Optional[int] = None,
    ) -> Dict:
        """
        處理所有圖片，偵測人臉並分群。

        Args:
            labels_path: labels.json 路徑
            output_path: face_clusters.json 輸出路徑
            eps: DBSCAN cosine 距離閾值（0.3~0.5）
            min_samples: DBSCAN 最少樣本數
            max_images: 最多處理幾張圖片（測試用）

        Returns:
            face_clusters.json 結構
        """
        # 載入 labels
        LOGGER.info("Loading labels from %s", labels_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        image_paths = []
        for r in labels_data.get("results", []):
            if "error" not in r:
                image_paths.append(r["path"])

        if max_images:
            image_paths = image_paths[:max_images]

        LOGGER.info("Total images to process: %d", len(image_paths))

        # 偵測所有人臉
        all_embeddings: List[np.ndarray] = []
        all_image_indices: List[int] = []  # 對應 image_paths 的 index
        all_bboxes: List[List[float]] = []
        image_faces: Dict[str, List[Dict]] = {}  # path → [{bbox, face_id}, ...]

        for i, path in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                LOGGER.info("Detecting faces: %d / %d", i + 1, len(image_paths))

            faces = self.detect_faces(path)
            if not faces:
                continue

            image_faces[path] = []
            for face in faces:
                idx = len(all_embeddings)
                all_embeddings.append(face["embedding"])
                all_image_indices.append(i)
                all_bboxes.append(face["bbox"])
                image_faces[path].append({
                    "bbox": face["bbox"],
                    "det_score": face["det_score"],
                    "embedding_index": idx,
                })

        total_faces = len(all_embeddings)
        LOGGER.info("Total faces detected: %d in %d images", total_faces, len(image_faces))

        if total_faces == 0:
            LOGGER.warning("No faces detected!")
            return {"n_clusters": 0, "n_faces": 0, "images": {}, "clusters": {}}

        # 分群
        from sklearn.cluster import DBSCAN

        embeddings_matrix = np.array(all_embeddings)  # (N, 512)
        # InsightFace embeddings 已經 L2-normalized，cosine = dot product
        # DBSCAN metric="cosine" 會用 1 - cosine_similarity
        LOGGER.info("Clustering %d faces with DBSCAN(eps=%.3f, min_samples=%d)",
                     total_faces, eps, min_samples)

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
        labels = clustering.fit_predict(embeddings_matrix)

        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        LOGGER.info("Clusters=%d, noise=%d", n_clusters, n_noise)

        # 建立結果
        # 1. 每個 cluster 的資訊
        clusters: Dict[str, Dict] = {}
        for cid in range(n_clusters):
            mask = labels == cid
            indices = np.where(mask)[0]
            # 找出這個 cluster 出現在哪些圖片
            cluster_images = set()
            for idx in indices:
                cluster_images.add(image_paths[all_image_indices[idx]])
            clusters[f"face_{cid}"] = {
                "id": cid,
                "count": int(mask.sum()),
                "images": list(cluster_images),
            }

        # 2. 每張圖片的 face 資訊
        images_result: Dict[str, List[Dict]] = {}
        for path, faces_info in image_faces.items():
            images_result[path] = []
            for face_info in faces_info:
                emb_idx = face_info["embedding_index"]
                cluster_id = int(labels[emb_idx])
                face_label = f"face_{cluster_id}" if cluster_id >= 0 else "unknown"
                images_result[path].append({
                    "bbox": face_info["bbox"],
                    "det_score": face_info["det_score"],
                    "face_id": face_label,
                    "cluster": cluster_id,
                })

        result = {
            "model": f"insightface/{self.model_name}",
            "n_images": len(image_paths),
            "n_faces": total_faces,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "eps": eps,
            "min_samples": min_samples,
            "clusters": clusters,
            "images": images_result,
        }

        # 存檔
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        LOGGER.info("Saved → %s", output_path)

        return result


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="本地人臉偵測與分群（InsightFace + DBSCAN）")
    p.add_argument("labels", help="labels.json 路徑")
    p.add_argument("-o", "--output", default="face_clusters.json", help="輸出檔案")
    p.add_argument("--model", default="buffalo_l", help="InsightFace 模型名稱")
    p.add_argument("--det-size", type=int, default=640, help="偵測圖片大小")
    p.add_argument("--eps", type=float, default=0.4, help="DBSCAN cosine 距離閾值（0.3~0.5）")
    p.add_argument("--min-samples", type=int, default=2, help="DBSCAN 最少樣本數")
    p.add_argument("--max-images", type=int, default=None, help="最多處理幾張（測試用）")
    p.add_argument("--device", default=None, help="強制裝置")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    _setup_logging(args.log_level)

    clusterer = FaceClusterer(
        model_name=args.model,
        det_size=args.det_size,
        device=args.device,
    )
    clusterer.process_all_images(
        labels_path=args.labels,
        output_path=args.output,
        eps=args.eps,
        min_samples=args.min_samples,
        max_images=args.max_images,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
