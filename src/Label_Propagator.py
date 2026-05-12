"""
Label_Propagator.py — 半監督標籤傳播

用已標注的圖片作為「種子」，把未標注的圖片分配到語義最相似的種子，
形成以標籤為中心的聚類。

流程：
  1. 載入 embeddings.pkl（全部向量）+ labels.json（已標注圖片）
  2. 計算未標注圖片與所有已標注圖片的 cosine similarity
  3. 分配到最相似的已標注圖片（種子）
  4. 輸出 clusters.json

用法：
  python Label_Propagator.py embeddings.pkl labels.json -o clusters.json
  python Label_Propagator.py embeddings.pkl labels.json --min-similarity 0.15
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

LOGGER = logging.getLogger("LabelPropagator")


class LabelPropagator:
    """用已標注圖片作為種子，聚類未標注圖片。"""

    def propagate(
        self,
        embeddings_path: str,
        labels_path: str,
        output_path: str,
        min_similarity: float = 0.10,
    ) -> Dict:
        """
        標籤傳播：把未標注圖片分配到最相似的已標注圖片。

        Args:
            embeddings_path: embeddings.pkl 路徑
            labels_path: labels.json 路徑（已標注圖片）
            output_path: clusters.json 輸出路徑
            min_similarity: 最低相似度閾值（低於此值歸為 unclustered）

        Returns:
            clusters.json 結構
        """
        # 載入 embeddings
        LOGGER.info("Loading embeddings from %s", embeddings_path)
        with open(embeddings_path, "rb") as f:
            emb_data = pickle.load(f)

        all_paths = emb_data["paths"]
        all_vectors = emb_data["vectors"]  # (N, dim), L2-normalized
        all_exifs = emb_data.get("exif", [{}] * len(all_paths))
        path_to_idx = {p: i for i, p in enumerate(all_paths)}

        LOGGER.info("Total images: %d", len(all_paths))

        # 載入 labels（成功的才有用）
        LOGGER.info("Loading labels from %s", labels_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        labeled_paths = set()
        path_to_label = {}
        for r in labels_data.get("results", []):
            if "error" not in r and r.get("text"):
                labeled_paths.add(r["path"])
                path_to_label[r["path"]] = r["text"]

        LOGGER.info("Labeled images: %d", len(labeled_paths))

        # 找出已標注和未標注的 index
        labeled_indices = [path_to_idx[p] for p in labeled_paths if p in path_to_idx]
        unlabeled_indices = [i for i, p in enumerate(all_paths) if p not in labeled_paths]

        LOGGER.info("Labeled in index: %d, Unlabeled: %d", len(labeled_indices), len(unlabeled_indices))

        if not labeled_indices:
            LOGGER.error("No labeled images found in embeddings!")
            return {}

        # 提取向量
        seed_vectors = all_vectors[labeled_indices]  # (S, dim)
        unlabeled_vectors = all_vectors[unlabeled_indices]  # (U, dim)

        # 計算 cosine similarity（向量已 L2-normalized，dot product = cosine）
        LOGGER.info("Computing similarity matrix (%d unlabeled × %d seeds)...", len(unlabeled_indices), len(labeled_indices))
        sim_matrix = unlabeled_vectors @ seed_vectors.T  # (U, S)

        # 找最近的種子
        best_seed_idx = np.argmax(sim_matrix, axis=1)  # (U,)
        best_sim = sim_matrix[np.arange(len(unlabeled_indices)), best_seed_idx]  # (U,)

        # 建立聚類
        clusters: Dict[str, Dict] = {}
        unclustered: List[Dict] = []

        # 先加入已標注的種子
        for i, idx in enumerate(labeled_indices):
            path = all_paths[idx]
            seed_id = f"seed_{i}"
            clusters[seed_id] = {
                "seed_path": path,
                "seed_label": path_to_label.get(path, ""),
                "members": [{
                    "path": path,
                    "similarity": 1.0,
                    "is_seed": True,
                }],
            }

        # 分配未標注圖片到最近的種子
        for i, (unlabeled_idx, seed_idx, sim) in enumerate(
            zip(unlabeled_indices, best_seed_idx, best_sim)
        ):
            path = all_paths[unlabeled_idx]
            seed_path = all_paths[labeled_indices[seed_idx]]

            if sim < min_similarity:
                unclustered.append({
                    "path": path,
                    "best_seed": seed_path,
                    "similarity": float(sim),
                })
                continue

            # 找到對應的 seed_id
            seed_id = None
            for sid, c in clusters.items():
                if c["seed_path"] == seed_path:
                    seed_id = sid
                    break

            if seed_id:
                clusters[seed_id]["members"].append({
                    "path": path,
                    "similarity": float(sim),
                    "is_seed": False,
                })

        # 統計
        total_clustered = sum(len(c["members"]) for c in clusters.values())
        total_unclustered = len(unclustered)

        LOGGER.info("Clusters: %d", len(clusters))
        LOGGER.info("Clustered: %d, Unclustered: %d", total_clustered, total_unclustered)

        # 輸出
        result = {
            "total_images": len(all_paths),
            "total_labeled": len(labeled_paths),
            "total_clustered": total_clustered,
            "total_unclustered": total_unclustered,
            "min_similarity": min_similarity,
            "clusters": clusters,
            "unclustered": unclustered,
        }

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
    p = argparse.ArgumentParser(description="半監督標籤傳播：用已標注圖片聚類未標注圖片")
    p.add_argument("embeddings", help="embeddings.pkl 路徑")
    p.add_argument("labels", help="labels.json 路徑（已標注圖片）")
    p.add_argument("-o", "--output", default="clusters.json", help="輸出 clusters.json")
    p.add_argument("--min-similarity", type=float, default=0.10, help="最低相似度閾值（預設 0.10）")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    _setup_logging(args.log_level)

    propagator = LabelPropagator()
    propagator.propagate(
        embeddings_path=args.embeddings,
        labels_path=args.labels,
        output_path=args.output,
        min_similarity=args.min_similarity,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
