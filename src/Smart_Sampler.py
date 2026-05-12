"""
Smart_Sampler.py — 語義聚類與代表採樣

讀取 Local_Indexer.py 產出的 embeddings.pkl，對 L2-normalized 向量進行
DBSCAN（自動類別數）或自適應 K-Means 聚類；從每個群中挑選最接近質心的 N
張照片。輸出 JSON 清單供 Vision_Labeler.py 使用。
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
from sklearn.cluster import DBSCAN, KMeans

LOGGER = logging.getLogger("SmartSampler")


class SmartSampler:
    def __init__(
        self,
        method: str = "dbscan",
        samples_per_cluster: int = 1,
        keep_noise: bool = True,
    ):
        method = method.lower()
        if method not in {"dbscan", "kmeans", "random"}:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.samples_per_cluster = max(1, int(samples_per_cluster))
        self.keep_noise = keep_noise

    # ---------- I/O ----------

    @staticmethod
    def load_embeddings(path: Path) -> Tuple[List[str], np.ndarray]:
        with open(path, "rb") as f:
            data = pickle.load(f)
        paths = list(data["paths"])
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        if len(paths) != len(vectors):
            raise ValueError(
                f"Length mismatch: paths={len(paths)} vectors={len(vectors)}"
            )
        return paths, vectors

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    # ---------- 聚類 ----------

    @staticmethod
    def _auto_k(n: int) -> int:
        return max(2, int(np.sqrt(n / 2)))

    def cluster(
        self,
        X: np.ndarray,
        eps: float = 0.25,
        min_samples: int = 3,
        k: Optional[int] = None,
        sample_ratio: float = 0.3,
    ) -> np.ndarray:
        if self.method == "random":
            # 隨機抽樣：返回 0/1 labels，1 表示被選中
            n = len(X)
            n_sample = max(1, int(n * sample_ratio))
            labels = np.full(n, -1, dtype=int)
            rng = np.random.RandomState(42)
            selected = rng.choice(n, size=n_sample, replace=False)
            labels[selected] = 0  # 所有選中的歸為同一類
            LOGGER.info("Random sampling: %d / %d (%.0f%%)", n_sample, n, sample_ratio * 100)
            return labels

        if self.method == "dbscan":
            LOGGER.info(
                "DBSCAN(eps=%.3f, min_samples=%d, metric=cosine) on %d vectors",
                eps, min_samples, len(X),
            )
            model = DBSCAN(
                eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1
            )
            return model.fit_predict(X)

        # k-means
        if k is None:
            k = self._auto_k(len(X))
        k = max(1, min(k, len(X)))
        LOGGER.info("KMeans(k=%d) on %d vectors", k, len(X))
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        return model.fit_predict(X)

    # ---------- 代表採樣 ----------

    def select_representatives(
        self, paths: List[str], X: np.ndarray, labels: np.ndarray
    ) -> List[Dict]:
        results: List[Dict] = []
        unique_labels = sorted({int(x) for x in labels.tolist()})
        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            if lab == -1:
                # 隨機抽樣模式：noise 是未被選中的，不保留
                if self.keep_noise and self.method != "random":
                    # 雜訊點各自視為獨特樣本，全數保留
                    for i in idx:
                        results.append({"cluster": -1, "path": paths[int(i)], "score": 1.0})
                continue

            cluster_X = X[idx]
            centroid = cluster_X.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            sims = cluster_X @ centroid  # 已 L2-normalized → cosine 相似度
            # 隨機抽樣模式：取全部選中的圖片
            if self.method == "random":
                n_select = len(idx)
            else:
                n_select = self.samples_per_cluster
            order = np.argsort(-sims)[:n_select]
            for o in order:
                results.append(
                    {
                        "cluster": int(lab),
                        "path": paths[int(idx[int(o)])],
                        "score": float(sims[int(o)]),
                        "cluster_size": int(len(idx)),
                    }
                )
        return results

    # ---------- 主流程 ----------

    def run(
        self,
        embeddings_path: os.PathLike | str,
        output_path: os.PathLike | str = "samples.json",
        eps: float = 0.25,
        min_samples: int = 3,
        k: Optional[int] = None,
        sample_ratio: float = 0.3,
    ) -> Dict:
        embeddings_path = Path(embeddings_path)
        output_path = Path(output_path)

        paths, vectors = self.load_embeddings(embeddings_path)
        if len(paths) == 0:
            LOGGER.warning("No embeddings; writing empty output.")
            payload = {"n_clusters": 0, "n_noise": 0, "samples": [], "paths": []}
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            return payload

        X = self._normalize(vectors)
        labels = self.cluster(X, eps=eps, min_samples=min_samples, k=k, sample_ratio=sample_ratio)

        n_clusters = len({int(l) for l in labels.tolist()} - {-1})
        n_noise = int((labels == -1).sum())
        LOGGER.info("Clusters=%d, noise=%d", n_clusters, n_noise)

        samples = self.select_representatives(paths, X, labels)
        payload = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "samples": samples,
            "paths": [s["path"] for s in samples],
            "method": self.method,
            "samples_per_cluster": self.samples_per_cluster,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        LOGGER.info("Wrote %d samples → %s", len(samples), output_path)
        return payload


# ---------- CLI -------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="對 CLIP 向量做聚類與代表採樣")
    p.add_argument("embeddings", help="Local_Indexer 產出的 embeddings.pkl")
    p.add_argument("-o", "--output", default="samples.json")
    p.add_argument("--method", default="dbscan", choices=["dbscan", "kmeans", "random"])
    p.add_argument("--n", type=int, default=1, help="每群採樣張數")
    p.add_argument("--eps", type=float, default=0.25, help="DBSCAN cosine 距離門檻")
    p.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples")
    p.add_argument("--k", type=int, default=None, help="KMeans k；省略則 sqrt(n/2)")
    p.add_argument("--sample-ratio", type=float, default=0.3, help="隨機抽樣比例（random 模式用，預設 0.3）")
    p.add_argument("--drop-noise", action="store_true", help="DBSCAN 雜訊點不保留")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    _setup_logging(args.log_level)
    sampler = SmartSampler(
        method=args.method,
        samples_per_cluster=args.n,
        keep_noise=not args.drop_noise,
    )
    sampler.run(
        args.embeddings,
        output_path=args.output,
        eps=args.eps,
        min_samples=args.min_samples,
        k=args.k,
        sample_ratio=args.sample_ratio,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
