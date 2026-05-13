"""
Face_Clusterer.py — 人臉偵測與分群（分層抽樣優化版）

策略：
  1. 按日期分層抽樣（同一天高機率相同的人）
  2. 每個日期抽 N 張做臉偵測
  3. 用抽樣結果做分群
  4. 其餘圖片分配到最近的群組

這樣大型相簿（數萬張）通常只需要做幾千張的人臉偵測就能建立完整索引。
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
        from_cache: bool = False,
        max_new: int = 0,
    ) -> Dict:
        """
        分層抽樣人臉分群。

        Args:
            labels_path: labels.json 路徑
            output_path: face_clusters.json 路徑
            eps: DBSCAN 距離閾值
            min_samples: DBSCAN 最少樣本數
            per_day: 每天抽幾張（預設 8）
            from_cache: 跳過 Phase 1 擴充，僅用既有 face_clusters.pkl
                        作為 Phase 2 輸入。用於補跑 Phase 3 而不影響 cluster id。
        """
        # 載入 labels
        LOGGER.info("Loading labels from %s", labels_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        image_paths = [r["path"] for r in labels_data.get("results", []) if "error" not in r]
        LOGGER.info("Total images: %d", len(image_paths))

        emb_path = Path(output_path).with_suffix(".pkl")

        if from_cache:
            # 跳過抽樣與 Phase 1 擴充；直接讀既有 cache 作為 Phase 2 輸入
            if not emb_path.exists():
                raise FileNotFoundError(
                    f"--from-cache 需要 {emb_path}，但檔案不存在。"
                    "請先正常跑一次 Face_Clusterer 產生 Phase 1 cache。"
                )
            LOGGER.info("[from-cache] Loading existing Phase 1 cache from %s", emb_path)
            with open(emb_path, "rb") as f:
                sample_data = pickle.load(f)
            LOGGER.info("[from-cache] Cache has %d faces from %d images",
                         len(sample_data["embeddings"]),
                         len(set(sample_data["image_paths"])))
        else:
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
                    rng = np.random.RandomState(hash(date) % 2**31)
                    indices = rng.choice(len(paths), size=per_day, replace=False)
                    sampled_paths.extend([paths[i] for i in indices])

            LOGGER.info("Sampled: %d images (%.1f%%) from %d dates",
                         len(sampled_paths), len(sampled_paths) / len(image_paths) * 100, len(date_groups))

            # Phase 1
            sample_data = self._detect_faces_batch(sampled_paths, emb_path, desc="Sampling")

        # Phase 2: 分群
        clusters, centroids = self._cluster(sample_data, eps, min_samples)

        # Phase 3: streaming detect + assign，max_new>0 時批次切割
        result = self._stream_phase3(image_paths, centroids, output_path, max_new=max_new)
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

            # 每 1000 張存 checkpoint，並把 new_data 折進 existing 然後重置
            # （避免 accumulator 一路膨脹到結束，雙倍 buffer 才釋放）
            if (i + 1) % 1000 == 0 and new_data["embeddings"]:
                existing["image_paths"].extend(new_data["image_paths"])
                existing["embeddings"].extend(new_data["embeddings"])
                existing["bboxes"].extend(new_data["bboxes"])
                existing["det_scores"].extend(new_data["det_scores"])
                new_data = {"image_paths": [], "embeddings": [], "bboxes": [], "det_scores": []}
                with open(emb_path, "wb") as f:
                    pickle.dump(existing, f, protocol=pickle.HIGHEST_PROTOCOL)
                LOGGER.info("[%s] Checkpoint: %d faces", desc, len(existing["embeddings"]))
                gc.collect()

        # 最終存檔
        existing["image_paths"].extend(new_data["image_paths"])
        existing["embeddings"].extend(new_data["embeddings"])
        existing["bboxes"].extend(new_data["bboxes"])
        existing["det_scores"].extend(new_data["det_scores"])
        with open(emb_path, "wb") as f:
            pickle.dump(existing, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("[%s] Done: %d faces in %d images", desc, len(existing["embeddings"]), len(set(existing["image_paths"])))

        return existing

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

    def _stream_phase3(self, image_paths: List[str], centroids: np.ndarray, output_path: str, max_new: int = 0) -> Dict:
        """
        Streaming Phase 3：偵測 + 立即指派 cluster + 寫 JSONL，整個過程不在記憶體
        累積 face embeddings。Bounded memory ≈ InsightFace 模型 + 當前 batch。

        斷點續跑：JSONL 已有的 path 跳過。
        """
        jsonl_path = Path(output_path).with_name("face_assignments.jsonl")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume: 收集 JSONL 已處理路徑
        processed: set = set()
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if "path" in rec:
                            processed.add(rec["path"])
                    except Exception:  # noqa: BLE001
                        continue
            LOGGER.info("[Phase 3 stream] Resuming: %d already in JSONL, %d new",
                         len(processed), len(image_paths) - len(processed))

        all_new = [p for p in image_paths if p not in processed]
        if max_new > 0 and len(all_new) > max_new:
            new_paths = all_new[:max_new]
            LOGGER.info("[Phase 3 stream] --max-new=%d: processing %d / %d remaining this run",
                         max_new, len(new_paths), len(all_new))
        else:
            new_paths = all_new

        if not new_paths:
            LOGGER.info("[Phase 3 stream] Nothing new to detect; building clusters from JSONL")
        else:
            LOGGER.info("[Phase 3 stream] Detecting %d images, writing to %s", len(new_paths), jsonl_path)

        n_centroids = len(centroids)

        # 偵測 + assign + 寫
        with open(jsonl_path, "a", encoding="utf-8") as out:
            for i, path in enumerate(new_paths):
                if (i + 1) % 200 == 0:
                    LOGGER.info("[Phase 3 stream] %d / %d", i + 1, len(new_paths))

                try:
                    faces = self.detect_faces(path)
                except Exception as e:  # noqa: BLE001
                    out.write(json.dumps({"path": path, "error": str(e)[:200]}) + "\n")
                    continue

                if not faces:
                    out.write(json.dumps({"path": path, "no_faces": True}) + "\n")
                    continue

                for face in faces:
                    emb = np.asarray(face["embedding"], dtype=np.float32)
                    emb = emb / (np.linalg.norm(emb) + 1e-12)
                    if n_centroids > 0:
                        sims = emb @ centroids.T
                        best = int(np.argmax(sims))
                        cid = best if sims[best] > 0.3 else -1
                    else:
                        cid = -1
                    rec = {
                        "path": path,
                        "face_id": f"face_{cid}" if cid >= 0 else "unknown",
                        "cluster": cid,
                        "bbox": [float(x) for x in face["bbox"]],
                        "det_score": float(face["det_score"]),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    # 丟棄 face dict（包含 512-dim embedding），不留在 memory

                if (i + 1) % 1000 == 0:
                    out.flush()
                    gc.collect()

        # 若還有未處理的 path（max_new 切割造成），不要重建 face_clusters.json
        # —— 等下一輪外層 wrapper 再呼叫，所有 JSONL 寫完才組裝
        remaining = len(all_new) - len(new_paths)
        if remaining > 0:
            LOGGER.info("[Phase 3 stream] %d images still pending; exiting without rebuild", remaining)
            return {"pending": remaining, "jsonl": str(jsonl_path)}

        return self._build_clusters_from_jsonl(jsonl_path, output_path, n_centroids)

    def _build_clusters_from_jsonl(self, jsonl_path: Path, output_path: str, n_centroids: int) -> Dict:
        """從 face_assignments.jsonl 串流組裝 face_clusters.json，記憶體只裝 dict 結構。"""
        LOGGER.info("Building face_clusters.json from %s", jsonl_path)
        clusters_info: Dict[str, Dict] = {
            f"face_{cid}": {"id": cid, "count": 0, "images": []} for cid in range(n_centroids)
        }
        cluster_seen_imgs: Dict[str, set] = {f"face_{cid}": set() for cid in range(n_centroids)}
        images_result: Dict[str, List[Dict]] = {}
        total_faces = 0
        n_noise = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if "error" in rec or "no_faces" in rec:
                    continue
                if "cluster" not in rec or "path" not in rec:
                    continue
                cid = rec["cluster"]
                path = rec["path"]
                fid = rec["face_id"]
                if cid >= 0 and cid < n_centroids:
                    key = f"face_{cid}"
                    if path not in cluster_seen_imgs[key]:
                        cluster_seen_imgs[key].add(path)
                        clusters_info[key]["images"].append(path)
                    clusters_info[key]["count"] += 1
                else:
                    n_noise += 1
                images_result.setdefault(path, []).append({
                    "bbox": rec["bbox"],
                    "det_score": rec["det_score"],
                    "face_id": fid,
                    "cluster": cid,
                })
                total_faces += 1

        result = {
            "model": f"insightface/{self.model_name}",
            "n_images": len(images_result),
            "n_faces": total_faces,
            "n_clusters": n_centroids,
            "n_noise": n_noise,
            "clusters": clusters_info,
            "images": images_result,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        LOGGER.info("Saved → %s (%d faces in %d clusters, %d noise)",
                     output_path, total_faces, n_centroids, n_noise)
        return result

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
    p.add_argument("--from-cache", action="store_true",
                   help="跳過 Phase 1 擴充，僅用既有 face_clusters.pkl 作 Phase 2 輸入。"
                        "用於補跑 Phase 3 不洗掉既有 cluster id 與命名。")
    p.add_argument("--max-new", type=int, default=0,
                   help="Phase 3 此次最多處理 N 張未處理影像就退出（0=不限）。"
                        "搭配外層 shell loop 做 subprocess 批次，每次退出讓 OS 強制回收"
                        "insightface/onnxruntime 的記憶體 leak。")
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
        from_cache=args.from_cache,
        max_new=args.max_new,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
