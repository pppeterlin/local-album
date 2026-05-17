"""
Pet_Detector.py — 寵物（貓）偵測 + 分群。

策略：
  1. YOLOv8n 偵測 cat bbox（COCO class 15）
  2. CLIP 對裁切的 bbox 做 image embedding
  3. DBSCAN 分群 → pet_clusters.json
  4. 每個 cluster 取信心分數最高的偵測當作縮圖

跟 Face_Clusterer 的差異：
  - 不分日期取樣（貓照片是子集，全跑 YOLO 也很快）
  - 沒有「Phase 1 cache → Phase 3 重跑」這種需求；單階段串流
  - JSONL 中介檔可斷點續跑

輸出檔案（METADATA_DIR/pets/ 下）：
  - pet_clusters.json    {clusters: {pet_id: {images, count, bboxes, ...}}}
  - pet_assignments.jsonl
  - pet_embeddings.pkl   （centroid 計算用）
  - pet_thumbs/<pet_id>.jpg

UI overlay（之後 server 才產生）：
  - pet_names.json, pet_merges.json, pet_moves.json,
    pet_removed.json, pet_skipped.json, pet_thumb_overrides.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PETS_DIR  # noqa: E402

LOGGER = logging.getLogger("PetDetector")

COCO_CAT_CLASS = 15  # YOLO/COCO: 15=cat, 16=dog
CLIP_MODEL_NAME = "ViT-B-32"  # 對裁切後的 cat bbox 已足夠分辨個體；用小模型加速
CLIP_PRETRAINED = "laion2b_s34b_b79k"


def _is_skip_path(p: str) -> bool:
    """跟 face_naming_server / Local_Indexer 對齊的系統路徑過濾。"""
    if not p:
        return True
    pl = p.lower().replace("\\", "/")
    fragments = (
        "/.thumbnails/", "/.trash/", "/.trashes/", "/.cache/",
        "/@eadir/", "/__macosx/", "/.spotlight-v100/", "/.fseventsd/",
    )
    for f in fragments:
        if f in pl:
            return True
    for part in p.split("/"):
        if part.startswith(".") and part not in (".", ".."):
            return True
    return False


class PetDetector:
    """貓偵測 + CLIP embedding + DBSCAN 分群。"""

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        clip_device: Optional[str] = None,
    ):
        self.yolo_model = yolo_model
        self.conf_threshold = conf_threshold
        self.clip_device = clip_device  # None → auto (mps/cuda/cpu)
        self._yolo = None
        self._clip_model = None
        self._clip_preprocess = None

    # ---- model loading (lazy) ----

    def _init_yolo(self):
        if self._yolo is not None:
            return
        from ultralytics import YOLO
        LOGGER.info("Loading YOLO %s ...", self.yolo_model)
        self._yolo = YOLO(self.yolo_model)
        LOGGER.info("YOLO ready")

    def _init_clip(self):
        if self._clip_model is not None:
            return
        import open_clip
        import torch
        if self.clip_device is None:
            if torch.backends.mps.is_available():
                self.clip_device = "mps"
            elif torch.cuda.is_available():
                self.clip_device = "cuda"
            else:
                self.clip_device = "cpu"
        LOGGER.info("Loading CLIP %s (%s) on %s ...", CLIP_MODEL_NAME, CLIP_PRETRAINED, self.clip_device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        model = model.to(self.clip_device).eval()
        self._clip_model = model
        self._clip_preprocess = preprocess
        LOGGER.info("CLIP ready")

    # ---- detection on one image ----

    def detect_cats(self, image_path: str) -> List[Dict]:
        """回傳 [{bbox: [x1,y1,x2,y2], conf: float, embedding: np.ndarray}]。"""
        self._init_yolo()
        self._init_clip()
        import cv2
        from PIL import Image
        import torch

        img = cv2.imread(image_path)
        if img is None:
            return []
        h, w = img.shape[:2]

        # YOLO 推論。max(h,w) 太大就 downscale 加速；最後 bbox 換算回原尺寸
        infer_img = img
        scale = 1.0
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            infer_img = cv2.resize(img, (int(w * scale), int(h * scale)))

        results = self._yolo.predict(
            infer_img, classes=[COCO_CAT_CLASS], conf=self.conf_threshold, verbose=False
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        # 取 bbox + 信心，crop 後送 CLIP
        out: List[Dict] = []
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        crops_pil = []
        bbox_orig = []
        for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
            # 還原到原圖座標
            x1o, y1o, x2o, y2o = [v / scale for v in (x1, y1, x2, y2)]
            # 加 padding 一點點，避免裁切太緊
            pad = 0.05
            bw, bh = x2o - x1o, y2o - y1o
            x1p = max(0, int(x1o - bw * pad))
            y1p = max(0, int(y1o - bh * pad))
            x2p = min(w, int(x2o + bw * pad))
            y2p = min(h, int(y2o + bh * pad))
            if x2p <= x1p or y2p <= y1p:
                continue
            crop_bgr = img[y1p:y2p, x1p:x2p]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crops_pil.append(Image.fromarray(crop_rgb))
            bbox_orig.append(([float(x1o), float(y1o), float(x2o), float(y2o)], float(conf)))

        if not crops_pil:
            return []

        # batch CLIP forward
        with torch.no_grad():
            batch = torch.stack([self._clip_preprocess(c) for c in crops_pil]).to(self.clip_device)
            embs = self._clip_model.encode_image(batch)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            embs = embs.cpu().numpy().astype(np.float32)

        for (bbox, conf), emb in zip(bbox_orig, embs):
            out.append({"bbox": bbox, "conf": conf, "embedding": emb})
        return out

    # ---- main pipeline ----

    def run(
        self,
        labels_path: str,
        output_path: str,
        eps: float = 0.35,
        min_samples: int = 2,
        max_new: int = 0,
    ) -> Dict:
        """完整 pipeline：偵測 → 串流寫 JSONL → DBSCAN → 寫 pet_clusters.json + 縮圖。"""
        LOGGER.info("Loading labels from %s", labels_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)
        image_paths = [
            r["path"] for r in labels_data.get("results", [])
            if "error" not in r and not _is_skip_path(r.get("path", ""))
        ]
        LOGGER.info("Total images: %d", len(image_paths))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path = out.parent / "pet_assignments.jsonl"

        # Phase 1: 偵測（streaming，JSONL 可斷點續跑）
        processed = set()
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if "path" in rec:
                            processed.add(rec["path"])
                    except Exception:  # noqa: BLE001
                        continue
            LOGGER.info("[Phase 1] Resuming: %d already detected, %d new",
                        len(processed), len(image_paths) - len(processed))

        new_paths = [p for p in image_paths if p not in processed]
        if max_new > 0 and len(new_paths) > max_new:
            this_run = new_paths[:max_new]
            LOGGER.info("[Phase 1] --max-new=%d: processing %d / %d remaining this run",
                        max_new, len(this_run), len(new_paths))
            new_paths = this_run

        if new_paths:
            with open(jsonl_path, "a", encoding="utf-8") as outf:
                for i, path in enumerate(new_paths):
                    if (i + 1) % 200 == 0:
                        LOGGER.info("[Phase 1] %d / %d", i + 1, len(new_paths))
                    try:
                        dets = self.detect_cats(path)
                    except Exception as e:  # noqa: BLE001
                        outf.write(json.dumps({"path": path, "error": str(e)[:200]}) + "\n")
                        continue
                    if not dets:
                        outf.write(json.dumps({"path": path, "no_cats": True}) + "\n")
                        continue
                    for d in dets:
                        rec = {
                            "path": path,
                            "bbox": d["bbox"],
                            "conf": d["conf"],
                            "embedding": d["embedding"].tolist(),
                        }
                        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if (i + 1) % 500 == 0:
                        outf.flush()
                        gc.collect()
        else:
            LOGGER.info("[Phase 1] Nothing new; skipping detection")

        # 若還沒跑完（max_new 切割），不要組裝 cluster
        all_remaining = [p for p in image_paths if p not in processed]
        if max_new > 0 and len(all_remaining) > len(new_paths):
            LOGGER.info("[Phase 1] %d images still pending; exiting without cluster build",
                        len(all_remaining) - len(new_paths))
            return {"pending": len(all_remaining) - len(new_paths)}

        # Phase 2: 從 JSONL 載入所有 embedding 做 DBSCAN
        return self._cluster_and_build(jsonl_path, output_path, eps, min_samples)

    def _cluster_and_build(
        self, jsonl_path: Path, output_path: str, eps: float, min_samples: int
    ) -> Dict:
        from sklearn.cluster import DBSCAN

        LOGGER.info("[Phase 2] Loading detections from %s", jsonl_path)
        detections: List[Dict] = []  # list of {path, bbox, conf, embedding (np.array)}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if "error" in rec or "no_cats" in rec or "embedding" not in rec:
                    continue
                rec["embedding"] = np.asarray(rec["embedding"], dtype=np.float32)
                detections.append(rec)

        if not detections:
            LOGGER.warning("No cat detections found; writing empty clusters file")
            out_data = {"clusters": {}, "images": {}, "total_cats": 0}
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            return out_data

        LOGGER.info("[Phase 2] %d cat detections from %d images",
                    len(detections), len(set(d["path"] for d in detections)))

        embs = np.stack([d["embedding"] for d in detections])
        LOGGER.info("[Phase 2] DBSCAN(eps=%.3f, min_samples=%d) on %d × %dd",
                    eps, min_samples, embs.shape[0], embs.shape[1])
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1).fit_predict(embs)

        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        LOGGER.info("[Phase 2] %d clusters, %d noise detections (singletons)", n_clusters, n_noise)

        # 組 cluster 結構
        clusters: Dict[str, Dict] = {}
        images_map: Dict[str, List[Dict]] = defaultdict(list)
        for det, lbl in zip(detections, labels):
            if lbl < 0:
                continue
            pid = f"pet_{lbl}"
            c = clusters.setdefault(pid, {"id": pid, "count": 0, "images": [], "bboxes": [], "confs": []})
            if det["path"] not in c["images"]:
                c["images"].append(det["path"])
            c["bboxes"].append(det["bbox"])
            c["confs"].append(det["conf"])
            c["count"] = len(c["images"])
            images_map[det["path"]].append({"pet_id": pid, "bbox": det["bbox"], "conf": det["conf"]})

        out_data = {
            "clusters": clusters,
            "images": {p: recs for p, recs in images_map.items()},
            "total_cats": sum(c["count"] for c in clusters.values()),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        LOGGER.info("[Phase 2] Wrote %s (%d clusters, %d images)",
                    output_path, len(clusters), len(out_data["images"]))

        # Phase 3: 縮圖（每個 cluster 取信心最高的偵測，裁切儲存）
        self._build_thumbnails(detections, labels, Path(output_path).parent / "pet_thumbs")
        return out_data

    def _build_thumbnails(self, detections: List[Dict], labels: np.ndarray, thumbs_dir: Path):
        import cv2
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        # group by cluster, pick best conf
        best_per_cluster: Dict[int, Dict] = {}
        for det, lbl in zip(detections, labels):
            if lbl < 0:
                continue
            cur = best_per_cluster.get(int(lbl))
            if cur is None or det["conf"] > cur["conf"]:
                best_per_cluster[int(lbl)] = det
        LOGGER.info("[Phase 3] Cropping %d thumbnails to %s", len(best_per_cluster), thumbs_dir)
        for lbl, det in best_per_cluster.items():
            img = cv2.imread(det["path"])
            if img is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            h, w = img.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            # 等比例縮到 max 256
            ch, cw = crop.shape[:2]
            if max(ch, cw) > 256:
                s = 256 / max(ch, cw)
                crop = cv2.resize(crop, (int(cw * s), int(ch * s)))
            cv2.imwrite(str(thumbs_dir / f"pet_{lbl}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Cat detection + clustering (parallel to Face_Clusterer)")
    p.add_argument("labels", help="labels.json from Vision_Labeler (provides image path list)")
    p.add_argument("-o", "--output", default=str(PETS_DIR / "pet_clusters.json"),
                   help=f"Output path (default: {PETS_DIR}/pet_clusters.json)")
    p.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO weights file or model id")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--eps", type=float, default=0.35, help="DBSCAN cosine-distance threshold")
    p.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples")
    p.add_argument("--max-new", type=int, default=0,
                   help="Process at most N new images then exit (for batch subprocesses)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    _setup_logging(args.log_level)

    det = PetDetector(yolo_model=args.yolo_model, conf_threshold=args.conf)
    det.run(args.labels, args.output, eps=args.eps, min_samples=args.min_samples, max_new=args.max_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
