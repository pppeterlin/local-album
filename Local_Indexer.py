"""
Local_Indexer.py — 本地照片向量化與特徵提取

針對 Apple Silicon (M4) MPS 加速最佳化：
  • 強制使用 MPS（若可用），fallback 至 CUDA / CPU。
  • DataLoader 多進程預處理（CPU），主進程 MPS 推論，I/O 與運算重疊。
  • os.scandir 迭代遞歸，避免一次讀入巨量路徑。
  • 批次推論並對特徵 L2-normalize（後續可用 cosine 相似度）。
  • 自動跳過損壞圖檔。

預設模型：open_clip ViT-H-14 (laion2b_s32b_b79k)，輸出維度 1024。
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile, ExifTags
from torch.utils.data import DataLoader, Dataset

import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # 允許處理大圖

LOGGER = logging.getLogger("LocalIndexer")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------- 高效目錄遍歷 -----------------------------------------------------

def iter_images(root: Path) -> Iterator[Path]:
    """以 os.scandir 進行非遞歸式深度優先掃描，記憶體佔用恆定。"""
    stack: List[str] = [str(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in SUPPORTED_EXTS:
                                yield Path(entry.path)
                    except OSError as e:
                        LOGGER.debug("Skip entry %s: %s", entry.path, e)
        except (PermissionError, FileNotFoundError, NotADirectoryError) as e:
            LOGGER.warning("Skip dir %s: %s", d, e)


# ---------- Dataset ----------------------------------------------------------

_EXIF_KEYS = {
    "DateTimeOriginal", "DateTimeDigitized", "DateTime",
    "Make", "Model", "LensModel",
    "ISOSpeedRatings", "FocalLength", "ExposureTime", "FNumber",
    "ImageWidth", "ImageLength",
    "PixelXDimension", "PixelYDimension",
    "Orientation",
}

_GPS_TAG_MAP = {v: k for k, v in ExifTags.GPSTAGS.items()}
_IFD_TAG_MAP = {v: k for k, v in ExifTags.TAGS.items()}


def _extract_exif(img: Image.Image) -> Dict:
    """Extract human-readable EXIF metadata from a PIL Image. Returns {} on failure."""
    try:
        raw = img.getexif()
    except Exception:  # noqa: BLE001
        return {}
    if not raw:
        return {}

    meta: Dict = {}

    # Standard IFD tags
    for tag_id, value in raw.items():
        name = _IFD_TAG_MAP.get(tag_id, str(tag_id))
        if name not in _EXIF_KEYS:
            continue
        # Convert bytes/IFD to str for JSON serializability
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace").strip("\x00")
            except Exception:  # noqa: BLE001
                continue
        # Rational numbers → float
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            if value.denominator:
                value = float(value.numerator) / float(value.denominator)
            else:
                value = float(value.numerator)
        meta[name] = value

    # GPS IFD (nested)
    gps_ifd = raw.get_ifd(0x8825)  # GPSInfo tag
    if gps_ifd:
        gps: Dict = {}
        for tag_id, value in gps_ifd.items():
            name = _GPS_TAG_MAP.get(tag_id, str(tag_id))
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace").strip("\x00")
                except Exception:  # noqa: BLE001
                    continue
            if hasattr(value, "numerator") and hasattr(value, "denominator"):
                if value.denominator:
                    value = float(value.numerator) / float(value.denominator)
                else:
                    value = float(value.numerator)
            # Tuples of Rationals (GPS coords) → list of floats
            if isinstance(value, tuple):
                value = tuple(
                    float(v.numerator) / float(v.denominator)
                    if hasattr(v, "numerator") else float(v)
                    for v in value
                )
            gps[name] = value
        if gps:
            meta["GPSInfo"] = gps

    return meta


# ---------- Dataset ----------------------------------------------------------

class _ImageDataset(Dataset):
    """於 worker 進程載入圖檔並執行 CLIP preprocess。同時擷取 EXIF metadata。"""

    def __init__(self, paths: List[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[int, Optional[torch.Tensor], Dict]:
        p = self.paths[idx]
        try:
            with Image.open(p) as img:
                # Extract EXIF BEFORE converting to RGB (some tags lost on convert)
                exif_meta = _extract_exif(img)
                img = img.convert("RGB")
                tensor = self.preprocess(img)
            return idx, tensor, exif_meta
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("Skip corrupt image %s: %s", p, e)
            return idx, None, {}


def _collate(batch):
    """過濾掉預處理失敗者；回傳 (indices, stacked_tensor, exif_list)。"""
    valid = [(i, t, e) for i, t, e in batch if t is not None]
    if not valid:
        return [], None, []
    idxs, tensors, exifs = zip(*valid)
    return list(idxs), torch.stack(tensors, dim=0), list(exifs)


# ---------- 主類別 ----------------------------------------------------------

class LocalIndexer:
    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        batch_size: int = 8,
        num_workers: int = 4,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = self._select_device(device)
        LOGGER.info("Using device: %s", self.device)

        # 限制 CPU 線程數，避免與 DataLoader workers 互搶
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        except Exception:  # noqa: BLE001
            pass

        LOGGER.info("Loading %s / %s ...", model_name, pretrained)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.embedding_dim = int(self.model.visual.output_dim)
        LOGGER.info("Model ready (dim=%d)", self.embedding_dim)

    # ---- device --------------------------------------------------------

    @staticmethod
    def _select_device(device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOGGER.warning("MPS / CUDA 不可用，退化使用 CPU（會非常慢）。")
        return torch.device("cpu")

    # ---- encoding ------------------------------------------------------

    @torch.inference_mode()
    def _encode_batch(self, batch: torch.Tensor) -> np.ndarray:
        batch = batch.to(self.device, non_blocking=True)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # MPS → float32 後再轉回 CPU/numpy
        return feats.detach().to("cpu", dtype=torch.float32).numpy()

    # ---- main API ------------------------------------------------------

    def index_directory(
        self,
        root: os.PathLike | str,
        output_path: os.PathLike | str = "embeddings.pkl",
        log_every: int = 200,
        incremental: bool = False,
    ) -> dict:
        root = Path(root).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)

        # 增量模式：載入既有 embeddings，建立已索引路徑集合
        existing_paths: set = set()
        existing_data: Optional[dict] = None
        if incremental and output_path.exists():
            LOGGER.info("Incremental mode: loading existing %s ...", output_path)
            with open(output_path, "rb") as f:
                existing_data = pickle.load(f)
            existing_paths = set(existing_data["paths"])
            LOGGER.info("Found %d existing embeddings", len(existing_paths))

        LOGGER.info("Scanning %s ...", root)
        all_paths = list(iter_images(root))
        total = len(all_paths)
        LOGGER.info("Found %d candidate images", total)

        # 增量模式：只處理新圖片
        if incremental and existing_paths:
            new_paths = [p for p in all_paths if str(p) not in existing_paths]
            LOGGER.info("New images to index: %d (skipping %d existing)", len(new_paths), len(existing_paths))
        else:
            new_paths = all_paths

        if not new_paths and not existing_paths:
            empty = {
                "paths": [],
                "vectors": np.zeros((0, self.embedding_dim), dtype=np.float32),
                "exif": [],
                "model": f"{self.model_name}/{self.pretrained}",
                "dim": self.embedding_dim,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(empty, f, protocol=pickle.HIGHEST_PROTOCOL)
            return empty

        if not new_paths:
            LOGGER.info("No new images to index — keeping existing embeddings")
            return existing_data

        dataset = _ImageDataset(new_paths, self.preprocess)
        # MPS 不支援 pin_memory；CUDA 才開
        pin = self.device.type == "cuda"
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=pin,
            persistent_workers=self.num_workers > 0,
        )

        out_paths: List[str] = []
        out_vecs: List[np.ndarray] = []
        out_exifs: List[Dict] = []
        processed = 0
        skipped = 0

        for idxs, tensor, exifs in loader:
            if tensor is None:
                skipped += self.batch_size  # 估計值
                continue
            feats = self._encode_batch(tensor)
            for i, v, e in zip(idxs, feats, exifs):
                out_paths.append(str(new_paths[i]))
                out_vecs.append(v)
                out_exifs.append(e)
            processed += len(idxs)
            if processed // log_every != (processed - len(idxs)) // log_every:
                LOGGER.info("Encoded %d / %d", processed, len(new_paths))

        skipped = len(new_paths) - processed
        LOGGER.info("Done. encoded=%d skipped=%d", processed, skipped)

        new_vectors = (
            np.stack(out_vecs, axis=0).astype(np.float32)
            if out_vecs
            else np.zeros((0, self.embedding_dim), dtype=np.float32)
        )

        # 增量模式：合併既有 + 新增
        if incremental and existing_data:
            merged_paths = existing_data["paths"] + out_paths
            merged_vectors = np.concatenate([existing_data["vectors"], new_vectors], axis=0)
            merged_exifs = existing_data["exif"] + out_exifs
            LOGGER.info("Merged: %d existing + %d new = %d total", len(existing_data["paths"]), len(out_paths), len(merged_paths))
        else:
            merged_paths = out_paths
            merged_vectors = new_vectors
            merged_exifs = out_exifs

        result = {
            "paths": merged_paths,
            "vectors": merged_vectors,
            "exif": merged_exifs,
            "model": f"{self.model_name}/{self.pretrained}",
            "dim": self.embedding_dim,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Saved %d embeddings → %s", len(merged_paths), output_path)
        return result


# ---------- CLI -------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="本地 CLIP 向量化（MPS 最佳化）")
    parser.add_argument("root", help="待掃描的根目錄")
    parser.add_argument("-o", "--output", default="embeddings.pkl")
    parser.add_argument("--model", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, choices=["mps", "cuda", "cpu"])
    parser.add_argument("--incremental", action="store_true", help="增量索引：跳過已存在的圖片，合併既有 embeddings")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    indexer = LocalIndexer(
        model_name=args.model,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    indexer.index_directory(args.root, args.output, incremental=args.incremental)
    return 0


if __name__ == "__main__":
    sys.exit(main())
