"""
Photo_Search.py — 本地照片語義搜尋

基於 CLIP 文字-圖像跨模態檢索：
  • 載入 embeddings.pkl（Stage 1 產出的向量索引）
  • 將查詢文字用 CLIP text encoder 編碼成向量
  • 與所有圖片向量計算 cosine similarity
  • 回傳 top-k 最相關的照片

支援組合查詢：語義搜尋 + EXIF 篩選（日期、相機型號）

用法：
  python Photo_Search.py embeddings.pkl --query "咖啡" --top 5
  python Photo_Search.py embeddings.pkl --query "夕陽" --date-from 2023-08-01 --date-to 2023-12-31
  python Photo_Search.py embeddings.pkl --query "海邊" --camera iPhone
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import open_clip

LOGGER = logging.getLogger("PhotoSearcher")


class PhotoSearcher:
    """基於 CLIP 向量的照片語義搜尋器。"""

    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: Optional[str] = None,
    ):
        self.device = self._select_device(device)
        LOGGER.info("Loading CLIP %s / %s on %s ...", model_name, pretrained, self.device)

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(model_name)
        LOGGER.info("Model ready")

    @staticmethod
    def _select_device(device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def encode_text(self, text: str) -> np.ndarray:
        """將查詢文字編碼成 L2-normalized 向量 (1, dim)。"""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)

    def search(
        self,
        query: str,
        embeddings_path: str,
        top_k: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        camera: Optional[str] = None,
    ) -> List[Dict]:
        """
        語義搜尋照片。

        Args:
            query: 查詢文字（支援中英文）
            embeddings_path: embeddings.pkl 路徑
            top_k: 回傳前 k 個結果
            date_from: 篩選起始日期 (YYYY-MM-DD)
            date_to: 篩選結束日期 (YYYY-MM-DD)
            camera: 篩選相機型號（模糊匹配）

        Returns:
            [{"path": str, "score": float, "exif": dict}, ...]
        """
        # 載入 embeddings
        LOGGER.info("Loading embeddings from %s", embeddings_path)
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)

        paths = data["paths"]
        vectors = data["vectors"]  # shape (N, dim), L2-normalized
        exif_list = data.get("exif", [{}] * len(paths))

        LOGGER.info("Index: %d images, dim=%d", len(paths), vectors.shape[1])

        # 建立篩選 mask
        mask = np.ones(len(paths), dtype=bool)
        filter_desc = []

        if date_from or date_to:
            mask, desc = self._filter_by_date(exif_list, mask, date_from, date_to)
            filter_desc.append(desc)

        if camera:
            mask, desc = self._filter_by_camera(exif_list, mask, camera)
            filter_desc.append(desc)

        filtered_count = mask.sum()
        if filtered_count == 0:
            LOGGER.warning("No images match the filters")
            return []

        if filter_desc:
            LOGGER.info("Filters: %s → %d images", ", ".join(filter_desc), filtered_count)

        # 編碼查詢文字
        LOGGER.info("Encoding query: \"%s\"", query)
        text_vec = self.encode_text(query)  # (1, dim)

        # 計算 cosine similarity（向量已 L2-normalized，dot product = cosine）
        # 只計算符合篩選條件的向量
        filtered_vectors = vectors[mask]  # (M, dim)
        similarities = (filtered_vectors @ text_vec.T).squeeze(-1)  # (M,)

        # 取 top-k
        k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        # 映射回原始 index
        original_indices = np.where(mask)[0]

        results = []
        for idx in top_indices:
            orig_idx = original_indices[idx]
            results.append({
                "path": paths[orig_idx],
                "score": float(similarities[idx]),
                "exif": exif_list[orig_idx],
            })

        return results

    @staticmethod
    def _filter_by_date(
        exif_list: List[Dict],
        mask: np.ndarray,
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> Tuple[np.ndarray, str]:
        """依 EXIF DateTimeOriginal 篩選。"""
        dt_from = datetime.strptime(date_from, "%Y-%m-%d") if date_from else None
        dt_to = datetime.strptime(date_to, "%Y-%m-%d") if date_to else None

        new_mask = mask.copy()
        for i, exif in enumerate(exif_list):
            if not mask[i]:
                continue
            dt_str = exif.get("DateTimeOriginal")
            if not dt_str:
                new_mask[i] = False
                continue
            try:
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                new_mask[i] = False
                continue
            if dt_from and dt < dt_from:
                new_mask[i] = False
            if dt_to and dt > dt_to:
                new_mask[i] = False

        parts = []
        if date_from:
            parts.append(f"from {date_from}")
        if date_to:
            parts.append(f"to {date_to}")
        return new_mask, f"date {' '.join(parts)}"

    @staticmethod
    def _filter_by_camera(
        exif_list: List[Dict],
        mask: np.ndarray,
        camera: str,
    ) -> Tuple[np.ndarray, str]:
        """依 EXIF Make/Model 模糊篩選。"""
        camera_lower = camera.lower()
        new_mask = mask.copy()
        for i, exif in enumerate(exif_list):
            if not mask[i]:
                continue
            make = (exif.get("Make") or "").lower()
            model = (exif.get("Model") or "").lower()
            if camera_lower not in make and camera_lower not in model:
                new_mask[i] = False
        return new_mask, f"camera={camera}"


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="本地照片語義搜尋（基於 CLIP 向量）")
    p.add_argument("embeddings", help="embeddings.pkl 路徑")
    p.add_argument("-q", "--query", required=True, help="查詢文字（中英文皆可）")
    p.add_argument("--top", type=int, default=10, help="回傳前 k 個結果（預設 10）")
    p.add_argument("--model", default="ViT-H-14", help="CLIP 模型名稱")
    p.add_argument("--pretrained", default="laion2b_s32b_b79k", help="CLIP pretrained weights")
    p.add_argument("--device", default=None, help="強制裝置 (mps/cuda/cpu)")
    p.add_argument("--date-from", default=None, help="篩選起始日期 (YYYY-MM-DD)")
    p.add_argument("--date-to", default=None, help="篩選結束日期 (YYYY-MM-DD)")
    p.add_argument("--camera", default=None, help="篩選相機型號（模糊匹配）")
    p.add_argument("--json", action="store_true", help="以 JSON 格式輸出")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    _setup_logging(args.log_level)

    searcher = PhotoSearcher(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    results = searcher.search(
        query=args.query,
        embeddings_path=args.embeddings,
        top_k=args.top,
        date_from=args.date_from,
        date_to=args.date_to,
        camera=args.camera,
    )

    if not results:
        print("No results found.")
        return 0

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"\n🔍 \"{args.query}\" — Top {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            score_pct = r["score"] * 100
            path = r["path"]
            filename = Path(path).name
            exif = r["exif"]

            # 格式化 EXIF 資訊
            exif_parts = []
            if exif.get("DateTimeOriginal"):
                exif_parts.append(exif["DateTimeOriginal"])
            cam = " ".join(filter(None, [exif.get("Make"), exif.get("Model")]))
            if cam:
                exif_parts.append(cam)
            exif_str = " │ ".join(exif_parts) if exif_parts else ""

            print(f"  {i}. [{score_pct:5.1f}%] {filename}")
            print(f"     {path}")
            if exif_str:
                print(f"     📷 {exif_str}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
