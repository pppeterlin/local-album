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
import re
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

    def create_label_embeddings(
        self,
        labels_path: str,
        output_path: str,
        api_key: str,
        base_url: str,
    ) -> Dict:
        """
        為所有標注建立 CLIP 向量（翻譯成英文後編碼）。

        Args:
            labels_path: labels.json 路徑
            output_path: label_embeddings.pkl 輸出路徑
            api_key: LLM API key（翻譯用）
            base_url: LLM API base URL

        Returns:
            {"paths": [...], "vectors": np.ndarray, "labels": [...]}
        """
        LOGGER.info("Loading labels from %s", labels_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        paths = []
        labels = []
        for r in labels_data.get("results", []):
            if "error" not in r and r.get("text"):
                paths.append(r["path"])
                labels.append(r["text"])

        LOGGER.info("Translating %d labels to English...", len(labels))
        translated = []
        for i, label in enumerate(labels):
            if (i + 1) % 50 == 0:
                LOGGER.info("Translating %d / %d", i + 1, len(labels))
            try:
                en = self._translate_to_english(label, api_key, base_url)
                translated.append(en)
            except Exception as e:
                LOGGER.warning("Translation failed for %s: %s", paths[i], e)
                translated.append("")

        LOGGER.info("Encoding %d labels with CLIP...", len(translated))
        vectors = []
        for i in range(0, len(translated), 32):
            batch = translated[i:i+32]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.append(feats.cpu().numpy().astype(np.float32))

        vectors = np.concatenate(vectors, axis=0)

        result = {
            "paths": paths,
            "vectors": vectors,
            "labels": labels,
            "translations": translated,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Saved %d label embeddings → %s", len(paths), output_path)
        return result

    @staticmethod
    def _has_cjk(text: str) -> bool:
        """檢查文字是否包含中日韓字元。"""
        return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))

    @staticmethod
    def _translate_to_english(text: str, api_key: str, base_url: str, model: str) -> str:
        """用 LLM 將中文翻譯成英文（供 CLIP text encoder 使用）。"""
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a translator. Translate the user's text to English. Output ONLY the translation, nothing else."},
                {"role": "user", "content": text},
            ],
            max_tokens=100,
        )
        translated = resp.choices[0].message.content.strip()
        LOGGER.info("Translated: \"%s\" → \"%s\"", text, translated)
        return translated

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
        labels_path: Optional[str] = None,
        top_k: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        camera: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        translate_model: Optional[str] = None,
        clip_weight: float = 0.6,
    ) -> List[Dict]:
        """
        混合語義搜尋：CLIP 向量相似度 + 標注關鍵字匹配。

        自動偵測語言：
        - 中文查詢 → 翻譯成英文給 CLIP，標注用原文匹配
        - 英文查詢 → 直接給 CLIP，標注用原文匹配

        Args:
            query: 查詢文字（中英文皆可）
            embeddings_path: embeddings.pkl 路徑
            labels_path: labels.json 路徑（可選，啟用關鍵字搜尋）
            top_k: 回傳前 k 個結果
            date_from: 篩選起始日期 (YYYY-MM-DD)
            date_to: 篩選結束日期 (YYYY-MM-DD)
            camera: 篩選相機型號（模糊匹配）
            api_key: LLM API key（中文翻譯用）
            base_url: LLM API base URL（翻譯用）
            translate_model: 翻譯用模型名稱
            clip_weight: CLIP 相似度權重（0~1）

        Returns:
            [{"path": str, "score": float, "clip_score": float, "keyword_score": float, "label": str, "exif": dict}, ...]
        """
        # 載入 embeddings
        LOGGER.info("Loading embeddings from %s", embeddings_path)
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)

        paths = data["paths"]
        vectors = data["vectors"]  # shape (N, dim), L2-normalized
        exif_list = data.get("exif", [{}] * len(paths))

        LOGGER.info("Index: %d images, dim=%d", len(paths), vectors.shape[1])

        # 載入 labels.json 建立 path→label 對照表（關鍵字搜尋用）
        path_to_label = {}
        if labels_path and Path(labels_path).exists():
            LOGGER.info("Loading labels from %s", labels_path)
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
            for r in labels_data.get("results", []):
                p = r.get("path", "")
                t = r.get("text", "")
                if p and t:
                    path_to_label[p] = t
            LOGGER.info("Loaded %d labels for keyword matching", len(path_to_label))

        # 載入 label embeddings（語義搜尋用）
        label_emb_path = Path(embeddings_path).parent / "label_embeddings.pkl"
        label_paths = []
        label_vectors = None
        label_texts = {}
        if label_emb_path.exists():
            LOGGER.info("Loading label embeddings from %s", label_emb_path)
            with open(label_emb_path, "rb") as f:
                lemb = pickle.load(f)
            label_paths = lemb["paths"]
            label_vectors = lemb["vectors"]
            for p, t in zip(lemb["paths"], lemb.get("labels", [])):
                label_texts[p] = t
            LOGGER.info("Label embeddings: %d", len(label_paths))
        else:
            LOGGER.info("No label_embeddings.pkl found — using CLIP-only search")

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

        # 編碼查詢文字（中文自動翻譯成英文給 CLIP）
        search_query = query
        if self._has_cjk(query) and api_key:
            try:
                search_query = self._translate_to_english(
                    query, api_key,
                    base_url or "https://api.openai.com/v1",
                    translate_model or "gpt-4o-mini",
                )
            except Exception as e:
                LOGGER.warning("Translation failed: %s — using original query", e)
        elif self._has_cjk(query) and not api_key:
            LOGGER.warning("Query contains CJK but no API key — CLIP search may be less accurate")

        LOGGER.info("Encoding query: \"%s\"", search_query)
        text_vec = self.encode_text(search_query)  # (1, dim)

        # 計算 cosine similarity（向量已 L2-normalized，dot product = cosine）
        # 只計算符合篩選條件的向量
        filtered_vectors = vectors[mask]  # (M, dim)
        clip_similarities = (filtered_vectors @ text_vec.T).squeeze(-1)  # (M,)

        # 計算關鍵字匹配分數（如果有 labels）
        keyword_similarities = np.zeros(len(clip_similarities), dtype=np.float32)
        if path_to_label:
            query_lower = query.lower()
            filtered_paths = [paths[i] for i in np.where(mask)[0]]
            for i, p in enumerate(filtered_paths):
                label = path_to_label.get(p, "").lower()
                if query_lower in label:
                    # 完全匹配給 1.0，部分匹配按比例
                    keyword_similarities[i] = 1.0

        # 混合分數
        if path_to_label:
            # 有 labels：加權組合
            clip_w = clip_weight
            keyword_w = 1.0 - clip_weight
            combined_scores = clip_w * clip_similarities + keyword_w * keyword_similarities
            LOGGER.info("Hybrid search: CLIP weight=%.1f, keyword weight=%.1f", clip_w, keyword_w)
        else:
            # 無 labels：純 CLIP
            combined_scores = clip_similarities
            LOGGER.info("CLIP-only search (no labels provided)")

        # 取 top-k
        k = min(top_k, len(combined_scores))
        top_indices = np.argpartition(combined_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(combined_scores[top_indices])[::-1]]

        # 映射回原始 index
        original_indices = np.where(mask)[0]

        results = []
        for idx in top_indices:
            orig_idx = original_indices[idx]
            path = paths[orig_idx]
            results.append({
                "path": path,
                "score": float(combined_scores[idx]),
                "clip_score": float(clip_similarities[idx]),
                "keyword_score": float(keyword_similarities[idx]),
                "label": path_to_label.get(path, ""),
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
    p.add_argument("--clip-weight", type=float, default=0.6, help="CLIP 權重（0~1，預設 0.6）")
    p.add_argument("--api-key", default=None, help="LLM API key（中文翻譯用，預設讀 VISION_API_KEY）")
    p.add_argument("--base-url", default=None, help="LLM API base URL（翻譯用）")
    p.add_argument("--translate-model", default=None, help="翻譯用模型（預設讀 VISION_MODEL）")
    p.add_argument("--json", action="store_true", help="以 JSON 格式輸出")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    _setup_logging(args.log_level)

    searcher = PhotoSearcher(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    # API key for translation (CLI > env)
    api_key = args.api_key or os.environ.get("VISION_API_KEY")
    base_url = args.base_url or os.environ.get("VISION_BASE_URL")
    translate_model = args.translate_model or os.environ.get("VISION_MODEL")

    # 自動偵測 labels.json（data/labels/ 下）
    project_dir = Path(__file__).resolve().parent.parent
    labels_path = project_dir / "data" / "labels" / "labels.json"
    if not labels_path.exists():
        labels_path = None
        LOGGER.info("No labels.json found in project dir — using CLIP-only search")
    else:
        LOGGER.info("Auto-detected labels: %s", labels_path)

    results = searcher.search(
        query=args.query,
        embeddings_path=args.embeddings,
        labels_path=str(labels_path) if labels_path else None,
        top_k=args.top,
        date_from=args.date_from,
        date_to=args.date_to,
        camera=args.camera,
        api_key=api_key,
        base_url=base_url,
        translate_model=translate_model,
        clip_weight=args.clip_weight,
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
            clip_pct = r.get("clip_score", 0) * 100
            keyword_pct = r.get("keyword_score", 0) * 100
            path = r["path"]
            filename = Path(path).name
            label = r.get("label", "")
            exif = r["exif"]

            # 格式化 EXIF 資訊
            exif_parts = []
            if exif.get("DateTimeOriginal"):
                exif_parts.append(exif["DateTimeOriginal"])
            cam = " ".join(filter(None, [exif.get("Make"), exif.get("Model")]))
            if cam:
                exif_parts.append(cam)
            exif_str = " │ ".join(exif_parts) if exif_parts else ""

            # 分數顯示
            if keyword_pct > 0:
                score_display = f"[{score_pct:5.1f}% = CLIP {clip_pct:.1f}% + KW {keyword_pct:.1f}%]"
            else:
                score_display = f"[{score_pct:5.1f}%]"

            print(f"  {i}. {score_display} {filename}")
            print(f"     {path}")
            if label:
                # 截斷長標注
                label_preview = label[:80] + "..." if len(label) > 80 else label
                print(f"     🏷️  {label_preview}")
            if exif_str:
                print(f"     📷 {exif_str}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
