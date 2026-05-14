"""
Photo_Index.py — 統一照片索引引擎

整合四維索引：
  • 人臉（Who）：face_clusters.json
  • 時間（When）：EXIF DateTimeOriginal
  • 地點（Where）：EXIF GPS → 反向地理編碼
  • 主體（What）：CLIP 向量 + 標注文字

支援組合查詢，輸出 JSON 或人類可讀格式。

用法（從專案根目錄執行）：
  python src/Photo_Index.py build \
      --labels data/labels/photos.json \
      --faces data/faces/face_clusters.json \
      --embeddings data/embeddings/photos.pkl
  python src/Photo_Index.py search "beach 2023"
  python src/Photo_Index.py name-face face_0 grandma
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
from typing import Dict, List, Optional, Tuple, Set, Tuple

import numpy as np

LOGGER = logging.getLogger("PhotoIndex")

# 地點名稱映射（GPS → 地名）
# 用戶可以自訂這個映射
LOCATION_NAMES_FILE = "location_names.json"
FACE_NAMES_FILE = "face_names.json"


class PhotoIndex:
    """統一照片索引引擎。"""

    def __init__(self, project_dir: str):
        # project_dir kept for backwards compat; actual locations come from _paths
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _paths import FACES_DIR, INDEX_DIR  # noqa: E402
        self.project_dir = Path(project_dir)
        self.index_path = INDEX_DIR / "photo_index.json"
        self.face_names_path = FACES_DIR / FACE_NAMES_FILE
        self.location_names_path = INDEX_DIR / LOCATION_NAMES_FILE
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.face_names_path.parent.mkdir(parents=True, exist_ok=True)

        # 載入索引
        self.index: Dict = {}

        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))

        # 載入人臉名稱映射
        self.face_names: Dict[str, str] = {}
        if self.face_names_path.exists():
            self.face_names = json.loads(self.face_names_path.read_text(encoding="utf-8"))

        # 載入地點名稱映射
        self.location_names: Dict[str, str] = {}
        if self.location_names_path.exists():
            self.location_names = json.loads(self.location_names_path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_overlay(path: Path, default):
        """讀 overlay JSON；不存在或壞掉就回 default。"""
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("Failed to read overlay %s: %s", path, e)
            return default

    def build(
        self,
        labels_path: str,
        faces_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
    ) -> Dict:
        """
        建立統一索引。

        Args:
            labels_path: labels.json 路徑
            faces_path: face_clusters.json 路徑（可選）
            embeddings_path: embeddings.pkl 路徑（可選，用於 EXIF）

        Returns:
            索引結構
        """
        LOGGER.info("Building unified index...")

        # 1. 載入 labels
        LOGGER.info("Loading labels from %s", labels_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        images: Dict[str, Dict] = {}
        for r in labels_data.get("results", []):
            if "error" not in r:
                path = r["path"]
                images[path] = {
                    "path": path,
                    "label": r.get("text", ""),
                    "faces": [],
                    "time": None,
                    "location": None,
                    "gps": None,
                }

        LOGGER.info("Loaded %d images from labels", len(images))

        # 2. 載入 EXIF（如果有 embeddings.pkl）
        if embeddings_path and Path(embeddings_path).exists():
            LOGGER.info("Loading EXIF from %s", embeddings_path)
            with open(embeddings_path, "rb") as f:
                emb_data = pickle.load(f)

            for path, exif in zip(emb_data["paths"], emb_data.get("exif", [])):
                if path in images:
                    # 時間
                    dt_str = exif.get("DateTimeOriginal")
                    if dt_str:
                        try:
                            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                            images[path]["time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                            images[path]["year"] = dt.year
                            images[path]["month"] = dt.month
                        except ValueError:
                            pass

                    # GPS
                    gps = exif.get("GPSInfo")
                    if gps:
                        images[path]["gps"] = gps
                        # 簡單的地理位置描述
                        lat = gps.get("GPSLatitude")
                        lng = gps.get("GPSLongitude")
                        if lat and lng:
                            images[path]["location"] = f"{lat:.4f},{lng:.4f}"

            LOGGER.info("Merged EXIF data")

        # 3. 載入人臉分群 + 套用人工 overlay（merges / moves / removed）
        if faces_path and Path(faces_path).exists():
            LOGGER.info("Loading face clusters from %s", faces_path)
            with open(faces_path, "r", encoding="utf-8") as f:
                faces_data = json.load(f)

            faces_dir = Path(faces_path).parent
            merges = self._load_overlay(faces_dir / "face_merges.json", {})
            moves_list = self._load_overlay(faces_dir / "face_moves.json", [])
            removed = self._load_overlay(faces_dir / "face_removed.json", {})
            LOGGER.info("Overlays: %d merges, %d moves, removed across %d clusters",
                         len(merges), len(moves_list), len(removed))

            def resolve(fid):
                seen = set()
                while fid in merges and fid not in seen:
                    seen.add(fid)
                    fid = merges[fid]
                return fid

            # moves index：(path, resolved_from_fid) → resolved_to_fid
            moves_idx: Dict[Tuple[str, str], str] = {}
            for m in moves_list:
                p, f, t = m.get("path"), m.get("from"), m.get("to")
                if p and f and t:
                    moves_idx[(p, resolve(f))] = resolve(t)

            # removed：按解析後 fid 聚合
            effective_removed: Dict[str, set] = {}
            for fid, paths in removed.items():
                eff = resolve(fid)
                effective_removed.setdefault(eff, set()).update(paths)

            applied = {"moved": 0, "removed": 0, "merged": 0}
            for path, faces in faces_data.get("images", {}).items():
                if path not in images:
                    continue
                for face in faces:
                    raw_fid = face["face_id"]
                    eff_fid = resolve(raw_fid)
                    if eff_fid != raw_fid:
                        applied["merged"] += 1
                    # apply move（特定 path 上指定 fid 的人臉被搬走）
                    move_key = (path, eff_fid)
                    if move_key in moves_idx:
                        eff_fid = moves_idx[move_key]
                        applied["moved"] += 1
                    # apply remove
                    if path in effective_removed.get(eff_fid, ()):
                        applied["removed"] += 1
                        continue
                    images[path]["faces"].append({
                        "id": eff_fid,
                        "name": self.face_names.get(eff_fid, eff_fid),
                        "det_score": face["det_score"],
                    })

            LOGGER.info("Merged face data (applied: %d merged, %d moved, %d removed)",
                         applied["merged"], applied["moved"], applied["removed"])

        # 4. 建立索引
        self.index = {
            "total_images": len(images),
            "built_at": datetime.now().isoformat(),
            "images": images,
        }

        # 存檔
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.index, indent=2, ensure_ascii=False))
        LOGGER.info("Index saved → %s (%d images)", self.index_path, len(images))

        return self.index

    def search(
        self,
        query: str = "",
        face_name: Optional[str] = None,
        faces_all: Optional[List[str]] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        location: Optional[str] = None,
        path_contains: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        組合查詢。

        Args:
            query: 語義查詢（用於標注文字匹配）
            face_name: 人臉名稱篩選
            year: 年份篩選
            month: 月份篩選
            location: 地點篩選（模糊匹配）
            top_k: 回傳前 k 個結果

        Returns:
            [{"path": str, "score": float, "faces": [...], "time": str, "location": str}, ...]
        """
        if not self.index:
            LOGGER.error("Index not built. Run 'build' first.")
            return []

        images = self.index.get("images", {})
        results = []

        for path, info in images.items():
            score = 0.0
            matched = True

            # 路徑子字串篩選（資料夾、檔名等）
            if path_contains:
                if path_contains not in path:
                    matched = False
                else:
                    score += 0.5

            # 人臉篩選（單一）
            img_faces = info.get("faces", [])
            face_ids = [f["id"] for f in img_faces]
            face_names_in_img = [f["name"] for f in img_faces]
            if face_name:
                if face_name not in face_ids and face_name not in face_names_in_img:
                    matched = False
                else:
                    score += 1.0

            # 多人合照篩選（AND，每個都要出現）
            if faces_all:
                for f in faces_all:
                    if f not in face_ids and f not in face_names_in_img:
                        matched = False
                        break
                else:
                    score += 1.0 * len(faces_all)

            # 年份篩選
            if year and info.get("year") != year:
                matched = False
            elif year:
                score += 0.5

            # 月份篩選
            if month and info.get("month") != month:
                matched = False
            elif month:
                score += 0.3

            # 地點篩選
            if location:
                loc = info.get("location", "")
                if location.lower() not in loc.lower():
                    matched = False
                else:
                    score += 0.5

            # 語義查詢（標注文字匹配）
            if query:
                label = info.get("label", "").lower()
                if query.lower() in label:
                    score += 1.0
                else:
                    # 不完全匹配，但不排除
                    score += 0.0

            if matched:
                results.append({
                    "path": path,
                    "score": score,
                    "label": info.get("label", ""),
                    "faces": info.get("faces", []),
                    "time": info.get("time"),
                    "location": info.get("location"),
                })

        # 排序（分數高的在前）
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def name_face(self, face_id: str, name: str) -> None:
        """為人臉命名。"""
        self.face_names[face_id] = name
        self.face_names_path.parent.mkdir(parents=True, exist_ok=True)
        self.face_names_path.write_text(json.dumps(self.face_names, indent=2, ensure_ascii=False))
        LOGGER.info("Named %s → %s", face_id, name)

    def get_unnamed_faces(self) -> List[Dict]:
        """取得未命名的人臉群組。"""
        if not self.index:
            return []

        # 從 face_clusters.json 載入
        faces_path = self.project_dir / "data" / "faces" / "face_clusters.json"
        if not faces_path.exists():
            return []

        with open(faces_path, "r", encoding="utf-8") as f:
            faces_data = json.load(f)

        unnamed = []
        for face_id, info in faces_data.get("clusters", {}).items():
            if face_id not in self.face_names:
                unnamed.append({
                    "id": face_id,
                    "count": info["count"],
                    "images": info["images"][:5],  # 前 5 張預覽
                })

        return unnamed

    def get_face_summary(self) -> Dict:
        """取得人臉摘要。"""
        if not self.index:
            return {}

        faces_path = self.project_dir / "data" / "faces" / "face_clusters.json"
        if not faces_path.exists():
            return {}

        with open(faces_path, "r", encoding="utf-8") as f:
            faces_data = json.load(f)

        summary = {}
        for face_id, info in faces_data.get("clusters", {}).items():
            name = self.face_names.get(face_id, face_id)
            summary[face_id] = {
                "name": name,
                "count": info["count"],
                "sample_images": info["images"][:3],
            }

        return summary


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="統一照片索引引擎")
    sub = p.add_subparsers(dest="command", required=True)

    # build 子命令
    build_p = sub.add_parser("build", help="建立索引")
    build_p.add_argument("--labels", required=True, help="labels.json 路徑")
    build_p.add_argument("--faces", default=None, help="face_clusters.json 路徑")
    build_p.add_argument("--embeddings", default=None, help="embeddings.pkl 路徑（用於 EXIF）")

    # search 子命令
    search_p = sub.add_parser("search", help="搜尋照片")
    search_p.add_argument("query", nargs="?", default="", help="語義查詢")
    search_p.add_argument("--face", default=None, help="人臉名稱篩選")
    search_p.add_argument("--year", type=int, default=None, help="年份篩選")
    search_p.add_argument("--month", type=int, default=None, help="月份篩選")
    search_p.add_argument("--location", default=None, help="地點篩選")
    search_p.add_argument("--top", type=int, default=20, help="回傳數量")
    search_p.add_argument("--json", action="store_true", help="JSON 輸出")

    # name-face 子命令
    name_p = sub.add_parser("name-face", help="為人臉命名")
    name_p.add_argument("face_id", help="人臉 ID (如 face_0)")
    name_p.add_argument("name", help="名稱")

    # list-faces 子命令
    sub.add_parser("list-faces", help="列出人臉")

    # status 子命令
    sub.add_parser("status", help="索引狀態")

    p.add_argument("--project-dir", default=".", help="專案目錄")
    p.add_argument("--log-level", default="INFO")

    args = p.parse_args(argv)
    _setup_logging(args.log_level)

    index = PhotoIndex(args.project_dir)

    if args.command == "build":
        index.build(
            labels_path=args.labels,
            faces_path=args.faces,
            embeddings_path=args.embeddings,
        )
        print(f"Index built: {index.index.get('total_images', 0)} images")

    elif args.command == "search":
        results = index.search(
            query=args.query,
            face_name=args.face,
            year=args.year,
            month=args.month,
            location=args.location,
            top_k=args.top,
        )
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(f"\n🔍 搜尋結果：{len(results)} 張\n")
            for i, r in enumerate(results, 1):
                faces = ", ".join(f["name"] for f in r["faces"]) if r["faces"] else "無人臉"
                time_str = r["time"] or "未知時間"
                print(f"  {i}. {Path(r['path']).name}")
                print(f"     👤 {faces}")
                print(f"     📅 {time_str}")
                if r["label"]:
                    label_preview = r["label"][:60] + "..." if len(r["label"]) > 60 else r["label"]
                    print(f"     🏷️  {label_preview}")
                print()

    elif args.command == "name-face":
        index.name_face(args.face_id, args.name)
        print(f"已命名：{args.face_id} → {args.name}")

    elif args.command == "list-faces":
        summary = index.get_face_summary()
        if not summary:
            print("尚未建立人臉索引。請先執行 build。")
        else:
            print(f"\n👥 人臉摘要：\n")
            for face_id, info in summary.items():
                print(f"  {face_id}: {info['name']} ({info['count']} 張)")

    elif args.command == "status":
        if not index.index:
            print("索引尚未建立。")
        else:
            print(f"總圖片：{index.index.get('total_images', 0)}")
            print(f"建立時間：{index.index.get('built_at', '未知')}")
            print(f"已命名人臉：{len(index.face_names)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
