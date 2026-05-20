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
from datetime import datetime, timedelta
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
    @staticmethod
    def _parse_filename_time(path: str) -> Optional[datetime]:
        """Best-effort time recovery from common filename patterns.
        Handles: IMG_YYYYMMDD_HHMMSS, IMG-YYYYMMDD-WA*, YYYY-MM-DD HH.MM.SS,
                 YYYYMMDD_HHMMSS, screenshots like Screenshot_YYYYMMDD-*.
        """
        import re as _re
        name = Path(path).name
        # YYYYMMDD_HHMMSS  (most common camera output)
        m = _re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})[_\- ]?(\d{2})(\d{2})(\d{2})(?!\d)", name)
        if m:
            y, mo, d, h, mi, s = (int(x) for x in m.groups())
        else:
            # YYYY-MM-DD HH.MM.SS or YYYY-MM-DD_HH-MM-SS
            m = _re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})[ _.\-](\d{2})[.\-_](\d{2})[.\-_](\d{2})", name)
            if m:
                y, mo, d, h, mi, s = (int(x) for x in m.groups())
            else:
                # Date only: YYYYMMDD or YYYY-MM-DD — use noon to land on the correct calendar day
                m = _re.search(r"(?<!\d)(\d{4})[-_]?(\d{2})[-_]?(\d{2})(?!\d)", name)
                if not m:
                    return None
                y, mo, d = (int(x) for x in m.groups())
                h, mi, s = 12, 0, 0
        try:
            return datetime(y, mo, d, h, mi, s)
        except ValueError:
            return None

    def _fill_missing_time(self, images: Dict[str, Dict]) -> None:
        """For images without ``time``, try (1) PIL EXIF re-read from disk,
        (2) filename pattern parsing, (3) file mtime as last resort.

        Stamps ``time``, ``year``, ``month`` and a ``time_source`` tag so
        downstream consumers can tell how trustworthy the timestamp is.
        """
        missing = [p for p, info in images.items() if not info.get("time")]
        if not missing:
            return
        LOGGER.info("Backfilling time for %d photos without EXIF…", len(missing))

        try:
            from PIL import Image, ExifTags
            tag_map = {v: k for k, v in ExifTags.TAGS.items()}
            DTO = tag_map.get("DateTimeOriginal")
            DTD = tag_map.get("DateTimeDigitized")
            DT  = tag_map.get("DateTime")
            DATE_TAGS = [t for t in (DTO, DTD, DT) if t]
        except ImportError:
            Image, DATE_TAGS = None, []

        stats = {"exif": 0, "filename": 0, "mtime": 0, "missing_file": 0}
        for i, path in enumerate(missing):
            if i and i % 5000 == 0:
                LOGGER.info("  …%d / %d", i, len(missing))
            p = Path(path)
            if not p.exists():
                stats["missing_file"] += 1
                continue

            dt = None; source = None

            # 1. PIL EXIF re-read
            if Image is not None:
                try:
                    with Image.open(p) as im:
                        ex = im.getexif()
                        for tid in DATE_TAGS:
                            if not tid: continue
                            v = ex.get(tid)
                            if not v: continue
                            try:
                                dt = datetime.strptime(str(v).strip(), "%Y:%m:%d %H:%M:%S")
                                source = "exif"
                                break
                            except ValueError:
                                pass
                except Exception:  # noqa: BLE001
                    pass

            # 2. Filename pattern
            if dt is None:
                dt = self._parse_filename_time(path)
                if dt is not None:
                    source = "filename"

            # 3. mtime fallback
            if dt is None:
                try:
                    dt = datetime.fromtimestamp(p.stat().st_mtime)
                    source = "mtime"
                except Exception:  # noqa: BLE001
                    continue

            stats[source] = stats.get(source, 0) + 1
            images[path]["time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            images[path]["year"] = dt.year
            images[path]["month"] = dt.month
            images[path]["time_source"] = source

        LOGGER.info(
            "Time backfill done. EXIF re-read: %d, filename: %d, mtime: %d, missing file: %d",
            stats["exif"], stats["filename"], stats["mtime"], stats["missing_file"],
        )

    @staticmethod
    def _parse_gps(gps: dict) -> Optional[Tuple[float, float]]:
        """
        Convert a PIL-style EXIF GPSInfo dict to (lat, lng) decimal degrees.

        PIL leaves GPSInfo as tag-numbered (1/2/3/4 for ref/lat/ref/lng);
        keys may be int or str depending on whether the dict has been
        round-tripped through JSON. Values for lat/lng are DMS tuples
        (degrees, minutes, seconds).
        """
        if not isinstance(gps, dict):
            return None

        def g(*keys):
            for k in keys:
                if k in gps:
                    return gps[k]
            return None

        lat_ref = g(1, "1", "GPSLatitudeRef") or "N"
        lat_dms = g(2, "2", "GPSLatitude")
        lng_ref = g(3, "3", "GPSLongitudeRef") or "E"
        lng_dms = g(4, "4", "GPSLongitude")
        if not lat_dms or not lng_dms:
            return None
        try:
            # DMS tuple → decimal degrees
            if hasattr(lat_dms, "__len__") and len(lat_dms) == 3:
                lat = float(lat_dms[0]) + float(lat_dms[1]) / 60 + float(lat_dms[2]) / 3600
            else:
                lat = float(lat_dms)  # already decimal
            if hasattr(lng_dms, "__len__") and len(lng_dms) == 3:
                lng = float(lng_dms[0]) + float(lng_dms[1]) / 60 + float(lng_dms[2]) / 3600
            else:
                lng = float(lng_dms)
        except (TypeError, ValueError, IndexError):
            return None
        if str(lat_ref).upper().startswith("S"):
            lat = -lat
        if str(lng_ref).upper().startswith("W"):
            lng = -lng
        # Discard nonsense values (some EXIF writers leave (0,0,0))
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return None
        if lat == 0 and lng == 0:
            return None
        return lat, lng

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

    def _backfill_location_names(self, images: Dict[str, Dict]) -> None:
        """
        為 ``images`` 中有 GPS 的照片回填 ``location_name``。

        策略：
        1. 收集所有唯一 (lat, lng) 座標（rounded 到小數點後 4 位 ≈ 11m 精度）
        2. 一次性呼叫 ``reverse_geocoder.search`` 拿地名（離線、純 numpy）
        3. 若 ``location_names.json`` 有自訂覆寫（key=``"lat,lng"``），用使用者值
        """
        try:
            import reverse_geocoder as rg
        except ImportError:
            LOGGER.warning(
                "reverse_geocoder not installed; skipping --with-location. "
                "Install with: pip install reverse-geocoder"
            )
            return

        coord_to_paths: Dict[Tuple[float, float], List[str]] = {}
        for path, info in images.items():
            gps = info.get("gps")
            if not gps:
                continue
            lat, lng = gps.get("GPSLatitude"), gps.get("GPSLongitude")
            if lat is None or lng is None:
                continue
            key = (round(float(lat), 4), round(float(lng), 4))
            coord_to_paths.setdefault(key, []).append(path)

        if not coord_to_paths:
            LOGGER.info("No GPS coordinates found; nothing to reverse-geocode.")
            return

        coords = list(coord_to_paths.keys())
        LOGGER.info("Reverse-geocoding %d unique coordinates (%d photos)...",
                    len(coords), sum(len(p) for p in coord_to_paths.values()))
        # mode=1 uses single-thread; cheaper for our scale + avoids the noisy
        # "Loading formatted geocoded file..." reload on subsequent calls.
        results = rg.search(coords, mode=1)

        # Build coord → "City, CC" map, apply user overrides from location_names.json
        coord_to_name: Dict[Tuple[float, float], str] = {}
        for coord, r in zip(coords, results):
            city = r.get("name", "").strip()
            cc = r.get("cc", "").strip()
            coord_to_name[coord] = f"{city}, {cc}" if city and cc else (city or cc or "Unknown")

        # User overrides ("lat,lng" key) win over auto-geocoded names
        overrides_applied = 0
        for coord in coords:
            key_str = f"{coord[0]:.4f},{coord[1]:.4f}"
            if key_str in self.location_names:
                coord_to_name[coord] = self.location_names[key_str]
                overrides_applied += 1

        # Stamp location_name onto every image
        stamped = 0
        for coord, paths in coord_to_paths.items():
            name = coord_to_name[coord]
            for path in paths:
                images[path]["location_name"] = name
                stamped += 1

        LOGGER.info(
            "Reverse-geocoded %d photos across %d places (%d user overrides applied)",
            stamped, len(set(coord_to_name.values())), overrides_applied,
        )

    def build(
        self,
        labels_paths=None,
        faces_path: Optional[str] = None,
        embeddings_paths=None,
        with_location: bool = False,
        # Back-compat single-string aliases
        labels_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
    ) -> Dict:
        """
        建立統一索引。

        Args:
            labels_paths: list of labels.json file paths OR directories (auto-discover *.json).
                Each photo root may have its own labels sidecar; the union is consumed.
            faces_path: face_clusters.json 路徑（全域）
            embeddings_paths: list of embeddings.pkl files OR directories.
            with_location: 啟用反向地理編碼（離線，需 ``reverse-geocoder`` 套件）
                為每張有 GPS 的照片回填 ``location_name`` 欄位（如 ``Taipei, TW``）。
                用戶可在 ``index/location_names.json`` 用 ``"lat,lng" → "name"``
                覆寫自動結果（精度為小數點後 4 位）。

        Returns:
            索引結構
        """
        # Back-compat: accept the old single-string form too
        if labels_path and not labels_paths:
            labels_paths = [labels_path]
        if embeddings_path and not embeddings_paths:
            embeddings_paths = [embeddings_path]
        labels_paths = labels_paths or []
        embeddings_paths = embeddings_paths or []

        # Resolve each input to a concrete list of files (expand directories)
        def _expand(paths_or_dirs, glob_pat: str) -> List[Path]:
            files: List[Path] = []
            for arg in paths_or_dirs:
                p = Path(arg)
                if p.is_dir():
                    files.extend(sorted(x for x in p.glob(glob_pat) if not x.name.startswith(".")))
                elif p.exists():
                    files.append(p)
                else:
                    LOGGER.warning("Skipping missing path: %s", p)
            return files

        labels_files = _expand(labels_paths, "*.json")
        if not labels_files:
            raise RuntimeError("No labels file given. Pass --labels at least once.")

        LOGGER.info("Building unified index from %d labels file(s)...", len(labels_files))

        images: Dict[str, Dict] = {}
        # 1. 載入 labels（multi-source: union all results, dedupe by path）
        for lp in labels_files:
            LOGGER.info("Loading labels from %s", lp)
            with open(lp, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
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

        LOGGER.info("Loaded %d images across all labels files", len(images))

        # 2. 載入 EXIF — embeddings_paths 是 list，每個元素可以是檔案或目錄
        pkl_files: List[Path] = _expand(embeddings_paths, "*.pkl")
        if pkl_files:

            exif_merged = 0
            gps_extracted = 0
            for pkl in pkl_files:
                LOGGER.info("Loading EXIF from %s", pkl)
                with open(pkl, "rb") as f:
                    emb_data = pickle.load(f)
                for path, exif in zip(emb_data.get("paths", []), emb_data.get("exif", [])):
                    if not (path in images and exif):
                        continue
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
                    # GPS：PIL 給的 GPSInfo 是 tag-numbered dict（key 為 int 或 str("1"/"2")）。
                    # tag 1=GPSLatitudeRef, 2=GPSLatitude (DMS tuple), 3=GPSLongitudeRef,
                    # 4=GPSLongitude (DMS tuple)
                    gps_raw = exif.get("GPSInfo")
                    if gps_raw:
                        latlng = self._parse_gps(gps_raw)
                        if latlng is not None:
                            lat, lng = latlng
                            images[path]["gps"] = {"GPSLatitude": lat, "GPSLongitude": lng}
                            images[path]["location"] = f"{lat:.4f},{lng:.4f}"
                            gps_extracted += 1
                    if images[path].get("time") or images[path].get("gps"):
                        exif_merged += 1
            LOGGER.info("Merged EXIF data: %d photos got time/GPS (%d with GPS)",
                        exif_merged, gps_extracted)

        # 2.5 補時間：對沒 time 的照片，從磁碟讀 EXIF / 解析檔名 / fallback mtime
        # (PIL 讀 EXIF 不解 pixel 資料很快，~5ms/file × 50k 約 4 分鐘，one-time cost)
        self._fill_missing_time(images)

        # 2.6 反向地理編碼（GPS → 城市/國家），可選
        if with_location:
            self._backfill_location_names(images)

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

    # ------------------------------------------------------------------
    # Timeline events (v0.6 P3)
    # ------------------------------------------------------------------

    def build_timeline_events(
        self,
        gap_hours: float = 6.0,
        min_photos: int = 3,
    ) -> Dict:
        """
        Aggregate photo_index into "event" clusters and write
        ``index/timeline_events.json``.

        Algorithm (sliding window over time):
          1. Take every photo with a ``time`` field, sort ascending
          2. Walk in order; whenever the gap between consecutive photos
             exceeds ``gap_hours`` (default 6 h), start a new segment
          3. Drop segments shorter than ``min_photos`` (default 3)
          4. For each surviving segment compute:
             - start / end timestamps + duration_days
             - top 3 face_ids by appearance count
             - top 3 place names (from location_name)
             - cover photo (medoid heuristic: middle-time photo,
               preferring one with at least one face)

        The output is saved next to photo_index.json so the server can
        serve /api/timeline without recomputing.
        """
        if not self.index:
            from_disk = json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else None
            if not from_disk:
                raise RuntimeError("photo_index not built; run `build` first")
            self.index = from_disk

        images = self.index.get("images", {})
        # Sort all timed photos ascending; carry (path, dt, info) tuples for speed
        timed: List[Tuple[str, datetime, Dict]] = []
        for path, info in images.items():
            t = info.get("time")
            if not t: continue
            try:
                dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            timed.append((path, dt, info))
        timed.sort(key=lambda x: x[1])
        LOGGER.info("Timeline source: %d timed photos", len(timed))

        gap = timedelta(hours=gap_hours)
        segments: List[List[Tuple[str, datetime, Dict]]] = []
        current: List[Tuple[str, datetime, Dict]] = []
        for entry in timed:
            if current and (entry[1] - current[-1][1]) > gap:
                if len(current) >= min_photos:
                    segments.append(current)
                current = []
            current.append(entry)
        if len(current) >= min_photos:
            segments.append(current)
        LOGGER.info("Aggregated into %d events (gap=%.1fh, min_photos=%d)",
                    len(segments), gap_hours, min_photos)

        events: List[Dict] = []
        for seg in segments:
            paths = [e[0] for e in seg]
            start_dt, end_dt = seg[0][1], seg[-1][1]
            duration_days = max(1, (end_dt.date() - start_dt.date()).days + 1)

            # Top faces (count appearances; faces is overlay-resolved already)
            face_counts: Dict[str, int] = {}
            for _, _, info in seg:
                for f in info.get("faces", []):
                    fid = f.get("face_id")
                    if fid:
                        face_counts[fid] = face_counts.get(fid, 0) + 1
            top_faces = [fid for fid, _ in
                         sorted(face_counts.items(), key=lambda kv: -kv[1])[:3]]

            # Top places — order by appearance count
            place_counts: Dict[str, int] = {}
            for _, _, info in seg:
                name = info.get("location_name")
                if name:
                    place_counts[name] = place_counts.get(name, 0) + 1
            top_places = [n for n, _ in
                          sorted(place_counts.items(), key=lambda kv: -kv[1])[:3]]

            # Cover heuristic: prefer middle-time photo with faces; fallback to plain middle
            mid_idx = len(seg) // 2
            cover = seg[mid_idx][0]
            with_faces = [p for p, _, info in seg if info.get("faces")]
            if with_faces:
                # Pick the with-faces photo nearest the middle
                mid_path = seg[mid_idx][0]
                if mid_path in with_faces:
                    cover = mid_path
                else:
                    cover = min(with_faces, key=lambda p: abs(paths.index(p) - mid_idx))

            events.append({
                "id": f"evt_{start_dt.strftime('%Y%m%d_%H%M%S')}",
                "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end":   end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_days": duration_days,
                "photo_count": len(paths),
                "photos": paths,
                "top_faces": top_faces,
                "top_places": top_places,
                "cover": cover,
            })

        # newest first for serve
        events.sort(key=lambda e: e["start"], reverse=True)

        out = {
            "built_at": datetime.now().isoformat(),
            "gap_hours": gap_hours,
            "min_photos": min_photos,
            "total_events": len(events),
            "events": events,
        }
        out_path = self.index_path.parent / "timeline_events.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        LOGGER.info("Timeline saved → %s (%d events)", out_path, len(events))
        return out

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
    # --labels and --embeddings can be repeated (one per photo root sidecar)
    # AND/OR point to a directory (auto-discover *.json / *.pkl inside).
    build_p.add_argument("--labels", required=True, action="append", default=[],
                         help="labels.json 檔或包含多個 *.json 的目錄（可重複，每個對應一個 photo root）")
    build_p.add_argument("--faces", default=None, help="face_clusters.json 路徑（全域）")
    build_p.add_argument("--embeddings", action="append", default=[],
                         help="embeddings .pkl 檔或目錄（可重複；用於 EXIF）")
    build_p.add_argument(
        "--with-location", action="store_true",
        help="啟用反向地理編碼，為有 GPS 的照片回填 location_name（需 reverse-geocoder 套件）",
    )

    # build-timeline 子命令
    tl_p = sub.add_parser("build-timeline", help="從現有 photo_index 聚合「事件」並寫 timeline_events.json")
    tl_p.add_argument("--gap-hours", type=float, default=6.0, help="切段門檻（相鄰照片間隔小時，預設 6）")
    tl_p.add_argument("--min-photos", type=int, default=3, help="最小事件照片數（預設 3）")

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
            labels_paths=args.labels,
            faces_path=args.faces,
            embeddings_paths=args.embeddings,
            with_location=args.with_location,
        )
        print(f"Index built: {index.index.get('total_images', 0)} images")

    elif args.command == "build-timeline":
        out = index.build_timeline_events(
            gap_hours=args.gap_hours,
            min_photos=args.min_photos,
        )
        print(f"Timeline built: {out['total_events']} events "
              f"(gap={out['gap_hours']}h, min_photos={out['min_photos']})")

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
