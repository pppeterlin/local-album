#!/usr/bin/env python3
"""
face_naming_server.py — 人臉命名伺服器 v4

功能：
  • 真分頁（prev/next，每頁取代上一頁，記憶體可控）
  • 命名、編輯、取消命名
  • 略過：後端持久化，自動排到最後
  • 合併：非破壞性折疊（source 隱藏、images 合進 target；可在 face_merges.json 反查）
  • 展開群組、移除/恢復照片
"""

import json
import os
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PROJECT_ROOT as PROJECT_DIR, FACES_DIR, PETS_DIR, INDEX_DIR  # noqa: E402
import _auth  # noqa: E402
from locale_zh import zh_place as _zh_place_raw  # noqa: E402


def _zh(name):
    """Translate reverse-geocoded place name to 繁中 (user override file aware)."""
    return _zh_place_raw(name, INDEX_DIR / "location_names_zh.json")

# In-memory caches of derived JSON files (photo_index, timeline_events, memories).
# Keyed by file mtime so edits via the Photo_Index CLI hot-reload without restart.
_PHOTO_INDEX_CACHE: dict = {"mtime": 0.0, "data": None}
_TIMELINE_CACHE:    dict = {"mtime": 0.0, "data": None}
_PETS_CACHE:        dict = {"mtime": 0.0, "data": None}
_PATH_TO_PETS_CACHE: dict = {"mtime": 0.0, "data": None}
_PATH_TO_FACES_CACHE: dict = {"mtime": 0.0, "data": None}


def _mtime_cached(path: Path, cache: dict):
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if cache["mtime"] != mtime:
        cache["data"] = json.loads(path.read_text(encoding="utf-8"))
        cache["mtime"] = mtime
    return cache["data"]


def _load_photo_index():
    return _mtime_cached(INDEX_DIR / "photo_index.json", _PHOTO_INDEX_CACHE)


def _load_timeline():
    return _mtime_cached(INDEX_DIR / "timeline_events.json", _TIMELINE_CACHE)


def _load_pets():
    return _mtime_cached(PETS_DIR / "pet_clusters.json", _PETS_CACHE)


def _path_to_faces() -> Dict[str, List[str]]:
    """Map photo path → list of face_id strings (from photo_index 'faces').
    Cached by photo_index.json mtime so memories/overview/timeline don't
    have to rebuild this 57k-entry map on every request."""
    idx_path = INDEX_DIR / "photo_index.json"
    if not idx_path.exists():
        return {}
    mtime = idx_path.stat().st_mtime
    if _PATH_TO_FACES_CACHE["mtime"] != mtime:
        idx = _load_photo_index() or {}
        m: Dict[str, List[str]] = {}
        for p, info in idx.get("images", {}).items():
            ids = [f.get("id", "") for f in info.get("faces", []) if f.get("id")]
            if ids:
                m[p] = ids
        _PATH_TO_FACES_CACHE["data"] = m
        _PATH_TO_FACES_CACHE["mtime"] = mtime
    return _PATH_TO_FACES_CACHE["data"] or {}


def _path_to_pets() -> Dict[str, List[str]]:
    """Map photo path → list of pet_id strings for ALL photos.
    Cached by pet_clusters.json mtime so it auto-refreshes on rebuild."""
    pets = _load_pets()
    if not pets:
        return {}
    pets_path = PETS_DIR / "pet_clusters.json"
    mtime = pets_path.stat().st_mtime
    if _PATH_TO_PETS_CACHE["mtime"] != mtime:
        m: Dict[str, List[str]] = {}
        for p, entries in (pets.get("images") or {}).items():
            ids = [e.get("pet_id", "") for e in entries if e.get("pet_id")]
            if ids:
                m[p] = ids
        _PATH_TO_PETS_CACHE["data"] = m
        _PATH_TO_PETS_CACHE["mtime"] = mtime
    return _PATH_TO_PETS_CACHE["data"] or {}

# Template directory (HTML extracted from this file in v0.5).
# Read on every request — files are small, OS page cache handles it,
# and re-read enables editing templates without restarting the server.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(name: str) -> str:
    html = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    if "<!--V05_ASSETS-->" in html:
        assets = (TEMPLATES_DIR / "_assets.html").read_text(encoding="utf-8")
        html = html.replace("<!--V05_ASSETS-->", assets)
    return html

# File path sets per kind ("face" | "pet"). Both share the same on-disk
# schema (clusters / names / merges / moves / removed / skipped / thumb
# overrides + a thumbs directory). Pets live under METADATA_DIR/pets/.
KIND_PATHS = {
    "face": {
        "clusters": FACES_DIR / "face_clusters.json",
        "names": FACES_DIR / "face_names.json",
        "removed": FACES_DIR / "face_removed.json",
        "skipped": FACES_DIR / "face_skipped.json",
        "merges": FACES_DIR / "face_merges.json",
        "moves": FACES_DIR / "face_moves.json",
        "thumb_overrides": FACES_DIR / "face_thumb_overrides.json",
        "thumbs": FACES_DIR / "face_thumbs",
        "id_prefix": "face",
    },
    "pet": {
        "clusters": PETS_DIR / "pet_clusters.json",
        "names": PETS_DIR / "pet_names.json",
        "removed": PETS_DIR / "pet_removed.json",
        "skipped": PETS_DIR / "pet_skipped.json",
        "merges": PETS_DIR / "pet_merges.json",
        "moves": PETS_DIR / "pet_moves.json",
        "thumb_overrides": PETS_DIR / "pet_thumb_overrides.json",
        "thumbs": PETS_DIR / "pet_thumbs",
        "id_prefix": "pet",
    },
}

# 圖片縮圖 cache 兩種共用（縮圖跟分類無關）
IMG_THUMB_CACHE = FACES_DIR / "img_thumb_cache"
PAGE_SIZE = 24  # divisible by 2/3/4/6 so most grid breakpoints fill cleanly

# --- backward-compat aliases (face) — old code paths still reference these names
FACES_FILE = KIND_PATHS["face"]["clusters"]
NAMES_FILE = KIND_PATHS["face"]["names"]
REMOVED_FILE = KIND_PATHS["face"]["removed"]
SKIPPED_FILE = KIND_PATHS["face"]["skipped"]
MERGES_FILE = KIND_PATHS["face"]["merges"]
MOVES_FILE = KIND_PATHS["face"]["moves"]
THUMB_OVERRIDES_FILE = KIND_PATHS["face"]["thumb_overrides"]
THUMBS_DIR = KIND_PATHS["face"]["thumbs"]


def load_json(path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default if default is not None else {}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# Kind-aware loaders (kind="face" | "pet"). face_naming_server originally
# was face-only with global FACES_FILE/etc.; we keep the old wrappers as
# face-specific aliases so legacy code paths keep working.
def load_clusters(kind):   return load_json(KIND_PATHS[kind]["clusters"])
def load_names_k(kind):    return load_json(KIND_PATHS[kind]["names"])
def load_removed_k(kind):  return load_json(KIND_PATHS[kind]["removed"], {})
def load_skipped_k(kind):  return load_json(KIND_PATHS[kind]["skipped"], [])
def load_merges_k(kind):   return load_json(KIND_PATHS[kind]["merges"], {})
def load_moves_k(kind):    return load_json(KIND_PATHS[kind]["moves"], [])
def save_names_k(kind, n):   save_json(KIND_PATHS[kind]["names"], n)
def save_removed_k(kind, r): save_json(KIND_PATHS[kind]["removed"], r)
def save_skipped_k(kind, s): save_json(KIND_PATHS[kind]["skipped"], s)
def save_merges_k(kind, m):  save_json(KIND_PATHS[kind]["merges"], m)
def save_moves_k(kind, m):   save_json(KIND_PATHS[kind]["moves"], m)

# Face-specific aliases (legacy)
def load_faces():   return load_clusters("face")
def load_names():   return load_names_k("face")
def load_removed(): return load_removed_k("face")
def load_skipped(): return load_skipped_k("face")
def load_merges():  return load_merges_k("face")
def load_moves():   return load_moves_k("face")
def save_names(n):   save_names_k("face", n)
def save_removed(r): save_removed_k("face", r)
def save_skipped(s): save_skipped_k("face", s)
def save_merges(m):  save_merges_k("face", m)
def save_moves(m):   save_moves_k("face", m)


def _next_user_id(clusters_data, moves, kind: str = "face") -> str:
    """產生下一個 user-created id（face_uN / pet_uN）。避開既有 cluster 與 moves target。"""
    prefix = KIND_PATHS[kind]["id_prefix"]
    existing = set((clusters_data.get("clusters") or {}).keys())
    for m in moves:
        t = m.get("to", "")
        if t:
            existing.add(t)
    n = 1
    while f"{prefix}_u{n}" in existing:
        n += 1
    return f"{prefix}_u{n}"


def _next_user_face_id(faces, moves) -> str:
    """Legacy alias for face kind."""
    return _next_user_id(faces, moves, "face")


SKIP_PATH_FRAGMENTS = (
    "/.thumbnails/", "/.trash/", "/.trashes/", "/.cache/",
    "/@eadir/", "/__macosx/", "/.spotlight-v100/", "/.fseventsd/",
)


def _is_skip_path(p: str) -> bool:
    if not p:
        return True
    pl = p.lower().replace("\\", "/")
    for frag in SKIP_PATH_FRAGMENTS:
        if frag in pl:
            return True
    # 任何路徑段以 . 開頭（如 /.thumbnails/、/foo/.cache/...）
    for part in p.split("/"):
        if part.startswith(".") and part not in (".", ".."):
            return True
    return False


def _resolve_target(merges, fid):
    """跟著 merge 鏈走到最終 target（防止 A→B→C 失效）。"""
    seen = set()
    while fid in merges and fid not in seen:
        seen.add(fid)
        fid = merges[fid]
    return fid




class Handler(SimpleHTTPRequestHandler):
    # --- auth helpers --------------------------------------------------
    def _user(self):
        if not hasattr(self, "_cached_user"):
            self._cached_user = _auth.extract_user_from_headers(self.headers)
        return self._cached_user

    def _perms(self):
        if not hasattr(self, "_cached_perms"):
            self._cached_perms = _auth.get_user_perms(self._user())
        return self._cached_perms

    def _relationships(self):
        if not hasattr(self, "_cached_rels"):
            self._cached_rels = _auth.load_relationships()
        return self._cached_rels

    def _personalize_name(self, fid: str, canonical: str) -> str:
        """套用 relationships.json + self-as-我 規則回傳顯示名稱。"""
        viewer_id = self._perms().get("identity") or ""
        return _auth.display_name_for(viewer_id, fid, canonical, self._relationships())

    def _require_login(self):
        """Return False (and serve login) if no valid session. True → continue."""
        if self._user():
            return True
        # /login HTML + /api/login POST allowed unauthenticated
        return False

    def _require_admin(self):
        p = self._perms()
        if not p["is_admin"]:
            self.send_error(403, "admin only")
            return False
        return True

    def _can_see_cluster(self, fid: str) -> bool:
        p = self._perms()
        if p["is_admin"]:
            return True
        return fid in p["allowed_faces"]

    def _can_see_image(self, img_path: str) -> bool:
        p = self._perms()
        if p["is_admin"]:
            return True
        merges = load_merges()
        faces = load_faces().get("images", {}).get(img_path, [])
        face_ids = [_resolve_target(merges, f.get("face_id", "")) for f in faces]
        # 套用 moves：被搬走的 face record 應該追新 target
        moves = load_moves()
        moves_map = {}
        for m in moves:
            if m.get("path") != img_path:
                continue
            f_final = _resolve_target(merges, m.get("from", ""))
            t_final = _resolve_target(merges, m.get("to", ""))
            moves_map[f_final] = t_final
        effective = [moves_map.get(fid, fid) for fid in face_ids]
        # v0.6: also pass pet_ids so /img_thumb/ + /image/ can serve pet-only
        # photos to viewers whose group whitelists those pets.
        pet_ids = _path_to_pets().get(img_path, [])
        return _auth.can_see_photo(p, img_path, effective, pet_ids)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # public routes
        if path in ("/login", "/login/"):
            self.html(self.generate_login_html())
            return
        if path == "/api/me":
            user = self._user()
            perms = self._perms()
            self.json_response({"user": user, "is_admin": perms["is_admin"], "is_viewer": perms["is_viewer"]})
            return

        # everything else requires login
        if not self._require_login():
            if path.startswith("/api/"):
                self.send_error(401, "login required")
            else:
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()
            return

        if path in ("/", "/index.html"):
            self.html(self.generate_html())

        elif path in ("/admin", "/admin/"):
            if not self._require_admin():
                return
            self.html(self.generate_admin_html())

        elif path in ("/overview", "/overview/"):
            self.html(_load_template("overview.html"))

        elif path == "/api/overview":
            self.json_response(self._build_overview_payload())

        elif path in ("/timeline", "/timeline/"):
            self.html(_load_template("timeline.html"))

        elif path == "/api/timeline":
            self.json_response(self._build_timeline_payload())

        elif path == "/api/timeline_event":
            eid = qs.get("id", [""])[0]
            self.json_response(self._build_timeline_event_detail(eid))

        elif path in ("/memories", "/memories/"):
            self.html(_load_template("memories.html"))

        elif path == "/api/memories":
            self.json_response(self._build_memories_payload())

        elif path == "/api/admin/users":
            if not self._require_admin(): return
            self.json_response(self._admin_list_users())

        elif path == "/api/admin/groups":
            if not self._require_admin(): return
            self.json_response(self._admin_list_groups())

        elif path == "/api/admin/faces":
            if not self._require_admin(): return
            self.json_response(self._admin_list_faces())

        elif path == "/api/admin/pets":
            if not self._require_admin(): return
            self.json_response(self._admin_list_pets())

        elif path == "/api/admin/paths":
            if not self._require_admin(): return
            self.json_response(self._admin_list_paths())

        elif path == "/api/admin/relationship-graph":
            if not self._require_admin(): return
            self.json_response({
                "graph": _auth.load_relationship_graph(),
                "family_types": list(_auth.FAMILY_EDGE_RULES.keys()),
                "non_family_types": sorted(_auth.NON_FAMILY_TYPES),
                # rules so the dialog can show default aliases as placeholders
                "family_rules": {k: list(v) for k, v in _auth.FAMILY_EDGE_RULES.items()},
            })

        elif path.startswith("/image/"):
            img = self.fix_path(path[7:])
            if not self._can_see_image(img):
                self.send_error(403); return
            self.serve_file(img, "image/jpeg")

        elif path.startswith("/thumb/") or path.startswith("/pet_thumb/"):
            # cluster thumbnail; face by default, /pet_thumb/ for pet kind.
            # 寵物 cluster 對所有 viewer 公開，face 仍受 _can_see_cluster 規範。
            if path.startswith("/pet_thumb/"):
                fid_filename = unquote(path[len("/pet_thumb/"):])
                kind = "pet"
            else:
                fid_filename = unquote(path[len("/thumb/"):])
                kind = "face"
            fid = fid_filename.rsplit(".", 1)[0]
            if kind == "face" and not self._can_see_cluster(fid):
                self.send_error(403); return
            thumb = KIND_PATHS[kind]["thumbs"] / fid_filename
            self.serve_file(thumb, "image/jpeg")

        elif path.startswith("/img_thumb/"):
            try:
                w = int(qs.get("w", ["256"])[0])
            except ValueError:
                w = 256
            w = max(64, min(w, 1024))
            orig = self.fix_path(path[11:])
            if not self._can_see_image(orig):
                self.send_error(403); return
            self.serve_img_thumb(orig, w)

        elif path == "/api/page":
            page = int(qs.get("page", [0])[0])
            flt = qs.get("filter", ["all"])[0]
            kind = qs.get("kind", ["face"])[0]
            if kind not in KIND_PATHS and kind != "all": kind = "face"
            self.json_response(self.get_page(page, flt, kind=kind))

        elif path == "/api/clusters":
            # 輕量列表，給合併下拉選單用
            kind = qs.get("kind", ["face"])[0]
            if kind not in KIND_PATHS and kind != "all": kind = "face"
            data = self.get_all_sorted("all", kind=kind)
            self.json_response([
                {"id": r["id"], "name": r["name"], "count": r["count"]}
                for r in data
            ])

        elif path == "/api/stats":
            kind = qs.get("kind", ["face"])[0]
            if kind not in KIND_PATHS and kind != "all": kind = "face"
            self.json_response(self.get_stats(kind=kind))

        elif path == "/api/cluster_meta":
            fid = qs.get("fid", [""])[0]
            kind = qs.get("kind", ["face"])[0]
            # 單一 cluster 必須是 face 或 pet；"all" 不合理 → fallback face
            if kind not in KIND_PATHS: kind = "face"
            self.json_response(self.get_cluster_meta(fid, kind=kind))

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self.read_body()

        # login / logout — open routes
        if path == "/api/login":
            username = (body.get("username") or "").strip()
            password = body.get("password") or ""
            users = _auth.load_users()
            u = users.get(username)
            if not u or not _auth.verify_password(password, u.get("password_hash", "")):
                self.send_response(401)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "invalid credentials"}).encode())
                return
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Set-Cookie", _auth.cookie_header(username))
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "user": username, "is_admin": u.get("role") == "admin"}).encode())
            return
        if path == "/api/logout":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Set-Cookie", _auth.cookie_header(None))
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            return

        # TODO(v0.3+): /api/me/password — let logged-in users change their own
        # password without needing admin / CLI access. Body: {old, new}.
        # Verify old via _auth.verify_password against load_users()[user],
        # update password_hash via _auth.hash_password, save_users.
        # UI side: add a small "改密碼" link in the userbar.

        # 從這裡開始都是 mutation，至少要登入
        if not self._user():
            self.send_error(401, "login required"); return

        # body 帶 kind 表示對 face 或 pet 動作（向下相容，預設 face）
        kind = body.get("kind") or "face"
        if kind not in KIND_PATHS:
            kind = "face"

        # /api/set_thumb：admin 全 cluster 可改；非 admin 只能改自己的 identity cluster
        # （讓 user 可以從自己照片裡挑一張當頭像，無須 admin）
        if path == "/api/set_thumb":
            perms = self._perms()
            fid = body.get("face_id") or ""
            img_path = body.get("image_path")
            if not perms["is_admin"]:
                if kind != "face" or not fid or fid != perms.get("identity"):
                    self.send_error(403, "只能修改自己 (identity) 的頭像")
                    return
            self.json_response(self._set_custom_thumb(fid, img_path, kind=kind))
            return

        # 其餘 mutation 都需要 admin
        if not self._require_admin():
            return

        if path == "/api/name":
            names = load_names_k(kind)
            fid, name = body.get("face_id"), body.get("name", "")
            if fid and name:
                names[fid] = name
            elif fid and fid in names:
                del names[fid]
            save_names_k(kind, names)
            self.json_response({"ok": True})

        elif path == "/api/skip":
            skipped = load_skipped_k(kind)
            fid = body.get("face_id")
            if fid:
                if fid in skipped:
                    skipped.remove(fid)
                else:
                    skipped.append(fid)
            save_skipped_k(kind, skipped)
            self.json_response({"ok": True, "skipped": fid in skipped})

        elif path == "/api/skip-batch":
            # body: {kind, face_ids: [...]} — idempotent，全部加入 skipped 集合
            skipped = load_skipped_k(kind)
            ids = body.get("face_ids") or []
            added = 0
            for fid in ids:
                if isinstance(fid, str) and fid and fid not in skipped:
                    skipped.append(fid)
                    added += 1
            save_skipped_k(kind, skipped)
            self.json_response({"ok": True, "added": added, "total_skipped": len(skipped)})

        elif path == "/api/merge":
            merges = load_merges_k(kind)
            names = load_names_k(kind)
            src, tgt = body.get("source_id"), body.get("target_id")
            if src and tgt and src != tgt:
                final = _resolve_target(merges, tgt)
                if final != src:
                    merges[src] = final
                    save_merges_k(kind, merges)
                    if final in names:
                        names[src] = names[final]
                        save_names_k(kind, names)
            self.json_response({"ok": True})

        elif path == "/api/unmerge":
            merges = load_merges_k(kind)
            fid = body.get("face_id")
            if fid and fid in merges:
                del merges[fid]
                save_merges_k(kind, merges)
            self.json_response({"ok": True})

        elif path == "/api/merge-batch":
            # body: {source_ids: [list], target_id, kind}
            merges = load_merges_k(kind)
            names = load_names_k(kind)
            srcs = body.get("source_ids") or []
            tgt = body.get("target_id")
            if not isinstance(srcs, list) or not tgt:
                self.json_response({"ok": False, "error": "需要 source_ids[] 與 target_id"})
                return
            final = _resolve_target(merges, tgt)
            merged_count = 0
            for src in srcs:
                if not isinstance(src, str) or not src or src == final:
                    continue
                # 避免循環：解 src 自己再看
                src_final = _resolve_target(merges, src)
                if src_final == final:
                    continue
                merges[src] = final
                if final in names:
                    names[src] = names[final]
                merged_count += 1
            save_merges_k(kind, merges)
            save_names_k(kind, names)
            self.json_response({"ok": True, "merged": merged_count, "target": final})

        elif path == "/api/remove":
            removed = load_removed_k(kind)
            fid, img_path = body.get("face_id"), body.get("image_path")
            if fid and img_path:
                removed.setdefault(fid, [])
                if img_path not in removed[fid]:
                    removed[fid].append(img_path)
                save_removed_k(kind, removed)
            self.json_response({"ok": True})

        elif path == "/api/restore":
            removed = load_removed_k(kind)
            fid, img_path = body.get("face_id"), body.get("image_path")
            if fid and img_path and fid in removed and img_path in removed[fid]:
                removed[fid].remove(img_path)
                save_removed_k(kind, removed)
            self.json_response({"ok": True})

        elif path == "/api/move":
            moves = load_moves_k(kind)
            from_id = body.get("from_id")
            to_id = body.get("to_id")
            img_path = body.get("image_path")
            new_name = (body.get("new_name") or "").strip()
            if to_id == "__new__":
                to_id = _next_user_id(load_clusters(kind), moves, kind)
                if new_name:
                    names = load_names_k(kind)
                    names[to_id] = new_name
                    save_names_k(kind, names)
            if from_id and to_id and img_path and from_id != to_id:
                moves = [m for m in moves if m.get("path") != img_path]
                moves.append({"path": img_path, "from": from_id, "to": to_id})
                save_moves_k(kind, moves)
            self.json_response({"ok": True, "to_id": to_id})

        elif path == "/api/unmove":
            moves = load_moves_k(kind)
            img_path = body.get("image_path")
            if img_path:
                moves = [m for m in moves if m.get("path") != img_path]
                save_moves_k(kind, moves)
            self.json_response({"ok": True})

        elif path == "/api/move-batch":
            moves = load_moves_k(kind)
            from_id = body.get("from_id")
            to_id = body.get("to_id")
            paths = body.get("paths", [])
            new_name = (body.get("new_name") or "").strip()
            if to_id == "__new__":
                to_id = _next_user_id(load_clusters(kind), moves, kind)
                if new_name:
                    names = load_names_k(kind)
                    names[to_id] = new_name
                    save_names_k(kind, names)
            if from_id and to_id and paths and from_id != to_id:
                ps = set(paths)
                moves = [m for m in moves if m.get("path") not in ps]
                for p in paths:
                    moves.append({"path": p, "from": from_id, "to": to_id})
                save_moves_k(kind, moves)
            self.json_response({"ok": True, "moved": len(paths), "to_id": to_id})


        elif path == "/api/admin/users":
            # body: {username, password, role, groups[]}
            self.json_response(self._admin_create_user(body))

        elif path.startswith("/api/admin/users/"):
            # body: {action: "update"|"delete", role?, groups?, password?}
            uname = unquote(path[len("/api/admin/users/"):])
            self.json_response(self._admin_modify_user(uname, body))

        elif path.startswith("/api/admin/groups/"):
            # body: {action: "upsert"|"delete", allowed_faces?, blocked_paths?}
            gname = unquote(path[len("/api/admin/groups/"):])
            self.json_response(self._admin_modify_group(gname, body))

        elif path == "/api/admin/relationship-graph":
            # body: {nodes: [face_id], edges: [{from, to, type, alias_from?, alias_to?}]}
            self.json_response(self._admin_save_relationship_graph(body))

        else:
            self.send_error(404)

    # ---- helpers ----

    def fix_path(self, p):
        p = unquote(p)
        while "//" in p:
            p = p.replace("//", "/")
        if not p.startswith("/"):
            p = "/" + p
        return p

    def read_body(self):
        cl = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(cl)) if cl else {}

    def json_response(self, data):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def html(self, content):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        # During v0.5 active dev: force browsers to re-fetch HTML so CSS/JS edits show up
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(content.encode())

    def serve_img_thumb(self, orig_path: str, width: int):
        """回傳 orig_path 的縮圖（長邊 width）；磁碟 cache。"""
        import hashlib
        from pathlib import Path as _P
        orig = _P(orig_path)
        if not orig.exists() or not orig.is_file():
            self.send_error(404, str(orig_path))
            return
        key = hashlib.md5(orig_path.encode("utf-8")).hexdigest()
        IMG_THUMB_CACHE.mkdir(parents=True, exist_ok=True)
        cache_file = IMG_THUMB_CACHE / f"{key}_{width}.jpg"
        if not cache_file.exists() or cache_file.stat().st_mtime < orig.stat().st_mtime:
            try:
                from PIL import Image, ImageOps
            except ImportError:
                self.send_error(500, "Pillow not installed")
                return
            try:
                with Image.open(orig_path) as im:
                    im = ImageOps.exif_transpose(im)
                    im.thumbnail((width, width), Image.LANCZOS)
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    im.save(cache_file, "JPEG", quality=82, optimize=True)
            except Exception as e:  # noqa: BLE001
                self.send_error(500, f"thumb failed: {e}")
                return
        self.send_response(200)
        self.send_header("Content-type", "image/jpeg")
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        with open(cache_file, "rb") as f:
            self.wfile.write(f.read())

    def serve_file(self, path, mime):
        if isinstance(path, str):
            path = Path(path)
        if path.exists() and path.is_file():
            self.send_response(200)
            self.send_header("Content-type", mime)
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, str(path))

    # ---- cluster metadata (mtime per photo, cached) ----

    _MTIME_CACHE: dict = {}

    def get_cluster_meta(self, fid: str, kind: str = "face") -> dict:
        """為單一 cluster 的圖片回傳 mtime / 年 / 月，並按時間 desc 排序。"""
        if not fid:
            return {"id": fid, "images": [], "removed": []}
        all_data = self.get_all_sorted("all", kind=kind)
        target = next((c for c in all_data if c["id"] == fid), None)
        if not target:
            return {"id": fid, "images": [], "removed": []}

        from concurrent.futures import ThreadPoolExecutor
        from datetime import datetime
        import os as _os

        cache = type(self)._MTIME_CACHE

        def fetch(p: str):
            if p in cache:
                return cache[p]
            try:
                ts = _os.stat(p).st_mtime
            except (OSError, ValueError):
                ts = 0.0
            cache[p] = ts
            return ts

        all_paths = list(target["images"]) + list(target.get("removed", []))
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(fetch, all_paths))

        def enrich(p: str) -> dict:
            ts = cache.get(p, 0.0)
            if ts > 0:
                dt = datetime.fromtimestamp(ts)
                return {"path": p, "ts": ts, "year": dt.year, "month": dt.month}
            return {"path": p, "ts": 0, "year": None, "month": None}

        active = sorted([enrich(p) for p in target["images"]],
                        key=lambda x: x["ts"], reverse=True)
        # viewer 不需要也不該看到「已移除」清單（避免路徑洩漏）
        if self._perms()["is_admin"]:
            removed = sorted([enrich(p) for p in target.get("removed", [])],
                             key=lambda x: x["ts"], reverse=True)
        else:
            removed = []
        return {"id": fid, "images": active, "removed": removed}

    # ---- thumb crop helper ----

    def _set_custom_thumb(self, fid: str, img_path: str, kind: str = "face") -> dict:
        """裁切指定照片裡屬於 fid (含合併源) 的人臉/寵物、覆寫 thumbs/<fid>.jpg"""
        if not fid or not img_path:
            return {"ok": False, "error": "missing face_id or image_path"}
        from pathlib import Path as _P
        if not _P(img_path).exists():
            return {"ok": False, "error": f"image not found: {img_path}"}

        # 解析合併源
        merges = load_merges_k(kind)
        merged_in = {fid}
        for src in merges.keys():
            if _resolve_target(merges, src) == fid:
                merged_in.add(src)

        # 在 clusters.images[img_path] 找對應的 record
        clusters_data = load_clusters(kind) or {}
        rec = (clusters_data.get("images", {}) or {}).get(img_path, [])
        target_rec = None
        id_key = "face_id" if kind == "face" else "pet_id"
        for r in rec:
            if r.get(id_key) in merged_in or r.get("face_id") in merged_in:  # 容錯
                target_rec = r
                break
        if not target_rec:
            return {"ok": False,
                    "error": f"找不到屬於 {fid} 的偵測於 {img_path}"}

        # 裁切（face 用 generate_face_thumbs，pet 用 OpenCV 直接 crop）
        out_path = KIND_PATHS[kind]["thumbs"] / f"{fid}.jpg"
        if kind == "face":
            try:
                from generate_face_thumbs import crop_face_thumb
            except ImportError:
                return {"ok": False, "error": "無法載入 crop_face_thumb"}
            ok = crop_face_thumb(img_path, target_rec["bbox"], out_path)
        else:
            ok = self._crop_pet_thumb(img_path, target_rec["bbox"], out_path)
        if not ok:
            return {"ok": False, "error": "裁切失敗（圖檔損壞或 bbox 無效）"}

        # 記錄手動指定
        overrides_file = KIND_PATHS[kind]["thumb_overrides"]
        overrides = load_json(overrides_file, {})
        overrides[fid] = img_path
        save_json(overrides_file, overrides)
        return {"ok": True, "thumb_path": str(out_path), "image_path": img_path}

    @staticmethod
    def _crop_pet_thumb(img_path: str, bbox: list, out_path) -> bool:
        """簡單的 bbox crop（沒有 face alignment），給寵物用。"""
        try:
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                return False
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = img.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return False
            crop = img[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            if max(ch, cw) > 256:
                s = 256 / max(ch, cw)
                crop = cv2.resize(crop, (int(cw * s), int(ch * s)))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- overview / timeline / memories (v0.6) ----

    def _build_overview_payload(self) -> dict:
        """Slim per-photo records for the /overview page, viewer-filtered.

        Returns {"photos": [...], "total": int} or {"error": ...} when
        the index hasn't been built yet (admin should run
        `Photo_Index.py build` to populate it).
        """
        idx = _load_photo_index()
        if not idx:
            return {
                "error": "photo_index not built",
                "hint": "uv run python src/Photo_Index.py build --labels ... --faces ... --embeddings ... --with-location",
                "photos": [],
                "total": 0,
            }
        perms = self._perms()
        pets_map = _path_to_pets()
        photos = []
        for path, info in idx.get("images", {}).items():
            if not info.get("time"):
                continue  # no timestamp → can't place on timeline
            face_ids = [f.get("id", "") for f in info.get("faces", []) if f.get("id")]
            if not _auth.can_see_photo(perms, path, face_ids, pets_map.get(path, [])):
                continue
            photos.append({
                "path": path,
                "time": info["time"],
                "year": info.get("year"),
                "month": info.get("month"),
                "location_name": _zh(info.get("location_name")),
            })
        # newest first
        photos.sort(key=lambda p: p["time"], reverse=True)
        return {"photos": photos, "total": len(photos)}

    def _build_timeline_payload(self) -> dict:
        """Viewer-filtered timeline events.

        For each event, drop any photos the viewer can't see; if fewer than 3
        survive, drop the entire event. Recompute top_faces / top_places /
        cover from the filtered photo set so privacy is preserved end-to-end.
        """
        tl = _load_timeline()
        if not tl:
            return {
                "error": "timeline not built",
                "hint": "uv run python src/Photo_Index.py build-timeline",
                "events": [],
                "total": 0,
            }
        idx = _load_photo_index()
        if not idx:
            return {"error": "photo_index not built", "events": [], "total": 0}

        perms = self._perms()
        pets_map = _path_to_pets()
        path_to_faces = _path_to_faces()  # mtime-cached, see _path_to_faces()

        filtered_events = []
        for evt in tl.get("events", []):
            visible = [p for p in evt["photos"]
                       if _auth.can_see_photo(perms, p, path_to_faces.get(p, []), pets_map.get(p, []))]
            if len(visible) < 3:
                continue
            # Recompute top_faces / top_places from visible photos only
            face_counts: Dict[str, int] = {}
            place_counts: Dict[str, int] = {}
            for p in visible:
                info = idx["images"].get(p, {})
                for fid in path_to_faces.get(p, []):
                    face_counts[fid] = face_counts.get(fid, 0) + 1
                ln = info.get("location_name")
                if ln:
                    place_counts[ln] = place_counts.get(ln, 0) + 1
            top_faces  = [k for k, _ in sorted(face_counts.items(),  key=lambda kv: -kv[1])[:3]]
            top_places = [_zh(k) for k, _ in sorted(place_counts.items(), key=lambda kv: -kv[1])[:3]]
            # Cover: keep original if still visible; else middle of filtered list
            cover = evt["cover"] if evt["cover"] in visible else visible[len(visible) // 2]
            filtered_events.append({
                "id": evt["id"],
                "start": evt["start"],
                "end": evt["end"],
                "duration_days": evt["duration_days"],
                "photo_count": len(visible),
                "top_faces": top_faces,
                "top_places": top_places,
                "cover": cover,
                # photos list is omitted from the listing payload to keep
                # responses small; fetched lazily via /api/timeline?event=<id>
            })

        return {"events": filtered_events, "total": len(filtered_events)}

    def _build_timeline_event_detail(self, event_id: str) -> dict:
        """Full per-event photo list (viewer-filtered) for the expand modal."""
        if not event_id:
            return {"error": "missing id"}
        tl = _load_timeline()
        idx = _load_photo_index()
        if not tl or not idx:
            return {"error": "timeline or index not built"}
        evt = next((e for e in tl.get("events", []) if e["id"] == event_id), None)
        if not evt:
            return {"error": "event not found"}
        perms = self._perms()
        pets_map = _path_to_pets()
        visible = []
        for p in evt["photos"]:
            face_ids = [f.get("id", "") for f in idx["images"].get(p, {}).get("faces", []) if f.get("id")]
            if _auth.can_see_photo(perms, p, face_ids, pets_map.get(p, [])):
                info = idx["images"].get(p, {})
                visible.append({"path": p, "time": info.get("time"), "location_name": _zh(info.get("location_name"))})
        return {
            "id": evt["id"],
            "start": evt["start"],
            "end": evt["end"],
            "duration_days": evt["duration_days"],
            "top_places": [_zh(p) for p in evt.get("top_places", [])],
            "photos": visible,
        }

    # ------------------------------------------------------------------
    # Memories (v0.6 P4)
    # ------------------------------------------------------------------
    # Mixed-source themed cards regenerated on each request. Sources:
    #   - On-this-day        — same MM-DD across past years
    #   - Trip               — timeline_events.json: events with location + >=2 days
    #   - Topic-by-month     — keyword sets matched against label text
    #   - Place memory       — random named location_name, all-time
    #   - Person memory      — random named face × month
    #   - Season             — month group (春 3-5 / 夏 6-8 / 秋 9-11 / 冬 12-2) × year
    #   - Weekend            — random year-month, only photos taken on Sat/Sun
    #   - Pet                — random named pet × month (if pet_clusters exists)
    # Title pattern follows the user's preferred form: "{year} 年 {m} 月 + 主題".
    # Each card carries its photo path list inline so click→display is instant.

    def _build_memories_payload(self) -> dict:
        idx = _load_photo_index()
        if not idx:
            return {
                "error": "photo_index not built",
                "hint": "uv run python src/Photo_Index.py build --labels ... --faces ... --embeddings ... --with-location",
                "cards": [], "total": 0,
            }
        import random as _r, hashlib as _h
        from datetime import date
        perms = self._perms()

        MAX_PHOTOS_PER_CARD = 30  # keep cards focused; sample/rank when source has more
        images_map = idx.get("images", {})

        def _sample(paths, score_fn=None, n=MAX_PHOTOS_PER_CARD):
            """Cap to n photos, then return sorted by time asc.

            - score_fn given → take top-n by descending score (used for topic
              cards where label-keyword density indicates relevance)
            - else            → random sample n (used for season/weekend/place/
              person/pet where every match is equally relevant)
            """
            if len(paths) <= n:
                picked = list(paths)
            elif score_fn:
                picked = sorted(paths, key=score_fn, reverse=True)[:n]
            else:
                picked = _r.sample(paths, n)
            return sorted(picked, key=lambda p: images_map.get(p, {}).get("time", ""))

        # Reuse mtime-cached path→faces / path→pets maps (vs rebuilding 57k
        # entries every request — saved ~3s on Chun's library).
        path_to_faces = _path_to_faces()
        pets_map = _path_to_pets()
        # Memoize visibility within this request — every generator below asks
        # about the same photos repeatedly, and can_see_photo does multiple
        # set intersections each call.
        _visible_cache: Dict[str, bool] = {}
        def can_see(p):
            v = _visible_cache.get(p)
            if v is None:
                v = _auth.can_see_photo(perms, p, path_to_faces.get(p, []), pets_map.get(p, []))
                _visible_cache[p] = v
            return v

        cards: List[dict] = []
        today = date.today()

        # --- On This Day -------------------------------------------------
        otd_by_year: Dict[int, List[str]] = {}
        for p, info in idx["images"].items():
            if info.get("month") == today.month and info.get("day_of_month_match", None) is None:
                t = info.get("time")
                if not t: continue
                if t[8:10] == f"{today.day:02d}" and can_see(p):
                    y = info.get("year")
                    if y and y != today.year:
                        otd_by_year.setdefault(y, []).append(p)
        for y, paths in sorted(otd_by_year.items(), reverse=True):
            if len(paths) < 3: continue
            picked = _sample(paths)
            cards.append({
                "id": f"mem_otd_{y}",
                "type": "on_this_day",
                "title": f"{today.year - y} 年前的今天",
                "subtitle": f"{y} 年 {today.month} 月 {today.day} 日",
                "cover": picked[len(picked) // 2],
                "photos": picked,
                "photo_count": len(picked),
                "accent": "primary",
            })

        # --- Trips (from timeline_events.json) ---------------------------
        tl = _load_timeline()
        if tl:
            trip_pool = [e for e in tl.get("events", [])
                         if e.get("top_places") and e.get("duration_days", 0) >= 2]
            _r.shuffle(trip_pool)
            picked = 0
            for evt in trip_pool:
                visible = [p for p in evt["photos"] if can_see(p)]
                if len(visible) < 5: continue
                place = _zh((evt["top_places"] or ["?"])[0])
                start_d = evt["start"][:10]
                y, m = start_d[:4], int(start_d[5:7])
                # Trip photos are all from one trip, so random sample is fine
                sampled = _sample(visible)
                # Prefer original event cover if it survived sampling, else middle
                cover = evt["cover"] if evt["cover"] in sampled else sampled[len(sampled)//2]
                cards.append({
                    "id": f"mem_trip_{evt['id']}",
                    "type": "trip",
                    "title": f"{y} 年 {m} 月 · {place}",
                    "subtitle": f"{evt['duration_days']} 天",
                    "cover": cover,
                    "photos": sampled,
                    "photo_count": len(sampled),
                    "accent": "secondary",
                })
                picked += 1
                if picked >= 3: break

        # --- Topic-by-month (label keyword matching with exclude lists) ---
        # Each topic has BOTH include AND exclude lists. A photo matches the
        # topic iff its label contains an include keyword AND NO exclude.
        # The exclude lists kill common substring false-positives like
        #   "海" → "劉海" (bangs hairstyle), "上海" (Shanghai), "海苔" (nori),
        #          "海鮮" (seafood), "海派", "海報", "海關" etc.
        TOPICS = {
            "美食": {
                "icon": "utensils-crossed",
                "include": ["美食","食物","料理","甜點","蛋糕","烤肉","火鍋","餐廳",
                            "拉麵","壽司","披薩","漢堡","便當","下午茶","佳餚","早餐",
                            "晚餐","午餐","炒飯","炒麵","餃子","包子","燒烤","咖啡廳",
                            "甜品","點心","菜餚","小吃","飯店餐", "麵食","湯品"],
                "exclude": [],
            },
            "山景": {
                "icon": "mountain",
                "include": ["山脈","山頂","登山","山巒","山稜","山林","高山","山谷",
                            "山坡","山色","山岳","群山","山景","眺望群山","山中","山間"],
                "exclude": ["中山","山田","山口","山本","山內","山下","河山","江山",
                            "山寨","山貨","靠山","座山","山雞","山豬","山藥"],
            },
            "海景": {
                "icon": "waves",
                "include": ["海邊","沙灘","海浪","海岸","海洋","海面","海景","海濱",
                            "海域","浪花","大海","海水","海湾","海灣","海滨","海滩",
                            "出海","海平面","遠眺海面"],
                "exclude": ["劉海","刘海","瀏海","上海","海苔","海鮮","海鱼","海派",
                            "海綿","海绵","海報","海报","海內","海关","海關","海軍",
                            "海军","海運","海运","海拔","腦海","脑海","海量","海口"],
            },
            "夜景": {
                "icon": "moon",
                "include": ["夜景","夜晚","霓虹","夜市","夜色","燈光秀","煙火","煙花",
                            "華燈","夜空","深夜","入夜","夜幕","萬家燈火"],
                "exclude": ["子夜","夜店","夜半","夜路","夜深人靜","夜以繼日"],
            },
            "建築": {
                "icon": "landmark",
                "include": ["古蹟","寺廟","教堂","神社","佛塔","古建築","老建築",
                            "傳統建築","歷史建築","現代建築","摩天大樓","紀念碑",
                            "宮殿","城堡","廟宇"],
                "exclude": [],
            },
            "派對": {
                "icon": "party-popper",
                "include": ["派對","慶祝","生日派對","生日蛋糕","聚餐","聚會","節日",
                            "婚禮","婚宴","晚宴","歡聚","派对","狂歡","派对场景"],
                "exclude": [],
            },
            "天空": {
                "icon": "cloud-sun",
                "include": ["日落","日出","彩霞","雲海","藍天","晚霞","彩虹","夕陽",
                            "朝陽","雲彩","火燒雲","雲層","碧空","蒼穹"],
                "exclude": [],
            },
            "街景": {
                "icon": "traffic-cone",
                "include": ["街道","街景","巷弄","商店街","老街","市集","市區","鬧區",
                            "鬧街","街頭","步行街","購物街","街市","街角"],
                "exclude": ["街道辦","街頭巷尾(成語)","華爾街","百老匯"],
            },
            "植物": {
                "icon": "flower",
                "include": ["花海","櫻花","油桐花","花圃","花園","花田","盆栽","綠葉",
                            "綠樹","花朵","鮮花","玫瑰","向日葵","杜鵑","繡球花",
                            "百合","薰衣草","落葉","秋葉","紅葉"],
                "exclude": ["花費","花了","花光","花錢","花心","花樣"],
            },
        }
        # Number of DISTINCT include keywords a label must hit. >=2 kills the
        # vast majority of "passing mention" false positives (e.g. a kindergarten
        # decoration description that says 彩虹 once doesn't get into 天空).
        TOPIC_MIN_DISTINCT = 2

        import re as _re
        # Compile per-topic regexes once (alternation is one pass through label
        # vs N substring checks — ~5× faster on 57k photos).
        _topic_inc_re: Dict[str, "_re.Pattern"] = {}
        _topic_exc_re: Dict[str, Optional["_re.Pattern"]] = {}
        for _topic, _cfg in TOPICS.items():
            _topic_inc_re[_topic] = _re.compile("|".join(_re.escape(kw) for kw in _cfg["include"]))
            _topic_exc_re[_topic] = (_re.compile("|".join(_re.escape(kw) for kw in _cfg["exclude"]))
                                     if _cfg["exclude"] else None)

        def topic_unique_hits(label, topic):
            """Count DISTINCT include keywords in label for this topic.
            Returns 0 if any exclude keyword is present."""
            if not label: return 0
            excl = _topic_exc_re[topic]
            if excl and excl.search(label): return 0
            return len(set(_topic_inc_re[topic].findall(label)))

        topic_names = list(TOPICS.keys())
        _r.shuffle(topic_names)
        topic_picked = 0
        for topic in topic_names:
            if topic_picked >= 3: break
            cfg = TOPICS[topic]
            # bucket by (year, month, photo) carrying the unique-hit count for ranking
            buckets: Dict[Tuple[int,int], List[Tuple[str, int]]] = {}
            for p, info in idx["images"].items():
                if not info.get("year") or not info.get("month"):
                    continue
                hits = topic_unique_hits(info.get("label") or "", topic)
                if hits < TOPIC_MIN_DISTINCT:
                    continue
                if not can_see(p): continue
                buckets.setdefault((info["year"], info["month"]), []).append((p, hits))
            candidates = [(ym, ps) for ym, ps in buckets.items() if len(ps) >= 6]
            if not candidates: continue
            ym, paths_with_hits = _r.choice(candidates)
            y, m = ym
            paths = [p for p, _ in paths_with_hits]
            # Rank by unique-hit count (already computed) so the cover + first
            # photos are the MOST on-topic ones.
            hit_map = dict(paths_with_hits)
            picked = _sample(paths, score_fn=lambda p: hit_map.get(p, 0))
            cards.append({
                "id": f"mem_topic_{topic}_{y}_{m:02d}",
                "type": "topic",
                "icon": cfg["icon"],
                "title": f"{y} 年 {m} 月 · {topic}",
                "subtitle": "",
                "cover": picked[len(picked) // 2],
                "photos": picked,
                "photo_count": len(picked),
                "accent": "accent",
            })
            topic_picked += 1

        # --- Place memory (random named location, all-time) --------------
        place_buckets: Dict[str, List[str]] = {}
        for p, info in idx["images"].items():
            name = info.get("location_name")
            if name and can_see(p):
                place_buckets.setdefault(name, []).append(p)
        place_candidates = [(n, ps) for n, ps in place_buckets.items() if len(ps) >= 10]
        if place_candidates:
            name, paths = _r.choice(place_candidates)
            picked = _sample(paths)
            cards.append({
                "id": f"mem_place_{_h.md5(name.encode()).hexdigest()[:8]}",
                "type": "place",
                "title": f"在 {_zh(name)}",
                "subtitle": "跨年度回顧",
                "cover": picked[len(picked) // 2],
                "photos": picked,
                "photo_count": len(picked),
                "accent": "info",
            })

        # --- Person recent (random named face's most-active recent month) ---
        try:
            fnames = json.loads((FACES_DIR / "face_names.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            fnames = {}
        if fnames:
            # Group photos per face per (year, month)
            face_buckets: Dict[Tuple[str, int, int], List[str]] = {}
            for p, info in idx["images"].items():
                if not (info.get("year") and info.get("month")) or not can_see(p): continue
                for fid in path_to_faces.get(p, []):
                    if fid in fnames:
                        face_buckets.setdefault((fid, info["year"], info["month"]), []).append(p)
            person_candidates = [(fid, y, m, ps) for (fid, y, m), ps in face_buckets.items() if len(ps) >= 8]

            # Bias toward people the VIEWER actually knows: anyone they've
            # given an alias to in relationships.json + their identity face.
            # Otherwise mom (face_5) gets random "與某舅舅" cards which is
            # uninteresting; with this filter she sees her closest people.
            viewer_id = perms.get("identity") or ""
            rels = self._relationships() or {}
            important_faces = set((rels.get(viewer_id) or {}).keys())
            if viewer_id:
                important_faces.add(viewer_id)
            if important_faces:
                preferred = [c for c in person_candidates if c[0] in important_faces]
                if preferred:
                    person_candidates = preferred  # fallback to all only if nobody known has enough photos
            if person_candidates:
                fid, y, m, paths = _r.choice(person_candidates)
                picked = _sample(paths)
                # Apply per-viewer aliases (relationship graph): from mom's
                # account, the cluster named "我自己" should display as "君".
                display_name = self._personalize_name(fid, fnames[fid])
                # Self-card: "與 我" reads weirdly in Chinese — use a softer
                # nostalgic phrasing when the card is about the viewer themself.
                is_self = (fid == (perms.get("identity") or ""))
                if is_self:
                    title = f"{y} 年 {m} 月 · 那時候的我"
                else:
                    title = f"{y} 年 {m} 月 · 與 {display_name}"
                cards.append({
                    "id": f"mem_person_{fid}_{y}_{m:02d}",
                    "type": "person",
                    "title": title,
                    "subtitle": "",
                    "cover": picked[len(picked) // 2],
                    "photos": picked,
                    "photo_count": len(picked),
                    "accent": "warning",
                })

        # --- Season (month group × year) --------------------------------
        SEASONS = {"春": (3,4,5), "夏": (6,7,8), "秋": (9,10,11), "冬": (12,1,2)}
        season_buckets: Dict[Tuple[str,int], List[str]] = {}
        for p, info in idx["images"].items():
            y, m = info.get("year"), info.get("month")
            if not (y and m) or not can_see(p): continue
            for sname, months in SEASONS.items():
                if m in months:
                    # Winter (12,1,2): bucket 1/2 under previous year (consistent season)
                    bucket_y = y - 1 if sname == "冬" and m in (1, 2) else y
                    season_buckets.setdefault((sname, bucket_y), []).append(p)
                    break
        season_candidates = [(s, y, ps) for (s, y), ps in season_buckets.items()
                             if len(ps) >= 30 and y < today.year]
        _r.shuffle(season_candidates)
        if season_candidates:
            s, y, paths = season_candidates[0]
            picked = _sample(paths)
            cards.append({
                "id": f"mem_season_{s}_{y}",
                "type": "season",
                "title": f"{y} 年的{s}天",
                "subtitle": "",
                "cover": picked[len(picked) // 2],
                "photos": picked,
                "photo_count": len(picked),
                "accent": "success",
            })

        # --- Weekend (random year-month with photos taken on Sat/Sun) ----
        from datetime import datetime as _dt
        weekend_buckets: Dict[Tuple[int,int], List[str]] = {}
        for p, info in idx["images"].items():
            t = info.get("time")
            if not t or not info.get("year") or not info.get("month") or not can_see(p):
                continue
            try:
                wd = _dt.strptime(t, "%Y-%m-%d %H:%M:%S").weekday()  # Mon=0 .. Sun=6
            except ValueError:
                continue
            if wd >= 5:  # Sat or Sun
                weekend_buckets.setdefault((info["year"], info["month"]), []).append(p)
        weekend_candidates = [(y, m, ps) for (y, m), ps in weekend_buckets.items() if len(ps) >= 12]
        _r.shuffle(weekend_candidates)
        if weekend_candidates:
            y, m, paths = weekend_candidates[0]
            picked = _sample(paths)
            cards.append({
                "id": f"mem_weekend_{y}_{m:02d}",
                "type": "weekend",
                "title": f"{y} 年 {m} 月 · 週末",
                "subtitle": "",
                "cover": picked[len(picked) // 2],
                "photos": picked,
                "photo_count": len(picked),
                "accent": "info",
            })

        # --- Pet (named pet × month) ------------------------------------
        try:
            pnames = json.loads((PETS_DIR / "pet_names.json").read_text(encoding="utf-8"))
            pets_data = json.loads((PETS_DIR / "pet_clusters.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pnames, pets_data = {}, {}
        # pet_clusters has {"images": {path: [{pet_id: ..., bbox: ...}]}}
        if pnames and pets_data.get("images"):
            path_to_pets: Dict[str, List[str]] = {}
            for p, pets in pets_data["images"].items():
                for entry in pets:
                    pid = entry.get("pet_id")
                    if pid and pid in pnames:
                        path_to_pets.setdefault(p, []).append(pid)
            pet_buckets: Dict[Tuple[str,int,int], List[str]] = {}
            for p, info in idx["images"].items():
                if not (info.get("year") and info.get("month")) or not can_see(p):
                    continue
                for pid in path_to_pets.get(p, []):
                    pet_buckets.setdefault((pid, info["year"], info["month"]), []).append(p)
            pet_candidates = [(pid, y, m, ps) for (pid, y, m), ps in pet_buckets.items() if len(ps) >= 6]
            if pet_candidates:
                pid, y, m, paths = _r.choice(pet_candidates)
                picked = _sample(paths)
                cards.append({
                    "id": f"mem_pet_{pid}_{y}_{m:02d}",
                    "type": "pet",
                    "title": f"{y} 年 {m} 月 · {pnames[pid]}",
                    "subtitle": "",
                    "cover": picked[len(picked) // 2],
                    "photos": picked,
                    "photo_count": len(picked),
                    "accent": "secondary",
                })

        # Shuffle all cards together so on-this-day mixes with the rest
        _r.shuffle(cards)

        return {"cards": cards, "total": len(cards), "date": today.isoformat()}

    # ---- admin: users / groups / visibility helpers ----

    @staticmethod
    def _admin_list_users():
        users = _auth.load_users()
        return [
            {
                "username": n,
                "role": u.get("role", "viewer"),
                "identity": u.get("identity", ""),
                "groups": list(u.get("groups", [])),
            }
            for n, u in sorted(users.items())
        ]

    @staticmethod
    def _admin_list_groups():
        groups = _auth.load_groups()
        return [
            {
                "name": n,
                "allowed_faces": list(g.get("allowed_faces", [])),
                "allowed_pets": list(g.get("allowed_pets", [])),
                "blocked_paths": list(g.get("blocked_paths", [])),
            }
            for n, g in sorted(groups.items())
        ]

    @staticmethod
    def _admin_list_clusters_named(kind: str):
        """共用：列出某 kind 已命名的 cluster（給 admin 群組編輯做白名單）。"""
        data = load_clusters(kind) or {}
        names = load_names_k(kind)
        merges = load_merges_k(kind)
        clusters = data.get("clusters", {}) or {}
        merge_back: dict[str, list[str]] = defaultdict(list)
        for src in list(merges.keys()):
            final = _resolve_target(merges, src)
            if final != src:
                merge_back[final].append(src)
        thumbs_dir = KIND_PATHS[kind]["thumbs"]
        result = []
        for fid, info in clusters.items():
            if fid in merges:
                continue
            name = names.get(fid, "")
            if not name:
                continue
            count = len(info.get("images", []) or [])
            for s in merge_back.get(fid, []):
                count += len(clusters.get(s, {}).get("images", []) or [])
            thumb_file = thumbs_dir / f"{fid}.jpg"
            thumb_ver = int(thumb_file.stat().st_mtime) if thumb_file.exists() else 0
            result.append({"id": fid, "name": name, "count": count, "thumb_ver": thumb_ver})
        result.sort(key=lambda r: (-r["count"], r["name"]))
        return result

    @staticmethod
    def _admin_list_faces():
        return Handler._admin_list_clusters_named("face")

    @staticmethod
    def _admin_list_pets():
        return Handler._admin_list_clusters_named("pet")

    @staticmethod
    def _admin_list_paths():
        """列出 metadata 中曾索引到的 top-level 目錄前綴，供 blocked_paths 勾選。

        策略：掃 face_clusters.json 內所有 image 路徑，取前 N 段路徑作為候選。
        """
        faces = load_faces() or {}
        clusters = faces.get("clusters", {}) or {}
        # 只列「照片庫根目錄下一層」的資料夾，作為封鎖路徑候選。
        # 例：/Volumes/970EvoP2T/Chun/Pictures/8_Xiaomi_Mi13Ultra/...
        #     → 取 /Volumes/970EvoP2T/Chun/Pictures/8_Xiaomi_Mi13Ultra/
        # split('/') 後 index 5 就是 Pictures 下一層的資料夾名。
        counts: dict[str, int] = defaultdict(int)
        for info in clusters.values():
            for p in info.get("images", []) or []:
                if _is_skip_path(p):
                    continue
                parts = p.split("/")
                if len(parts) >= 7:  # 至少要有一個檔案在這層下面
                    top = "/".join(parts[:6]) + "/"
                    counts[top] += 1
        return [
            {"path": p, "count": c}
            for p, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if c > 0
        ]

    def _admin_create_user(self, body: dict):
        username = (body.get("username") or "").strip()
        password = body.get("password") or ""
        role = body.get("role") or "viewer"
        groups = body.get("groups") or []
        identity = (body.get("identity") or "").strip()
        if not username or not password:
            return {"ok": False, "error": "username 與 password 必填"}
        if role not in ("admin", "viewer"):
            return {"ok": False, "error": "權限必須是 admin 或 viewer"}
        if not isinstance(groups, list):
            return {"ok": False, "error": "groups 必須是陣列"}
        users = _auth.load_users()
        if username in users:
            return {"ok": False, "error": f"使用者 '{username}' 已存在"}
        users[username] = {
            "password_hash": _auth.hash_password(password),
            "role": role,
            "identity": identity,
            "groups": [g for g in groups if isinstance(g, str) and g.strip()],
        }
        _auth.save_users(users)
        return {"ok": True, "username": username}

    def _admin_modify_user(self, username: str, body: dict):
        action = body.get("action") or "update"
        users = _auth.load_users()
        if username not in users:
            return {"ok": False, "error": f"沒有使用者 '{username}'"}
        if action == "delete":
            # 防呆：不能刪除自己（避免把所有 admin 鎖在外面的常見情境）
            if username == self._user():
                return {"ok": False, "error": "不能刪除自己"}
            # 若是最後一個 admin，禁止
            target = users[username]
            if target.get("role") == "admin":
                admins = [n for n, u in users.items() if u.get("role") == "admin"]
                if len(admins) <= 1:
                    return {"ok": False, "error": "不能刪除最後一個 admin"}
            del users[username]
            _auth.save_users(users)
            return {"ok": True}
        # update
        u = users[username]
        if "role" in body:
            new_role = body.get("role")
            if new_role not in ("admin", "viewer"):
                return {"ok": False, "error": "權限必須是 admin 或 viewer"}
            # 防呆：不能把唯一 admin 降權
            if u.get("role") == "admin" and new_role != "admin":
                admins = [n for n, x in users.items() if x.get("role") == "admin"]
                if len(admins) <= 1:
                    return {"ok": False, "error": "不能降權最後一個 admin"}
            u["role"] = new_role
        if "groups" in body:
            groups = body.get("groups") or []
            if not isinstance(groups, list):
                return {"ok": False, "error": "groups 必須是陣列"}
            u["groups"] = [g for g in groups if isinstance(g, str) and g.strip()]
        if "identity" in body:
            u["identity"] = (body.get("identity") or "").strip()
        if body.get("password"):
            u["password_hash"] = _auth.hash_password(body["password"])
        _auth.save_users(users)
        return {"ok": True}

    def _admin_modify_group(self, name: str, body: dict):
        action = body.get("action") or "upsert"
        groups = _auth.load_groups()
        if action == "delete":
            if name not in groups:
                return {"ok": False, "error": f"沒有群組 '{name}'"}
            del groups[name]
            _auth.save_groups(groups)
            # 順便把所有 user.groups 裡的這個名字清掉，避免懸空引用
            users = _auth.load_users()
            changed = False
            for u in users.values():
                if name in u.get("groups", []):
                    u["groups"] = [g for g in u["groups"] if g != name]
                    changed = True
            if changed:
                _auth.save_users(users)
            return {"ok": True}
        # upsert
        if not name or not name.strip():
            return {"ok": False, "error": "群組名不能為空"}
        allowed_faces = body.get("allowed_faces") or []
        allowed_pets = body.get("allowed_pets") or []
        blocked_paths = body.get("blocked_paths") or []
        if not isinstance(allowed_faces, list) or not isinstance(allowed_pets, list) or not isinstance(blocked_paths, list):
            return {"ok": False, "error": "allowed_faces / allowed_pets / blocked_paths 必須是陣列"}
        groups[name] = {
            "allowed_faces": [x for x in allowed_faces if isinstance(x, str) and x.strip()],
            "allowed_pets":  [x for x in allowed_pets  if isinstance(x, str) and x.strip()],
            "blocked_paths": [x for x in blocked_paths if isinstance(x, str) and x.strip()],
        }
        _auth.save_groups(groups)
        return {"ok": True}

    def _admin_save_relationship_graph(self, body: dict):
        """儲存圖 + 從圖自動推導 relationships.json。"""
        nodes = body.get("nodes") or []
        edges = body.get("edges") or []
        positions = body.get("positions") or {}
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return {"ok": False, "error": "nodes / edges 必須是陣列"}
        if not isinstance(positions, dict):
            positions = {}
        clean_nodes = [n for n in nodes if isinstance(n, str) and n.strip()]
        clean_edges: list[dict] = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            f, t, typ = e.get("from"), e.get("to"), e.get("type")
            if not (isinstance(f, str) and isinstance(t, str) and isinstance(typ, str)):
                continue
            f, t, typ = f.strip(), t.strip(), typ.strip()
            if not f or not t or not typ or f == t:
                continue
            clean = {"from": f, "to": t, "type": typ}
            if e.get("alias_from"):
                clean["alias_from"] = str(e["alias_from"]).strip()
            if e.get("alias_to"):
                clean["alias_to"] = str(e["alias_to"]).strip()
            clean_edges.append(clean)
        # positions: 只保留還在 nodes 內的 fid，且 x/y 為數字
        node_set = set(clean_nodes)
        clean_positions: dict[str, dict] = {}
        for fid, p in positions.items():
            if fid not in node_set or not isinstance(p, dict):
                continue
            x, y = p.get("x"), p.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                clean_positions[fid] = {"x": float(x), "y": float(y)}
        graph = {"nodes": clean_nodes, "edges": clean_edges, "positions": clean_positions}
        _auth.save_relationship_graph(graph)
        # 自動推導 relationships.json（家人邊才產 alias；非家人邊只在圖上顯示）
        derived = _auth.derive_relationships(graph)
        _auth.save_relationships(derived)
        return {"ok": True, "nodes": len(clean_nodes), "edges": len(clean_edges),
                "aliases": sum(len(v) for v in derived.values())}

    # ---- data assembly ----

    def get_all_sorted(self, flt, kind: str = "face"):
        # kind="all" = viewer 統一視圖：face + pet 合併，每筆 item 自帶 kind
        if kind == "all":
            face_items = self.get_all_sorted(flt, "face")
            pet_items = self.get_all_sorted(flt, "pet")
            for r in face_items:
                r["kind"] = "face"
            for r in pet_items:
                r["kind"] = "pet"
            combined = face_items + pet_items
            identity = self._perms().get("identity") or ""
            # 登入者自己 (identity) 的 cluster 排第一
            combined.sort(key=lambda r: (
                r["skipped"],
                not (identity and r["id"] == identity),
                -r["count"],
            ))
            return combined

        faces = load_clusters(kind) or {}
        names = load_names_k(kind)
        removed = load_removed_k(kind)
        skipped = set(load_skipped_k(kind))
        merges = load_merges_k(kind)
        moves = load_moves_k(kind)
        clusters = faces.get("clusters", {})
        thumbs_dir = KIND_PATHS[kind]["thumbs"]

        # 反查：每個最終 target 收哪些 source
        merge_back: dict[str, list[str]] = defaultdict(list)
        for src in list(merges.keys()):
            final = _resolve_target(merges, src)
            if final != src:
                merge_back[final].append(src)

        # 索引 moves：from_id → {path: to_id}, to_id → [paths]
        moves_out: dict[str, dict[str, str]] = defaultdict(dict)
        moves_in: dict[str, list[str]] = defaultdict(list)
        for m in moves:
            f, t, p = m.get("from"), m.get("to"), m.get("path")
            if f and t and p:
                # 如果 from 已被合併走，解析到最終 target
                f_final = _resolve_target(merges, f)
                t_final = _resolve_target(merges, t)
                moves_out[f_final][p] = t_final
                moves_in[t_final].append(p)

        def _thumb_ver(fid: str) -> int:
            f = thumbs_dir / f"{fid}.jpg"
            return int(f.stat().st_mtime) if f.exists() else 0

        result = []
        for fid, info in clusters.items():
            # source cluster 被折疊到 target，不獨立顯示
            if fid in merges:
                continue

            own_imgs = list(info.get("images", []))
            merged_srcs = merge_back.get(fid, [])
            for s in merged_srcs:
                own_imgs.extend(clusters.get(s, {}).get("images", []))

            # 過濾系統/縮圖路徑 + 去重保序
            seen = set()
            all_imgs = [x for x in own_imgs
                        if not _is_skip_path(x) and not (x in seen or seen.add(x))]

            # 被移出此群（顯示層級）：moves.from == fid 且 to != fid
            moved_away = set(moves_out.get(fid, {}).keys())

            # 被移入此群
            moved_in = moves_in.get(fid, [])
            for p in moved_in:
                if p not in seen:
                    all_imgs.append(p)
                    seen.add(p)

            # 已移除（target + 所有 source 的）
            rem = list(removed.get(fid, []))
            for s in merged_srcs:
                for img in removed.get(s, []):
                    if img not in rem:
                        rem.append(img)
            rem_set = set(rem)

            active = [img for img in all_imgs
                      if img not in rem_set and img not in moved_away]

            result.append({
                "id": fid,
                "name": names.get(fid, ""),
                "count": len(active),
                "original_count": len(all_imgs),
                "images": active,
                "removed": rem,
                "skipped": fid in skipped,
                "merged_from": merged_srcs,
                "moved_in_count": len([p for p in moved_in if p not in rem_set]),
                "moved_away_count": len(moved_away),
                "user_created": False,
                "thumb_v": _thumb_ver(fid),
            })

        # User-created clusters (透過 batch move 產生的新 group)：
        # 它們不在 clusters dict 裡，但被 moves 指為 to，需要補上 synthetic 條目
        native_ids = set(clusters.keys())
        synthetic_targets = set()
        for tgt in moves_in.keys():
            if tgt not in native_ids and tgt not in merges:
                synthetic_targets.add(tgt)
        for fid in synthetic_targets:
            paths = []
            seen = set()
            for p in moves_in[fid]:
                if _is_skip_path(p):
                    continue
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
            rem = list(removed.get(fid, []))
            rem_set = set(rem)
            active = [p for p in paths if p not in rem_set]
            result.append({
                "id": fid,
                "name": names.get(fid, ""),
                "count": len(active),
                "original_count": len(paths),
                "images": active,
                "removed": rem,
                "skipped": fid in skipped,
                "merged_from": [],
                "moved_in_count": len(active),
                "moved_away_count": 0,
                "user_created": True,
                "thumb_v": _thumb_ver(fid),
            })

        # filter
        if flt == "named":
            result = [r for r in result if r["name"]]
        elif flt == "unnamed":
            result = [r for r in result if not r["name"] and not r["skipped"]]
        elif flt == "skipped":
            result = [r for r in result if r["skipped"]]
        # "all" → 全留

        # 權限：非 admin 要過 cluster + per-image 雙層過濾
        perms = self._perms()
        if not perms["is_admin"]:
            if kind == "pet":
                # 寵物的可見性規則：
                #   - 預設（allowed_pets 為空）：所有 cluster 可見；
                #     per-image 仍受 blocked_paths 限制。
                #   - 設了 allowed_pets（白名單模式）：只看 cluster ∈ allowed_pets，
                #     而且這些 cluster 的照片 **解鎖 blocked_paths**
                #     （pet unlock，類似 face identity 的特權；
                #      admin 明確授權該 pet 就等於放行該 pet 的照片）。
                allowed_pets = perms.get("allowed_pets") or set()
                whitelist_mode = bool(allowed_pets)
                if whitelist_mode:
                    result = [r for r in result if r["id"] in allowed_pets]

                def _ok_pet(path, cluster_id):
                    # 白名單模式 + cluster 是被允許的 → pet unlock 解鎖路徑封鎖
                    if whitelist_mode and cluster_id in allowed_pets:
                        return True
                    return not _auth.path_blocked(path, perms["blocked_paths"])

                filtered: list[dict] = []
                for r in result:
                    visible_imgs = [p for p in r["images"] if _ok_pet(p, r["id"])]
                    if not visible_imgs:
                        continue
                    r["images"] = visible_imgs
                    r["count"] = len(visible_imgs)
                    r["removed"] = [p for p in r["removed"] if _ok_pet(p, r["id"])]
                    filtered.append(r)
                result = filtered
            else:
                # 人臉：套完整 can_see_photo 規則（含 identity 特權 / blocked_paths / 無臉=隱藏）
                # 1) cluster：只留 allowed_faces 對應的（含本人 identity，已在 allowed_faces 裡）
                result = [r for r in result if r["id"] in perms["allowed_faces"]]

                # 2) per-image：用 _auth.can_see_photo 規則
                img_to_faces: dict[str, list[str]] = {}
                for img_path, recs in (faces.get("images") or {}).items():
                    img_to_faces[img_path] = [
                        _resolve_target(merges, r.get("face_id", "")) for r in recs
                    ]
                moves_per_path: dict[str, list[tuple[str, str]]] = defaultdict(list)
                for m in moves:
                    p = m.get("path")
                    if not p:
                        continue
                    f_final = _resolve_target(merges, m.get("from", ""))
                    t_final = _resolve_target(merges, m.get("to", ""))
                    moves_per_path[p].append((f_final, t_final))

                def _effective_faces(path: str) -> list[str]:
                    base = list(img_to_faces.get(path, []))
                    for f_final, t_final in moves_per_path.get(path, []):
                        base = [t_final if fid == f_final else fid for fid in base]
                    return base

                filtered = []
                for r in result:
                    visible_imgs = [
                        p for p in r["images"]
                        if _auth.can_see_photo(perms, p, _effective_faces(p))
                    ]
                    if not visible_imgs:
                        continue
                    r["images"] = visible_imgs
                    r["count"] = len(visible_imgs)
                    r["removed"] = [
                        p for p in r["removed"]
                        if _auth.can_see_photo(perms, p, _effective_faces(p))
                    ]
                    filtered.append(r)
                result = filtered

        # 排序：未略過在前 → 登入者本人 cluster 第一 → 其餘依 count desc → 略過在後
        identity = self._perms().get("identity") or ""
        result.sort(key=lambda r: (
            r["skipped"],
            not (identity and r["id"] == identity),
            -r["count"],
        ))
        return result

    def get_page(self, page, flt, kind: str = "face"):
        all_data = self.get_all_sorted(flt, kind=kind)
        total = len(all_data)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        page = max(0, min(page, total_pages - 1))
        start = page * PAGE_SIZE
        end = start + PAGE_SIZE
        items = all_data[start:end]
        # 個人化顯示名稱：只對人臉 cluster 套用 relationships.json
        # （寵物名稱保持 canonical，沒有 perspective alias）
        for r in items:
            item_kind = r.get("kind", kind)
            if item_kind == "face":
                r["name"] = self._personalize_name(r["id"], r.get("name", ""))
        return {
            "page": page,
            "page_size": PAGE_SIZE,
            "total": total,
            "total_pages": total_pages,
            "items": items,
        }

    def get_stats(self, kind: str = "face"):
        # all 模式：把 face + pet 的 stats 加總（viewer 不在乎拆 face/pet）
        if kind == "all":
            f = self.get_stats("face")
            p = self.get_stats("pet")
            return {k: f.get(k, 0) + p.get(k, 0) for k in ("total", "displayed", "named", "skipped", "merges")}
        names = load_names_k(kind)
        skipped = load_skipped_k(kind)
        merges = load_merges_k(kind)
        clusters_data = load_clusters(kind) or {}
        total = len(clusters_data.get("clusters", {}))
        perms = self._perms()
        if perms["is_admin"]:
            return {
                "total": total,
                "displayed": total - len(merges),
                "named": len(names),
                "skipped": len(skipped),
                "merges": len(merges),
            }
        if kind == "pet":
            # 寵物全部 cluster 對 viewer 都公開
            return {
                "total": total,
                "displayed": total - len(merges),
                "named": len(names),
                "skipped": 0,
                "merges": 0,
            }
        # face viewer
        allowed = perms["allowed_faces"]
        return {
            "total": len(allowed),
            "displayed": len(allowed),
            "named": sum(1 for fid in allowed if names.get(fid)),
            "skipped": 0,
            "merges": 0,
        }

    def generate_login_html(self):
        return _load_template("login.html")

    def generate_html(self):
        user = self._user() or ""
        perms = self._perms()
        is_admin = "true" if perms["is_admin"] else "false"
        # 偵測哪些 kind 有資料；UI 才會顯示對應的 toggle 按鈕
        has_face = (FACES_DIR / "face_clusters.json").exists()
        has_pet  = (PETS_DIR / "pet_clusters.json").exists()
        my_identity = perms.get("identity") or ""
        inject = (
            f"<script>"
            f"window.CURRENT_USER='{user}';"
            f"window.IS_ADMIN={is_admin};"
            f"window.IDENTITY='{my_identity}';"  # 自己對應的 face_id；空字串 = 訪客
            f"window.KIND='face';"
            f"window.HAS_FACE={'true' if has_face else 'false'};"
            f"window.HAS_PET={'true' if has_pet else 'false'};"
            f"</script>"
        )
        return self._generate_html_template().replace(
            "<!--USER_INJECT-->", inject)

    def generate_admin_html(self):
        user = self._user() or ""
        return _load_template("admin.html").replace("__USER__", user)

    def _generate_html_template(self):
        return _load_template("main.html")


def main():
    port = 8765
    # Bind 0.0.0.0 so家用 LAN 設備（手機、平板等）能存取
    bind_host = os.environ.get("BIND_HOST", "0.0.0.0")

    # 啟動前確認有 admin
    users = _auth.load_users()
    has_admin = any(u.get("role") == "admin" for u in users.values())
    if not has_admin:
        print("=" * 60)
        print("⚠️  尚未建立 admin 帳號。請先跑：")
        print()
        print("  uv run python src/manage_users.py user-add <name> \\")
        print("      --role admin --password '<your_pw>'")
        print()
        print(f"  使用者/群組設定檔位於 {_auth.AUTH_DIR}")
        print("=" * 60)
        return

    # TODO(v0.3+): HTTPS / external access
    #   - LAN HTTP is fine for家用; to expose externally use Tailscale Serve
    #     (`tailscale serve --bg http://localhost:8765`) for automatic TLS +
    #     identity, or Cloudflare Tunnel.
    #   - If we ever bind directly to a public interface, swap HTTPServer for
    #     ssl-wrapped server and read certs from METADATA_DIR/auth/{cert,key}.pem.
    server = HTTPServer((bind_host, port), Handler)
    visible = "127.0.0.1" if bind_host == "127.0.0.1" else "<LAN ip>"
    print(f"人臉命名伺服器 v5：http://{visible}:{port}   (bind {bind_host})")
    print(f"使用者: {len(users)}, admin: {sum(1 for u in users.values() if u.get('role')=='admin')}")
    print("按 Ctrl+C 結束")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        names = load_names()
        print(f"\n結束。已命名 {len(names)} 個群組")


if __name__ == "__main__":
    main()
