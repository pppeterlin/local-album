"""
Simple auth for face_naming_server.

Storage: users.json + groups.json under METADATA_DIR/auth/.
Sessions: HMAC-signed cookies (stateless), 1 year expiry baked in
payload. No DB, no server-side session table.

Permission model (per photo P for user U):
  1. admin role                              → always visible
  2. U.identity ∈ faces(P)                   → visible (本人特權，beats blocked_paths)
  3. P.path matches U.blocked_paths          → hidden
  4. P has no detected faces                 → hidden for non-admin
                                               (privacy-by-default for
                                                un-curated content)
  5. U.allowed_faces ∩ faces(P) non-empty    → visible (face unlock via 群組)
  6. otherwise                               → hidden

U.allowed_faces and U.blocked_paths are unions over U's groups.
U.identity (optional face_id) — only this single face_id gets the
"override blocked_paths" privilege. Other allowed_faces (from groups) do not:
they're filtered out if the photo's path is in blocked_paths.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from http.cookies import SimpleCookie
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import METADATA_DIR  # noqa: E402

AUTH_DIR = METADATA_DIR / "auth"
USERS_FILE = AUTH_DIR / "users.json"
GROUPS_FILE = AUTH_DIR / "groups.json"
SECRET_FILE = AUTH_DIR / ".session_secret"
# Per-viewer display-name aliases (e.g. mom sees chun as "兒子").
# Schema: {viewer_face_id: {target_face_id: "alias", ...}}
# Self alias (same key as viewer) overrides the default "我".
# Local file under METADATA_DIR/auth (already outside the repo / gitignored).
# Auto-derived from RELATIONSHIP_GRAPH_FILE on save; can also be hand-edited.
RELATIONSHIPS_FILE = AUTH_DIR / "relationships.json"

# Underlying relationship graph (nodes + typed edges) edited via /admin UI.
# Schema: {nodes: [face_id], edges: [{from, to, type, alias_from?, alias_to?}]}
#   - type:  one of FAMILY_EDGE_RULES keys, NON_FAMILY_TYPES, or a custom string
#   - alias_from: override what `from` displays as when `to` is the viewer
#   - alias_to:   override what `to` displays as when `from` is the viewer
RELATIONSHIP_GRAPH_FILE = AUTH_DIR / "relationship_graph.json"

SESSION_TTL_SEC = 365 * 24 * 60 * 60  # 1 year
COOKIE_NAME = "lasession"

# --- secret -----------------------------------------------------------------

def _secret() -> bytes:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes()
    s = secrets.token_bytes(32)
    SECRET_FILE.write_bytes(s)
    try:
        os.chmod(SECRET_FILE, 0o600)
    except OSError:
        pass
    return s

# --- password hashing -------------------------------------------------------

def hash_password(pw: str) -> str:
    """scrypt with random salt; returns `scrypt$N$r$p$salt$hash` (base64)."""
    N, r, p = 2**14, 8, 1
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=N, r=r, p=p, maxmem=64 * 1024 * 1024)
    return f"scrypt${N}${r}${p}${base64.b64encode(salt).decode()}${base64.b64encode(h).decode()}"

def verify_password(pw: str, stored: str) -> bool:
    try:
        algo, N, r, p, salt_b64, hash_b64 = stored.split("$")
        if algo != "scrypt":
            return False
        N, r, p = int(N), int(r), int(p)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        got = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=N, r=r, p=p, maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(got, expected)
    except Exception:  # noqa: BLE001
        return False

# --- session cookie ---------------------------------------------------------

def make_session_cookie(username: str) -> str:
    payload = {"u": username, "exp": int(time.time()) + SESSION_TTL_SEC}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(_secret(), raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw).decode() + "." + base64.urlsafe_b64encode(sig).decode()

def parse_session_cookie(val: str) -> str | None:
    if not val or "." not in val:
        return None
    try:
        raw_b64, sig_b64 = val.split(".", 1)
        raw = base64.urlsafe_b64decode(raw_b64)
        sig = base64.urlsafe_b64decode(sig_b64)
        expected = hmac.new(_secret(), raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(raw)
        if payload.get("exp", 0) < time.time():
            return None
        return payload.get("u")
    except Exception:  # noqa: BLE001
        return None

def extract_user_from_headers(headers) -> str | None:
    raw_cookie = headers.get("Cookie", "")
    if not raw_cookie:
        return None
    sc = SimpleCookie()
    try:
        sc.load(raw_cookie)
    except Exception:  # noqa: BLE001
        return None
    morsel = sc.get(COOKIE_NAME)
    if not morsel:
        return None
    return parse_session_cookie(morsel.value)

def cookie_header(username: str | None) -> str:
    """Set-Cookie header value. username=None → expire the cookie."""
    if username is None:
        return f"{COOKIE_NAME}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax"
    val = make_session_cookie(username)
    return f"{COOKIE_NAME}={val}; Path=/; Max-Age={SESSION_TTL_SEC}; HttpOnly; SameSite=Lax"

# --- users / groups ---------------------------------------------------------

def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default

def _save_json(path: Path, data):
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def load_users() -> dict:    return _load_json(USERS_FILE, {})
def save_users(d): _save_json(USERS_FILE, d)
def load_groups() -> dict:   return _load_json(GROUPS_FILE, {})
def save_groups(d): _save_json(GROUPS_FILE, d)
def load_relationships() -> dict: return _load_json(RELATIONSHIPS_FILE, {})
def save_relationships(d): _save_json(RELATIONSHIPS_FILE, d)
def load_relationship_graph() -> dict: return _load_json(RELATIONSHIP_GRAPH_FILE, {"nodes": [], "edges": []})
def save_relationship_graph(d): _save_json(RELATIONSHIP_GRAPH_FILE, d)


# Family edge rules. Convention: edge (from → to, type) means
# "from is the {type} of to" — e.g. type=父 → from is to's father.
# Each rule returns (forward, backward):
#   - forward:  what `from` looks like to `to`         (viewer = to)
#   - backward: what `to` looks like to `from`         (viewer = from)
# Gender-neutral receiver terms (孫子女 / 兄姐 / 子女 / 配偶);
# users can override per-edge via alias_from / alias_to.
# Reverse relationships (子, 女, 孫, …) are just edges drawn in the opposite
# direction with the parent-side type — no need to define them here.
FAMILY_EDGE_RULES: dict[str, tuple[str, str]] = {
    "父": ("爸爸", "子女"),
    "母": ("媽媽", "子女"),
    "夫": ("丈夫", "妻子"),
    "妻": ("妻子", "丈夫"),
    "兄": ("哥哥", "弟妹"),
    "姐": ("姐姐", "弟妹"),
    "弟": ("弟弟", "兄姐"),
    "妹": ("妹妹", "兄姐"),
    "爺爺": ("爺爺", "孫子女"),
    "奶奶": ("奶奶", "孫子女"),
    "外公": ("外公", "外孫子女"),
    "外婆": ("外婆", "外孫子女"),
}

# Non-family types — pure visualization, no alias derivation.
NON_FAMILY_TYPES: set[str] = {
    "同學-國小", "同學-國中", "同學-高中", "同學-大學", "同事", "朋友",
}


def derive_relationships(graph: dict) -> dict:
    """從 relationship_graph 推導 relationships.json。
    只處理 FAMILY_EDGE_RULES 範圍內的邊；非家人邊不影響 alias。
    每個邊上的 alias_from / alias_to 覆寫預設詞。
    回傳 {viewer_face_id: {target_face_id: alias_string}}。
    """
    result: dict[str, dict[str, str]] = {}
    for edge in graph.get("edges", []):
        f, t, typ = edge.get("from"), edge.get("to"), edge.get("type")
        if not f or not t or not typ:
            continue
        rule = FAMILY_EDGE_RULES.get(typ)
        if not rule:
            continue  # non-family or custom → no auto alias
        forward_default, backward_default = rule
        forward = edge.get("alias_to") or forward_default      # `from` 從 `to` 視角看到的稱呼
        backward = edge.get("alias_from") or backward_default  # `to` 從 `from` 視角看到的稱呼
        result.setdefault(t, {})[f] = forward
        result.setdefault(f, {})[t] = backward
    return result


def display_name_for(
    viewer_identity: str,
    target_face_id: str,
    canonical_name: str,
    relationships: dict,
) -> str:
    """個人化顯示名稱。優先序：
      1. relationships[viewer][target] 存在 → 用 alias（可覆寫 self 預設）
      2. target == viewer → 「我」
      3. 否則 → canonical_name
    """
    if not viewer_identity:
        return canonical_name
    rel = relationships.get(viewer_identity, {})
    if target_face_id in rel:
        return rel[target_face_id]
    if target_face_id == viewer_identity:
        return "我"
    return canonical_name

# --- permission resolution --------------------------------------------------

def get_user_perms(username: str | None) -> dict:
    """Resolve effective perms for a user.
    Returns: {is_admin, is_viewer, identity, allowed_faces:set, allowed_pets:set, blocked_paths:list}.
      - identity: single face_id (本人); ""=訪客. Has override-blocked-paths privilege.
      - allowed_faces: union over user's groups (includes identity for cluster-list
        purposes so the 本人 cluster is always visible).
      - allowed_pets: union over user's groups. **Empty = no restriction** (寵物預設
        全公開，與 allowed_faces 不同；只要任一群組有指定 allowed_pets 就轉為白名單模式)。
    None / unknown user → is_viewer=False (denies everything except login).
    """
    empty = {"is_admin": False, "is_viewer": False, "identity": "",
             "allowed_faces": set(), "allowed_pets": set(), "blocked_paths": []}
    if not username:
        return empty
    users = load_users()
    u = users.get(username)
    if not u:
        return empty
    if u.get("role") == "admin":
        return {"is_admin": True, "is_viewer": True, "identity": (u.get("identity") or "").strip(),
                "allowed_faces": set(), "allowed_pets": set(), "blocked_paths": []}
    groups = load_groups()
    allowed_faces: set[str] = set()
    allowed_pets: set[str] = set()
    blocked: list[str] = []
    for g in u.get("groups", []):
        gd = groups.get(g, {})
        for fid in gd.get("allowed_faces", []):
            allowed_faces.add(fid)
        for pid in gd.get("allowed_pets", []):
            allowed_pets.add(pid)
        for bp in gd.get("blocked_paths", []):
            if bp not in blocked:
                blocked.append(bp)
    identity = (u.get("identity") or "").strip()
    # 本人 cluster 一定要能看到 → 確保 identity 也在 allowed_faces（用於 cluster 篩選）
    # 注意：blocked_paths 的 override 特權只給 identity，不給其他 allowed_faces
    if identity:
        allowed_faces.add(identity)
    return {"is_admin": False, "is_viewer": True, "identity": identity,
            "allowed_faces": allowed_faces, "allowed_pets": allowed_pets,
            "blocked_paths": blocked}

def path_blocked(path: str, blocked_paths: list[str]) -> bool:
    return any(path.startswith(bp) for bp in blocked_paths)

def can_see_photo(perms: dict, path: str, face_ids: list[str], pet_ids: list[str] = ()) -> bool:
    """Decide visibility for a single photo.

    Rule order (first matching wins):
      1. admin                                           → visible
      2. identity ∈ faces(P)                             → visible  (本人特權，beats blocked_paths)
      3. allowed_pets ∩ pets(P) (with whitelist semantic) → visible  (寵物明確分享，beats blocked_paths)
      4. P.path ∈ blocked_paths                          → hidden
      5. allowed_faces ∩ faces(P) (non-empty)            → visible
      6. else                                            → hidden

    Why pets beat blocked_paths (rule 3 above rule 4):
        A pet only appears in ``allowed_pets`` because the admin explicitly
        whitelisted it for this group ("這隻是我們的貓，家人都可以看"). The
        blocked_paths list is for "Chun's personal phone dump", which usually
        intersects ALL paths in a single-drive library. Without rule 3, mom
        sees zero 奶茶 photos because every shot of the cat is in one of
        Chun's blocked camera dirs.
    """
    if perms["is_admin"]:
        return True
    if not perms["is_viewer"]:
        return False
    # 本人特權：identity 出現在照片裡 → 永遠可見，beats blocked_paths
    identity = perms.get("identity") or ""
    if identity and identity in face_ids:
        return True
    # 寵物明確分享：allowed_pets 為空 = 公開；非空 = 白名單。Beats blocked_paths.
    if pet_ids:
        allowed_pets = perms.get("allowed_pets") or set()
        if not allowed_pets or allowed_pets.intersection(pet_ids):
            return True
    # 路徑黑名單 → 擋
    if path_blocked(path, perms["blocked_paths"]):
        return False
    # 群組允許的人臉出現在照片 → 可見
    if face_ids and perms["allowed_faces"].intersection(face_ids):
        return True
    # 無人臉、無寵物（或寵物未授權）的照片 → admin only
    return False
