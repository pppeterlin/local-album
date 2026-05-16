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

# --- permission resolution --------------------------------------------------

def get_user_perms(username: str | None) -> dict:
    """Resolve effective perms for a user.
    Returns: {is_admin, is_viewer, identity, allowed_faces:set, blocked_paths:list}.
      - identity: single face_id (本人); ""=訪客. Has override-blocked-paths privilege.
      - allowed_faces: union over user's groups (includes identity for cluster-list
        purposes so the 本人 cluster is always visible).
    None / unknown user → is_viewer=False (denies everything except login).
    """
    if not username:
        return {"is_admin": False, "is_viewer": False, "identity": "", "allowed_faces": set(), "blocked_paths": []}
    users = load_users()
    u = users.get(username)
    if not u:
        return {"is_admin": False, "is_viewer": False, "identity": "", "allowed_faces": set(), "blocked_paths": []}
    if u.get("role") == "admin":
        return {"is_admin": True, "is_viewer": True, "identity": (u.get("identity") or "").strip(),
                "allowed_faces": set(), "blocked_paths": []}
    groups = load_groups()
    allowed: set[str] = set()
    blocked: list[str] = []
    for g in u.get("groups", []):
        gd = groups.get(g, {})
        for fid in gd.get("allowed_faces", []):
            allowed.add(fid)
        for bp in gd.get("blocked_paths", []):
            if bp not in blocked:
                blocked.append(bp)
    identity = (u.get("identity") or "").strip()
    # 本人 cluster 一定要能看到 → 確保 identity 也在 allowed_faces（用於 cluster 篩選）
    # 注意：blocked_paths 的 override 特權只給 identity，不給其他 allowed_faces
    if identity:
        allowed.add(identity)
    return {"is_admin": False, "is_viewer": True, "identity": identity,
            "allowed_faces": allowed, "blocked_paths": blocked}

def path_blocked(path: str, blocked_paths: list[str]) -> bool:
    return any(path.startswith(bp) for bp in blocked_paths)

def can_see_photo(perms: dict, path: str, face_ids: list[str]) -> bool:
    """Decide visibility for a single photo. See module docstring for rule order."""
    if perms["is_admin"]:
        return True
    if not perms["is_viewer"]:
        return False
    # 本人特權：identity 出現在照片裡 → 永遠可見，beats blocked_paths
    identity = perms.get("identity") or ""
    if identity and identity in face_ids:
        return True
    # 路徑黑名單 → 擋
    if path_blocked(path, perms["blocked_paths"]):
        return False
    if not face_ids:
        # no-face photo → admin only
        # TODO(v0.3+): support per-group "public_paths" or "allow_no_face" so
        # landscape / food shots in approved dirs can also reach viewers
        # (e.g. share /Pictures/Travel/2024 with everyone, no face required).
        return False
    # 群組允許的人臉出現在照片 → 可見
    if perms["allowed_faces"].intersection(face_ids):
        return True
    return False  # 嚴格私有：未授權的人臉照片不顯示
