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
from collections import defaultdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PROJECT_ROOT as PROJECT_DIR, FACES_DIR  # noqa: E402
import _auth  # noqa: E402

FACES_FILE = FACES_DIR / "face_clusters.json"
NAMES_FILE = FACES_DIR / "face_names.json"
REMOVED_FILE = FACES_DIR / "face_removed.json"
SKIPPED_FILE = FACES_DIR / "face_skipped.json"
MERGES_FILE = FACES_DIR / "face_merges.json"
MOVES_FILE = FACES_DIR / "face_moves.json"
THUMB_OVERRIDES_FILE = FACES_DIR / "face_thumb_overrides.json"
THUMBS_DIR = FACES_DIR / "face_thumbs"
IMG_THUMB_CACHE = FACES_DIR / "img_thumb_cache"
PAGE_SIZE = 20


def load_json(path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default if default is not None else {}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_faces():   return load_json(FACES_FILE)
def load_names():   return load_json(NAMES_FILE)
def load_removed(): return load_json(REMOVED_FILE, {})
def load_skipped(): return load_json(SKIPPED_FILE, [])
def load_merges():  return load_json(MERGES_FILE, {})
def load_moves():   return load_json(MOVES_FILE, [])
def save_names(n):  save_json(NAMES_FILE, n)
def save_removed(r): save_json(REMOVED_FILE, r)
def save_skipped(s): save_json(SKIPPED_FILE, s)
def save_merges(m): save_json(MERGES_FILE, m)
def save_moves(m):  save_json(MOVES_FILE, m)


def _next_user_face_id(faces, moves) -> str:
    """產生下一個 user-created face id (`face_u1`, `face_u2`, ...)，避開既有 cluster 與既有 moves target。"""
    existing = set((faces.get("clusters") or {}).keys())
    for m in moves:
        t = m.get("to", "")
        if t:
            existing.add(t)
    n = 1
    while f"face_u{n}" in existing:
        n += 1
    return f"face_u{n}"


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


ADMIN_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>管理 · 人臉命名工具</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#111;color:#e0e0e0;padding:20px;max-width:1100px;margin:0 auto}
a{color:#4fc3f7;text-decoration:none}
.userbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;color:#888;font-size:13px}
.userbar .who{color:#4fc3f7}
.userbar a,.userbar button{padding:6px 12px;background:#222;color:#888;border:1px solid #333;border-radius:6px;font-size:12px;cursor:pointer;text-decoration:none;margin-left:6px}
.userbar a:hover,.userbar button:hover{background:#2a2a2a;color:#ccc}
h1{font-size:22px;color:#4fc3f7;margin-bottom:14px}
.tabs{display:flex;gap:6px;margin-bottom:18px;border-bottom:1px solid #333}
.tabs button{padding:10px 18px;background:none;border:none;color:#888;font-size:14px;cursor:pointer;border-bottom:2px solid transparent}
.tabs button.active{color:#4fc3f7;border-color:#4fc3f7}
.panel{display:none}
.panel.active{display:block}
.card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:10px;padding:16px;margin-bottom:14px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.row > *{flex:0 0 auto}
.row label{color:#999;font-size:12px}
input[type=text],input[type=password],select{background:#222;color:#fff;border:1px solid #333;border-radius:5px;padding:7px 10px;font-size:13px}
input:focus,select:focus{border-color:#4fc3f7;outline:none}
button.primary{background:#4fc3f7;color:#000;border:none;padding:7px 14px;border-radius:5px;font-weight:600;cursor:pointer;font-size:13px}
button.primary:hover{background:#3fb3e7}
button.danger{background:#5a2828;color:#ff8a8a;border:1px solid #6a3838;padding:6px 10px;border-radius:5px;cursor:pointer;font-size:12px}
button.danger:hover{background:#6a3030}
button.ghost{background:#222;color:#999;border:1px solid #333;padding:6px 10px;border-radius:5px;cursor:pointer;font-size:12px}
button.ghost:hover{background:#2a2a2a;color:#ccc}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px 8px;text-align:left;border-bottom:1px solid #2a2a2a;vertical-align:middle}
th{color:#888;font-weight:500;font-size:12px}
.tag{display:inline-block;padding:2px 8px;background:#222;color:#9cc;border-radius:10px;font-size:11px;margin:2px 3px 2px 0}
.tag.admin{background:#3a2a4a;color:#d6c}
.muted{color:#666;font-size:12px}
.msg{padding:10px 14px;border-radius:6px;margin:10px 0;font-size:13px;display:none}
.msg.ok{background:#1e3a1e;color:#9c9}
.msg.err{background:#3a1e1e;color:#f99}
/* group editor */
.editor h3{font-size:14px;color:#9cf;margin:10px 0 8px}
.face-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;max-height:380px;overflow-y:auto;padding:6px;background:#111;border:1px solid #2a2a2a;border-radius:6px}
.face-item{position:relative;background:#1a1a1a;border:2px solid transparent;border-radius:6px;cursor:pointer;overflow:hidden;text-align:center}
.face-item.on{border-color:#4fc3f7}
.face-item img{width:100%;height:90px;object-fit:cover;display:block;background:#000}
.face-item .meta{padding:4px 6px;font-size:11px;color:#bbb;line-height:1.3}
.face-item .meta .n{color:#9cc;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.face-item.on::after{content:"✓";position:absolute;top:4px;right:6px;background:#4fc3f7;color:#000;width:18px;height:18px;border-radius:50%;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center}
.path-list{max-height:240px;overflow-y:auto;background:#111;border:1px solid #2a2a2a;border-radius:6px;padding:6px}
.path-row{display:flex;gap:8px;align-items:center;padding:6px 8px;border-radius:4px}
.path-row:hover{background:#1a1a1a}
.path-row label{cursor:pointer;flex:1;font-size:12px;color:#ccc;word-break:break-all}
.path-row .c{color:#666;font-size:11px}
.checkbox-row{display:flex;gap:14px;flex-wrap:wrap;background:#111;border:1px solid #2a2a2a;border-radius:6px;padding:8px 10px}
.checkbox-row label{font-size:13px;color:#ccc;cursor:pointer;display:inline-flex;align-items:center;gap:4px}
input[type=checkbox]{accent-color:#4fc3f7}
.section-label{font-size:12px;color:#777;margin:14px 0 6px}
.actions-bar{display:flex;gap:8px;margin-top:12px;justify-content:flex-end}
.group-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.group-header h2{font-size:16px;color:#4fc3f7}
.summary{color:#888;font-size:12px;margin-bottom:8px}
</style>
</head><body>
<div class="userbar">
  <div>👤 <span class="who">__USER__</span> <span style="color:#666;font-size:11px">(admin)</span></div>
  <div>
    <a href="/">← 回相簿</a>
    <button onclick="logout()">登出</button>
  </div>
</div>

<h1>⚙️ 管理</h1>
<div class="msg" id="msg"></div>

<div class="tabs">
  <button class="active" data-tab="users" onclick="switchTab('users')">使用者</button>
  <button data-tab="groups" onclick="switchTab('groups')">群組</button>
</div>

<!-- USERS -->
<div class="panel active" id="panel-users">
  <div class="card">
    <h3 style="font-size:14px;color:#9cf;margin-bottom:10px">新增使用者</h3>
    <div class="row">
      <input id="nu-name" type="text" placeholder="使用者名稱" autocomplete="off">
      <input id="nu-pw" type="password" placeholder="密碼" autocomplete="new-password">
      <label>權限 <select id="nu-role">
        <option value="viewer">viewer</option>
        <option value="admin">admin</option>
      </select></label>
      <button class="primary" onclick="createUser()">新增</button>
    </div>
    <div class="section-label">身份（這個使用者本人是誰？本人照片會自動可見，且不受群組黑名單限制）</div>
    <select id="nu-identity" style="min-width:240px"></select>
    <div class="section-label">群組（viewer 才有意義；群組可多個使用者共用）</div>
    <div class="checkbox-row" id="nu-groups"></div>
  </div>

  <div class="card">
    <table id="users-table"><thead>
      <tr><th>名稱</th><th>權限</th><th>身份</th><th>群組</th><th style="text-align:right">動作</th></tr>
    </thead><tbody></tbody></table>
  </div>
</div>

<!-- GROUPS -->
<div class="panel" id="panel-groups">
  <div class="card">
    <h3 style="font-size:14px;color:#9cf;margin-bottom:10px">新增群組</h3>
    <div class="row">
      <input id="ng-name" type="text" placeholder="群組名稱（如 family）" autocomplete="off">
      <button class="primary" onclick="createGroup()">新增</button>
    </div>
  </div>
  <div id="groups-list"></div>
</div>

<script>
let USERS=[], GROUPS=[], FACES=[], PATHS=[];
const $ = (id) => document.getElementById(id);

function showMsg(text, ok){
  const m = $('msg');
  m.textContent = text;
  m.className = 'msg ' + (ok ? 'ok' : 'err');
  m.style.display = 'block';
  setTimeout(()=>{ m.style.display='none'; }, 3500);
}
function api(method, url, body){
  const opts = {method, headers:{'Content-Type':'application/json'}};
  if(body!==undefined) opts.body = JSON.stringify(body);
  return fetch(url, opts).then(r=>r.json());
}
function logout(){ fetch('/api/logout',{method:'POST'}).then(()=>location.href='/login'); }

function switchTab(t){
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('active', b.dataset.tab===t));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active', p.id==='panel-'+t));
  if(t==='groups'){ renderGroups(); }
}

// ---- USERS ----
function escapeHtml(s){ return (s||'').replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"})[c]); }

function renderGroupCheckboxes(container, selected){
  const sel = new Set(selected||[]);
  container.innerHTML = GROUPS.length
    ? GROUPS.map(g=>`<label><input type="checkbox" value="${escapeHtml(g.name)}" ${sel.has(g.name)?'checked':''}>${escapeHtml(g.name)}</label>`).join('')
    : '<span class="muted">（尚未建立群組，先到「群組」分頁新增）</span>';
}

function renderIdentityOptions(sel, selected){
  // 訪客 + 所有命名的 face cluster
  const cur = selected || '';
  sel.innerHTML = '<option value="">訪客（無預設可見照片）</option>'
    + FACES.map(f=>`<option value="${escapeHtml(f.id)}" ${cur===f.id?'selected':''}>${escapeHtml(f.name)} · ${f.count} 張</option>`).join('');
}

function identityLabel(fid){
  if(!fid) return '<span class="muted">訪客</span>';
  const f = FACES.find(x=>x.id===fid);
  return f ? `<span class="tag">${escapeHtml(f.name)}</span>` : `<span class="tag muted">${escapeHtml(fid)}（已失效）</span>`;
}

function renderUsers(){
  const tb = document.querySelector('#users-table tbody');
  tb.innerHTML = USERS.map(u=>{
    const isAdmin = u.role==='admin';
    const groupTags = (u.groups||[]).map(g=>`<span class="tag">${escapeHtml(g)}</span>`).join('') || '<span class="muted">—</span>';
    return `<tr data-name="${escapeHtml(u.username)}">
      <td><b>${escapeHtml(u.username)}</b></td>
      <td><span class="tag ${isAdmin?'admin':''}">${u.role}</span></td>
      <td>${identityLabel(u.identity)}</td>
      <td>${groupTags}</td>
      <td style="text-align:right;white-space:nowrap">
        <button class="ghost" onclick="editUser('${escapeHtml(u.username)}')">編輯</button>
        <button class="danger" onclick="deleteUser('${escapeHtml(u.username)}')">刪除</button>
      </td>
    </tr>`;
  }).join('');
  renderGroupCheckboxes($('nu-groups'), []);
  renderIdentityOptions($('nu-identity'), '');
}

function createUser(){
  const username = $('nu-name').value.trim();
  const password = $('nu-pw').value;
  const role = $('nu-role').value;
  const identity = $('nu-identity').value;
  const groups = Array.from($('nu-groups').querySelectorAll('input:checked')).map(x=>x.value);
  if(!username || !password){ showMsg('username 與密碼必填', false); return; }
  api('POST','/api/admin/users',{username,password,role,identity,groups}).then(r=>{
    if(r.ok){ showMsg('✓ 已新增 '+username, true); $('nu-name').value=''; $('nu-pw').value=''; loadAll(); }
    else showMsg(r.error||'新增失敗', false);
  });
}

function editUser(name){
  const u = USERS.find(x=>x.username===name);
  if(!u) return;
  const tr = document.querySelector(`#users-table tr[data-name="${CSS.escape(name)}"]`);
  const groupChecks = GROUPS.map(g=>`<label style="display:inline-flex;gap:3px;align-items:center;margin-right:10px;font-size:12px"><input type="checkbox" value="${escapeHtml(g.name)}" ${u.groups.includes(g.name)?'checked':''}>${escapeHtml(g.name)}</label>`).join('') || '<span class="muted">（無群組）</span>';
  const idOpts = '<option value="">訪客（無預設可見照片）</option>'
    + FACES.map(f=>`<option value="${escapeHtml(f.id)}" ${u.identity===f.id?'selected':''}>${escapeHtml(f.name)} · ${f.count} 張</option>`).join('');
  tr.innerHTML = `<td colspan="5">
    <div class="row" style="margin-bottom:8px">
      <b>${escapeHtml(name)}</b>
      <label>權限 <select id="ed-role-${escapeHtml(name)}">
        <option value="viewer" ${u.role==='viewer'?'selected':''}>viewer</option>
        <option value="admin" ${u.role==='admin'?'selected':''}>admin</option>
      </select></label>
      <input type="password" id="ed-pw-${escapeHtml(name)}" placeholder="新密碼（留空則不改）" autocomplete="new-password">
    </div>
    <div class="row" style="margin-bottom:8px">
      <label>身份 <select id="ed-id-${escapeHtml(name)}" style="min-width:240px">${idOpts}</select></label>
    </div>
    <div style="margin-bottom:8px">${groupChecks}</div>
    <div style="text-align:right">
      <button class="ghost" onclick="loadUsers()">取消</button>
      <button class="primary" onclick="saveUser('${escapeHtml(name)}')">儲存</button>
    </div>
  </td>`;
}

function saveUser(name){
  const role = $('ed-role-'+name).value;
  const identity = $('ed-id-'+name).value;
  const pw = $('ed-pw-'+name).value;
  const tr = document.querySelector(`#users-table tr[data-name="${CSS.escape(name)}"]`);
  const groups = Array.from(tr.querySelectorAll('input[type=checkbox]:checked')).map(x=>x.value);
  const body = {action:'update', role, identity, groups};
  if(pw) body.password = pw;
  api('POST','/api/admin/users/'+encodeURIComponent(name), body).then(r=>{
    if(r.ok){ showMsg('✓ 已更新 '+name, true); loadUsers(); }
    else showMsg(r.error||'儲存失敗', false);
  });
}

function deleteUser(name){
  if(!confirm('確定刪除使用者「'+name+'」？')) return;
  api('POST','/api/admin/users/'+encodeURIComponent(name),{action:'delete'}).then(r=>{
    if(r.ok){ showMsg('✓ 已刪除 '+name, true); loadUsers(); }
    else showMsg(r.error||'刪除失敗', false);
  });
}

// ---- GROUPS ----
function renderGroups(){
  const wrap = $('groups-list');
  if(GROUPS.length===0){ wrap.innerHTML='<div class="card muted">尚未建立群組。</div>'; return; }
  wrap.innerHTML = GROUPS.map(g=>{
    const allowed = new Set(g.allowed_faces||[]);
    const blocked = new Set(g.blocked_paths||[]);
    const faceGrid = FACES.map(f=>{
      const on = allowed.has(f.id);
      return `<div class="face-item ${on?'on':''}" data-fid="${escapeHtml(f.id)}" onclick="toggleFace(this)">
        <img src="/thumb/${escapeHtml(f.id)}.jpg?v=${f.thumb_ver}" loading="lazy">
        <div class="meta"><span class="n">${escapeHtml(f.name)}</span><span style="color:#666">${f.count}</span></div>
      </div>`;
    }).join('') || '<span class="muted">（尚無命名的人臉群組）</span>';
    const pathRows = PATHS.map(p=>{
      const id = 'p-'+g.name+'-'+btoa(unescape(encodeURIComponent(p.path))).replace(/=/g,'');
      return `<div class="path-row">
        <input type="checkbox" id="${id}" value="${escapeHtml(p.path)}" ${blocked.has(p.path)?'checked':''}>
        <label for="${id}">${escapeHtml(p.path)}<span class="c"> · ${p.count} 張</span></label>
      </div>`;
    }).join('') || '<span class="muted">（沒有可選路徑）</span>';
    return `<div class="card editor" data-group="${escapeHtml(g.name)}">
      <div class="group-header">
        <h2>${escapeHtml(g.name)}</h2>
        <button class="danger" onclick="deleteGroup('${escapeHtml(g.name)}')">刪除群組</button>
      </div>
      <div class="summary">允許看到 <b>${g.allowed_faces.length}</b> 個人臉群組 · 封鎖 <b>${g.blocked_paths.length}</b> 個路徑前綴</div>

      <h3>可看到的人臉群組（allowed_faces）</h3>
      <div class="face-grid">${faceGrid}</div>

      <h3>封鎖路徑前綴（blocked_paths）</h3>
      <div class="path-list">${pathRows}</div>

      <div class="actions-bar">
        <button class="primary" onclick="saveGroup('${escapeHtml(g.name)}')">儲存</button>
      </div>
    </div>`;
  }).join('');
}

function toggleFace(el){ el.classList.toggle('on'); }

function createGroup(){
  const name = $('ng-name').value.trim();
  if(!name){ showMsg('群組名稱必填', false); return; }
  api('POST','/api/admin/groups/'+encodeURIComponent(name),{action:'upsert',allowed_faces:[],blocked_paths:[]}).then(r=>{
    if(r.ok){ showMsg('✓ 已新增群組 '+name, true); $('ng-name').value=''; loadAll(); }
    else showMsg(r.error||'新增失敗', false);
  });
}

function saveGroup(name){
  const card = document.querySelector(`.card[data-group="${CSS.escape(name)}"]`);
  const allowed_faces = Array.from(card.querySelectorAll('.face-item.on')).map(x=>x.dataset.fid);
  const blocked_paths = Array.from(card.querySelectorAll('.path-list input:checked')).map(x=>x.value);
  api('POST','/api/admin/groups/'+encodeURIComponent(name),{action:'upsert',allowed_faces,blocked_paths}).then(r=>{
    if(r.ok){ showMsg('✓ 已更新群組 '+name, true); loadAll(); }
    else showMsg(r.error||'儲存失敗', false);
  });
}

function deleteGroup(name){
  if(!confirm('確定刪除群組「'+name+'」？\\n所有使用者對這個群組的引用也會被移除。')) return;
  api('POST','/api/admin/groups/'+encodeURIComponent(name),{action:'delete'}).then(r=>{
    if(r.ok){ showMsg('✓ 已刪除群組 '+name, true); loadAll(); }
    else showMsg(r.error||'刪除失敗', false);
  });
}

// ---- bootstrap ----
function loadUsers(){
  return fetch('/api/admin/users').then(r=>r.json()).then(d=>{ USERS=d; renderUsers(); });
}
function loadAll(){
  return Promise.all([
    fetch('/api/admin/users').then(r=>r.json()),
    fetch('/api/admin/groups').then(r=>r.json()),
    fetch('/api/admin/faces').then(r=>r.json()),
    fetch('/api/admin/paths').then(r=>r.json()),
  ]).then(([u,g,f,p])=>{
    USERS=u; GROUPS=g; FACES=f; PATHS=p;
    renderUsers();
    if(document.querySelector('.tabs button.active').dataset.tab==='groups') renderGroups();
  });
}
loadAll();
</script>
</body></html>"""


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
        return _auth.can_see_photo(p, img_path, effective)

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

        elif path == "/api/admin/users":
            if not self._require_admin(): return
            self.json_response(self._admin_list_users())

        elif path == "/api/admin/groups":
            if not self._require_admin(): return
            self.json_response(self._admin_list_groups())

        elif path == "/api/admin/faces":
            if not self._require_admin(): return
            self.json_response(self._admin_list_faces())

        elif path == "/api/admin/paths":
            if not self._require_admin(): return
            self.json_response(self._admin_list_paths())

        elif path.startswith("/image/"):
            img = self.fix_path(path[7:])
            if not self._can_see_image(img):
                self.send_error(403); return
            self.serve_file(img, "image/jpeg")

        elif path.startswith("/thumb/"):
            # face cluster thumbnail; only allowed for clusters the user can see
            fid_filename = unquote(path[7:])
            fid = fid_filename.rsplit(".", 1)[0]
            if not self._can_see_cluster(fid):
                self.send_error(403); return
            thumb = THUMBS_DIR / fid_filename
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
            self.json_response(self.get_page(page, flt))

        elif path == "/api/clusters":
            # 輕量列表，給合併下拉選單用
            data = self.get_all_sorted("all")
            self.json_response([
                {"id": r["id"], "name": r["name"], "count": r["count"]}
                for r in data
            ])

        elif path == "/api/stats":
            self.json_response(self.get_stats())

        elif path == "/api/cluster_meta":
            fid = qs.get("fid", [""])[0]
            self.json_response(self.get_cluster_meta(fid))

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

        # everything else is admin-only (mutations)
        if not self._user():
            self.send_error(401, "login required"); return
        if not self._require_admin():
            return

        if path == "/api/name":
            names = load_names()
            fid, name = body.get("face_id"), body.get("name", "")
            if fid and name:
                names[fid] = name
            elif fid and fid in names:
                del names[fid]
            save_names(names)
            self.json_response({"ok": True})

        elif path == "/api/skip":
            skipped = load_skipped()
            fid = body.get("face_id")
            if fid:
                if fid in skipped:
                    skipped.remove(fid)
                else:
                    skipped.append(fid)
            save_skipped(skipped)
            self.json_response({"ok": True, "skipped": fid in skipped})

        elif path == "/api/merge":
            merges = load_merges()
            names = load_names()
            src, tgt = body.get("source_id"), body.get("target_id")
            if src and tgt and src != tgt:
                final = _resolve_target(merges, tgt)
                if final != src:
                    merges[src] = final
                    save_merges(merges)
                    # 命名傳遞：source 也採用 target 的名字（讓 photo_index 重建後一致）
                    if final in names:
                        names[src] = names[final]
                        save_names(names)
            self.json_response({"ok": True})

        elif path == "/api/unmerge":
            merges = load_merges()
            fid = body.get("face_id")
            if fid and fid in merges:
                del merges[fid]
                save_merges(merges)
            self.json_response({"ok": True})

        elif path == "/api/remove":
            removed = load_removed()
            fid, img_path = body.get("face_id"), body.get("image_path")
            if fid and img_path:
                removed.setdefault(fid, [])
                if img_path not in removed[fid]:
                    removed[fid].append(img_path)
                save_removed(removed)
            self.json_response({"ok": True})

        elif path == "/api/restore":
            removed = load_removed()
            fid, img_path = body.get("face_id"), body.get("image_path")
            if fid and img_path and fid in removed and img_path in removed[fid]:
                removed[fid].remove(img_path)
                save_removed(removed)
            self.json_response({"ok": True})

        elif path == "/api/move":
            moves = load_moves()
            from_id = body.get("from_id")
            to_id = body.get("to_id")
            img_path = body.get("image_path")
            new_name = (body.get("new_name") or "").strip()
            if to_id == "__new__":
                to_id = _next_user_face_id(load_faces(), moves)
                if new_name:
                    names = load_names()
                    names[to_id] = new_name
                    save_names(names)
            if from_id and to_id and img_path and from_id != to_id:
                moves = [m for m in moves if m.get("path") != img_path]
                moves.append({"path": img_path, "from": from_id, "to": to_id})
                save_moves(moves)
            self.json_response({"ok": True, "to_id": to_id})

        elif path == "/api/unmove":
            moves = load_moves()
            img_path = body.get("image_path")
            if img_path:
                moves = [m for m in moves if m.get("path") != img_path]
                save_moves(moves)
            self.json_response({"ok": True})

        elif path == "/api/move-batch":
            moves = load_moves()
            from_id = body.get("from_id")
            to_id = body.get("to_id")
            paths = body.get("paths", [])
            new_name = (body.get("new_name") or "").strip()
            # 特殊 sentinel：建立新群組，optional new_name 直接寫進 face_names.json
            if to_id == "__new__":
                to_id = _next_user_face_id(load_faces(), moves)
                if new_name:
                    names = load_names()
                    names[to_id] = new_name
                    save_names(names)
            if from_id and to_id and paths and from_id != to_id:
                ps = set(paths)
                moves = [m for m in moves if m.get("path") not in ps]
                for p in paths:
                    moves.append({"path": p, "from": from_id, "to": to_id})
                save_moves(moves)
            self.json_response({"ok": True, "moved": len(paths), "to_id": to_id})

        elif path == "/api/set_thumb":
            fid = body.get("face_id")
            img_path = body.get("image_path")
            result = self._set_custom_thumb(fid, img_path)
            self.json_response(result)

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

    def get_cluster_meta(self, fid: str) -> dict:
        """為單一 cluster 的圖片回傳 mtime / 年 / 月，並按時間 desc 排序。"""
        if not fid:
            return {"id": fid, "images": [], "removed": []}
        all_data = self.get_all_sorted("all")
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
        removed = sorted([enrich(p) for p in target.get("removed", [])],
                         key=lambda x: x["ts"], reverse=True)
        return {"id": fid, "images": active, "removed": removed}

    # ---- thumb crop helper ----

    def _set_custom_thumb(self, face_id: str, img_path: str) -> dict:
        """裁切指定照片裡屬於 face_id (含合併源) 的人臉、覆寫 face_thumbs/<fid>.jpg"""
        if not face_id or not img_path:
            return {"ok": False, "error": "missing face_id or image_path"}
        from pathlib import Path as _P
        if not _P(img_path).exists():
            return {"ok": False, "error": f"image not found: {img_path}"}

        # 解析合併源（找出視覺上「同一個人」的所有 face_id）
        merges = load_merges()
        # 反查：哪些 source 合進這個 target
        merged_in = {face_id}
        for src, tgt in merges.items():
            final = _resolve_target(merges, src)
            if final == face_id:
                merged_in.add(src)

        # 在 face_clusters.images[img_path] 找對應的 face record
        faces = load_faces()
        rec = (faces.get("images", {}) or {}).get(img_path, [])
        target_face = None
        for f in rec:
            if f.get("face_id") in merged_in:
                target_face = f
                break
        if not target_face:
            return {"ok": False,
                    "error": f"找不到屬於 {face_id} 的人臉於 {img_path}（可能是 face detector 沒偵測到）"}

        # 裁切
        try:
            from generate_face_thumbs import crop_face_thumb
        except ImportError:
            return {"ok": False, "error": "無法載入 crop_face_thumb"}

        out_path = THUMBS_DIR / f"{face_id}.jpg"
        ok = crop_face_thumb(img_path, target_face["bbox"], out_path)
        if not ok:
            return {"ok": False, "error": "裁切失敗（圖檔損壞或 bbox 無效）"}

        # 記錄手動指定
        overrides = load_json(THUMB_OVERRIDES_FILE, {})
        overrides[face_id] = img_path
        save_json(THUMB_OVERRIDES_FILE, overrides)
        return {"ok": True, "thumb_path": str(out_path), "image_path": img_path}

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
                "blocked_paths": list(g.get("blocked_paths", [])),
            }
            for n, g in sorted(groups.items())
        ]

    @staticmethod
    def _admin_list_faces():
        """已命名的 cluster 清單（admin 才會用到，給群組編輯選 allowed_faces）。"""
        faces = load_faces() or {}
        names = load_names()
        merges = load_merges()
        clusters = faces.get("clusters", {}) or {}
        # 反查每個 final target 的圖片數（含被合併進來的 source）
        merge_back: dict[str, list[str]] = defaultdict(list)
        for src in list(merges.keys()):
            final = _resolve_target(merges, src)
            if final != src:
                merge_back[final].append(src)
        result = []
        for fid, info in clusters.items():
            if fid in merges:
                continue  # 被合併走的不獨立顯示
            name = names.get(fid, "")
            if not name:
                continue  # 只列出已命名的，admin 才知道誰是誰
            count = len(info.get("images", []) or [])
            for s in merge_back.get(fid, []):
                count += len(clusters.get(s, {}).get("images", []) or [])
            thumb_file = THUMBS_DIR / f"{fid}.jpg"
            thumb_ver = int(thumb_file.stat().st_mtime) if thumb_file.exists() else 0
            result.append({"id": fid, "name": name, "count": count, "thumb_ver": thumb_ver})
        result.sort(key=lambda r: (-r["count"], r["name"]))
        return result

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
        blocked_paths = body.get("blocked_paths") or []
        if not isinstance(allowed_faces, list) or not isinstance(blocked_paths, list):
            return {"ok": False, "error": "allowed_faces / blocked_paths 必須是陣列"}
        groups[name] = {
            "allowed_faces": [x for x in allowed_faces if isinstance(x, str) and x.strip()],
            "blocked_paths": [x for x in blocked_paths if isinstance(x, str) and x.strip()],
        }
        _auth.save_groups(groups)
        return {"ok": True}

    # ---- data assembly ----

    def get_all_sorted(self, flt):
        faces = load_faces()
        names = load_names()
        removed = load_removed()
        skipped = set(load_skipped())
        merges = load_merges()
        moves = load_moves()
        clusters = faces.get("clusters", {})

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
            f = THUMBS_DIR / f"{fid}.jpg"
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
            # 1) cluster：只留 allowed_faces 對應的（含本人 identity，已在 allowed_faces 裡）
            result = [r for r in result if r["id"] in perms["allowed_faces"]]

            # 2) per-image：用 _auth.can_see_photo 規則
            #    - 本人在照片裡 → 顯示 (beats blocked_paths)
            #    - 路徑被擋 → 隱藏
            #    - 群組 allowed_faces 與照片人臉有交集 → 顯示
            # 預先建 path -> 已解析的 face_id 清單（含 merges + moves）
            img_to_faces: dict[str, list[str]] = {}
            for img_path, recs in (faces.get("images") or {}).items():
                img_to_faces[img_path] = [
                    _resolve_target(merges, r.get("face_id", "")) for r in recs
                ]
            # 套用 moves：from_face → to_face per image
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

            filtered: list[dict] = []
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

        # 排序：未略過在前（依 count desc）、略過在後（依 count desc）
        result.sort(key=lambda r: (r["skipped"], -r["count"]))
        return result

    def get_page(self, page, flt):
        all_data = self.get_all_sorted(flt)
        total = len(all_data)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        page = max(0, min(page, total_pages - 1))
        start = page * PAGE_SIZE
        end = start + PAGE_SIZE
        return {
            "page": page,
            "page_size": PAGE_SIZE,
            "total": total,
            "total_pages": total_pages,
            "items": all_data[start:end],
        }

    def get_stats(self):
        names = load_names()
        skipped = load_skipped()
        merges = load_merges()
        faces = load_faces()
        total = len(faces.get("clusters", {}))
        perms = self._perms()
        if perms["is_admin"]:
            return {
                "total": total,
                "displayed": total - len(merges),
                "named": len(names),
                "skipped": len(skipped),
                "merges": len(merges),
            }
        allowed = perms["allowed_faces"]
        return {
            "total": len(allowed),
            "displayed": len(allowed),
            "named": sum(1 for fid in allowed if names.get(fid)),
            "skipped": 0,
            "merges": 0,
        }

    def generate_login_html(self):
        return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>登入 · 人臉命名工具</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#111;color:#e0e0e0;
  min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.card{background:#1a1a1a;padding:30px;border-radius:12px;border:1px solid #333;max-width:360px;width:100%}
h1{font-size:20px;margin-bottom:18px;color:#4fc3f7;text-align:center}
input{width:100%;padding:11px 12px;background:#222;color:#fff;border:1px solid #333;border-radius:6px;font-size:15px;margin-bottom:12px}
input:focus{border-color:#4fc3f7;outline:none}
button{width:100%;padding:12px;background:#4fc3f7;color:#000;border:none;border-radius:6px;font-size:15px;font-weight:600;cursor:pointer}
button:hover{background:#3fb3e7}
.err{color:#ef5350;font-size:13px;margin-top:8px;text-align:center;min-height:20px}
</style>
</head><body>
<form class="card" onsubmit="event.preventDefault();login()">
  <h1>👥 人臉命名工具</h1>
  <input id="u" type="text" placeholder="使用者名稱" autofocus autocomplete="username">
  <input id="p" type="password" placeholder="密碼" autocomplete="current-password">
  <button type="submit">登入</button>
  <div class="err" id="err"></div>
</form>
<script>
function login(){
  fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({username:document.getElementById('u').value,password:document.getElementById('p').value})})
    .then(r=>r.json()).then(d=>{
      if(d.ok) location.href='/';
      else document.getElementById('err').textContent=d.error||'登入失敗';
    });
}
</script>
</body></html>"""

    def generate_html(self):
        user = self._user() or ""
        perms = self._perms()
        is_admin = "true" if perms["is_admin"] else "false"
        inject = f"<script>window.CURRENT_USER='{user}';window.IS_ADMIN={is_admin};</script>"
        return self._generate_html_template().replace(
            "<!--USER_INJECT-->", inject)

    def generate_admin_html(self):
        user = self._user() or ""
        return ADMIN_HTML_TEMPLATE.replace("__USER__", user)

    def _generate_html_template(self):
        return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>人臉命名工具</title>
<!--USER_INJECT-->
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#111;color:#e0e0e0;padding:20px}
h1{text-align:center;margin-bottom:6px}
.subtitle{text-align:center;color:#888;margin-bottom:14px;font-size:14px}
.userbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;color:#888;font-size:13px;flex-wrap:wrap;gap:8px}
.userbar .who{color:#4fc3f7}
.userbar button{padding:6px 12px;background:#222;color:#888;border:1px solid #333;border-radius:6px;font-size:12px;cursor:pointer}
.userbar button:hover{background:#2a2a2a;color:#ccc}
.toolbar{display:flex;justify-content:center;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.toolbar button{padding:7px 14px;background:#222;color:#888;border:1px solid #333;border-radius:6px;cursor:pointer;font-size:13px;min-height:36px}
.toolbar button:hover{background:#2a2a2a}
.toolbar button.active{background:#4fc3f7;color:#000;border-color:#4fc3f7}
.admin-only{display:none}
body.admin .admin-only{display:initial}
body.admin .admin-only.flex{display:flex}
body.admin .admin-only.actions{display:flex}
@media (max-width:600px){
  body{padding:10px}
  .grid{grid-template-columns:1fr !important}
  .pager{flex-wrap:wrap}
  .expand-modal .expand-box{max-height:calc(100vh - 20px) !important;width:100vw !important}
  .expand-header{padding:10px 14px}
  .expand-header h3{font-size:15px}
  .modal-box{padding:18px}
  .face-thumb{width:80px !important;height:80px !important}
  .card-top{grid-template-columns:1fr 80px !important}
}
.stats{text-align:center;margin-bottom:16px;color:#888;font-size:13px}
.pager{display:flex;justify-content:center;align-items:center;gap:10px;margin:18px 0}
.pager button{padding:8px 16px;background:#222;color:#ccc;border:1px solid #333;border-radius:6px;cursor:pointer}
.pager button:hover:not(:disabled){background:#2a2a2a}
.pager button:disabled{opacity:.3;cursor:not-allowed}
.pager input{width:56px;padding:6px;text-align:center;background:#222;color:#fff;border:1px solid #333;border-radius:6px}
.pager .info{color:#888;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:16px}
/* viewer-only tile mode: 大頭像 + 名字，整塊可點 */
body:not(.admin) .grid{grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:14px}
body:not(.admin) .tile{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:14px;padding:14px 10px;cursor:pointer;text-align:center;transition:transform .15s,border-color .15s;user-select:none;-webkit-tap-highlight-color:transparent}
body:not(.admin) .tile:hover{transform:translateY(-2px);border-color:#4fc3f7}
body:not(.admin) .tile:active{transform:scale(.97)}
body:not(.admin) .tile .avatar{width:100%;aspect-ratio:1;border-radius:50%;object-fit:cover;background:#000;display:block;margin:0 auto 10px;border:2px solid #2a2a2a}
body:not(.admin) .tile:hover .avatar{border-color:#4fc3f7}
body:not(.admin) .tile .tname{font-size:15px;color:#e0e0e0;font-weight:500;line-height:1.3;word-break:break-word}
body:not(.admin) .tile .tcount{color:#666;font-size:12px;margin-top:3px}
@media (max-width:480px){
  body:not(.admin) .grid{grid-template-columns:repeat(2,1fr);gap:10px}
  body:not(.admin) .tile{padding:10px 8px}
  body:not(.admin) .tile .tname{font-size:14px}
}
@media (min-width:481px) and (max-width:768px){
  body:not(.admin) .grid{grid-template-columns:repeat(3,1fr)}
}
/* viewer 也要能蓋掉桌面 1fr 強制：因為 600px @media 把 grid 強制成 1fr */
@media (max-width:600px){
  body:not(.admin) .grid{grid-template-columns:repeat(2,1fr) !important}
}
.list{display:flex;flex-direction:column;gap:6px}
.list-row{display:grid;grid-template-columns:48px 90px 1fr 80px auto;align-items:center;gap:14px;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:8px 14px}
.list-row:hover{border-color:#444}
.list-row .lr-thumb{width:40px;height:40px;border-radius:50%;object-fit:cover;border:2px solid #4fc3f7}
.list-row .lr-id{color:#888;font-size:12px;font-family:monospace}
.list-row .lr-name{color:#4fc3f7;font-weight:600;font-size:15px;cursor:text;padding:4px 8px;border-radius:4px;min-height:28px}
.list-row .lr-name:hover{background:#222}
.list-row .lr-name-input{width:100%;padding:5px 8px;background:#222;color:#fff;border:1px solid #4fc3f7;border-radius:4px;font-size:14px}
.list-row .lr-count{color:#888;font-size:13px;text-align:right}
.list-row .lr-actions{display:flex;gap:6px}
.list-row .lr-actions button{padding:5px 10px;font-size:12px;border-radius:4px;border:1px solid #444;background:#222;color:#ccc;cursor:pointer}
.list-row .lr-actions button:hover{background:#2a2a2a}
.card{background:#1a1a1a;border-radius:10px;border:2px solid #2a2a2a;overflow:hidden;transition:border-color .2s}
.card:hover{border-color:#444}
.card.named{border-color:#4fc3f7}
.card.skipped{opacity:.55;border-color:#444;background:#161616}
.card.skipped .face-meta h3{color:#777}
.card-top{display:grid;grid-template-columns:1fr 100px;gap:12px;padding:14px}
.face-meta h3{color:#4fc3f7;font-size:17px;margin-bottom:4px}
.face-meta .count{color:#888;font-size:13px}
.face-meta .named-label{color:#4fc3f7;font-weight:600;font-size:15px;margin-top:6px}
.face-meta .merged-badge{display:inline-block;margin-top:6px;padding:2px 6px;background:#3a2a4a;color:#ce93d8;font-size:11px;border-radius:4px}
.face-thumb{width:100px;height:100px;border-radius:50%;object-fit:cover;border:3px solid #4fc3f7}
.card.skipped .face-thumb{border-color:#555}
.photo-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;padding:0 3px}
.photo-grid img{width:100%;aspect-ratio:1;object-fit:cover;cursor:pointer;transition:opacity .15s}
.photo-grid img:hover{opacity:.8}
.expand-bar{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:#1f1f1f;border-top:1px solid #2a2a2a;cursor:pointer;font-size:13px;color:#888}
.expand-bar:hover{background:#252525;color:#ccc}
.expand-photos{display:grid;grid-template-columns:repeat(6,1fr);gap:4px}
.expand-photos .thumb-wrap{position:relative}
.expand-photos img{width:100%;aspect-ratio:1;object-fit:cover;cursor:pointer}
.expand-photos .thumb-wrap{position:relative}
.expand-photos .thumb-actions{position:absolute;top:2px;right:2px;display:flex;gap:3px;opacity:0;transition:opacity .15s}
.expand-photos .thumb-wrap:hover .thumb-actions{opacity:1}
.expand-photos .thumb-wrap.selectable img{cursor:pointer}
.expand-photos .thumb-wrap.selected{outline:3px solid #4fc3f7;outline-offset:-3px}
.expand-photos .thumb-wrap.selected::after{content:'✓';position:absolute;top:4px;left:4px;width:22px;height:22px;background:#4fc3f7;color:#000;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700}
.expand-tools{display:flex;gap:8px;padding:4px 0 10px;align-items:center}
.expand-tools button{padding:5px 12px;font-size:12px;border-radius:5px;border:1px solid #444;background:#222;color:#ccc;cursor:pointer}
.expand-tools button.active{background:#4fc3f7;color:#000;border-color:#4fc3f7}
.expand-tools .hint{color:#666;font-size:11px;margin-left:4px}
.select-bar{display:none;position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1a1a1a;border:2px solid #4fc3f7;border-radius:30px;padding:10px 22px;box-shadow:0 4px 24px rgba(0,0,0,.6);gap:12px;align-items:center;z-index:200}
.select-bar.active{display:flex}
.select-bar .count{color:#4fc3f7;font-weight:600}
.select-bar button{padding:7px 14px;border-radius:18px;border:none;cursor:pointer;font-size:13px;font-weight:600}
.select-bar .btn-go{background:#4fc3f7;color:#000}
.select-bar .btn-cancel{background:#333;color:#ccc}
.expand-photos .thumb-actions button{width:22px;height:22px;border-radius:50%;border:none;cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center;color:#fff;padding:0}
.expand-photos .move-btn{background:rgba(76,175,80,.85)}
.expand-photos .remove-btn{background:rgba(239,83,80,.85)}
.expand-photos .thumb-btn{background:rgba(255,193,7,.85)}
.actions{display:flex;gap:6px;padding:10px 14px;border-top:1px solid #2a2a2a;flex-wrap:wrap}
.actions input{flex:1;min-width:120px;padding:7px 10px;border:1px solid #333;border-radius:5px;background:#222;color:#fff;font-size:13px}
.actions input:focus{border-color:#4fc3f7;outline:none}
.btn{padding:7px 12px;border-radius:5px;cursor:pointer;font-size:13px;border:none}
.btn-save{background:#4fc3f7;color:#000;font-weight:600}
.btn-edit{background:#333;color:#4fc3f7;border:1px solid #444}
.btn-skip{background:#333;color:#888;border:1px solid #444}
.btn-merge{background:#333;color:#ab47bc;border:1px solid #444}
.btn-undo{background:#333;color:#ef5350;border:1px solid #444}
.modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.85);z-index:100;justify-content:center;align-items:center}
.modal.active{display:flex}
.modal-box{background:#1a1a1a;padding:24px;border-radius:12px;max-width:480px;width:90%;border:1px solid #333}
.expand-modal{align-items:flex-start;padding:24px 0}
.expand-modal .expand-box{background:#161616;border:1px solid #333;border-radius:12px;width:min(1400px,95vw);max-height:calc(100vh - 48px);display:flex;flex-direction:column}
.expand-header{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;border-bottom:1px solid #2a2a2a;flex-shrink:0}
.expand-header h3{color:#4fc3f7;font-size:18px}
.expand-header .meta{color:#888;font-size:13px;margin-left:10px}
.expand-header .close{background:#333;color:#ccc;border:none;width:34px;height:34px;border-radius:50%;cursor:pointer;font-size:16px}
.expand-header .close:hover{background:#444}
.expand-body{padding:14px 20px;overflow-y:auto;flex:1}
.expand-modal .expand-photos{grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}
.expand-modal .expand-photos .thumb-wrap img{border-radius:6px}
.expand-modal .removed-grid{grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:6px}
.year-filter{display:flex;gap:8px;align-items:center;padding:8px 0 12px;flex-wrap:wrap}
.year-filter select{padding:6px 10px;background:#222;color:#fff;border:1px solid #333;border-radius:6px;font-size:13px}
.year-filter .clear{padding:4px 10px;background:#333;color:#888;border:none;border-radius:4px;cursor:pointer;font-size:12px}
.year-section{margin-bottom:14px}
.year-header{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:#1f2a30;color:#4fc3f7;font-weight:600;border-radius:6px;cursor:pointer;margin-bottom:6px;font-size:14px;
  position:sticky;top:0;z-index:5;box-shadow:0 2px 6px rgba(0,0,0,.4)}
.year-header:hover{background:#253339}
.year-header .toggle{transition:transform .15s}
.year-section.collapsed .year-header .toggle{transform:rotate(-90deg)}
.year-section.collapsed .year-body{display:none}
.modal-box h3{margin-bottom:16px}
.modal-box input[type=text]{width:100%;padding:8px 10px;background:#222;color:#fff;border:1px solid #333;border-radius:6px;margin-bottom:8px}
.modal-box select{width:100%;padding:10px;margin-bottom:16px;background:#222;color:#fff;border:1px solid #333;border-radius:6px;max-height:300px}
.modal-actions{display:flex;gap:8px;justify-content:flex-end}
.spinner{text-align:center;padding:40px;color:#666}
.removed-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:4px;padding:10px 14px}
.removed-grid .thumb-wrap{position:relative}
.removed-grid img{width:100%;aspect-ratio:1;object-fit:cover;opacity:.4}
.removed-grid .restore-btn{position:absolute;top:2px;right:2px;width:22px;height:22px;border-radius:50%;background:rgba(76,175,80,.85);color:#fff;border:none;cursor:pointer;font-size:13px;display:flex;align-items:center;justify-content:center}
</style>
</head>
<body>

<div class="userbar">
  <div>👤 <span class="who" id="whoami"></span> <span id="role-tag" style="color:#888;font-size:11px"></span></div>
  <div>
    <a href="/admin" class="admin-only" style="padding:6px 12px;background:#222;color:#888;border:1px solid #333;border-radius:6px;font-size:12px;text-decoration:none;margin-right:6px">⚙️ 管理</a>
    <button onclick="logout()">登出</button>
  </div>
</div>

<h1 id="mainTitle">👥 家庭相簿</h1>
<div class="subtitle admin-only" id="subtitle"></div>

<div class="toolbar admin-only">
  <button class="active" data-filter="all" onclick="setFilter('all')">全部</button>
  <button data-filter="unnamed" onclick="setFilter('unnamed')">未命名</button>
  <button data-filter="named" onclick="setFilter('named')">已命名</button>
  <button data-filter="skipped" onclick="setFilter('skipped')">已略過</button>
  <button id="viewToggle" onclick="toggleView()" style="display:none;margin-left:14px">📋 清單模式</button>
</div>

<div class="stats admin-only" id="stats"></div>

<div class="pager" id="pagerTop"></div>
<div class="grid" id="grid"></div>
<div class="spinner" id="spinner" style="display:none">載入中...</div>
<div class="pager" id="pagerBottom"></div>

<div class="modal" id="mergeModal">
  <div class="modal-box">
    <h3>合併到哪個群組？</h3>
    <input type="text" id="mergeFilter" placeholder="輸入名稱或 ID 過濾..." oninput="filterMergeOptions()">
    <select id="mergeTarget" size="10"></select>
    <div class="modal-actions">
      <button class="btn btn-skip" onclick="closeMerge()">取消</button>
      <button class="btn btn-save" onclick="confirmMerge()">合併</button>
    </div>
  </div>
</div>

<div class="modal expand-modal" id="expandModal" onclick="if(event.target===this)closeExpand()">
  <div class="expand-box">
    <div class="expand-header">
      <div>
        <h3 id="expandTitle"></h3>
        <span class="meta" id="expandMeta"></span>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <button id="expandSelectBtn" class="admin-only" onclick="if(openExpandFid)toggleSelectMode(openExpandFid)"
                style="padding:6px 12px;font-size:13px;border-radius:6px;border:1px solid #444;background:#222;color:#ccc;cursor:pointer">🔲 多選</button>
        <button class="close" onclick="closeExpand()">✕</button>
      </div>
    </div>
    <div class="expand-body" id="expandBody"></div>
  </div>
</div>

<div class="select-bar" id="selectBar">
  <span class="count" id="selectCount">0</span>
  <span style="color:#aaa">張已選</span>
  <button class="btn-go" onclick="moveSelected()">→ 移到別群</button>
  <button class="btn-cancel" onclick="exitSelectMode()">取消</button>
</div>

<div class="modal" id="moveModal">
  <div class="modal-box">
    <h3>把這張照片移到哪個群組？</h3>
    <div id="movePreview" style="text-align:center;margin-bottom:12px"></div>
    <input type="text" id="moveFilter" placeholder="輸入名稱或 ID 過濾..." oninput="filterMoveOptions()">
    <select id="moveTarget" size="10"></select>
    <div style="display:flex;gap:6px;align-items:center;margin-top:10px;padding:10px;background:#222;border-radius:6px">
      <input type="text" id="newGroupName" placeholder="新群組名稱（可選）"
             style="flex:1;padding:7px 10px;background:#1a1a1a;color:#fff;border:1px solid #333;border-radius:4px;font-size:13px"
             onkeydown="if(event.key==='Enter')moveToNew()">
      <button class="btn btn-merge" onclick="moveToNew()" title="建立新群組並把選的照片放進去">✨ 移到新群組</button>
    </div>
    <div class="modal-actions" style="margin-top:12px">
      <button class="btn btn-skip" onclick="closeMove()">取消</button>
      <button class="btn btn-save" onclick="confirmMove()">移到既有群組</button>
    </div>
  </div>
</div>

<script>
let ITEMS = [];
let currentPage = 0;
let totalPages = 1;
let totalCount = 0;
let filter = 'all';
let viewMode = 'grid';  // 'grid' | 'list'，list 只有 named filter 下可用
let mergeSource = '';
let moveSource = '';   // 來源 cluster id
let movePath = '';     // 單張移動的圖片路徑（null 代表 batch）
let allClusters = [];  // 給合併/移動下拉用，按需 fetch
let openExpandFid = null;       // 目前 modal 展開的 cluster id（單一）
let openExpandMeta = null;      // {images:[{path,ts,year,month}], removed:[]}
let selectMode = null;          // 多選模式作用中的 cluster id
let selectedPaths = new Set();  // 已選中的照片路徑
let yearFilter = '';            // '' = 全部
let monthFilter = '';           // '' = 全部
let collapsedYears = new Set(); // 已摺疊（key = String(year) 或 '無日期'）
let materializedYears = new Set(); // 已 render 出 img DOM 的年份
let removedSectionCollapsed = true;   // 「已移除」區段預設折疊
let removedSectionMaterialized = false;
let unloadTimers = new Map();   // year -> setTimeout id（用於折疊後 TTL 釋放 DOM）
const UNLOAD_TTL_MS = 3 * 60 * 1000;  // 折疊超過 3 分鐘 → 移除 img DOM

// 初始化 user / role 顯示
document.getElementById('whoami').textContent = window.CURRENT_USER || '(未登入)';
document.getElementById('role-tag').textContent = window.IS_ADMIN ? '(admin)' : '(viewer · 唯讀)';
if(window.IS_ADMIN){ document.body.classList.add('admin'); }

function logout(){
  fetch('/api/logout',{method:'POST'}).then(()=>location.href='/login');
}

loadStats();
loadPage(0);

// Esc 關 modal
document.addEventListener('keydown', e=>{
  if(e.key !== 'Escape') return;
  if(document.getElementById('expandModal').classList.contains('active')){
    closeExpand();
  } else if(document.getElementById('moveModal').classList.contains('active')){
    closeMove();
  } else if(document.getElementById('mergeModal').classList.contains('active')){
    closeMerge();
  }
});

function loadStats(){
  fetch('/api/stats').then(r=>r.json()).then(d=>{
    document.getElementById('subtitle').textContent =
      `共 ${d.total} 個原始群組，合併後 ${d.displayed} 個可顯示`;
    document.getElementById('stats').innerHTML =
      `<span style="color:#4fc3f7">${d.named}</span> 已命名 · ` +
      `<span style="color:#888">${d.skipped}</span> 已略過 · ` +
      `<span style="color:#ce93d8">${d.merges}</span> 已合併`;
  });
}

function loadPage(page){
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('grid').innerHTML = '';
  return fetch(`/api/page?page=${page}&filter=${filter}`).then(r=>r.json()).then(d=>{
    ITEMS = d.items;
    currentPage = d.page;
    totalPages = d.total_pages;
    totalCount = d.total;
    renderGrid();
    renderPager();
    document.getElementById('spinner').style.display = 'none';
    // 如果 expand modal 還開著，刷新內容
    if(openExpandFid) renderExpandBody();
  });
}

function renderPager(){
  const html = `
    <button onclick="loadPage(0)" ${currentPage===0?'disabled':''}>«</button>
    <button onclick="loadPage(${currentPage-1})" ${currentPage===0?'disabled':''}>‹</button>
    <span class="info">第
      <input type="number" min="1" max="${totalPages}" value="${currentPage+1}"
             onchange="loadPage(this.value-1)"> / ${totalPages} 頁
      · 共 ${totalCount} 筆</span>
    <button onclick="loadPage(${currentPage+1})" ${currentPage>=totalPages-1?'disabled':''}>›</button>
    <button onclick="loadPage(${totalPages-1})" ${currentPage>=totalPages-1?'disabled':''}>»</button>
  `;
  document.getElementById('pagerTop').innerHTML = html;
  document.getElementById('pagerBottom').innerHTML = html;
}

function renderGrid(){
  const grid = document.getElementById('grid');
  if(viewMode === 'list' && filter === 'named'){
    grid.className = 'list';
    grid.innerHTML = ITEMS.map(c => renderListRow(c)).join('');
  } else {
    grid.className = 'grid';
    grid.innerHTML = ITEMS.map(c => renderCard(c)).join('');
  }
}

function renderListRow(c){
  const fid = c.id;
  const nameClick = window.IS_ADMIN ? `onclick="editNameInline('${fid}')"` : '';
  const actions = window.IS_ADMIN ? `
    <div class="lr-actions">
      <button onclick="editNameInline('${fid}')">✏️ 改名</button>
      <button onclick="openMerge('${fid}')">🔗 合併</button>
      <button onclick="undoAction('${fid}')">↩ 取消命名</button>
    </div>` : '<div></div>';
  return `
    <div class="list-row" id="row_${fid}">
      <img class="lr-thumb" src="/thumb/${fid}.jpg?v=${c.thumb_v||0}" onerror="this.style.visibility='hidden'">
      <div class="lr-id">${fid}</div>
      <div class="lr-name" id="name_${fid}" ${nameClick}>${c.name}</div>
      <div class="lr-count">${c.count} 張</div>
      ${actions}
    </div>
  `;
}

function editNameInline(fid){
  const c = ITEMS.find(x=>x.id===fid);
  if(!c) return;
  const cell = document.getElementById('name_'+fid);
  if(!cell) return;
  const current = c.name || '';
  cell.innerHTML = `<input class="lr-name-input" id="inp_${fid}" value="${current.replace(/"/g,'&quot;')}"
    onblur="commitInlineName('${fid}')"
    onkeydown="if(event.key==='Enter')commitInlineName('${fid}');else if(event.key==='Escape')cancelInlineName('${fid}')">`;
  const inp = document.getElementById('inp_'+fid);
  inp.focus();
  inp.select();
}

function commitInlineName(fid){
  const inp = document.getElementById('inp_'+fid);
  if(!inp) return;
  const v = inp.value.trim();
  const c = ITEMS.find(x=>x.id===fid);
  if(!c) return;
  if(!v){ cancelInlineName(fid); return; }
  if(v === c.name){ cancelInlineName(fid); return; }
  post('/api/name',{face_id:fid,name:v}).then(()=>{
    c.name = v;
    document.getElementById('name_'+fid).textContent = v;
    loadStats();
  });
}

function cancelInlineName(fid){
  const c = ITEMS.find(x=>x.id===fid);
  if(!c) return;
  const cell = document.getElementById('name_'+fid);
  if(cell) cell.textContent = c.name || '';
}

function toggleView(){
  viewMode = viewMode === 'grid' ? 'list' : 'grid';
  document.getElementById('viewToggle').textContent =
    viewMode === 'list' ? '🖼 圖片模式' : '📋 清單模式';
  renderGrid();
}

function renderCard(c){
  const fid = c.id;
  // viewer：簡化成大頭像 + 名字的 tile，整塊可點 → openExpand
  if(!window.IS_ADMIN){
    const label = c.name || fid;
    return `
      <div class="tile" onclick="openExpand('${fid}')" role="button" tabindex="0"
           onkeydown="if(event.key==='Enter'||event.key===' ')openExpand('${fid}')">
        <img class="avatar" src="/thumb/${fid}.jpg?v=${c.thumb_v||0}" alt="${label.replace(/"/g,'&quot;')}" loading="lazy" onerror="this.style.visibility='hidden'">
        <div class="tname">${label}</div>
      </div>`;
  }
  const isNamed = !!c.name;
  const isSkipped = !!c.skipped;
  const cls = 'card' + (isNamed?' named':'') + (isSkipped?' skipped':'');

  const previewCount = Math.min(Math.max(6, c.images.length), 8);
  const previewImgs = c.images.slice(0,previewCount).map(img=>
    `<img src="/img_thumb/${img}?w=200" loading="lazy" decoding="async"
          onclick="window.open('/image/${img}')">`
  ).join('');

  let actions = '';
  if(window.IS_ADMIN){
    if(isSkipped){
      actions = `<span style="color:#666">已略過</span>
        <button class="btn btn-edit" onclick="toggleSkip('${fid}')">↩ 恢復</button>`;
    } else if(isNamed){
      actions = `
        <button class="btn btn-edit" onclick="editName('${fid}')">✏️ 改名</button>
        <button class="btn btn-undo" onclick="undoAction('${fid}')">↩ 取消命名</button>
        <button class="btn btn-merge" onclick="openMerge('${fid}')">🔗 合併</button>
      `;
    } else {
      actions = `
        <input type="text" id="inp_${fid}" placeholder="輸入名稱..." onkeydown="if(event.key==='Enter')saveName('${fid}')">
        <button class="btn btn-save" onclick="saveName('${fid}')">💾</button>
        <button class="btn btn-skip" onclick="toggleSkip('${fid}')">⏭️ 略過</button>
        <button class="btn btn-merge" onclick="openMerge('${fid}')">🔗 合併</button>
      `;
    }
  }

  const badges = [];
  if((c.merged_from||[]).length > 0)
    badges.push(`<div class="merged-badge">已合併 ${c.merged_from.length} 群</div>`);
  if(c.moved_in_count > 0)
    badges.push(`<div class="merged-badge" style="background:#2a4a3a;color:#a5d6a7">← 移入 ${c.moved_in_count} 張</div>`);
  if(c.moved_away_count > 0)
    badges.push(`<div class="merged-badge" style="background:#4a3a2a;color:#ffcc80">→ 移出 ${c.moved_away_count} 張</div>`);
  const mergedBadge = badges.join('');

  return `
    <div class="${cls}" id="card_${fid}">
      <div class="card-top">
        <div class="face-meta">
          <h3>${c.name || fid}</h3>
          ${c.name ? `<div style="color:#666;font-size:11px;font-family:monospace;margin-top:2px">${fid}</div>` : ''}
          <div class="count">${c.count} 張${c.count!==c.original_count?' (原 '+c.original_count+')':''}</div>
          ${mergedBadge}
        </div>
        <img class="face-thumb" src="/thumb/${fid}.jpg?v=${c.thumb_v||0}" onerror="this.style.display='none'">
      </div>
      <div class="photo-grid">${previewImgs}</div>
      <div class="expand-bar" onclick="openExpand('${fid}')">
        <span>📂 展開查看全部 ${c.count} 張</span>
        <span>↗</span>
      </div>
      <div class="actions">${actions}</div>
    </div>
  `;
}

function openExpand(fid){
  openExpandFid = fid;
  openExpandMeta = null;
  yearFilter = '';
  monthFilter = '';
  collapsedYears.clear();
  materializedYears.clear();
  removedSectionCollapsed = true;
  removedSectionMaterialized = false;
  for(const t of unloadTimers.values()) clearTimeout(t);
  unloadTimers.clear();
  if(selectMode && selectMode !== fid){
    selectMode = null;
    selectedPaths.clear();
    renderSelectBar();
  }
  document.getElementById('expandBody').innerHTML = '<div style="padding:30px;color:#888;text-align:center">載入中…</div>';
  document.getElementById('expandModal').classList.add('active');
  fetch(`/api/cluster_meta?fid=${encodeURIComponent(fid)}`).then(r=>r.json()).then(meta=>{
    if(openExpandFid !== fid) return;
    openExpandMeta = meta;
    // 預設：只展開最新年度，其餘折疊。「無日期」也預設折疊。
    const yearSet = new Set();
    let hasUndated = false;
    for(const x of meta.images){
      if(x.year) yearSet.add(String(x.year));
      else hasUndated = true;
    }
    const ys = [...yearSet].sort((a,b)=>Number(b)-Number(a));
    const latest = ys[0];
    collapsedYears = new Set(ys.filter(y => y !== latest));
    if(hasUndated) collapsedYears.add('無日期');
    materializedYears = new Set(latest ? [latest] : []);
    renderExpandBody();
  });
}

function closeExpand(){
  document.getElementById('expandModal').classList.remove('active');
  openExpandFid = null;
  openExpandMeta = null;
  collapsedYears.clear();
  materializedYears.clear();
  removedSectionCollapsed = true;
  removedSectionMaterialized = false;
  for(const t of unloadTimers.values()) clearTimeout(t);
  unloadTimers.clear();
  if(selectMode){
    selectMode = null;
    selectedPaths.clear();
    renderSelectBar();
  }
}

function renderExpandBody(){
  if(!openExpandFid || !openExpandMeta) return;
  const c = ITEMS.find(x=>x.id===openExpandFid);
  if(!c){ closeExpand(); return; }
  const fid = c.id;
  const inSelect = selectMode === fid;

  // viewer 只顯示名字（不露 face_id），admin 則同時顯示 ID 方便除錯
  document.getElementById('expandTitle').textContent =
    window.IS_ADMIN ? (c.name ? `${c.name}  (${fid})` : fid)
                    : (c.name || '');
  // viewer 不顯示「(原 N)」避免暴露被擋掉的照片數
  document.getElementById('expandMeta').textContent =
    window.IS_ADMIN && c.count !== c.original_count
      ? `${c.count} 張 (原 ${c.original_count})`
      : `${c.count} 張`;

  // 套 filter
  const allItems = openExpandMeta.images || [];
  const filtered = allItems.filter(x => {
    if(yearFilter !== '' && String(x.year||'') !== String(yearFilter)) return false;
    if(monthFilter !== '' && String(x.month||'') !== String(monthFilter)) return false;
    return true;
  });

  // 取得可選年份（從未過濾的清單）
  const yearSet = new Set();
  for(const x of allItems) if(x.year) yearSet.add(x.year);
  const years = [...yearSet].sort((a,b)=>b-a);
  const yearOptions = years.map(y => `<option value="${y}" ${String(y)===String(yearFilter)?'selected':''}>${y}</option>`).join('');
  const monthOptions = Array.from({length:12},(_,i)=>i+1).map(m =>
    `<option value="${m}" ${String(m)===String(monthFilter)?'selected':''}>${m} 月</option>`).join('');

  // header 多選按鈕的狀態同步
  const headerBtn = document.getElementById('expandSelectBtn');
  if(headerBtn){
    headerBtn.textContent = inSelect ? '✓ 多選中' : '🔲 多選';
    headerBtn.style.background = inSelect ? '#4fc3f7' : '#222';
    headerBtn.style.color = inSelect ? '#000' : '#ccc';
    headerBtn.style.borderColor = inSelect ? '#4fc3f7' : '#444';
  }
  const tools = `
    ${inSelect ? '<div class="expand-tools"><span class="hint">點縮圖切換選取；底部 bar 操作</span></div>' : ''}
    <div class="year-filter">
      <span style="color:#888">篩選：</span>
      <select onchange="yearFilter=this.value; renderExpandBody();">
        <option value="">全部年份</option>
        ${yearOptions}
        ${allItems.some(x=>!x.year) ? `<option value="null" ${yearFilter==='null'?'selected':''}>無日期</option>` : ''}
      </select>
      <select onchange="monthFilter=this.value; renderExpandBody();">
        <option value="">全部月份</option>
        ${monthOptions}
      </select>
      ${(yearFilter||monthFilter) ? '<button class="clear" onclick="yearFilter=\\'\\';monthFilter=\\'\\';renderExpandBody();">清除</button>' : ''}
      <span style="color:#888;margin-left:auto">顯示 ${filtered.length} / ${allItems.length}</span>
    </div>`;

  // 按年分組（同年內已按 ts desc），key 統一為 string
  const byYear = new Map();
  for(const x of filtered){
    const key = String(x.year || '無日期');
    if(!byYear.has(key)) byYear.set(key, []);
    byYear.get(key).push(x);
  }
  const yearKeys = [...byYear.keys()].sort((a,b)=>{
    if(a==='無日期') return 1; if(b==='無日期') return -1;
    return Number(b) - Number(a);
  });

  function renderThumb(x){
    const img = x.path;
    const safeImg = img.replace(/'/g,"\\'");
    const isSel = inSelect && selectedPaths.has(img);
    const wrapCls = 'thumb-wrap' + (inSelect?' selectable':'') + (isSel?' selected':'');
    const imgClick = inSelect
      ? `onclick="toggleThumbSelect('${fid}','${safeImg}')"`
      : `onclick="window.open('/image/${img}')" style="cursor:pointer"`;
    const actions = (inSelect || !window.IS_ADMIN) ? '' : `
      <div class="thumb-actions">
        <button class="thumb-btn" onclick="event.stopPropagation();setAsThumb('${fid}','${safeImg}')" title="設為群組特寫照">⭐</button>
        <button class="move-btn" onclick="event.stopPropagation();openMove('${fid}','${safeImg}')" title="移到別群">→</button>
        <button class="remove-btn" onclick="event.stopPropagation();removeImg('${fid}','${safeImg}')" title="移除">✕</button>
      </div>`;
    return `<div class="${wrapCls}" data-thumb="${fid}|${img}">
      <img src="/img_thumb/${img}?w=300" loading="lazy" decoding="async" ${imgClick}>
      ${actions}
    </div>`;
  }

  const sections = yearKeys.map(y => {
    const items = byYear.get(y);
    const collapsed = collapsedYears.has(y);
    // Materialize only if expanded OR previously expanded and TTL still alive
    let materialized = materializedYears.has(y);
    // 防呆：展開但還沒 materialize → 自動補
    if(!collapsed && !materialized){ materializedYears.add(y); materialized = true; }
    const bodyHtml = materialized
      ? `<div class="expand-photos">${items.map(renderThumb).join('')}</div>`
      : `<div style="color:#666;font-size:12px;padding:6px 4px">點選展開以載入</div>`;
    return `<div class="year-section ${collapsed?'collapsed':''}" data-year="${y}">
      <div class="year-header" onclick="toggleYear('${y}')">
        <span>${y} <span style="color:#888;font-weight:400;font-size:12px">(${items.length} 張)</span></span>
        <span class="toggle">▼</span>
      </div>
      <div class="year-body">${bodyHtml}</div>
    </div>`;
  }).join('');

  const removed = openExpandMeta.removed || [];
  let removedSection = '';
  if(removed.length > 0){
    const collapsed = removedSectionCollapsed;
    let mat = removedSectionMaterialized;
    if(!collapsed && !mat){ removedSectionMaterialized = true; mat = true; }
    const removedImgs = removed.map(x =>
      `<div class="thumb-wrap">
        <img src="/img_thumb/${x.path}?w=200" loading="lazy" decoding="async">
        <button class="restore-btn" onclick="restoreImg('${fid}','${x.path.replace(/'/g,"\\'")}')" title="恢復">↩</button>
      </div>`
    ).join('');
    const body = mat
      ? `<div class="removed-grid">${removedImgs}</div>`
      : `<div style="color:#666;font-size:12px;padding:6px 4px">點選展開以載入</div>`;
    removedSection = `<div class="year-section ${collapsed?'collapsed':''}" data-removed="1" style="margin-top:14px">
      <div class="year-header" style="background:#3a2424;color:#ef9a9a" onclick="toggleRemovedSection()">
        <span>已移除 <span style="color:#888;font-weight:400;font-size:12px">(${removed.length} 張)</span></span>
        <span class="toggle">▼</span>
      </div>
      <div class="year-body">${body}</div>
    </div>`;
  }

  document.getElementById('expandBody').innerHTML = `${tools}${sections}${removedSection}`;
}

function toggleRemovedSection(){
  if(removedSectionCollapsed){
    removedSectionCollapsed = false;
    removedSectionMaterialized = true;
  } else {
    removedSectionCollapsed = true;
    // TTL：3 min 後若仍 collapsed → 釋放 DOM
    setTimeout(()=>{
      if(!openExpandFid) return;
      if(!removedSectionCollapsed) return;
      removedSectionMaterialized = false;
      renderExpandBody();
    }, UNLOAD_TTL_MS);
  }
  renderExpandBody();
}

function toggleYear(y){
  y = String(y);
  if(collapsedYears.has(y)){
    // 展開：取消預定的 unload + materialize（若尚未）
    collapsedYears.delete(y);
    if(unloadTimers.has(y)){ clearTimeout(unloadTimers.get(y)); unloadTimers.delete(y); }
    materializedYears.add(y);
  } else {
    // 折疊：留 DOM、但 TTL 過後釋放
    collapsedYears.add(y);
    if(unloadTimers.has(y)) clearTimeout(unloadTimers.get(y));
    const t = setTimeout(()=>{
      if(!openExpandFid) return;
      if(!collapsedYears.has(y)) return;  // 已被重新展開
      materializedYears.delete(y);
      unloadTimers.delete(y);
      renderExpandBody();
    }, UNLOAD_TTL_MS);
    unloadTimers.set(y, t);
  }
  renderExpandBody();
}

function refreshExpandMeta(){
  if(!openExpandFid) return Promise.resolve();
  return fetch(`/api/cluster_meta?fid=${encodeURIComponent(openExpandFid)}`)
    .then(r=>r.json()).then(meta=>{
      if(openExpandFid){ openExpandMeta = meta; renderExpandBody(); }
    });
}

function toggleSelectMode(fid){
  if(selectMode === fid){
    selectMode = null;
    selectedPaths.clear();
  } else {
    selectMode = fid;
    selectedPaths.clear();
  }
  renderExpandBody();
  renderSelectBar();
}

function exitSelectMode(){
  selectMode = null;
  selectedPaths.clear();
  if(openExpandFid) renderExpandBody();
  renderSelectBar();
}

function toggleThumbSelect(fid, img){
  if(selectMode !== fid) return;
  if(selectedPaths.has(img)) selectedPaths.delete(img);
  else selectedPaths.add(img);
  // 只更新該縮圖 class，不重渲整個 grid
  const el = document.querySelector(`[data-thumb="${fid}|${img}"]`);
  if(el) el.classList.toggle('selected', selectedPaths.has(img));
  renderSelectBar();
}

function renderSelectBar(){
  const bar = document.getElementById('selectBar');
  if(selectedPaths.size === 0){
    bar.classList.remove('active');
    return;
  }
  bar.classList.add('active');
  document.getElementById('selectCount').textContent = selectedPaths.size;
}

function moveSelected(){
  if(!selectMode || selectedPaths.size === 0) return;
  moveSource = selectMode;
  movePath = null;  // null = batch
  document.getElementById('moveFilter').value = '';
  const ngn = document.getElementById('newGroupName'); if(ngn) ngn.value = '';
  document.getElementById('movePreview').innerHTML =
    `<div style="color:#4fc3f7;font-size:15px">批次移動 ${selectedPaths.size} 張</div>
     <div style="color:#888;font-size:12px;margin-top:4px">來自 ${moveSource}</div>`;
  fetch('/api/clusters').then(r=>r.json()).then(list=>{
    allClusters = list.filter(c=>c.id !== moveSource);
    renderMoveOptions(allClusters);
    document.getElementById('moveModal').classList.add('active');
    setTimeout(()=>document.getElementById('moveFilter').focus(),50);
  });
}

function saveName(fid){
  const inp = document.getElementById('inp_'+fid);
  if(!inp||!inp.value.trim()) return;
  post('/api/name',{face_id:fid,name:inp.value.trim()}).then(()=>{
    loadStats(); loadPage(currentPage);
  });
}

function editName(fid){
  const c = ITEMS.find(x=>x.id===fid);
  if(!c) return;
  const current = c.name;
  // 簡化：直接清空名稱、重渲染、focus 輸入框
  c.name = '';
  renderGrid();
  setTimeout(()=>{
    const inp=document.getElementById('inp_'+fid);
    if(inp){inp.value=current; inp.focus(); inp.select();}
  },30);
}

function undoAction(fid){
  post('/api/name',{face_id:fid,name:''}).then(()=>{
    loadStats(); loadPage(currentPage);
  });
}

function toggleSkip(fid){
  post('/api/skip',{face_id:fid}).then(()=>{
    loadStats(); loadPage(currentPage);
  });
}

function openMerge(fid){
  mergeSource=fid;
  document.getElementById('mergeFilter').value='';
  // fetch 完整列表（按需）
  fetch('/api/clusters').then(r=>r.json()).then(list=>{
    allClusters = list.filter(c=>c.id!==fid);
    renderMergeOptions(allClusters);
    document.getElementById('mergeModal').classList.add('active');
    setTimeout(()=>document.getElementById('mergeFilter').focus(),50);
  });
}

function renderMergeOptions(list){
  const sel = document.getElementById('mergeTarget');
  sel.innerHTML = list.map(c=>
    `<option value="${c.id}">${c.id}${c.name?' · '+c.name:''} (${c.count} 張)</option>`
  ).join('');
}

function filterMergeOptions(){
  const q = document.getElementById('mergeFilter').value.toLowerCase();
  if(!q){ renderMergeOptions(allClusters); return; }
  const filtered = allClusters.filter(c=>
    c.id.toLowerCase().includes(q) || (c.name||'').toLowerCase().includes(q)
  );
  renderMergeOptions(filtered);
}

function confirmMerge(){
  const tgt = document.getElementById('mergeTarget').value;
  if(!tgt) return;
  post('/api/merge',{source_id:mergeSource,target_id:tgt}).then(()=>{
    closeMerge();
    loadStats();
    loadPage(currentPage);
  });
}

function closeMerge(){document.getElementById('mergeModal').classList.remove('active');}

function openMove(fid, imgPath){
  moveSource = fid;
  movePath = imgPath;
  document.getElementById('moveFilter').value = '';
  const ngn = document.getElementById('newGroupName'); if(ngn) ngn.value = '';
  document.getElementById('movePreview').innerHTML =
    `<img src="/img_thumb/${imgPath}?w=300" style="max-height:140px;border-radius:6px;border:2px solid #333">
     <div style="color:#888;font-size:12px;margin-top:6px">來自 ${fid}</div>`;
  fetch('/api/clusters').then(r=>r.json()).then(list=>{
    allClusters = list.filter(c=>c.id !== fid);
    renderMoveOptions(allClusters);
    document.getElementById('moveModal').classList.add('active');
    setTimeout(()=>document.getElementById('moveFilter').focus(),50);
  });
}

function renderMoveOptions(list){
  const sel = document.getElementById('moveTarget');
  sel.innerHTML = list.map(c=>
    `<option value="${c.id}">${c.id}${c.name?' · '+c.name:''} (${c.count} 張)</option>`
  ).join('');
}

function filterMoveOptions(){
  const q = document.getElementById('moveFilter').value.toLowerCase();
  const filtered = q ? allClusters.filter(c=>
    c.id.toLowerCase().includes(q) || (c.name||'').toLowerCase().includes(q)
  ) : allClusters;
  renderMoveOptions(filtered);
}

function confirmMove(){
  const tgt = document.getElementById('moveTarget').value;
  if(!tgt) return;
  doMove(tgt);
}

function moveToNew(){
  const nm = (document.getElementById('newGroupName').value || '').trim();
  doMove('__new__', nm);
}

function doMove(tgt, newName){
  const isBatch = movePath === null;
  const batchBody = {from_id:moveSource, to_id:tgt, paths:[...selectedPaths]};
  const singleBody = {from_id:moveSource, to_id:tgt, image_path:movePath};
  if(newName){ batchBody.new_name = newName; singleBody.new_name = newName; }
  const req = isBatch ? post('/api/move-batch', batchBody) : post('/api/move', singleBody);
  req.then(r=>r.json()).then(d=>{
    closeMove();
    if(isBatch){
      selectMode = null;
      selectedPaths.clear();
      renderSelectBar();
    }
    loadStats();
    loadPage(currentPage).then(()=>{ if(openExpandFid) refreshExpandMeta(); });
    if(tgt === '__new__' && d && d.to_id){
      const nm = (document.getElementById('newGroupName')||{}).value || '';
      setTimeout(()=>{
        alert(nm.trim()
          ? `已建立新群組「${nm.trim()}」 (${d.to_id})`
          : `已建立新群組 ${d.to_id}，請後續手動命名。`);
      }, 100);
    }
  });
}

function closeMove(){document.getElementById('moveModal').classList.remove('active');}

function setAsThumb(fid, img){
  fetch('/api/set_thumb', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({face_id: fid, image_path: img})
  }).then(r=>r.json()).then(d=>{
    if(!d.ok){ alert('設定失敗：' + (d.error || '未知錯誤')); return; }
    // cache-bust 換新的縮圖
    const ts = Date.now();
    document.querySelectorAll(`img.face-thumb[src*="/thumb/${fid}.jpg"]`).forEach(el=>{
      el.src = `/thumb/${fid}.jpg?t=${ts}`;
    });
    // expand modal 內也可能有 face thumb（暫時沒，但 face naming 卡片有）
    const lrThumb = document.querySelector(`.lr-thumb[src*="/thumb/${fid}.jpg"]`);
    if(lrThumb) lrThumb.src = `/thumb/${fid}.jpg?t=${ts}`;
  });
}

function removeImg(fid,img){
  post('/api/remove',{face_id:fid,image_path:img}).then(()=>{
    const c=ITEMS.find(x=>x.id===fid);
    if(c){
      c.images=c.images.filter(i=>i!==img);
      c.removed=[...(c.removed||[]),img];
      c.count=c.images.length;
      renderGrid();
    }
    if(openExpandFid===fid) refreshExpandMeta();
  });
}

function restoreImg(fid,img){
  post('/api/restore',{face_id:fid,image_path:img}).then(()=>{
    const c=ITEMS.find(x=>x.id===fid);
    if(c){
      c.images.push(img);
      c.removed=(c.removed||[]).filter(i=>i!==img);
      c.count=c.images.length;
      renderGrid();
    }
    if(openExpandFid===fid) refreshExpandMeta();
  });
}

function setFilter(f){
  filter=f;
  document.querySelectorAll('.toolbar button[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter===f));
  // 清單模式只在 named filter 下可用
  const tgl = document.getElementById('viewToggle');
  if(f === 'named'){
    tgl.style.display = '';
  } else {
    tgl.style.display = 'none';
    if(viewMode === 'list'){
      viewMode = 'grid';
      tgl.textContent = '📋 清單模式';
    }
  }
  loadPage(0);
}

function post(url,data){
  return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
}
</script>
</body>
</html>"""


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
