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
from collections import defaultdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

PROJECT_DIR = Path(__file__).resolve().parent.parent
FACES_DIR = PROJECT_DIR / "data" / "faces"
FACES_FILE = FACES_DIR / "face_clusters.json"
NAMES_FILE = FACES_DIR / "face_names.json"
REMOVED_FILE = FACES_DIR / "face_removed.json"
SKIPPED_FILE = FACES_DIR / "face_skipped.json"
MERGES_FILE = FACES_DIR / "face_merges.json"
MOVES_FILE = FACES_DIR / "face_moves.json"
THUMBS_DIR = FACES_DIR / "face_thumbs"
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


def _resolve_target(merges, fid):
    """跟著 merge 鏈走到最終 target（防止 A→B→C 失效）。"""
    seen = set()
    while fid in merges and fid not in seen:
        seen.add(fid)
        fid = merges[fid]
    return fid


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self.html(self.generate_html())

        elif path.startswith("/image/"):
            img = self.fix_path(path[7:])
            self.serve_file(img, "image/jpeg")

        elif path.startswith("/thumb/"):
            thumb = THUMBS_DIR / unquote(path[7:])
            self.serve_file(thumb, "image/jpeg")

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

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self.read_body()

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
            # 把單張照片從 from_id 群移到 to_id 群
            moves = load_moves()
            from_id = body.get("from_id")
            to_id = body.get("to_id")
            img_path = body.get("image_path")
            if from_id and to_id and img_path and from_id != to_id:
                # 移除同一張的舊紀錄（避免重複）
                moves = [m for m in moves if m.get("path") != img_path]
                moves.append({"path": img_path, "from": from_id, "to": to_id})
                save_moves(moves)
            self.json_response({"ok": True})

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
            if from_id and to_id and paths and from_id != to_id:
                ps = set(paths)
                moves = [m for m in moves if m.get("path") not in ps]
                for p in paths:
                    moves.append({"path": p, "from": from_id, "to": to_id})
                save_moves(moves)
            self.json_response({"ok": True, "moved": len(paths)})

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

        result = []
        for fid, info in clusters.items():
            # source cluster 被折疊到 target，不獨立顯示
            if fid in merges:
                continue

            own_imgs = list(info.get("images", []))
            merged_srcs = merge_back.get(fid, [])
            for s in merged_srcs:
                own_imgs.extend(clusters.get(s, {}).get("images", []))

            # 去重保序
            seen = set()
            all_imgs = [x for x in own_imgs if not (x in seen or seen.add(x))]

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
            })

        # filter
        if flt == "named":
            result = [r for r in result if r["name"]]
        elif flt == "unnamed":
            result = [r for r in result if not r["name"] and not r["skipped"]]
        elif flt == "skipped":
            result = [r for r in result if r["skipped"]]
        # "all" → 全留

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
        return {
            "total": total,
            "displayed": total - len(merges),  # 不算被折疊的 source
            "named": len(names),
            "skipped": len(skipped),
            "merges": len(merges),
        }

    def generate_html(self):
        return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>人臉命名工具</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#111;color:#e0e0e0;padding:20px}
h1{text-align:center;margin-bottom:6px}
.subtitle{text-align:center;color:#888;margin-bottom:14px;font-size:14px}
.toolbar{display:flex;justify-content:center;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.toolbar button{padding:7px 14px;background:#222;color:#888;border:1px solid #333;border-radius:6px;cursor:pointer;font-size:13px}
.toolbar button:hover{background:#2a2a2a}
.toolbar button.active{background:#4fc3f7;color:#000;border-color:#4fc3f7}
.stats{text-align:center;margin-bottom:16px;color:#888;font-size:13px}
.pager{display:flex;justify-content:center;align-items:center;gap:10px;margin:18px 0}
.pager button{padding:8px 16px;background:#222;color:#ccc;border:1px solid #333;border-radius:6px;cursor:pointer}
.pager button:hover:not(:disabled){background:#2a2a2a}
.pager button:disabled{opacity:.3;cursor:not-allowed}
.pager input{width:56px;padding:6px;text-align:center;background:#222;color:#fff;border:1px solid #333;border-radius:6px}
.pager .info{color:#888;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:16px}
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
.expand-content{display:none;padding:10px 14px;background:#151515}
.expand-content.open{display:block}
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
.select-bar{display:none;position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1a1a1a;border:2px solid #4fc3f7;border-radius:30px;padding:10px 22px;box-shadow:0 4px 24px rgba(0,0,0,.6);gap:12px;align-items:center;z-index:90}
.select-bar.active{display:flex}
.select-bar .count{color:#4fc3f7;font-weight:600}
.select-bar button{padding:7px 14px;border-radius:18px;border:none;cursor:pointer;font-size:13px;font-weight:600}
.select-bar .btn-go{background:#4fc3f7;color:#000}
.select-bar .btn-cancel{background:#333;color:#ccc}
.expand-photos .thumb-actions button{width:22px;height:22px;border-radius:50%;border:none;cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center;color:#fff;padding:0}
.expand-photos .move-btn{background:rgba(76,175,80,.85)}
.expand-photos .remove-btn{background:rgba(239,83,80,.85)}
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

<h1>👥 人臉命名工具</h1>
<div class="subtitle" id="subtitle"></div>

<div class="toolbar">
  <button class="active" data-filter="all" onclick="setFilter('all')">全部</button>
  <button data-filter="unnamed" onclick="setFilter('unnamed')">未命名</button>
  <button data-filter="named" onclick="setFilter('named')">已命名</button>
  <button data-filter="skipped" onclick="setFilter('skipped')">已略過</button>
  <button id="viewToggle" onclick="toggleView()" style="display:none;margin-left:14px">📋 清單模式</button>
</div>

<div class="stats" id="stats"></div>

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
    <div class="modal-actions">
      <button class="btn btn-skip" onclick="closeMove()">取消</button>
      <button class="btn btn-save" onclick="confirmMove()">移動</button>
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
let expandedCards = new Set();  // 記住展開狀態，跨 re-render
let selectMode = null;          // 多選模式作用中的 cluster id
let selectedPaths = new Set();  // 已選中的照片路徑

loadStats();
loadPage(0);

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
  fetch(`/api/page?page=${page}&filter=${filter}`).then(r=>r.json()).then(d=>{
    ITEMS = d.items;
    currentPage = d.page;
    totalPages = d.total_pages;
    totalCount = d.total;
    renderGrid();
    renderPager();
    document.getElementById('spinner').style.display = 'none';
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
  return `
    <div class="list-row" id="row_${fid}">
      <img class="lr-thumb" src="/thumb/${fid}.jpg" onerror="this.style.visibility='hidden'">
      <div class="lr-id">${fid}</div>
      <div class="lr-name" id="name_${fid}" onclick="editNameInline('${fid}')">${c.name}</div>
      <div class="lr-count">${c.count} 張</div>
      <div class="lr-actions">
        <button onclick="editNameInline('${fid}')">✏️ 改名</button>
        <button onclick="openMerge('${fid}')">🔗 合併</button>
        <button onclick="undoAction('${fid}')">↩ 取消命名</button>
      </div>
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
  const isNamed = !!c.name;
  const isSkipped = !!c.skipped;
  const cls = 'card' + (isNamed?' named':'') + (isSkipped?' skipped':'');

  const previewCount = Math.min(Math.max(6, c.images.length), 8);
  const previewImgs = c.images.slice(0,previewCount).map(img=>
    `<img src="/image/${img}" onclick="window.open(this.src)">`
  ).join('');

  const inSelect = selectMode === fid;
  const allImgs = c.images.map(img=>{
    const safeImg = img.replace(/'/g,"\\'");
    const isSel = inSelect && selectedPaths.has(img);
    const wrapCls = 'thumb-wrap' + (inSelect?' selectable':'') + (isSel?' selected':'');
    const imgClick = inSelect
      ? `onclick="toggleThumbSelect('${fid}','${safeImg}')"`
      : `onclick="window.open('/image/${img}')"`;
    const actions = inSelect ? '' : `
      <div class="thumb-actions">
        <button class="move-btn" onclick="event.stopPropagation();openMove('${fid}','${safeImg}')" title="移到別群">→</button>
        <button class="remove-btn" onclick="event.stopPropagation();removeImg('${fid}','${safeImg}')" title="移除">✕</button>
      </div>`;
    return `<div class="${wrapCls}" data-thumb="${fid}|${img}">
      <img src="/image/${img}" ${imgClick}>
      ${actions}
    </div>`;
  }).join('');

  const expandTools = `
    <div class="expand-tools">
      <button class="${inSelect?'active':''}" onclick="toggleSelectMode('${fid}')">
        ${inSelect ? '✓ 多選中' : '🔲 多選'}
      </button>
      ${inSelect ? '<span class="hint">點縮圖切換選取；底部 bar 移動</span>' : ''}
    </div>`;

  const removedImgs = (c.removed||[]).map(img=>
    `<div class="thumb-wrap">
      <img src="/image/${img}">
      <button class="restore-btn" onclick="restoreImg('${fid}','${img.replace(/'/g,"\\'")}')" title="恢復">↩</button>
    </div>`
  ).join('');

  let actions = '';
  if(isSkipped){
    actions = `<span style="color:#666">已略過</span>
      <button class="btn btn-edit" onclick="toggleSkip('${fid}')">↩ 恢復</button>`;
  } else if(isNamed){
    actions = `
      <div class="named-label">✓ ${c.name}</div>
      <button class="btn btn-edit" onclick="editName('${fid}')">✏️ 編輯</button>
      <button class="btn btn-undo" onclick="undoAction('${fid}')">↩ 取消</button>
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

  const hasRemoved = (c.removed||[]).length > 0;
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
          <h3>${fid}</h3>
          <div class="count">${c.count} 張${c.count!==c.original_count?' (原 '+c.original_count+')':''}</div>
          ${mergedBadge}
        </div>
        <img class="face-thumb" src="/thumb/${fid}.jpg" onerror="this.style.display='none'">
      </div>
      <div class="photo-grid">${previewImgs}</div>
      <div class="expand-bar" onclick="toggleExpand('${fid}')">
        <span>📂 展開查看全部 ${c.count} 張</span>
        <span id="arrow_${fid}">▼</span>
      </div>
      <div class="expand-content ${expandedCards.has(fid)?'open':''}" id="expand_${fid}">
        ${expandTools}
        <div class="expand-photos">${allImgs}</div>
        ${hasRemoved?`<div style="padding:8px 0 4px;color:#666;font-size:12px">已移除 (${c.removed.length}):</div>
          <div class="removed-grid">${removedImgs}</div>`:''}
      </div>
      <div class="actions">${actions}</div>
    </div>
  `;
}

function toggleExpand(fid){
  const el = document.getElementById('expand_'+fid);
  const arrow = document.getElementById('arrow_'+fid);
  const open = el.classList.toggle('open');
  if(open) expandedCards.add(fid); else expandedCards.delete(fid);
  arrow.textContent = open?'▲':'▼';
}

function toggleSelectMode(fid){
  if(selectMode === fid){
    exitSelectMode();
  } else {
    selectMode = fid;
    selectedPaths.clear();
    expandedCards.add(fid);  // 保證該 card 是展開的
    renderGrid();
    renderSelectBar();
  }
}

function exitSelectMode(){
  selectMode = null;
  selectedPaths.clear();
  renderGrid();
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
  document.getElementById('movePreview').innerHTML =
    `<img src="/image/${imgPath}" style="max-height:140px;border-radius:6px;border:2px solid #333">
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
  const isBatch = movePath === null;
  const req = isBatch
    ? post('/api/move-batch',{from_id:moveSource, to_id:tgt, paths:[...selectedPaths]})
    : post('/api/move',{from_id:moveSource, to_id:tgt, image_path:movePath});
  req.then(()=>{
    closeMove();
    if(isBatch){
      selectMode = null;
      selectedPaths.clear();
      renderSelectBar();
    }
    loadStats();
    loadPage(currentPage);
  });
}

function closeMove(){document.getElementById('moveModal').classList.remove('active');}

function removeImg(fid,img){
  post('/api/remove',{face_id:fid,image_path:img}).then(()=>{
    // 局部更新（避免關掉展開）
    const c=ITEMS.find(x=>x.id===fid);
    if(c){
      c.images=c.images.filter(i=>i!==img);
      c.removed=[...(c.removed||[]),img];
      c.count=c.images.length;
      renderGrid();
    }
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
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"人臉命名伺服器 v4：http://127.0.0.1:{port}")
    print(f"分頁：每頁 {PAGE_SIZE} 個群組 · 略過 → 排到最後 · 合併 → 折疊到 target")
    print("按 Ctrl+C 結束")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        names = load_names()
        print(f"\n結束。已命名 {len(names)} 個群組")


if __name__ == "__main__":
    main()
