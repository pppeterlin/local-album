#!/usr/bin/env python3
"""
face_naming_server.py — 人臉命名伺服器 v3

功能：
  • 分頁載入（每頁 20 個群組，減少記憶體）
  • 四宮格卡片佈局
  • 展開群組、移除/恢復照片
  • 命名、略過、合併、重新編輯
"""

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

PROJECT_DIR = Path(__file__).resolve().parent.parent
FACES_DIR = PROJECT_DIR / "data" / "faces"
FACES_FILE = FACES_DIR / "face_clusters.json"
NAMES_FILE = FACES_DIR / "face_names.json"
REMOVED_FILE = FACES_DIR / "face_removed.json"
THUMBS_DIR = FACES_DIR / "face_thumbs"
PAGE_SIZE = 20


def load_json(path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default or {}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_faces(): return load_json(FACES_FILE)
def load_names(): return load_json(NAMES_FILE)
def load_removed(): return load_json(REMOVED_FILE, {})
def save_names(n): save_json(NAMES_FILE, n)
def save_removed(r): save_json(REMOVED_FILE, r)


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
            self.json_response(self.get_page(page))

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

        elif path == "/api/merge":
            names = load_names()
            src, tgt = body.get("source_id"), body.get("target_id")
            if src and tgt and tgt in names:
                names[src] = names[tgt]
                save_names(names)
            self.json_response({"ok": True})

        elif path == "/api/remove":
            removed = load_removed()
            fid, img_path = body.get("face_id"), body.get("image_path")
            if fid and img_path:
                if fid not in removed:
                    removed[fid] = []
                if img_path not in removed[fid]:
                    removed[fid].append(img_path)
                save_removed(removed)
            self.json_response({"ok": True})

        elif path == "/api/restore":
            removed = load_removed()
            fid, img_path = body.get("face_id"), body.get("image_path")
            if fid and img_path and fid in removed:
                if img_path in removed[fid]:
                    removed[fid].remove(img_path)
                save_removed(removed)
            self.json_response({"ok": True})

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

    def get_all_sorted(self):
        faces = load_faces()
        names = load_names()
        removed = load_removed()
        clusters = faces.get("clusters", {})

        result = []
        for fid, info in sorted(clusters.items(), key=lambda x: x[1]["count"], reverse=True):
            all_images = info["images"]
            removed_imgs = removed.get(fid, [])
            active_images = [img for img in all_images if img not in removed_imgs]
            result.append({
                "id": fid,
                "name": names.get(fid, ""),
                "count": len(active_images),
                "original_count": len(all_images),
                "images": active_images,
                "removed": removed_imgs,
            })
        return result

    def get_page(self, page):
        all_data = self.get_all_sorted()
        start = page * PAGE_SIZE
        end = start + PAGE_SIZE
        return {
            "page": page,
            "page_size": PAGE_SIZE,
            "total": len(all_data),
            "total_pages": (len(all_data) + PAGE_SIZE - 1) // PAGE_SIZE,
            "items": all_data[start:end],
        }

    def get_stats(self):
        names = load_names()
        removed = load_removed()
        faces = load_faces()
        total = len(faces.get("clusters", {}))
        named = len(names)
        return {"total": total, "named": named}

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
.sub{text-align:center;color:#666;margin-bottom:20px;font-size:14px}
.toolbar{display:flex;justify-content:center;gap:10px;margin-bottom:20px;flex-wrap:wrap}
.toolbar button{padding:6px 14px;border:1px solid #333;border-radius:6px;background:#1a1a1a;color:#ccc;cursor:pointer;font-size:13px}
.toolbar button.active{background:#4fc3f7;color:#000;border-color:#4fc3f7}
.stats{text-align:center;margin-bottom:16px;color:#888;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:16px}
.card{background:#1a1a1a;border-radius:10px;border:2px solid #2a2a2a;overflow:hidden;transition:border-color .2s}
.card:hover{border-color:#444}
.card.named{border-color:#4fc3f7}
.card.skipped{opacity:.4}
.card-top{display:grid;grid-template-columns:1fr 100px;gap:12px;padding:14px}
.face-meta h3{color:#4fc3f7;font-size:17px;margin-bottom:4px}
.face-meta .count{color:#888;font-size:13px}
.face-meta .named-label{color:#4fc3f7;font-weight:600;font-size:15px;margin-top:6px}
.face-thumb{width:100px;height:100px;border-radius:50%;object-fit:cover;border:3px solid #4fc3f7}
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
.expand-photos .remove-btn{position:absolute;top:2px;right:2px;width:22px;height:22px;border-radius:50%;background:rgba(239,83,80,.85);color:#fff;border:none;cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .15s}
.expand-photos .thumb-wrap:hover .remove-btn{opacity:1}
.actions{display:flex;gap:6px;padding:10px 14px;border-top:1px solid #2a2a2a;flex-wrap:wrap}
.actions input{flex:1;min-width:120px;padding:7px 10px;border:1px solid #333;border-radius:5px;background:#222;color:#fff;font-size:13px}
.actions input:focus{border-color:#4fc3f7;outline:none}
.btn{padding:7px 12px;border:none;border-radius:5px;cursor:pointer;font-size:12px;font-weight:500}
.btn-save{background:#4fc3f7;color:#000}
.btn-edit{background:#ffa726;color:#000}
.btn-skip{background:#333;color:#888;border:1px solid #444}
.btn-merge{background:#333;color:#ab47bc;border:1px solid #444}
.btn-undo{background:#333;color:#ef5350;border:1px solid #444}
.modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.85);z-index:100;justify-content:center;align-items:center}
.modal.active{display:flex}
.modal-box{background:#1a1a1a;padding:24px;border-radius:12px;max-width:400px;width:90%;border:1px solid #333}
.modal-box h3{margin-bottom:16px}
.modal-box select{width:100%;padding:10px;margin-bottom:16px;background:#222;color:#fff;border:1px solid #333;border-radius:6px}
.modal-actions{display:flex;gap:8px;justify-content:flex-end}
.load-more{text-align:center;padding:30px}
.load-more button{padding:12px 32px;font-size:15px;border:1px solid #4fc3f7;border-radius:8px;background:transparent;color:#4fc3f7;cursor:pointer}
.load-more button:hover{background:#4fc3f7;color:#000}
.spinner{text-align:center;padding:40px;color:#666}
.removed-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:4px;padding:10px 14px}
.removed-grid .thumb-wrap{position:relative}
.removed-grid img{width:100%;aspect-ratio:1;object-fit:cover;opacity:.4}
.removed-grid .restore-btn{position:absolute;top:2px;right:2px;width:20px;height:20px;border-radius:50%;background:rgba(76,175,80,.85);color:#fff;border:none;cursor:pointer;font-size:11px;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .15s}
.removed-grid .thumb-wrap:hover .restore-btn{opacity:1}
</style>
</head>
<body>

<h1>👥 人臉命名工具</h1>
<p class="sub" id="subtitle">載入中...</p>

<div class="toolbar">
  <button class="active" data-filter="all" onclick="setFilter('all')">全部</button>
  <button data-filter="unnamed" onclick="setFilter('unnamed')">未命名</button>
  <button data-filter="named" onclick="setFilter('named')">已命名</button>
</div>

<div class="stats" id="stats"></div>
<div class="grid" id="grid"></div>
<div class="load-more" id="loadMore" style="display:none">
  <button onclick="loadNextPage()">📂 載入更多</button>
</div>
<div class="spinner" id="spinner" style="display:none">載入中...</div>

<div class="modal" id="mergeModal">
  <div class="modal-box">
    <h3>🔗 合併到哪個群組？</h3>
    <select id="mergeTarget"></select>
    <div class="modal-actions">
      <button class="btn btn-skip" onclick="closeMerge()">取消</button>
      <button class="btn btn-save" onclick="confirmMerge()">確認合併</button>
    </div>
  </div>
</div>

<script>
let ALL_ITEMS = [];
let currentPage = 0;
let totalPages = 0;
let filter = 'all';
let mergeSource = '';
let state = {};

// 載入統計
fetch('/api/stats').then(r=>r.json()).then(d=>{
  document.getElementById('subtitle').textContent = `共 ${d.total} 個群組`;
});

// 載入第一頁
loadPage(0);

function loadPage(page){
  document.getElementById('spinner').style.display = 'block';
  fetch(`/api/page?page=${page}`).then(r=>r.json()).then(d=>{
    ALL_ITEMS = ALL_ITEMS.concat(d.items);
    totalPages = d.total_pages;
    currentPage = page;

    // 初始化 state
    d.items.forEach(c => { if(c.name) state[c.id] = 'named'; });

    renderGrid();
    updateStats();

    document.getElementById('spinner').style.display = 'none';
    document.getElementById('loadMore').style.display = (currentPage < totalPages - 1) ? 'block' : 'none';
  });
}

function loadNextPage(){
  document.getElementById('loadMore').style.display = 'none';
  loadPage(currentPage + 1);
}

function getFiltered(){
  return ALL_ITEMS.filter(c=>{
    const s = state[c.id] || '';
    if(filter==='named') return s==='named';
    if(filter==='unnamed') return !s;
    return true;
  });
}

function renderGrid(){
  const grid = document.getElementById('grid');
  grid.innerHTML = '';

  const items = getFiltered();
  updateStats();

  items.forEach(c=>{
    const fid = c.id;
    const s = state[fid] || '';
    const cls = s==='named'?'card named':'card';

    const previewCount = Math.min(Math.max(6, c.images.length), 8);
    const previewImgs = c.images.slice(0,previewCount).map(img=>
      `<img src="/image/${img}" onclick="window.open(this.src)">`
    ).join('');

    const allImgs = c.images.map(img=>
      `<div class="thumb-wrap">
        <img src="/image/${img}" onclick="window.open(this.src)">
        <button class="remove-btn" onclick="event.stopPropagation();removeImg('${fid}','${img.replace(/'/g,"\\'")}')" title="移除">✕</button>
      </div>`
    ).join('');

    const removedImgs = (c.removed||[]).map(img=>
      `<div class="thumb-wrap">
        <img src="/image/${img}">
        <button class="restore-btn" onclick="restoreImg('${fid}','${img.replace(/'/g,"\\'")}')" title="恢復">↩</button>
      </div>`
    ).join('');

    let actions = '';
    if(s==='named'){
      actions = `
        <div class="named-label">✓ ${c.name}</div>
        <button class="btn btn-edit" onclick="editName('${fid}')">✏️ 編輯</button>
        <button class="btn btn-undo" onclick="undoAction('${fid}')">↩️ 取消</button>
        <button class="btn btn-merge" onclick="openMerge('${fid}')">🔗 合併</button>
      `;
    } else {
      actions = `
        <input type="text" id="inp_${fid}" placeholder="輸入名稱..." onkeydown="if(event.key==='Enter')saveName('${fid}')">
        <button class="btn btn-save" onclick="saveName('${fid}')">💾</button>
        <button class="btn btn-skip" onclick="skipFace('${fid}')">⏭️ 略過</button>
        <button class="btn btn-merge" onclick="openMerge('${fid}')">🔗 合併</button>
      `;
    }

    const hasRemoved = (c.removed||[]).length > 0;

    grid.innerHTML += `
      <div class="${cls}" id="card_${fid}">
        <div class="card-top">
          <div class="face-meta">
            <h3>${fid}</h3>
            <div class="count">${c.count} 張${c.count!==c.original_count?' (原 '+c.original_count+')':''}</div>
          </div>
          <img class="face-thumb" src="/thumb/${fid}.jpg" onerror="this.style.display='none'">
        </div>
        <div class="photo-grid">${previewImgs}</div>
        <div class="expand-bar" onclick="toggleExpand('${fid}')">
          <span>📂 展開查看全部 ${c.count} 張</span>
          <span id="arrow_${fid}">▼</span>
        </div>
        <div class="expand-content" id="expand_${fid}">
          <div class="expand-photos">${allImgs}</div>
          ${hasRemoved?`
            <div style="padding:8px 0 4px;color:#666;font-size:12px">已移除 (${c.removed.length}):</div>
            <div class="removed-grid">${removedImgs}</div>
          `:''}
        </div>
        <div class="actions">${actions}</div>
      </div>
    `;
  });
}

function toggleExpand(fid){
  const el = document.getElementById('expand_'+fid);
  const arrow = document.getElementById('arrow_'+fid);
  el.classList.toggle('open');
  arrow.textContent = el.classList.contains('open')?'▲':'▼';
}

function saveName(fid){
  const inp = document.getElementById('inp_'+fid);
  if(!inp||!inp.value.trim()) return;
  post('/api/name',{face_id:fid,name:inp.value.trim()}).then(()=>{
    const c = ALL_ITEMS.find(x=>x.id===fid);
    if(c) c.name = inp.value.trim();
    state[fid]='named'; renderGrid();
  });
}

function editName(fid){ state[fid]=null; renderGrid(); setTimeout(()=>{
  const inp=document.getElementById('inp_'+fid);
  if(inp){inp.value=ALL_ITEMS.find(x=>x.id===fid)?.name||''; inp.focus(); inp.select();}
},50);}

function undoAction(fid){
  post('/api/name',{face_id:fid,name:''}).then(()=>{
    const c=ALL_ITEMS.find(x=>x.id===fid);
    if(c) c.name='';
    delete state[fid]; renderGrid();
  });
}

function skipFace(fid){ state[fid]='skipped'; renderGrid(); }

function openMerge(fid){
  mergeSource=fid;
  const sel=document.getElementById('mergeTarget');
  sel.innerHTML='';
  ALL_ITEMS.filter(c=>c.id!==fid).forEach(c=>{
    const opt=document.createElement('option');
    opt.value=c.id;
    opt.textContent=`${c.id} (${c.name||c.count+'張'})`;
    sel.appendChild(opt);
  });
  document.getElementById('mergeModal').classList.add('active');
}
function confirmMerge(){
  const tgt=document.getElementById('mergeTarget').value;
  if(!tgt)return;
  post('/api/merge',{source_id:mergeSource,target_id:tgt}).then(()=>{
    const c=ALL_ITEMS.find(x=>x.id===tgt);
    const s=ALL_ITEMS.find(x=>x.id===mergeSource);
    if(c&&s){s.name=c.name; state[mergeSource]='named';}
    closeMerge(); renderGrid();
  });
}
function closeMerge(){document.getElementById('mergeModal').classList.remove('active');}

function removeImg(fid,img){
  post('/api/remove',{face_id:fid,image_path:img}).then(()=>{
    const c=ALL_ITEMS.find(x=>x.id===fid);
    if(c){
      c.images=c.images.filter(i=>i!==img);
      if(!c.removed)c.removed=[];
      c.removed.push(img);
      c.count=c.images.length;
    }
    renderGrid();
  });
}

function restoreImg(fid,img){
  post('/api/restore',{face_id:fid,image_path:img}).then(()=>{
    const c=ALL_ITEMS.find(x=>x.id===fid);
    if(c){
      c.images.push(img);
      c.removed=(c.removed||[]).filter(i=>i!==img);
      c.count=c.images.length;
    }
    renderGrid();
  });
}

function setFilter(f){
  filter=f;
  document.querySelectorAll('.toolbar button').forEach(b=>b.classList.toggle('active',b.dataset.filter===f));
  renderGrid();
}

function updateStats(){
  const named=Object.values(state).filter(s=>s==='named').length;
  const total=ALL_ITEMS.length;
  document.getElementById('stats').innerHTML=
    `<span style="color:#4fc3f7">${named}</span> 已命名 | <span>${total-named}</span> 未處理 (共 ${total} 個群組)`;
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
    print(f"人臉命名伺服器 v3：http://127.0.0.1:{port}")
    print(f"分頁載入：每頁 {PAGE_SIZE} 個群組")
    print("按 Ctrl+C 結束")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n結束。")
        names = load_names()
        print(f"已命名 {len(names)} 個群組")


if __name__ == "__main__":
    main()
