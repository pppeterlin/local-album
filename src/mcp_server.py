#!/usr/bin/env python3
"""
mcp_server.py — Local Album MCP server (query-only).

Exposes the unified photo index to any MCP-compatible client (Claude
Code / Claude Desktop / etc.) over stdio. Pure read access — indexing,
labeling, and clustering remain CLI-only and are run separately.

Tools:
  • search_photos     — combined keyword + face + date + location search
  • list_faces        — face clusters (sorted by size)
  • get_photo         — full info for one image path
  • index_stats       — index health / counts

Run:
  uv run python src/mcp_server.py        # stdio MCP server

Claude Code registration (one-off):
  claude mcp add local-album \
      uv --directory "$(pwd)" run python src/mcp_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Reuse the index from Photo_Index.py (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Photo_Index import PhotoIndex  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    sys.stderr.write(
        "Missing dependency: mcp. Install with `uv sync` or `pip install mcp`.\n"
    )
    raise SystemExit(1) from e


PROJECT_ROOT = Path(__file__).resolve().parent.parent

mcp = FastMCP("local-album")


def _load_index() -> PhotoIndex:
    """Open the on-disk index. PhotoIndex.__init__ loads it lazily; we
    just verify the file exists and produce a friendly error if not."""
    idx = PhotoIndex(str(PROJECT_ROOT))
    if not idx.index:
        raise FileNotFoundError(
            f"No index at {idx.index_path}. Build it first with:\n"
            f"  uv run python src/Photo_Index.py build \\\n"
            f"      --labels data/labels/<name>.json \\\n"
            f"      --faces  data/faces/face_clusters.json \\\n"
            f"      --embeddings data/embeddings/<name>.pkl"
        )
    return idx


@mcp.tool()
def search_photos(
    query: str = "",
    face: str | None = None,
    year: int | None = None,
    month: int | None = None,
    location: str | None = None,
    path_contains: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Search the photo index by any combination of filters.

    Filters are combined with AND — each one narrows the result set.
    Use this to compose queries like "childhood photos with grandpa" in
    a single call (e.g. face='阿公', path_contains='小時候').

    Args:
        query:         Substring to match against image labels (case-insensitive).
        face:          Face cluster ID (e.g. "face_0") OR display name (e.g. "Mom").
        year:          Filter by EXIF year (e.g. 2023).
        month:         Filter by EXIF month (1-12).
        location:      Substring match against location field.
        path_contains: Substring match against the file path (folder name,
                       subdir, filename — useful for scoping to an album).
        top_k:         Max results (default 20).

    Returns:
        List of matches sorted by relevance, each with:
        {path, score, label, faces, time, location}
    """
    idx = _load_index()
    return idx.search(
        query=query,
        face_name=face,
        year=year,
        month=month,
        location=location,
        path_contains=path_contains,
        top_k=top_k,
    )


@mcp.tool()
def list_faces(
    named_only: bool = False,
    unnamed_only: bool = False,
    top_n: int = 100,
) -> list[dict[str, Any]]:
    """List face clusters in the library, largest first.

    Args:
        named_only:   Only clusters that have a human-assigned name.
        unnamed_only: Only clusters still awaiting a name.
        top_n:        Cap result size (default 100).

    Returns:
        [{id, name, count, sample_images}, ...]   sample_images are up to 3 paths.
    """
    idx = _load_index()
    faces_path = idx.project_dir / "data" / "faces" / "face_clusters.json"
    if not faces_path.exists():
        return []

    import json

    data = json.loads(faces_path.read_text(encoding="utf-8"))
    clusters = data.get("clusters", {})

    results = []
    for fid, info in clusters.items():
        name = idx.face_names.get(fid, "")
        if named_only and not name:
            continue
        if unnamed_only and name:
            continue
        results.append({
            "id": fid,
            "name": name,
            "count": info.get("count", 0),
            "sample_images": info.get("images", [])[:3],
        })

    results.sort(key=lambda x: x["count"], reverse=True)
    return results[:top_n]


@mcp.tool()
def get_photo(path: str) -> dict[str, Any]:
    """Return the full index record for a single photo by absolute path."""
    idx = _load_index()
    info = idx.index.get("images", {}).get(path)
    if not info:
        return {"error": f"Not in index: {path}"}
    return info


@mcp.tool()
def open_in_viewer(paths: list[str], max_open: int = 50) -> dict[str, Any]:
    """Open one or more image paths in the local OS's default image viewer.

    macOS opens them all in a single Preview window (sidebar shows the set).
    Linux/Windows open each in the default app (one window per file — most
    desktop environments don't expose a multi-image viewer via CLI).

    Args:
        paths:    List of absolute image paths (typically from search_photos).
        max_open: Hard cap to prevent runaway window spawning. Default 50.

    Returns:
        {opened: int, skipped_missing: [..], error: str | None}
    """
    import platform
    import shutil
    import subprocess

    paths = list(paths)[:max_open]
    existing = [p for p in paths if Path(p).exists()]
    missing = [p for p in paths if not Path(p).exists()]

    if not existing:
        return {"opened": 0, "skipped_missing": missing,
                "error": "no valid paths to open"}

    system = platform.system()
    try:
        if system == "Darwin":
            # `open -a Preview <files>` groups into one Preview window
            subprocess.run(["open", "-a", "Preview", *existing], check=True)
        elif system == "Windows":
            for p in existing:
                # start uses cmd.exe; the empty "" is the window title
                subprocess.Popen(["cmd", "/c", "start", "", p], shell=False)
        else:  # Linux / BSD
            opener = shutil.which("xdg-open") or shutil.which("gio")
            if not opener:
                return {"opened": 0, "skipped_missing": missing,
                        "error": "no xdg-open or gio found"}
            for p in existing:
                subprocess.Popen([opener, p])
    except Exception as e:  # noqa: BLE001
        return {"opened": 0, "skipped_missing": missing, "error": str(e)}

    return {"opened": len(existing), "skipped_missing": missing, "error": None}


@mcp.tool()
def index_stats() -> dict[str, Any]:
    """Summary of the current index: counts, build time, named/unnamed faces."""
    idx = _load_index()
    images = idx.index.get("images", {})
    with_time = sum(1 for i in images.values() if i.get("time"))
    with_gps = sum(1 for i in images.values() if i.get("gps"))
    with_faces = sum(1 for i in images.values() if i.get("faces"))

    faces_path = idx.project_dir / "data" / "faces" / "face_clusters.json"
    total_clusters = 0
    if faces_path.exists():
        import json
        total_clusters = len(json.loads(faces_path.read_text(encoding="utf-8")).get("clusters", {}))

    return {
        "total_images": len(images),
        "with_exif_time": with_time,
        "with_gps": with_gps,
        "with_faces": with_faces,
        "face_clusters_total": total_clusters,
        "face_clusters_named": len(idx.face_names),
        "built_at": idx.index.get("built_at"),
    }


if __name__ == "__main__":
    mcp.run()
