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
    """Open the on-disk index. Errors loudly if it doesn't exist."""
    idx = PhotoIndex(str(PROJECT_ROOT))
    if not idx.index_path.exists():
        raise FileNotFoundError(
            f"No index at {idx.index_path}. Build it first with:\n"
            f"  uv run python src/Photo_Index.py build \\\n"
            f"      --labels data/labels/<name>.json \\\n"
            f"      --faces  data/faces/face_clusters.json \\\n"
            f"      --embeddings data/embeddings/<name>.pkl"
        )
    idx.load_index()
    return idx


@mcp.tool()
def search_photos(
    query: str = "",
    face: str | None = None,
    year: int | None = None,
    month: int | None = None,
    location: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Search the photo index by any combination of filters.

    Args:
        query:    Substring to match against image labels (case-insensitive).
        face:     Face cluster ID (e.g. "face_0") OR display name (e.g. "Mom").
        year:     Filter by EXIF year (e.g. 2023).
        month:    Filter by EXIF month (1-12).
        location: Substring match against location field (lat,lng or place name).
        top_k:    Max results (default 20). Use a small number for previews.

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
