"""
共用路徑解析。所有產生物路徑由 METADATA_DIR 環境變數決定，
預設為專案根目錄下 data/，可放外接硬碟。

讀取順序：
  1. 環境變數 METADATA_DIR
  2. 專案根目錄下的 .env 檔 METADATA_DIR=
  3. fallback: <project_root>/data
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env_var(name: str) -> str | None:
    """先讀 os.environ，再 fallback 到專案根的 .env 檔。"""
    v = os.environ.get(name)
    if v:
        return v
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, val = line.split("=", 1)
        if k.strip() == name:
            return val.strip().strip('"').strip("'")
    return None


_md = _load_env_var("METADATA_DIR")
METADATA_DIR = Path(_md) if _md else (PROJECT_ROOT / "data")

FACES_DIR = METADATA_DIR / "faces"
PETS_DIR = METADATA_DIR / "pets"
LABELS_DIR = METADATA_DIR / "labels"
EMBEDDINGS_DIR = METADATA_DIR / "embeddings"
SAMPLES_DIR = METADATA_DIR / "samples"
INDEX_DIR = METADATA_DIR / "index"
