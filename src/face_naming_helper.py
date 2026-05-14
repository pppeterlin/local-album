#!/usr/bin/env python3
"""
face_naming_helper.py — 互動式人臉命名工具

顯示每個人臉群組的代表照片，讓使用者命名。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import PROJECT_ROOT as PROJECT_DIR, FACES_DIR, LABELS_DIR, EMBEDDINGS_DIR  # noqa: E402

FACES_FILE = FACES_DIR / "face_clusters.json"
NAMES_FILE = FACES_DIR / "face_names.json"


def load_faces():
    with open(FACES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_names():
    if NAMES_FILE.exists():
        return json.loads(NAMES_FILE.read_text(encoding="utf-8"))
    return {}


def save_names(names):
    NAMES_FILE.write_text(json.dumps(names, indent=2, ensure_ascii=False))


def open_images(paths):
    """用系統預設圖片檢視器開啟（macOS: Preview / Linux: xdg-open）。"""
    import platform
    cmd = ["open"] if platform.system() == "Darwin" else ["xdg-open"]
    for p in paths:
        subprocess.run(cmd + [p], check=False)


def main():
    faces_data = load_faces()
    names = load_names()
    clusters = faces_data.get("clusters", {})

    # 排序：大的群組在前
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )

    print("=" * 60)
    print("👥 人臉命名工具")
    print("=" * 60)
    print(f"\n共 {len(sorted_clusters)} 個群組\n")

    for face_id, info in sorted_clusters:
        current_name = names.get(face_id, "未命名")
        count = info["count"]
        sample_images = info["images"][:3]

        print(f"\n{'─' * 60}")
        print(f"  {face_id}: {current_name} ({count} 張)")
        print(f"  代表照片：")
        for img in sample_images:
            print(f"    • {Path(img).name}")
        print(f"{'─' * 60}")

        # 開啟代表照片
        print(f"\n  正在開啟代表照片...")
        open_images(sample_images)

        # 等待使用者輸入
        user_input = input(f"\n  輸入名稱（或 Enter 跳過，q 離開）: ").strip()

        if user_input.lower() == "q":
            print("\n結束命名。")
            break
        elif user_input:
            names[face_id] = user_input
            save_names(names)
            print(f"  ✓ 已命名：{face_id} → {user_input}")
        else:
            print(f"  跳過 {face_id}")

    # 更新 photo_index.json
    print("\n更新索引...")
    labels = LABELS_DIR / "labels.json"
    embeddings = EMBEDDINGS_DIR / "embeddings.pkl"
    os.system(
        f'cd "{PROJECT_DIR}" && uv run python src/Photo_Index.py build '
        f'--labels "{labels}" --faces "{FACES_FILE}" --embeddings "{embeddings}"'
    )

    print("\n完成！")
    print(f"已命名：{len(names)} / {len(clusters)}")


if __name__ == "__main__":
    main()
