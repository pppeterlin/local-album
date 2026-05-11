#!/usr/bin/env python3
"""Run remaining labeler tasks sequentially with auto-retry on timeout."""
import subprocess
import time
import json
import os

base = "/Users/chun/Documents/Python/Local Photo Labeler"
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with open(os.path.join(base, ".env")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

tasks = [
    ("samples_5_Xiaomi_MiMix3_remaining.json", "labels_5_Xiaomi_MiMix3.json", 9598),
    ("samples_8_Xiaomi_Mi13Ultra_remaining.json", "labels_8_Xiaomi_Mi13Ultra.json", 17234),
]

def get_count(labels_path):
    try:
        with open(labels_path) as f:
            return len(set(r['path'] for r in json.load(f)['results']))
    except Exception:
        return 0

for samples, labels, total in tasks:
    print(f"\n{'='*60}", flush=True)
    print(f"Starting: {samples} -> {labels}", flush=True)
    print(f"{'='*60}", flush=True)

    while True:
        labels_path = os.path.join(base, labels)
        current = get_count(labels_path)
        if current >= total:
            print(f"Already done: {current}/{total}", flush=True)
            break

        remaining = total - current
        print(f"{time.strftime('%H:%M:%S')}: {current}/{total} labeled, {remaining} remaining. Running...", flush=True)

        try:
            proc = subprocess.run(
                ["uv", "run", "python", "Xiaomi_Labeler.py", samples, "-o", labels,
                 "--no-reasoning", "--incremental", "--concurrency", "5"],
                cwd=base, env=env, timeout=580,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            if proc.returncode == 0:
                print(f"Done! {labels} complete.", flush=True)
                break
            else:
                print(f"Exit code {proc.returncode}, retrying...", flush=True)
        except subprocess.TimeoutExpired:
            print(f"Timeout, progress saved. Retrying...", flush=True)
            continue
        except Exception as e:
            print(f"Error: {e}, retrying...", flush=True)
            continue

print(f"\n{'='*60}", flush=True)
print(f"ALL DONE at {time.strftime('%H:%M:%S')}", flush=True)
print(f"{'='*60}", flush=True)
