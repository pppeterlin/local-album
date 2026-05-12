# Local Photo Labeler

A local-first pipeline for organizing large personal photo libraries.
It indexes photos with CLIP locally, samples a small representative
subset, sends only that subset to an **OpenAI-compatible Vision API**
for descriptive labels, clusters faces locally with InsightFace, and
unifies everything into a single searchable index.

The design goal is to **minimize cloud token spend and protect privacy**:
embeddings, faces, and full-resolution images never leave the machine;
only a privacy-scrubbed, downscaled subset is uploaded for labeling.

---

## Pipeline overview

```
   photos/                                                  data/
     │                                                        │
     │  1. Local_Indexer.py        (CLIP ViT-H-14, MPS/CUDA)  │
     ├─────────────────────────────────────────────► embeddings/*.pkl
     │                                                        │
     │  2. Smart_Sampler.py        (DBSCAN / KMeans / random) │
     │     ────────────────►                                  │
     │                                                        ▼
     │                                                  samples/*.json
     │                                                        │
     │  3. Vision_Labeler.py       (strip EXIF, resize, API)  │
     │     ─────────────────────────────────────────────►     │
     │                                                  labels/*.json
     │                                                        │
     │  4. Face_Clusterer.py       (InsightFace + DBSCAN)     │
     ├─────────────────────────────────────────────►   faces/face_clusters.json
     │                                                        │
     │  5. Photo_Index.py build    (merge Who/When/Where/What)│
     │                                                  index/photo_index.json
     │                                                        │
     │  6. Photo_Search.py / face_naming_server.py            │
     ▼                                                        ▼
     (CLI search)                                  (browser UI for naming faces)
```

Four orthogonal axes are indexed:

| Axis | Source | Component |
|---|---|---|
| **What** | CLIP image/text vectors + Vision-API labels | `Local_Indexer.py`, `Vision_Labeler.py` |
| **Who**  | Face embeddings + clustering | `Face_Clusterer.py` |
| **When** | EXIF `DateTimeOriginal` | extracted during indexing |
| **Where**| EXIF GPS | extracted during indexing |

---

## Quick start

```bash
# 1. Install dependencies (uv recommended, pip also works)
uv sync
# or:  pip install -r requirements.txt

# 2. Configure your Vision API
cp .env.example .env
# edit .env and set VISION_API_KEY (any OpenAI-compatible provider)

# 3. Run the full pipeline on one photo directory
scripts/run_pipeline.sh /path/to/photos

# 4. Cluster faces from the labels you just produced
uv run python src/Face_Clusterer.py data/labels/photos.json \
    -o data/faces/face_clusters.json

# 5. Generate face thumbnails and start the naming web UI
uv run python src/generate_face_thumbs.py
uv run python src/face_naming_server.py        # http://127.0.0.1:8765

# 6. Build the unified index, then search
uv run python src/Photo_Index.py build \
    --labels data/labels/photos.json \
    --faces  data/faces/face_clusters.json \
    --embeddings data/embeddings/photos.pkl

uv run python src/Photo_Search.py data/embeddings/photos.pkl \
    --query "sunset at the beach" --top 10
```

---

## Repository layout

```
.
├── src/                         All Python modules
│   ├── Local_Indexer.py         CLIP indexing
│   ├── Smart_Sampler.py         Clustering / sampling
│   ├── Vision_Labeler.py        Privacy preprocess + Vision API
│   ├── Face_Clusterer.py        Face detection + clustering
│   ├── Label_Propagator.py      Propagate labels via CLIP similarity
│   ├── Photo_Index.py           Unified Who/When/Where/What index
│   ├── Photo_Search.py          Semantic + keyword search CLI
│   ├── face_naming_server.py    Web UI for naming face clusters
│   ├── face_naming_helper.py    CLI alternative for naming
│   └── generate_face_thumbs.py  Face thumbnail extractor
├── scripts/
│   └── run_pipeline.sh          End-to-end driver (index → sample → label)
├── data/                        All generated artifacts (gitignored)
│   ├── embeddings/   *.pkl
│   ├── samples/      *.json
│   ├── labels/       *.json
│   ├── faces/        face_clusters.json, face_names.json, face_thumbs/
│   └── index/        photo_index.json
├── .env.example
├── pyproject.toml / uv.lock / requirements.txt
└── LICENSE  (MIT)
```

---

## Key parameters

Most defaults are conservative and pipeline-correct. The ones below are
the dials you'll most likely want to turn.

### `Smart_Sampler.py` — how much to send to the cloud

The most important knob for **cost control**. The cloud API is only
called for sampled photos; the rest get labels propagated locally.

| Flag | Default | What it does |
|---|---|---|
| `--method` | `dbscan` | `dbscan` (auto cluster count), `kmeans`, or `random` (uniform subset) |
| `--sample-ratio` | `0.3` | **Random mode only** — fraction of all photos to upload. `0.5` ≈ 50% sample, `0.1` ≈ 10× cost reduction. |
| `--n` | `1` | Photos per cluster to keep (DBSCAN/KMeans). Increase to cover intra-cluster variation. |
| `--eps` | `0.25` | DBSCAN cosine-distance threshold. Lower = tighter clusters, more samples. |
| `--min-samples` | `3` | DBSCAN noise threshold |

> **Tip:** `random` with `--sample-ratio 0.3–0.5` is the simplest and works
> well for diverse libraries. Switch to `dbscan` when you have many
> near-duplicates (burst shots, scans).

### `Face_Clusterer.py` — face detection + clustering

Uses a **per-day stratified sample**: same-day photos are very likely to
contain the same people, so detecting faces on a small subset and
propagating to the rest is dramatically faster.

| Flag | Default | What it does |
|---|---|---|
| `--per-day` | `8` | **How many photos per calendar day to run face detection on.** Raise for crowded events, lower for casual shooting. The single biggest speed/recall tradeoff. |
| `--model` | `buffalo_l` | InsightFace model pack (`buffalo_l` = ResNet-50, accurate; `buffalo_s` = faster) |
| `--det-size` | `640` | Detection input size |
| `--eps` | `0.4` | DBSCAN cosine threshold over face embeddings. Lower = more, tighter identities. |
| `--min-samples` | `2` | Minimum faces to form a cluster |

> **Rule of thumb:** for a library of N photos spanning D days,
> face detection runs on ≈ `min(N, D × --per-day)` images.
> 30k photos over 1000 days at `--per-day 8` ⇒ 8k detections, not 30k.

### `Vision_Labeler.py` — cloud labeling

| Flag | Default | What it does |
|---|---|---|
| `--concurrency` | `5` | Parallel API workers. Tune to your provider's rate limit. |
| `--max-long-edge` | `1024` | Images downscaled to this on the long edge before upload (privacy + cost) |
| `--jpeg-quality` | `90` | Re-encoded JPEG quality |
| `--max-tokens` | `500` | Response budget per image |
| `--incremental` | off | Skip images already in the output file; save every batch (crash-safe) |
| `--no-reasoning` | off | Disables `enable_thinking` (saves tokens on providers that support it; ignored by others) |

**Privacy:** every uploaded image is re-encoded with PIL after stripping
all EXIF (including GPS and timestamps). Originals are never sent.

### `Local_Indexer.py` — CLIP indexing

| Flag | Default | What it does |
|---|---|---|
| `--model` / `--pretrained` | `ViT-H-14` / `laion2b_s32b_b79k` | CLIP model (dim 1024). Drop to `ViT-B-32` for fast/cheap. |
| `--batch-size` | `8` | GPU batch size — lower if you OOM on 8/16 GB Macs |
| `--num-workers` | `4` | CPU DataLoader workers for preprocessing |
| `--device` | auto | `mps` / `cuda` / `cpu` — MPS is auto-detected on Apple Silicon |
| `--incremental` | off | Skip already-indexed paths; merge with existing pickle |

### `Photo_Search.py` — hybrid search

| Flag | Default | What it does |
|---|---|---|
| `--clip-weight` | `0.6` | Blend between CLIP cosine (1.0) and label keyword match (0.0). Lower if your labels are high quality, raise for purely visual queries. |
| `--date-from` / `--date-to` | — | EXIF date filter (YYYY-MM-DD) |
| `--camera` | — | Substring match against EXIF camera model |

CJK queries are auto-detected and translated to English via the same
Vision API (CLIP's text tower is English-trained).

---

## Face naming UI

```
uv run python src/face_naming_server.py   # http://127.0.0.1:8765
```

- Paginated (20 clusters per page), sorted by cluster size
- Inline thumbnail (best face per cluster, generated by `generate_face_thumbs.py`)
- Expandable grid of all photos in the cluster
- Remove / restore photos from a cluster (persists to `face_removed.json`)
- Merge two clusters (e.g., same person split across two IDs)
- Rename / undo

State is written to `data/faces/face_names.json` and
`data/faces/face_removed.json` immediately after every action.

---

## Environment variables

See [`.env.example`](.env.example). The only **required** one is
`VISION_API_KEY` — the rest have sensible defaults.

The Vision API just needs to be **OpenAI-compatible** (i.e. expose
`/v1/chat/completions` with image content parts). Works out of the box
with OpenAI, and with any provider that mirrors that interface.

---

## License

MIT — see [LICENSE](LICENSE).
