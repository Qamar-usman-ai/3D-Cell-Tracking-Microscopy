# Biohub – Cell Tracking During Development

Four complete, self-contained pipelines for **3D + time cell tracking** in light-sheet
microscopy, built for the Kaggle competition
[*Biohub – Cell Tracking During Development*](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

The task: given a 4D microscopy movie (time × z × y × x), **detect every cell centre in
every frame, link the same cell across consecutive frames, and detect mitosis (division)
events** so that full cell lineages can be reconstructed automatically — replacing the
hours of manual annotation biologists currently do.

---

## Competition & Scoring (how a submission is judged)

| Item | Detail |
|---|---|
| Input | `.zarr` time-lapse 3D volumes. Physical scale: `z = 1.625 µm/voxel`, `y = x = 0.40625 µm/voxel` |
| Output | `submission.csv` with two row types: `node` (cell detections: `t,z,y,x`) and `edge` (links between node ids) |
| Metric part 1 – **Edge Jaccard** | Predicted nodes are matched to ground-truth (GT) nodes per timepoint by optimal bipartite assignment on **scaled centroid distance ≤ 7 µm**. A predicted edge is a true positive when both endpoints match GT nodes that are themselves connected by a GT edge. `J_edge = TP / (TP + FP + FN)`, then **penalized** if you predict more nodes than GT estimates (GT is sparse, so over-detection hurts). |
| Metric part 2 – **Division Jaccard** | A division is a node with ≥ 2 outgoing edges. Each GT division is correct if the predicted graph contains a component that covers the pre-split stage and touches **both** daughter lineages. Micro-averaged Jaccard over all samples. |
| Final score | `score = 0.5 × EdgeJaccard + 0.5 × DivisionJaccard` |

> Because the metric matches per-timepoint and tolerates a 7 µm centroid error, the
> whole game is: **(1) detect as many true cell centres as possible without
> over-predicting, and (2) link them correctly across frames.**

---

## Repository structure

```
biohub-cell-tracking/
├── README.md                     ← you are here
├── requirements.txt              ← python dependencies
├── download_data.py              ← downloads the dataset using YOUR kaggle token
├── .gitignore                    ← keeps kaggle.json & data out of git
├── data/
│   └── README.md                 ← where to put kaggle.json
└── src/                          ← the four pipelines (each fully standalone)
    ├── pipeline_v1_dog_tracker.py          (DoG detection, score 0.78)
    ├── pipeline_v2_kalman_tracker.py       (Kalman filtering, score 0.56)
    ├── pipeline_v3_persistent_homology.py  (topology-based detection, score 0.68)
    └── pipeline_v4_anisgf_tracker.py       (AnisGF pre-filter + DoG, score 0.75)
```

Each pipeline in `src/` is **self-contained**: it reads the `.zarr` volumes directly
(using a fast blosc2 chunk reader with a zarr fallback), runs detection + linking,
and writes a valid `submission.csv`.

---

## Quick start (run any pipeline)

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. provide your Kaggle token (see data/README.md), then download the data
python download_data.py

# 3. run a pipeline (writes submission.csv in the working directory)
python src/pipeline_v1_dog_tracker.py
```

The pipelines auto-detect the `test/` folder inside the Kaggle input directory
(`/kaggle/input/...` on Kaggle, or wherever you place it locally — set `TEST_DIR`
env var to override).

---

## The four pipelines — what each one does and how it scores

### Version history at a glance

| File | Detector | Linker | Local score |
|---|---|---|---|
| `pipeline_v1_dog_tracker.py` | Difference-of-Gaussians (multi-scale) | two-pass Hungarian (velocity-weighted) | **0.78** |
| `pipeline_v2_kalman_tracker.py` | smoothed intensity peaks + intensity-weighted centroid refinement | constant-velocity **Kalman filter** | 0.56 |
| `pipeline_v3_persistent_homology.py` | **0-dimensional persistent homology** (topological) | two-pass Hungarian | 0.68 |
| `pipeline_v4_anisgf_tracker.py` | **Anisotropic Gaussian pre-filter** + multi-scale DoG | two-pass Hungarian | 0.75 |

### Common building blocks (present in all/most files)

1. **Chunked zarr I/O** — reads one timepoint at a time from
   `dataset.zarr/0/c/<t>/0/0/0` via blosc2 (fallback to `zarr`), so 100-frame movies
   fit in memory.
2. **Physical-unit geometry** — everything is measured in µm using the voxel scale
   `(1.625, 0.40625, 0.40625)`, so distance gates (`min_distance`, `max_link`) mean
   real-world cell sizes/movements, not voxels.
3. **Centroid refinement** — each detected peak is re-centred by intensity-weighted
   averaging inside a small window, which directly improves the 7 µm matching radius.
4. **Track-graph post-processing** — prune isolated nodes, drop tracks shorter than 4
   nodes, temporal position smoothing (midpoint / straight-line fit), and "gap
   recovery" that bridges 2-frame dropouts by interpolating intermediate nodes.
5. **A local re-implementation of the competition metric** (`match_per_timepoint`,
   `edge_jaccard`, `division_score`, `aggregate`) for offline validation against the
   training ground truth (`.geff` files).

### `pipeline_v1_dog_tracker.py` — DoG + Hungarian, **score 0.78** (best)

The strongest classical pipeline.
- **Detection** (`detect_blobs`): normalizes intensity to [0,1] using the 1st–99.7th
  percentile, then applies **multi-scale Difference-of-Gaussians** at two scale pairs
  (`1.5–4.0 µm` and `2.2–5.5 µm`, taking the element-wise maximum response), finds
  local maxima with a spherical non-maximum-suppression footprint of 3 µm, and keeps
  peaks above a small relative threshold.
- **Linking** (`link_twopass`): for each consecutive frame pair it builds a cost
  matrix of **physical distances between the previous position + half the previous
  velocity and the current detections**, and solves it twice (tight 6 µm gate first,
  then a second Hungarian pass on unmatched pairs at 8 µm). Matched velocities seed
  the next prediction.
- **Post-processing**: 1-frame gap closing (≤ 6 µm), prune isolated nodes, drop
  tracks < 4 nodes, straight-line smoothing (w = 0.8), and `recover_gap2` that
  interpolates nodes across 2-frame gaps (≤ 10.2 µm total, ≤ 4.4 µm per step).

### `pipeline_v2_kalman_tracker.py` — Kalman tracker, score 0.56

The most "tracking-engineered" pipeline — a per-cell **Kalman filter** with a
6-dimensional state `[z, y, x, vz, vy, vx]` (position + velocity in µm).
- **Detection**: downsample-by-4 pooling, light Gaussian smoothing, Otsu threshold,
  `peak_local_max`, then sub-voxel intensity-weighted refinement and score-sorted
  NMS at 4 µm.
- **Tracking** (`link_kalman`): every track predicts its next position with
  `F·x`, the innovation is corrected with the Kalman gain after each Hungarian
  assignment gated at 10 µm.
- **Extras**: optional **count calibration** — it measures how much a "generous"
  detector over-counts versus the GT `estimated_number_of_nodes` on train movies and
  caps per-frame detections accordingly; optional division detection (parent within
  9 µm, sister within 9 µm). Both were disabled in the submitted config, which is a
  big part of why it scores lowest here — but it is the cleanest base for adding a
  learned detector.

### `pipeline_v3_persistent_homology.py` — topological detection, score 0.68

Replaces *both* the band-pass filter and the arbitrary intensity threshold with a
**0-dimensional persistent-homology (superlevel-set) sweep**:
- Every voxel is activated in decreasing-intensity order while a union-find tracks
  connected components. A component is **born** at a local maximum and **dies** when
  it merges into a taller neighbour (elder rule).
- A peak's **persistence = birth − death** is its topological lifetime: real, isolated
  cell peaks persist over a wide threshold range; noise dies immediately. The only
  meaningful knob is `persistence_threshold`.
- Implemented with a Numba-JIT-compiled union-find core (falls back to pure Python),
  plus spatial-hash NMS. Same Hungarian linker and post-processing as v1.
- Conceptually the most principled detector; in practice the fixed-threshold/DoG
  tuning of v1/v4 edged it out on this data.

### `pipeline_v4_anisgf_tracker.py` — AnisGF + DoG, score 0.75

Hybrid: pre-filters each volume with an **anisotropic Gaussian** matched to the voxel
geometry (`σz = 2×σxy` in physical units, compensating the 4× z-vs-xy anisotropy)
before the same multi-scale DoG + two-pass Hungarian pipeline as v1. Almost matches
v1 with a different, arguably more physical front-end.

---

## Results (local scores)

| Pipeline | Score |
|---|---|
| v1 – DoG + Hungarian | **0.78** |
| v4 – AnisGF + DoG | 0.75 |
| v3 – Persistent homology | 0.68 |
| v2 – Kalman | 0.56 |

Typical runtime on the 4 test movies (CPU): v1 ≈ 5 min, v3 ≈ 3 min, v4 ≈ 6.5 min.

---

## Reproducing on Kaggle

1. Create a Notebook, enable **GPU/CPU** (≤ 12 h runtime, internet disabled is fine —
   data comes from the attached competition input).
2. Copy any `src/pipeline_v*.py` into a code cell, or upload it as a Notebook file.
3. The pipeline auto-finds `/kaggle/input/biohub-cell-tracking-during-development/test`
   and writes `submission.csv` at the notebook root, ready for submission.

## Ideas ranked by expected impact

1. **Learned detector** (StarDist-3D / Cellpose, attachable as a Kaggle dataset) —
   node recall is the biggest lever, and edge recall scales roughly with its square.
2. **Division detection turned on & tuned** on division-rich validation movies.
3. **Per-movie count budget** (v2's calibration idea) to dodge the over-prediction
   penalty without losing recall.
4. Longer temporal context for linking (tracklet-based global optimization instead of
   frame-to-frame Hungarian).

## Citation

Thibaut Goldsborough, Jordão Bragantini, Xiang Zhao, Gordon Leary, Teun Huijben,
Ilan da Silva Theodoro, Kyle Harrington, Chi-Li Chiu, Walter Reade, María Cruz, and
Loïc A. Royer. *Biohub - Cell Tracking During Development.*
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development, 2026. Kaggle.

## License

MIT (see LICENSE). Competition data remains subject to Kaggle's competition rules.
