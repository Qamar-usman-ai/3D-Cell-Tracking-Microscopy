import os, json, glob, time, gc
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    import blosc2
except Exception:
    blosc2 = None

try:
    import zarr
except Exception:
    zarr = None

try:
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    _SK = True
except Exception:
    _SK = False

# ---- Configurations & Constants ----
SCALE = np.array([1.625, 0.40625, 0.40625])  # z,y,x µm/voxel
GATE_UM = 7.0

XY_DS = 4
SMOOTH_SIGMA = 0.9
THRESH_REL = 0.32
MIN_PEAK_DIST = 2
NMS_RADIUS_UM = 4.0
REFINE_RZ, REFINE_RYX = 2, 5
MAX_LINK_UM = 10.0
DETECT_DIV = False
DIV_PARENT_UM, DIV_SISTER_UM = 9.0, 9.0
PRUNE_ISOLATED = True

USE_COUNT_CALIBRATION = True
GENEROUS_THRESH_REL = 0.10
CALIB_FRAMES = 6
BUDGET_SAFETY = 1.15

RUN_VALIDATION = True
VAL_SAMPLES = 2

# ---- Path Setup ----
def find_dirs():
    cands = ['/kaggle/input/biohub-cell-tracking-during-development',
             '/kaggle/input/competitions/biohub-cell-tracking-during-development']
    root = next((p for p in cands if os.path.isdir(os.path.join(p, 'test'))), None)
    if root is None:
        hits = glob.glob('/kaggle/input/**/test', recursive=True)
        root = os.path.dirname(hits[0]) if hits else cands[0]
    return os.path.join(root, 'train'), os.path.join(root, 'test')

TRAIN_DIR, TEST_DIR = find_dirs()
print('TRAIN:', TRAIN_DIR, os.path.isdir(TRAIN_DIR), '| TEST:', TEST_DIR, os.path.isdir(TEST_DIR))

# ==============================================================================
# 1 · I/O — Data Streaming & Reader
# ==============================================================================
def list_names(d):
    return sorted(p[:-5] for p in os.listdir(d) if p.endswith('.zarr')) if d and os.path.isdir(d) else []

def read_meta(zp):
    m = json.load(open(os.path.join(zp, '0', 'zarr.json')))
    return tuple(m['shape']), np.dtype(m['data_type'])

_CACHE = {}
def load_volume(zp, t, meta=None):
    try:
        if zp not in _CACHE:
            _CACHE[zp] = zarr.open(zp, mode='r')['0']
        return np.asarray(_CACHE[zp][t])
    except Exception:
        if meta is None:
            meta = (read_meta(zp),)
        shape, dtype = read_meta(zp)
        chunk = os.path.join(zp, '0', 'c', str(t), '0', '0', '0')
        return np.frombuffer(blosc2.decompress(open(chunk, 'rb').read()), dtype=dtype).reshape(shape[1:])

def read_geff(geff_path):
    g = zarr.open_group(geff_path, mode='r')
    n = {'node_id': np.asarray(g['nodes/ids'][:]).astype(np.int64)}
    for k in ('t', 'z', 'y', 'x'):
        n[k] = np.asarray(g[f'nodes/props/{k}/values'][:])
    nodes = pd.DataFrame(n)
    e = np.asarray(g['edges/ids'][:]).astype(np.int64)
    edges = pd.DataFrame(e, columns=['source_id', 'target_id']) if len(e) else pd.DataFrame(columns=['source_id', 'target_id'])
    est = np.nan
    try:
        meta = json.loads(open(os.path.join(geff_path, 'zarr.json')).read())
        def dig(o):
            if isinstance(o, dict):
                if 'estimated_number_of_nodes' in o: return o['estimated_number_of_nodes']
                for v in o.values():
                    r = dig(v)
                    if r is not None: return r
            return None
        v = dig(meta); est = float(v) if v else np.nan
    except Exception:
        pass
    return nodes, edges, est

test_names = list_names(TEST_DIR)
print(len(test_names), 'test movies')

# ==============================================================================
# 2 · Detection Engine
# ==============================================================================
def _pool(vol, f):
    if f <= 1: return vol.astype(np.float32)
    Z, Y, X = vol.shape; Y2, X2 = (Y // f) * f, (X // f) * f
    return vol[:, :Y2, :X2].astype(np.float32).reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))

def _thr(sm, rel):
    bg, hi = float(np.median(sm)), float(np.percentile(sm, 99.9))
    r = bg + rel * max(hi - bg, 1e-6)
    if _SK:
        try: return max(float(threshold_otsu(sm)), r)
        except Exception: pass
    return max(float(np.percentile(sm, 96.0)), r)

def _peaks(sm, thr, d):
    if _SK:
        return peak_local_max(sm, min_distance=int(d), threshold_abs=thr, exclude_border=False).astype(np.int32)
    mx = maximum_filter(sm, size=2 * int(d) + 1, mode='nearest')
    return np.argwhere((sm >= mx) & (sm > thr)).astype(np.int32)

def _refine(vol, zyx):
    Z, Y, X = vol.shape; z, y, x = (int(round(v)) for v in zyx)
    z0, z1 = max(0, z - REFINE_RZ), min(Z, z + REFINE_RZ + 1)
    y0, y1 = max(0, y - REFINE_RYX), min(Y, y + REFINE_RYX + 1)
    x0, x1 = max(0, x - REFINE_RYX), min(X, x + REFINE_RYX + 1)
    crop = vol[z0:z1, y0:y1, x0:x1].astype(np.float32); bg = float(crop.min())
    w = np.clip(crop - bg, 0, None); s = float(w.sum())
    if s <= 0: return np.array([z, y, x], float), 0.0
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    return np.array([(zz * w).sum(), (yy * w).sum(), (xx * w).sum()]) / s, float(crop.max() - bg)

def _nms(coords, scores, radius_um):
    if len(coords) <= 1: return coords, scores
    pts = coords * SCALE[None, :]; order = np.argsort(-scores)
    tree = cKDTree(pts); killed = np.zeros(len(coords), bool); keep = []
    for i in order:
        if killed[i]: continue
        keep.append(int(i)); killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    keep = np.array(keep); return coords[keep], scores[keep]

def detect(vol, sigma=SMOOTH_SIGMA, thresh_rel=THRESH_REL, min_dist=MIN_PEAK_DIST,
           nms_um=NMS_RADIUS_UM, topk=None):
    sm = gaussian_filter(_pool(vol, XY_DS), sigma) if sigma > 0 else _pool(vol, XY_DS)
    pk = _peaks(sm, _thr(sm, thresh_rel), min_dist)
    if len(pk) == 0: return np.zeros((0, 3)), np.zeros(0)
    full = pk.astype(float); full[:, 1] = full[:, 1] * XY_DS + (XY_DS - 1) / 2; full[:, 2] = full[:, 2] * XY_DS + (XY_DS - 1) / 2
    coords, scores = [], []
    for p in full:
        c, s = _refine(vol, p); coords.append(c); scores.append(s)
    coords, scores = _nms(np.array(coords), np.array(scores), nms_um)
    if topk is not None and len(coords) > topk:
        k = np.argsort(-scores)[:int(topk)]; coords, scores = coords[k], scores[k]
    return coords, scores

# ==============================================================================
# 3 · Kalman Filter Tracker Implementation
# ==============================================================================
class CellTracker:
    def __init__(self, node_id, pos, dt=1.0):
        self.node_id = node_id
        # State vector: [z, y, x, vz, vy, vx] in µm
        self.x = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0], dtype=float)
        
        # State Transition Matrix (F)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Measurement Matrix (H)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        
        # Covariance Matrices
        self.P = np.eye(6) * 10.0
        self.Q = np.eye(6) * 0.5   # Process noise
        self.R = np.eye(3) * 1.0   # Measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]

    def update(self, pos):
        y = pos - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

def link_kalman(active_trackers, curr_xyz):
    if len(active_trackers) == 0 or len(curr_xyz) == 0:
        return []

    # Predict physical positions (µm) for current frame
    predicted_um = np.array([tracker.predict() for tracker in active_trackers])
    curr_um = curr_xyz * SCALE[None, :]

    # Cost matrix computation
    D = np.sqrt(((predicted_um[:, None, :] - curr_um[None, :, :]) ** 2).sum(axis=2))
    BIG = 1e6
    cost = np.where(D > MAX_LINK_UM, BIG, D)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    matched_links = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < BIG:
            matched_links.append((r, c))
            active_trackers[r].update(curr_um[c])
            
    return matched_links

def divisions(prev_xyz, curr_xyz, links):
    if not DETECT_DIV or len(curr_xyz) <= len(prev_xyz): return []
    P = prev_xyz * SCALE[None]; C = curr_xyz * SCALE[None]
    matched = {c for _, c in links}; parent_of = {c: p for p, c in links}
    free = [j for j in range(len(curr_xyz)) if j not in matched]
    if not free: return []
    ptree = cKDTree(P); extra = []
    for j in free:
        d, p = ptree.query(C[j], k=1)
        if d > DIV_PARENT_UM: continue
        sis = [c for c, pp in parent_of.items() if pp == p]
        if sis and min(np.linalg.norm(C[s] - C[j]) for s in sis) <= DIV_SISTER_UM:
            extra.append((int(p), int(j)))
    return extra

# ==============================================================================
# 4 · Movie Tracking Pipeline
# ==============================================================================
COLS = ['dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']

def track_movie(zp, name, meta, topk=None):
    shape, dtype = meta; T = shape[0]
    node_rows, edge_rows, score = [], [], {}
    
    active_trackers = []
    prev_ids, prev_xyz = [], np.zeros((0, 3))
    nid, counts, ndiv = 1, [], 0

    for t in range(T):
        vol = load_volume(zp, t, meta)
        _thr_rel = GENEROUS_THRESH_REL if topk is not None else THRESH_REL
        coords, scores = detect(vol, thresh_rel=_thr_rel, topk=topk); del vol
        
        ids = list(range(nid, nid + len(coords))); nid += len(coords)
        for i, c, s in zip(ids, coords, scores):
            node_rows.append((name, 'node', i, t, float(c[0]), float(c[1]), float(c[2]), -1, -1))
            score[i] = float(s)

        if t > 0 and len(active_trackers):
            lk = link_kalman(active_trackers, coords)
            ex = divisions(prev_xyz, coords, lk)

            for p, c in lk:
                edge_rows.append((name, 'edge', -1, -1, -1, -1, -1, prev_ids[p], ids[c]))
            for p, c in ex:
                edge_rows.append((name, 'edge', -1, -1, -1, -1, -1, prev_ids[p], ids[c]))
            
            ndiv += len({p for p, _ in ex})

        # Initialize trackers for the current frame
        coords_um = coords * SCALE[None, :]
        active_trackers = [CellTracker(i, pos) for i, pos in zip(ids, coords_um)]
        prev_ids, prev_xyz = ids, coords
        counts.append(len(coords))

    nodes = pd.DataFrame(node_rows, columns=COLS)
    edges = pd.DataFrame(edge_rows, columns=COLS)

    if PRUNE_ISOLATED and len(edges):
        used = set(edges.source_id) | set(edges.target_id)
        nodes = nodes[nodes.node_id.isin(used)].reset_index(drop=True)

    return nodes, edges, dict(name=name, nodes=len(nodes), edges=len(edges),
                              cells_per_frame=float(np.mean(counts)) if counts else 0.0, div=ndiv, T=T)

# ==============================================================================
# 5 · Count Calibration
# ==============================================================================
calib_topk_per_frame = {}
CALIB_FACTOR = None

def generous_density(zp, meta, frames):
    shape, dtype = meta
    cnt = 0
    for t in frames:
        coords, _ = detect(load_volume(zp, int(t), meta), thresh_rel=GENEROUS_THRESH_REL)
        cnt += len(coords)
    return cnt / max(len(frames), 1)

if USE_COUNT_CALIBRATION and zarr is not None and os.path.isdir(TRAIN_DIR):
    train_names = list_names(TRAIN_DIR)
    picked, seen = [], set()
    for nm in train_names:
        e = nm.split('_')[0]
        if e in seen: continue
        seen.add(e); picked.append(nm)
        if len(picked) >= max(VAL_SAMPLES, 3): break
    ratios = []
    for nm in picked:
        try:
            gp = read_geff(os.path.join(TRAIN_DIR, nm + '.geff'))
            if gp is None or not np.isfinite(gp[2]): continue
            zp = os.path.join(TRAIN_DIR, nm + '.zarr'); meta = read_meta(zp)
            T = meta[0][0]
            frames = np.unique(np.linspace(0, T - 1, CALIB_FRAMES).astype(int))
            D = generous_density(zp, meta, frames)
            R = gp[2] / T
            if D > 0:
                ratios.append(R / D)
                print(f'  {nm[:22]:22s} true/fr={R:6.2f} generous/fr={D:6.2f} ratio={R/D:.3f}')
        except Exception as ex:
            print('  calib skip', nm, str(ex)[:50])
    if ratios:
        CALIB_FACTOR = float(np.median(ratios))
        print(f'calibration factor f = {CALIB_FACTOR:.3f} (generous detector over-counts by ~{1/CALIB_FACTOR:.2f}x)')
        for nm in test_names:
            zp = os.path.join(TEST_DIR, nm + '.zarr'); meta = read_meta(zp); T = meta[0][0]
            frames = np.unique(np.linspace(0, T - 1, CALIB_FRAMES).astype(int))
            D = generous_density(zp, meta, frames)
            calib_topk_per_frame[nm] = max(1, int(np.ceil(BUDGET_SAFETY * CALIB_FACTOR * D)))
        print('per-movie budgets:', {k: v for k, v in list(calib_topk_per_frame.items())[:6]})
    else:
        print('No usable training ratios; calibration disabled.')
else:
    print('Count calibration skipped (flag off or train/geff unavailable).')

# ==============================================================================
# 6 · Validation Proxy
# ==============================================================================
def _split(df):
    n = df[df.row_type == 'node'][['node_id', 't', 'z', 'y', 'x']]
    e = df[df.row_type == 'edge'][['source_id', 'target_id']].astype(int)
    return n, e

def match_nodes(pn, gn, r=GATE_UM):
    p2g = {}
    for t in sorted(set(pn.t) & set(gn.t)):
        p = pn[pn.t == t].reset_index(drop=True); g = gn[gn.t == t].reset_index(drop=True)
        if len(p) == 0 or len(g) == 0: continue
        D = np.sqrt(((p[['z','y','x']].values * SCALE)[:, None] - (g[['z','y','x']].values * SCALE)[None]) ** 2).sum(2)
        cost = np.where(D <= r, D, 1e6)
        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] < 1e6: p2g[int(p.loc[a, 'node_id'])] = int(g.loc[b, 'node_id'])
    return p2g

def proxy_score(pred_df, gn, ge, w=(0.5, 0.4, 0.1)):
    pn, pe = _split(pred_df); p2g = match_nodes(pn, gn)
    tp = len(p2g); fp = len(pn) - tp; fn = len(gn) - tp
    dp = tp / max(tp + fp, 1); dr = tp / max(tp + fn, 1); df1 = 2 * dp * dr / max(dp + dr, 1e-9)
    gset = set(map(tuple, ge[['source_id','target_id']].astype(int).values))
    pm = {(p2g[s], p2g[t]) for s, t in pe.values if s in p2g and t in p2g}
    etp = len(pm & gset); ep = etp / max(len(pm), 1); er = etp / max(len(gset), 1)
    ef1 = 2 * ep * er / max(ep + er, 1e-9)
    return round(w[0]*df1 + w[1]*ef1 + w[2]*1.0, 4), dict(node_recall=round(dr,3), node_prec=round(dp,3),
            edge_recall=round(er,3), edge_prec=round(ep,3), pred_nodes=len(pn), gt_nodes=len(gn))

if RUN_VALIDATION and zarr is not None and os.path.isdir(TRAIN_DIR):
    train_names = list_names(TRAIN_DIR)
    pick, seen = [], set()
    for nm in train_names:
        e = nm.split('_')[0]
        if e in seen: continue
        seen.add(e); pick.append(nm)
        if len(pick) >= VAL_SAMPLES: break
    rows = []
    for nm in pick:
        try:
            zp = os.path.join(TRAIN_DIR, nm + '.zarr'); meta = read_meta(zp)
            nodes, edges, st = track_movie(zp, nm, meta)
            gn, ge, _ = read_geff(os.path.join(TRAIN_DIR, nm + '.geff'))
            gn = gn[gn.t < meta[0][0]]; ge = ge[ge.source_id.isin(gn.node_id) & ge.target_id.isin(gn.node_id)]
            sc, br = proxy_score(pd.concat([nodes, edges], ignore_index=True), gn, ge)
            rows.append(dict(dataset=nm, embryo=nm[:4], proxy=sc, **br))
            print(f'  {nm[:24]:24s} proxy={sc} node_recall={br["node_recall"]} edge_recall={br["edge_recall"]}')
        except Exception as ex:
            print('  val skip', nm, str(ex)[:60])
    if rows: display(pd.DataFrame(rows))
else:
    print('Validation skipped (offline run or flag off).')

# ==============================================================================
# 7 · Export CSV & Schema Verification
# ==============================================================================
parts, stats = [], []
t0 = time.time()

for zp_name in test_names:
    zp = os.path.join(TEST_DIR, zp_name + '.zarr')
    if not os.path.exists(os.path.join(zp, '0', 'zarr.json')):
        print('  skip (no meta)', zp_name); continue
    meta = read_meta(zp)
    topk = calib_topk_per_frame.get(zp_name)
    nodes, edges, st = track_movie(zp, zp_name, meta, topk=topk)
    st['budget'] = topk; st['sec'] = round(time.time() - t0, 1)
    stats.append(st); parts += [nodes, edges]
    print(f"  {zp_name}: T={st['T']} nodes={st['nodes']} edges={st['edges']} "
          f"cells/frame={st['cells_per_frame']:.1f} budget={topk} ({st['sec']}s)")

submission = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
submission = submission[COLS]; submission.index.name = 'id'
submission.to_csv('submission.csv')

# Verification
assert set(test_names).issubset(set(submission.dataset.unique()) | set()), 'every test movie must appear'
nodes = submission[submission.row_type == 'node']; edges = submission[submission.row_type == 'edge']
assert (edges[['node_id','t','z','y','x']] == -1).all().all()
for ds, g in submission.groupby('dataset'):
    ids = set(g[g.row_type == 'node'].node_id); e = g[g.row_type == 'edge']
    assert (set(e.source_id) | set(e.target_id)).issubset(ids), f'dangling edge in {ds}'
    assert g[g.row_type == 'node'].node_id.is_unique, f'dup node_id in {ds}'

print(f"\nwrote submission.csv: {len(submission)} rows, {len(nodes)} nodes, {len(edges)} edges, checks passed ✅")
