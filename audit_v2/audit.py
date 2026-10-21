"""Two-level cross-split audit (decision A, 21-10-2026).

Exact      same source-image ID (.rf stem / manifest) or same MD5.
Copy (2a)  cos >= theta among FAISS top-k neighbours (queried in both directions), or
           flip-invariant pHash <= tau (exhaustive, numpy blocks of 2,000 rows). The
           operating point is theta = 0.98, tau = 0 (precision-conservative); 0.95 and 0.90
           are reported as sensitivity with the same tau.
Scene (2b) same file-name family and frame numbers within +-k (k = 2); no threshold.
Groups are connected components of all edges, within-split edges included (a chain
a1~a2 inside train with a2~b in val puts a1, a2, b in one group).

Usage: python -u -m audit_v2.audit ppeu ppeu11 sh17 A B C --cross ppeu:train=sh17:val
"""
import argparse
import itertools
import json
import re

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import features, pairs
from .features import wait_for_ram
from .registry import DATASETS_ROOT, REGISTRY, RESULTS, has_rf, source_id

TOPK = 5
BLOCK = 2000
COL_CHUNK = 8192
TAU_PH = 0
SCENE_K = 2
POINTS = {"t098": {"cos": 0.98, "ph": TAU_PH}, "t095": {"cos": 0.95, "ph": TAU_PH}, "t090": {"cos": 0.90, "ph": TAU_PH}}
OPERATING = "t098"
_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
_SEQ = re.compile(r"^(.*?)(\d+)((?:_(?:jpe?g|png))+)?$", re.I)


def popcount64(x):
    return _POP16[x.view(np.uint16)].reshape(*x.shape, 4).sum(axis=-1, dtype=np.int16)


def ham_block(ha, hb, col, colm):
    """Flip-invariant Hamming matrix between hash rows ha (n,4) and hb (m,4)."""
    a, am = ha[:, col][:, None], ha[:, colm][:, None]
    b, bm = hb[:, col][None, :], hb[:, colm][None, :]
    d = popcount64(a ^ b)
    np.minimum(d, popcount64(am ^ b), out=d)
    np.minimum(d, popcount64(a ^ bm), out=d)
    return d


def hamming_pairs(ha, hb, r_ph, r_dh=-1, same=False):
    """All (i, j) with ph <= r_ph or dh <= r_dh; same=True keeps i < j only."""
    I, J = [], []
    for r0 in range(0, len(ha), BLOCK):
        qa = ha[r0:r0 + BLOCK]
        for c0 in range(r0 if same else 0, len(hb), COL_CHUNK):
            qb = hb[c0:c0 + COL_CHUNK]
            m = ham_block(qa, qb, pairs.PH, pairs.PHM) <= r_ph
            if r_dh >= 0:
                m |= ham_block(qa, qb, pairs.DH, pairs.DHM) <= r_dh
            if same:
                m &= (np.arange(r0, r0 + len(qa))[:, None] < np.arange(c0, c0 + len(qb))[None, :])
            i, j = np.nonzero(m)
            I.append(i + r0); J.append(j + c0)
    return np.concatenate(I).astype(np.int64), np.concatenate(J).astype(np.int64)


def knn_pairs(ea, eb, k, thr, same=False):
    """top-k cosine neighbours in both directions, cos >= thr."""
    out = []
    for q, x, flip in ((ea, eb, False), (eb, ea, True)):
        idx = faiss.IndexFlatIP(x.shape[1])
        idx.add(np.ascontiguousarray(x, dtype=np.float32))
        kk = min(k + (1 if same else 0), len(x))
        D, L = idx.search(np.ascontiguousarray(q, dtype=np.float32), kk)
        qi = np.repeat(np.arange(len(q)), kk)
        m = (D.ravel() >= thr) & (L.ravel() >= 0)
        a, b = qi[m], L.ravel()[m].astype(np.int64)
        out.append((b, a) if flip else (a, b))
        if same:
            break  # symmetric
    i = np.concatenate([o[0] for o in out]); j = np.concatenate([o[1] for o in out])
    if same:
        keep = i != j
        i, j = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
    return i, j


def range_pairs(emb, thr, block=2000):
    """ALL pairs i < j with cos >= thr (exhaustive, blockwise inner products) - used for grouping,
    where a top-k list can miss edges inside dense clusters of near-identical frames."""
    I, J = [], []
    for r0 in range(0, len(emb), block):
        S = emb[r0:r0 + block] @ emb.T
        i, j = np.nonzero(S >= thr)
        i += r0
        m = i < j
        I.append(i[m]); J.append(j[m])
    return np.concatenate(I).astype(np.int64), np.concatenate(J).astype(np.int64)


def scene_stem(rel):
    name = rel.split("/")[-1]
    return source_id(name) if has_rf(name) else name.rsplit(".", 1)[0]


def scene_edges(rel, k=SCENE_K):
    """2b: same file-name family (stem minus trailing frame number) and |number difference| <= k; i < j."""
    fam, num = [], []
    for i, r in enumerate(rel):
        m = _SEQ.match(scene_stem(r))
        fam.append(m.group(1) if m else f"__{i}")
        num.append(int(m.group(2)) if m else 0)
    df = pd.DataFrame({"f": fam, "n": num, "i": np.arange(len(rel))}).sort_values(["f", "n", "i"])
    I, J = [], []
    for fm, g in df.groupby("f"):
        if fm.startswith("__") or k <= 0:
            continue
        a, idx = g.n.to_numpy(), g.i.to_numpy()
        for d in range(1, len(a)):
            m = (a[d:] - a[:-d]) <= k
            if not m.any():
                break
            I.append(idx[:-d][m]); J.append(idx[d:][m])
    if not I:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    i, j = np.concatenate(I), np.concatenate(J)
    return np.minimum(i, j), np.maximum(i, j)


def exact_keys(name, f):
    if REGISTRY[name][0].endswith(("PPE_Unified_v1_resplit", "PPE_Unified_v1.1", "PPE_Unified_v1fix")):
        man = pd.read_csv(DATASETS_ROOT / REGISTRY[name][0] / "manifest.csv").set_index("file")
        sid = man.loc[[r.split("/")[-1] for r in f["rel"]], "source_image_id"].to_numpy()
    else:
        sid = np.array([source_id(r) if has_rf(r) else f"__{k}" for k, r in enumerate(f["rel"])])
    return sid.astype(str), f["md5"]


def exact_pairs(ka, kb, same=False):
    """pairs sharing any key; ka/kb: tuples of key arrays."""
    out = set()
    for a, b in zip(ka, kb):
        pos = {}
        for j, v in enumerate(b):
            pos.setdefault(v, []).append(j)
        for i, v in enumerate(a):
            for j in pos.get(v, ()):
                if not same or i < j:
                    out.add((i, j))
    arr = np.array(sorted(out), dtype=np.int64).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


def edges(fa, fb, ka, kb, same=False, scene=None):
    """Candidate edges between fa and fb at the loosest point, with distances and tier flags.

    scene: optional (i, j) index arrays (local to fa/fb) of 2b edges.
    """
    th_min = min(p["cos"] for p in POINTS.values())
    ki, kj = knn_pairs(fa["emb"], fb["emb"], TOPK, th_min, same)
    hi, hj = hamming_pairs(fa["hashes"], fb["hashes"], max(p["ph"] for p in POINTS.values()), -1, same)
    ei, ej = exact_pairs(ka, kb, same)
    si, sj = scene if scene is not None else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    nb = np.int64(len(fb["emb"]))
    kkey, ekey, skey = ki * nb + kj, ei * nb + ej, si * nb + sj
    key = np.unique(np.concatenate([kkey, hi * nb + hj, ekey, skey]))
    i, j = key // nb, key % nb
    ph, dh, cos = pairs.distances(fa, fb, i, j)
    return pd.DataFrame({"i": i, "j": j, "ph": ph, "dh": dh, "cos": cos, "in_knn": np.isin(key, kkey),
                         "exact": np.isin(key, ekey), "scene": np.isin(key, skey)})


def flags(e, p):
    cos_t = e["in_knn"] & (e["cos"] >= p["cos"])
    ham_t = e["ph"] <= p["ph"]
    return cos_t, ham_t, cos_t | ham_t


def summarise(e, p):
    cos_t, ham_t, copy = flags(e, p)
    ex, sc = e["exact"], e["scene"]
    return {"exact": int(ex.sum()), "copy_2a": int(copy.sum()), "cos_tier": int(cos_t.sum()),
            "phash_tier": int(ham_t.sum()), "both_exact_2a": int((ex & copy).sum()),
            "exact_only": int((ex & ~copy).sum()), "copy_only": int((copy & ~ex).sum()),
            "scene_2b": int(sc.sum()), "scene_only": int((sc & ~ex & ~copy).sum()),
            "union_exact_2a": int((ex | copy).sum()), "union_all": int((ex | copy | sc).sum())}


def sub(f, m):
    return {k: v[m] for k, v in f.items()}


def _components(n, e, mask):
    m = mask.to_numpy()
    return connected_components(coo_matrix((np.ones(m.sum()), (e["i"][m], e["j"][m])), shape=(n, n)),
                                directed=False)[1]


def _spanning(comp, split):
    df = pd.DataFrame({"c": comp, "s": split})
    nsp = df.groupby("c")["s"].nunique()
    span = nsp[nsp > 1].index
    inspan = df.c.isin(span)
    return {"groups": int(comp.max() + 1), "spanning": int(len(span)), "images_in_spanning": int(inspan.sum()),
            "largest_spanning": int(df[inspan].c.value_counts().max()) if len(span) else 0}


def audit_dataset(name):
    wait_for_ram()
    f = features.load(name)
    split = f["split"]
    n = len(split)
    sid, md5 = exact_keys(name, f)
    si_all, sj_all = scene_edges(f["rel"])
    names = [s for s in ("train", "val", "test") if s in set(split)]
    idx = {s: np.flatnonzero(split == s) for s in names}
    local = {s: {g: k for k, g in enumerate(idx[s])} for s in names}

    def scene_local(s1, s2):
        a, b = local[s1], local[s2]
        I, J = [], []
        for x, y in zip(si_all.tolist(), sj_all.tolist()):
            if x in a and y in b:
                I.append(a[x]); J.append(b[y])
            elif y in a and x in b:
                I.append(a[y]); J.append(b[x])
        I, J = np.array(I, np.int64), np.array(J, np.int64)
        if s1 == s2:
            I, J = np.minimum(I, J), np.maximum(I, J)
        return I, J

    cross, within = {}, {}
    for s1, s2 in itertools.combinations(names, 2):
        a, b = idx[s1], idx[s2]
        e = edges(sub(f, a), sub(f, b), (sid[a], md5[a]), (sid[b], md5[b]), scene=scene_local(s1, s2))
        e["i"], e["j"] = a[e["i"]], b[e["j"]]
        cross[f"{s1}-{s2}"] = e
        print(f"  {name} {s1}-{s2}: {len(e)} candidate edges", flush=True)
    for s in names:
        a = idx[s]
        e = edges(sub(f, a), sub(f, a), (sid[a], md5[a]), (sid[a], md5[a]), same=True, scene=scene_local(s, s))
        e["i"], e["j"] = a[e["i"]], a[e["j"]]
        within[s] = e
    all_e = pd.concat(list(cross.values()) + list(within.values()), ignore_index=True)
    res = {"dataset": name, "images": int(n), "splits": {s: int(len(idx[s])) for s in names},
           "scene_k": SCENE_K, "points": {}}
    span_exact = _spanning(_components(n, all_e, all_e["exact"]), split)
    span_scene = _spanning(_components(n, all_e, all_e["exact"] | all_e["scene"]), split)
    for pname, p in POINTS.items():
        rows = {sp: summarise(e, p) for sp, e in cross.items()}
        for sp, e in cross.items():
            _, _, copy = flags(e, p)
            rows[sp]["leaked_images_2a"] = int(e.loc[copy | e["exact"], "j"].nunique())
            rows[sp]["leaked_images_all"] = int(e.loc[copy | e["exact"] | e["scene"], "j"].nunique())
        _, _, copy = flags(all_e, p)
        comp_2a = _components(n, all_e, all_e["exact"] | copy)
        comp_all = _components(n, all_e, all_e["exact"] | copy | all_e["scene"])
        res["points"][pname] = {
            "thresholds": p, "per_split_pair": rows,
            "cross_split": {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))} if rows else {},
            "groups_exact": span_exact, "groups_exact_2a": _spanning(comp_2a, split),
            "groups_exact_2b": span_scene, "groups_all": _spanning(comp_all, split)}
        np.save(RESULTS / f"components_{name}_{pname}.npy", comp_all)
    if cross:
        ce = pd.concat([e.assign(split_pair=sp) for sp, e in cross.items()])
        ce = ce.assign(a=f["rel"][ce["i"]], b=f["rel"][ce["j"]])
        ce.drop(columns=["i", "j"]).sort_values("cos", ascending=False).to_csv(
            RESULTS / f"edges_cross_split_{name}.csv", index=False)
    return res


def audit_cross(spec):
    """'X:sx=Y:sy' -> edges between split sx of X and split sy of Y (exact tier = MD5; 2b not applicable)."""
    left, right = spec.split("=")
    (xn, xs), (yn, ys) = left.split(":"), right.split(":")
    fx, fy = features.load(xn), features.load(yn)
    fx, fy = sub(fx, fx["split"] == xs), sub(fy, fy["split"] == ys)
    e = edges(fx, fy, (fx["md5"],), (fy["md5"],))
    res = {"spec": spec, "x_images": int(len(fx["rel"])), "y_images": int(len(fy["rel"])), "points": {}}
    for pname, p in POINTS.items():
        s = summarise(e, p)
        _, _, copy = flags(e, p)
        m = copy | e["exact"]
        s.update(x_images_matched=int(e.loc[m, "i"].nunique()), y_images_matched=int(e.loc[m, "j"].nunique()))
        res["points"][pname] = s
    e.assign(a=fx["rel"][e["i"]], b=fy["rel"][e["j"]]).drop(columns=["i", "j"]).sort_values(
        "cos", ascending=False).to_csv(RESULTS / f"edges_cross_{xn}-{xs}_{yn}-{ys}.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["ppeu", "ppeu11", "sh17", "A", "B", "C"])
    ap.add_argument("--cross", nargs="*", default=["ppeu:train=sh17:val", "ppeu11:train=sh17:val"])
    ap.add_argument("--out", default="audit_v2.json")
    a = ap.parse_args()
    out = {"topk": TOPK, "points": POINTS, "operating_point": OPERATING, "scene_k": SCENE_K,
           "within": {}, "cross": {}}
    for n in a.datasets:
        print(f"[audit] {n}", flush=True)
        out["within"][n] = audit_dataset(n)
        print(json.dumps({k: v["cross_split"] for k, v in out["within"][n]["points"].items()}), flush=True)
    for spec in a.cross:
        print(f"[audit] cross {spec}", flush=True)
        out["cross"][spec] = audit_cross(spec)
        print(json.dumps(out["cross"][spec]["points"]), flush=True)
    (RESULTS / a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
