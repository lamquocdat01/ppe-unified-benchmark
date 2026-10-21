"""Calibrate copy-level near-duplicate thresholds (decision Q5, 21-10-2026).

  positives  every pair inside a Roboflow augmentation group (<stem>.rf.<hash>) of the
             calibration set, after dropping groups that contain a pair with cos < 0.5
             (stem reused for different photos, the defect found in B)
  negatives  ~200k random CROSS-SOURCE pairs (A<->C, C<->SH17, A<->SH17 by default): images
             from different datasets cannot be the same scene. Pairs sharing an MD5 or a
             .rf stem (A and C share upstream photos) are known duplicates and dropped.
Deciding metrics: cosine of DINOv2 embeddings (theta) and flip-invariant pHash Hamming (tau).
dHash is reported as a reference column only. Each threshold is the loosest one with
precision >= 0.99 on these pairs; recall is reported there. The same thresholds are then
checked on the held-out set (B; negatives B<->A, B<->SH17).

Usage: python -u -m audit_v2.calibrate [--cal C] [--holdout B]
"""
import argparse
import itertools
import json

import numpy as np

from . import features, pairs
from .registry import RESULTS, has_rf, source_id

TARGET_P = 0.99
BAD_GROUP_COS = 0.5
N_NEG = 200_000
SEED = 42
COS_T = np.round(np.arange(0.30, 1.0001, 0.0025), 4)
HAM_T = np.arange(0, 33)
NEG_PAIRS = {"C": [("A", "C"), ("C", "sh17"), ("A", "sh17")],
             "B": [("B", "A"), ("B", "sh17")]}


def groups_of(f):
    sids = np.array([source_id(r) if has_rf(r) else f"__{k}" for k, r in enumerate(f["rel"])])
    return sids, np.unique(sids, return_inverse=True)[1]


def group_pairs(g):
    order = np.argsort(g, kind="stable")
    bounds = np.flatnonzero(np.diff(g[order])) + 1
    I, J = [], []
    for members in np.split(order, bounds):
        for a, b in itertools.combinations(sorted(members), 2):
            I.append(a); J.append(b)
    return np.array(I, np.int64), np.array(J, np.int64)


def positives(name):
    f = features.load(name)
    sids, g = groups_of(f)
    pi, pj = group_pairs(g)
    ph, dh, cos = pairs.distances(f, f, pi, pj)
    bad = np.unique(g[pi[cos < BAD_GROUP_COS]])
    keep = ~np.isin(g[pi], bad)
    info = {"images": int(len(g)), "groups": int(g.max() + 1), "groups_multi": int((np.bincount(g) > 1).sum()),
            "positive_pairs_all": int(len(pi)), "groups_dropped_cos_lt_0.5": int(len(bad)),
            "images_in_dropped_groups": int(np.isin(g, bad).sum()), "positive_pairs": int(keep.sum())}
    return {"ph": ph[keep], "dh": dh[keep], "cos": cos[keep]}, info


def negatives(combos, n, rng):
    per = n // len(combos)
    out = {"ph": [], "dh": [], "cos": []}
    info = {}
    cache = {}
    for x, y in combos:
        for k in (x, y):
            if k not in cache:
                f = features.load(k)
                cache[k] = (f, groups_of(f)[0])
        (fx, sx), (fy, sy) = cache[x], cache[y]
        i = rng.integers(0, len(sx), per); j = rng.integers(0, len(sy), per)
        has_stem = ~np.char.startswith(sx[i].astype(str), "__")  # "__<k>" = no .rf stem, not an ID
        known = (fx["md5"][i] == fy["md5"][j]) | ((sx[i] == sy[j]) & has_stem)
        i, j = i[~known], j[~known]
        ph, dh, cos = pairs.distances(fx, fy, i, j)
        for k, v in zip(("ph", "dh", "cos"), (ph, dh, cos)):
            out[k].append(v)
        info[f"{x}-{y}"] = {"pairs": int(len(i)), "known_duplicates_dropped": int(known.sum()),
                            "cos_p99": float(np.percentile(cos, 99)), "cos_max": float(cos.max()),
                            "ph_min": int(ph.min())}
    return {k: np.concatenate(v) for k, v in out.items()}, info


def curve(pos, neg, thresholds, le):
    pos = np.sort(pos); neg = np.sort(neg)
    rows = []
    for t in thresholds:
        if le:
            tp = np.searchsorted(pos, t, "right"); fp = np.searchsorted(neg, t, "right")
        else:
            tp = len(pos) - np.searchsorted(pos, t, "left"); fp = len(neg) - np.searchsorted(neg, t, "left")
        rows.append({"t": float(t), "tp": int(tp), "fp": int(fp), "recall": tp / len(pos),
                     "precision": tp / (tp + fp) if tp + fp else 1.0, "fpr": fp / len(neg)})
    return rows


def pick(rows):
    ok = [r for r in rows if r["precision"] >= TARGET_P and r["tp"] > 0]
    return max(ok, key=lambda r: r["recall"]) if ok else None


def at(rows, t):
    return next(r for r in rows if abs(r["t"] - t) < 1e-9)


def union(pos, neg, th):
    tp = int(pairs.is_dup(pos, th).sum()); fp = int(pairs.is_dup(neg, th).sum())
    return {"tp": tp, "fp": fp, "recall": tp / len(pos["cos"]), "precision": tp / (tp + fp) if tp + fp else 1.0,
            "fpr": fp / len(neg["cos"])}


def within_set(name, others, rng, n_neg=N_NEG, n_x=50_000):
    """First calibration (20/21-10-2026), kept for comparison: all .rf pairs of the set as positives,
    random different-group pairs of the SAME set (+ cross-source pairs) as negatives. Same-set
    negatives contain unlabelled near-duplicates (video frames), so precision here is a lower bound."""
    f = features.load(name)
    sids, g = groups_of(f)
    pi, pj = group_pairs(g)
    i = rng.integers(0, len(g), n_neg * 2); j = rng.integers(0, len(g), n_neg * 2)
    m = g[i] != g[j]
    i, j = i[m][:n_neg], j[m][:n_neg]
    pos = dict(zip(("ph", "dh", "cos"), pairs.distances(f, f, pi, pj)))
    neg = dict(zip(("ph", "dh", "cos"), pairs.distances(f, f, i, j)))
    xneg, _ = negatives([(name, o) for o in others], n_x, rng)
    neg = {k: np.concatenate([neg[k], xneg[k]]) for k in neg}
    return {"positives": int(len(pi)), "negatives_same_set": int(len(i)), "negatives_cross_source": int(len(xneg["cos"])),
            "curves": {"cos": curve(pos["cos"], neg["cos"], COS_T, le=False),
                       "ph": curve(pos["ph"], neg["ph"], HAM_T, le=True)}}


OPERATING_POINT = {"cos": 0.98, "phash": 0, "topk": 5, "scene_k": 2,
                   "role": "copy-level (2a) operating point, precision-conservative: theta 0.98 reaches P >= 0.99 "
                           "on held-out B with same-set negatives (within_set_holdout); the cross-source "
                           "calibration (theta 0.485, tau 16) merges 99.98% of PPE-Unified into one group "
                           "(see threshold_sensitivity). Scene-level duplicates are handled by 2b "
                           "(file-name sequence, frame numbers within +-2), not by a threshold.",
                   "decided": "2026-10-21, option A"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default="C")
    ap.add_argument("--holdout", default="B")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {"definition": "copy-level: exact OR cos >= theta OR pHash <= tau (dHash reference only)",
           "target_precision": TARGET_P, "calibration_set": a.cal, "holdout_set": a.holdout, "seed": SEED,
           "positives": f".rf groups of the set, groups with any pair cos < {BAD_GROUP_COS} dropped",
           "negatives": "random cross-source pairs; MD5/stem-identical pairs dropped", "sets": {}}
    data = {}
    for name in (a.cal, a.holdout):
        pos, pinfo = positives(name)
        neg, ninfo = negatives(NEG_PAIRS[name], N_NEG, rng)
        data[name] = (pos, neg)
        out["sets"][name] = {"positives": pinfo, "negatives": ninfo,
                             "curves": {"cos": curve(pos["cos"], neg["cos"], COS_T, le=False),
                                        "ph": curve(pos["ph"], neg["ph"], HAM_T, le=True),
                                        "dh": curve(pos["dh"], neg["dh"], HAM_T, le=True)}}
        print(name, json.dumps({"positives": pinfo, "negatives": ninfo}), flush=True)
    cal, hold = out["sets"][a.cal]["curves"], out["sets"][a.holdout]["curves"]
    chosen = {}
    for k in ("cos", "ph", "dh"):
        r = pick(cal[k])
        chosen[k] = None if r is None else {
            "threshold": r["t"], "role": "reference" if k == "dh" else "decision",
            "calibration": {x: r[x] for x in ("precision", "recall", "fpr", "tp", "fp")},
            "holdout": {x: at(hold[k], r["t"])[x] for x in ("precision", "recall", "fpr", "tp", "fp")}}
    out["thresholds"] = chosen
    th = {"cos": chosen["cos"]["threshold"] if chosen["cos"] else None,
          "ph": int(chosen["ph"]["threshold"]) if chosen["ph"] else None, "dh": None}
    out["copy_level_rule"] = {"thresholds": th, "calibration": union(*data[a.cal], th),
                              "holdout": union(*data[a.holdout], th)}
    ws = within_set(a.holdout, ["A", "sh17"], np.random.default_rng(SEED))
    r = at(ws["curves"]["cos"], OPERATING_POINT["cos"])
    ws["at_operating_point"] = {"cos": OPERATING_POINT["cos"], **{x: r[x] for x in ("precision", "recall", "tp", "fp")}}
    out["within_set_holdout"] = ws
    out["operating_point"] = OPERATING_POINT
    sens = RESULTS / "resplit_v11"
    out["threshold_sensitivity"] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(sens.glob("resplit_*.json"))]
    for t in out["threshold_sensitivity"]:
        t.pop("class_ratio_pct", None)
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez_compressed(RESULTS / "calibration_distances.npz",
                        **{f"{s}_{grp}_{k}": v for s, (p, n) in data.items()
                           for grp, dd in (("pos", p), ("neg", n)) for k, v in dd.items()})
    print(json.dumps({"thresholds": chosen, "copy_level_rule": out["copy_level_rule"]}, indent=1))


if __name__ == "__main__":
    main()
