"""Calibrate perceptual thresholds on augmentation groups with known identity.

Ground truth: Roboflow export names <stem>.rf.<hash> -> files sharing a stem are
augmented copies of one photograph (positives); pairs from different stems are
negatives. Two negative regimes are reported:
  exhaustive - every cross-group pair in the dataset (the regime an audit runs in;
               any genuinely duplicated photo uploaded under two stems counts as FP,
               so this precision is a lower bound)
  random     - uniformly sampled cross-group pairs, balanced 1:1 against positives

Thresholds are chosen on the calibration set (C) at exhaustive precision >= 0.99 and
checked on the held-out set (B).

Usage: python -m audit_v2.calibrate [--cal C] [--holdout B]
"""
import argparse
import itertools
import json

import numpy as np

from . import features, pairs
from .registry import RESULTS, source_id

TARGET_P = 0.99
PH_T = np.arange(0, 17)
DH_T = np.arange(0, 17)
COS_T = np.round(np.arange(0.80, 1.0001, 0.005), 3)
N_RANDOM = 200_000
SEED = 42


def groups_of(f):
    sids = np.array([source_id(r) for r in f["rel"]])
    _, g = np.unique(sids, return_inverse=True)
    return g


def positive_pairs(g):
    order = np.argsort(g, kind="stable")
    bounds = np.flatnonzero(np.diff(g[order])) + 1
    I, J = [], []
    for members in np.split(order, bounds):
        for a, b in itertools.combinations(sorted(members), 2):
            I.append(a); J.append(b)
    return np.array(I, dtype=np.int64), np.array(J, dtype=np.int64)


def curve(pos_d, neg_d, n_rand_d, thresholds, le=True):
    """Return per-threshold TP, FP (exhaustive), recall, precision, FPR (random), balanced precision."""
    rows = []
    for t in thresholds:
        if le:
            tp = int((pos_d <= t).sum()); fp = int((neg_d <= t).sum()); fr = float((n_rand_d <= t).mean())
        else:
            tp = int((pos_d >= t).sum()); fp = int((neg_d >= t).sum()); fr = float((n_rand_d >= t).mean())
        rec = tp / len(pos_d)
        rows.append({"t": float(t), "tp": tp, "fp": fp, "recall": rec,
                     "precision": tp / (tp + fp) if tp + fp else 1.0,
                     "fpr_random": fr, "precision_balanced": rec / (rec + fr) if rec + fr else 1.0})
    return rows


def pick(rows, le=True):
    ok = [r for r in rows if r["precision"] >= TARGET_P and r["tp"] > 0]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["recall"], -r["t"] if le else r["t"]))


def evaluate(name):
    f = features.load(name)
    g = groups_of(f)
    pi, pj = positive_pairs(g)
    pos = dict(zip(("ph", "dh", "cos"), pairs.distances(f, f, pi, pj)))
    cand = pairs.candidates(f, r_ph=int(PH_T.max()), r_dh=int(DH_T.max()), r_cos=float(COS_T.min()))
    negm = g[cand["i"]] != g[cand["j"]]
    md5_same = f["md5"][cand["i"]] == f["md5"][cand["j"]]
    neg = {k: cand[k][negm] for k in ("ph", "dh", "cos")}
    rng = np.random.default_rng(SEED)
    ri = rng.integers(0, len(g), N_RANDOM * 2); rj = rng.integers(0, len(g), N_RANDOM * 2)
    m = g[ri] != g[rj]
    ri, rj = ri[m][:N_RANDOM], rj[m][:N_RANDOM]
    rnd = dict(zip(("ph", "dh", "cos"), pairs.distances(f, f, ri, rj)))
    n = len(g)
    info = {"dataset": name, "images": int(n), "groups": int(g.max() + 1),
            "groups_multi": int((np.bincount(g) > 1).sum()),
            "positive_pairs": int(len(pi)), "negative_pairs_total": int(n * (n - 1) // 2 - len(pi)),
            "negative_candidates": int(negm.sum()),
            "negative_md5_identical": int((negm & md5_same).sum()), "random_negatives": int(len(ri))}
    curves = {"ph": curve(pos["ph"], neg["ph"], rnd["ph"], PH_T),
              "dh": curve(pos["dh"], neg["dh"], rnd["dh"], DH_T),
              "cos": curve(pos["cos"], neg["cos"], rnd["cos"], COS_T, le=False)}
    return f, g, pos, cand, negm, curves, info


def union_metrics(pos, cand, negm, th):
    tp = int(pairs.is_dup(pos, th).sum())
    fp = int((pairs.is_dup(cand, th) & negm).sum())
    return {"tp": tp, "fp": fp, "recall": tp / len(pos["ph"]), "precision": tp / (tp + fp) if tp + fp else 1.0}


def grid_union(pos, cand, negm):
    """Best union rule (max recall at exhaustive precision >= TARGET_P) on a coarse grid."""
    best = None
    for tp_ in list(range(0, 17, 2)) + [None]:
        for td in list(range(0, 17, 2)) + [None]:
            for tc in [round(x, 2) for x in np.arange(0.85, 1.0, 0.01)] + [None]:
                th = {"ph": tp_, "dh": td, "cos": tc}
                if all(v is None for v in th.values()):
                    continue
                mtr = union_metrics(pos, cand, negm, th)
                if mtr["precision"] >= TARGET_P and (best is None or mtr["recall"] > best[1]["recall"]):
                    best = (th, mtr)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default="C")
    ap.add_argument("--holdout", default="B")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"target_precision": TARGET_P, "calibration_set": a.cal, "holdout_set": a.holdout, "sets": {}}
    ev = {}
    for name in (a.cal, a.holdout):
        f, g, pos, cand, negm, curves, info = evaluate(name)
        ev[name] = (pos, cand, negm)
        out["sets"][name] = {"info": info, "curves": curves}
        print(name, json.dumps(info), flush=True)
    cal = out["sets"][a.cal]["curves"]
    single = {"ph": pick(cal["ph"]), "dh": pick(cal["dh"]), "cos": pick(cal["cos"], le=False)}
    out["single_metric"] = {k: (v and {"threshold": v["t"], "precision": v["precision"], "recall": v["recall"]})
                            for k, v in single.items()}
    best = grid_union(*ev[a.cal])
    th = best[0] if best else {k: (single[k]["t"] if single[k] else None) for k in single}
    out["union_rule"] = {"thresholds": th,
                         "calibration": union_metrics(*ev[a.cal], th),
                         "holdout": union_metrics(*ev[a.holdout], th)}
    for k, v in single.items():
        if v:
            hold = next(r for r in out["sets"][a.holdout]["curves"][k] if r["t"] == v["t"])
            out["single_metric"][k]["holdout"] = {"precision": hold["precision"], "recall": hold["recall"]}
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"single": out["single_metric"], "union": out["union_rule"]}, indent=1))


if __name__ == "__main__":
    main()
