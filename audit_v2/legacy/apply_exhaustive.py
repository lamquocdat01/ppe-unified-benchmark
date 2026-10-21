"""Apply the calibrated two-tier audit within datasets (across splits) and across datasets.

Exact tier      : same source-image ID (.rf. stem) or identical MD5.
Perceptual tier : calibrated union rule from results/audit_v2/calibration.json.

Usage: python -m audit_v2.apply ppeu sh17 A B C --cross ppeu:train=sh17
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import features, pairs
from .registry import RESULTS, has_rf, source_id


def thresholds():
    cal = json.loads((RESULTS / "calibration.json").read_text(encoding="utf-8"))
    return cal["union_rule"]["thresholds"]


def _radius(th):
    return dict(r_ph=th["ph"] if th.get("ph") is not None else -1,
                r_dh=th["dh"] if th.get("dh") is not None else -1,
                r_cos=th["cos"] if th.get("cos") is not None else 1.01)


def exact_keys(f):
    sid = np.array([source_id(r) for r in f["rel"]])
    rf = np.array([has_rf(r) for r in f["rel"]])
    # a stem only identifies a photo when the platform suffix is present
    sid = np.where(rf, sid, np.array([f"__{k}" for k in range(len(sid))]))
    return sid, f["md5"]


def exact_pairs(f):
    """All (i<j) pairs sharing source ID or MD5."""
    sid, md5 = exact_keys(f)
    out = set()
    for key in (sid, md5):
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        for gid in np.flatnonzero(cnt > 1):
            out.update(itertools.combinations(sorted(np.flatnonzero(inv == gid)), 2))
    if not out:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a = np.array(sorted(out), dtype=np.int64)
    return a[:, 0], a[:, 1]


def components(n, i, j):
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def audit_within(name, th):
    f = features.load(name)
    split = f["split"]
    n = len(split)
    ei, ej = exact_pairs(f)
    cand = pairs.candidates(f, **_radius(th))
    dm = pairs.is_dup(cand, th)
    pi, pj = cand["i"][dm], cand["j"][dm]
    key_e = set(zip(ei.tolist(), ej.tolist()))
    key_p = set(zip(pi.tolist(), pj.tolist()))
    rows = []
    split_names = sorted(set(split.tolist()), key=lambda s: ["train", "val", "test"].index(s)
                         if s in ("train", "val", "test") else 9)
    for s1, s2 in itertools.combinations(split_names, 2):
        sel = lambda ks: {(a, b) for a, b in ks if {split[a], split[b]} == {s1, s2}}
        E, P = sel(key_e), sel(key_p)
        U = E | P
        img2 = {b if split[b] == s2 else a for a, b in U}
        rows.append({"dataset": name, "split_pair": f"{s1}-{s2}", "exact_pairs": len(E),
                     "perceptual_pairs": len(P), "perceptual_only_pairs": len(P - E),
                     "exact_missed_by_perceptual": len(E - P), "union_pairs": len(U),
                     "leaked_images_in_second": len(img2)})
    # component view: groups (exact) vs components (exact + perceptual) spanning > 1 split
    allp = np.array(sorted(key_e | key_p), dtype=np.int64).reshape(-1, 2)
    comp_u = components(n, allp[:, 0], allp[:, 1]) if len(allp) else np.arange(n)
    ea = np.array(sorted(key_e), dtype=np.int64).reshape(-1, 2)
    comp_e = components(n, ea[:, 0], ea[:, 1]) if len(ea) else np.arange(n)

    def spanning(comp):
        df = pd.DataFrame({"c": comp, "s": split})
        k = df.groupby("c")["s"].nunique()
        return int((k > 1).sum()), int(df[df.c.isin(k[k > 1].index)].shape[0])

    se, se_img = spanning(comp_e)
    su, su_img = spanning(comp_u)
    summary = {"dataset": name, "images": n, "splits": {s: int((split == s).sum()) for s in split_names},
               "thresholds": th,
               "exact_cross_split_pairs": sum(r["exact_pairs"] for r in rows),
               "union_cross_split_pairs": sum(r["union_pairs"] for r in rows),
               "perceptual_only_cross_split_pairs": sum(r["perceptual_only_pairs"] for r in rows),
               "exact_components_spanning": se, "exact_images_in_spanning": se_img,
               "union_components_spanning": su, "union_images_in_spanning": su_img,
               "per_split_pair": rows}
    ex = summary["exact_cross_split_pairs"]
    summary["additional_pct_vs_exact"] = (100.0 * summary["perceptual_only_cross_split_pairs"] / ex) if ex else None
    # pair list for inspection / re-split
    cross = split[pi] != split[pj]
    pd.DataFrame({"a": f["rel"][pi[cross]], "a_split": split[pi[cross]],
                  "b": f["rel"][pj[cross]], "b_split": split[pj[cross]],
                  "ph": cand["ph"][dm][cross], "dh": cand["dh"][dm][cross], "cos": cand["cos"][dm][cross],
                  "exact": [(a, b) in key_e for a, b in zip(pi[cross].tolist(), pj[cross].tolist())]}
                 ).sort_values("cos", ascending=False).to_csv(RESULTS / f"pairs_{name}.csv", index=False)
    np.save(RESULTS / f"components_{name}.npy", comp_u)
    return summary


def audit_cross(spec, th):
    """spec 'X:split=Y' -> pairs between images of X in that split and all images of Y."""
    left, right = spec.split("=")
    xname, xsplit = (left.split(":") + [None])[:2]
    fx, fy = features.load(xname), features.load(right)
    if xsplit:
        m = fx["split"] == xsplit
        fx = {k: v[m] for k, v in fx.items()}
    cand = pairs.candidates(fx, fy, **_radius(th))
    dm = pairs.is_dup(cand, th)
    md5_y = set(fy["md5"].tolist())
    exact = sum(1 for h in fx["md5"] if h in md5_y)
    pd.DataFrame({"x": fx["rel"][cand["i"][dm]], "y": fy["rel"][cand["j"][dm]],
                  "ph": cand["ph"][dm], "dh": cand["dh"][dm], "cos": cand["cos"][dm]}
                 ).sort_values("cos", ascending=False).to_csv(RESULTS / f"pairs_cross_{xname}{('-' + xsplit) if xsplit else ''}_{right}.csv", index=False)
    return {"spec": spec, "x_images": int(len(fx["rel"])), "y_images": int(len(fy["rel"])),
            "exact_md5_matches": exact, "perceptual_pairs": int(dm.sum()),
            "x_images_matched": int(len(np.unique(cand["i"][dm]))),
            "y_images_matched": int(len(np.unique(cand["j"][dm])))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["ppeu", "sh17", "A", "B", "C"])
    ap.add_argument("--cross", nargs="*", default=["ppeu:train=sh17"])
    a = ap.parse_args()
    th = thresholds()
    res = {"thresholds": th, "within": {}, "cross": {}}
    for n in a.datasets:
        res["within"][n] = audit_within(n, th)
        s = res["within"][n]
        print(n, {k: s[k] for k in ("exact_cross_split_pairs", "union_cross_split_pairs",
                                    "perceptual_only_cross_split_pairs", "exact_components_spanning",
                                    "union_components_spanning")}, flush=True)
    for spec in a.cross:
        res["cross"][spec] = audit_cross(spec, th)
        print(spec, res["cross"][spec], flush=True)
    (RESULTS / "audit_v2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    pd.DataFrame([r for s in res["within"].values() for r in s["per_split_pair"]]).to_csv(
        RESULTS / "audit_v2_per_split_pair.csv", index=False)


if __name__ == "__main__":
    main()
