"""Group-aware stratified re-split of PPE-Unified with near-duplicate groups (v1.1, decision Q6).

Groups = connected components of
  exact      same source_image_id (manifest) or same MD5
  copy (2a)  cos >= theta for ANY pair (exhaustive; a top-k list misses edges in dense
             frame clusters), or flip-invariant pHash <= tau
  scene (2b) same file-name family and frame numbers within +-k (no threshold)
The split itself is the unchanged greedy iterative stratification + repair of
Final Project/Data Set_scripts/stratified_resplit.py (seed 42); only the group key changes.

Usage:
  python -u -m audit_v2.resplit --theta 0.95 --tau 4 --k 2 [--write PPE_SH17/PPE_Unified_v1.1]
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import features
from .audit import exact_keys, exact_pairs, hamming_pairs, range_pairs, scene_edges
from .pairs import PH, PHM, ham_flip
from .registry import DATASETS_ROOT, REGISTRY, RESULTS

ALGO_DIR = Path(os.environ.get("PPE_SCRIPTS", Path(__file__).resolve().parent.parent / "scripts"))


def near_dup_groups(name, theta, tau, k):
    f = features.load(name)
    n = len(f["rel"])
    sid, md5 = exact_keys(name, f)
    E = {"exact": exact_pairs((sid, md5), (sid, md5), same=True),
         "copy_cos": range_pairs(f["emb"], theta)}
    hi, hj = hamming_pairs(f["hashes"], f["hashes"], tau, -1, same=True)
    m = ham_flip(f["hashes"], f["hashes"], hi, hj, PH, PHM) <= tau
    E["copy_phash"] = (hi[m], hj[m])
    E["scene"] = scene_edges(f["rel"], k)
    I = np.concatenate([e[0] for e in E.values()]); J = np.concatenate([e[1] for e in E.values()])
    comp = connected_components(coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)), directed=False)[1]
    files = [r.split("/")[-1] for r in f["rel"]]
    return f, files, comp, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ppeu")
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--tau", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--write", default=None, help="destination relative to THS_DATASETS (hardlinks)")
    ap.add_argument("--algo", choices=["bg", "v1"], default="bg",
                    help="bg: background-aware extension (stratify_bg, reproduces v1 on singleton groups); "
                         "v1: the unchanged v1 algorithm")
    a = ap.parse_args()
    tag = a.tag or f"t{a.theta}_p{a.tau}_k{a.k}"
    sys.path.insert(0, str(ALGO_DIR))
    import stratified_resplit as sr  # noqa: E402 - the unchanged v1 algorithm
    import verify_dataset as vd  # noqa: E402

    src = DATASETS_ROOT / REGISTRY[a.name][0]
    f, files, comp, E = near_dup_groups(a.name, a.theta, a.tau, a.k)
    comp_of = dict(zip(files, comp))
    class_names = sr.load_class_names(src)
    nc = len(class_names)
    req = sr.min_req_table(class_names)
    man, file_info, boxes_per_file, groups_v1 = sr.collect(src, class_names)

    groups = {}
    for fn in man["file"]:
        if fn not in file_info:
            continue
        gid = f"g{comp_of[fn]:06d}"
        g = groups.setdefault(gid, {"files": [], "boxes": defaultdict(int), "n_img": 0, "old_split": None})
        g["files"].append(fn); g["n_img"] += 1
        for c, nb in boxes_per_file[fn].items():
            g["boxes"][c] += nb
        g["old_split"] = file_info[fn]["split"]
    for g in groups.values():
        g["boxes"] = dict(g["boxes"])

    n_img = sum(g["n_img"] for g in groups.values())
    quota = {"train": int(round(n_img * sr.TARGET_RATIO["train"])), "val": int(round(n_img * sr.TARGET_RATIO["val"]))}
    quota["test"] = n_img - quota["train"] - quota["val"]
    bg_steps = None
    if a.algo == "bg":
        from . import stratify_bg
        r = stratify_bg.split(groups, boxes_per_file, class_names)
        assign, box, img, bg, det, quota, bg_steps = (r["assign"], r["box"], r["img"], r["bg"], r["detail"],
                                                      r["quota"], r["bg_balance_steps"])
    else:
        rng = random.Random(sr.SEED)
        assign, totals, _, n_bg = sr.greedy_stratify(groups, nc, class_names, quota, req, rng, verbose=False)
        assign, box, img, bg, cur, det = sr.repair(assign, groups, nc, class_names, totals, quota, req, rng,
                                                   n_bg=n_bg, verbose=False)

    # v1 reference numbers from the current split
    box1 = {s: Counter() for s in sr.SPLITS}
    img1 = Counter()
    for fn, info in file_info.items():
        img1[info["split"]] += 1
        for c, nb in boxes_per_file[fn].items():
            box1[info["split"]][c] += nb

    def ratios(bx):
        out = {}
        for cid, cn in enumerate(class_names):
            t = sum(bx[s][cid] for s in sr.SPLITS)
            out[cn] = {s: (100.0 * bx[s][cid] / t if t else 0.0) for s in sr.SPLITS}
        return out

    r1, r2 = ratios(box1), ratios(box)
    dev = lambda r: max(abs(v[s] - 100 * sr.TARGET_RATIO[s]) for cn, v in r.items() if cn not in sr.EXEMPT_CLASSES
                        for s in sr.SPLITS)
    sizes = Counter(g["n_img"] for g in groups.values())
    sz = np.array(sorted((g["n_img"] for g in groups.values()), reverse=True))
    file_split = {fn: assign[gid] for gid, g in groups.items() for fn in g["files"]}
    split_arr = np.array([file_split[fn] for fn in files])
    leak = {kname: int((split_arr[e[0]] != split_arr[e[1]]).sum()) for kname, e in E.items()}
    min_viol = [f"{class_names[c]}@{s}: {box[s][c]} < {sr.minimum_for(c, s, req)}"
                for c in range(nc) for s in ("val", "test")
                if sr.minimum_for(c, s, req) and box[s][c] < sr.minimum_for(c, s, req)]
    res = {"tag": tag, "theta": a.theta, "tau_phash": a.tau, "scene_k": a.k,
           "edges": {kname: int(len(e[0])) for kname, e in E.items()},
           "groups_v1": len(groups_v1), "groups_v11": len(groups),
           "group_size_hist": {str(k): v for k, v in sorted(sizes.items())},
           "largest_groups": sz[:10].tolist(), "largest_pct": round(100 * sz[0] / n_img, 2),
           "images_v1": dict(img1), "images_v11": dict(img), "quota": quota,
           "image_ratio_v11_pct": {s: round(100 * img[s] / n_img, 2) for s in sr.SPLITS},
           "max_class_ratio_dev_pp": {"v1": round(dev(r1), 2), "v11": round(dev(r2), 2)},
           "class_ratio_pct": {cn: {"v1": {s: round(r1[cn][s], 1) for s in sr.SPLITS},
                                    "v11": {s: round(r2[cn][s], 1) for s in sr.SPLITS}} for cn in class_names},
           "min_support_violations": min_viol, "penalty": det,
           "cross_split_edges_after": leak,
           "background_per_split": dict(bg), "algorithm": a.algo, "bg_balance_steps": bg_steps,
           "source": a.name}
    out_dir = RESULTS / "resplit_v11"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"resplit_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    pd.DataFrame({"file": files, "group_v11": [f"g{c:06d}" for c in comp], "split_v1": f["split"],
                  "split_v11": split_arr}).to_csv(out_dir / f"assignment_{tag}.csv", index=False)
    print(json.dumps({k: res[k] for k in ("tag", "groups_v11", "largest_groups", "largest_pct", "images_v11",
                                          "max_class_ratio_dev_pp", "min_support_violations",
                                          "cross_split_edges_after")}), flush=True)
    if a.write:
        if min_viol or any(leak.values()):
            raise SystemExit("not writing: minimum-support violations or cross-split edges remain")
        dst = DATASETS_ROOT / a.write
        sr.materialize(src, dst, assign, groups, file_info, man, mode="hardlink", normalize_size=None)
        for extra in ("FIXES.csv",):
            if (src / extra).exists():
                shutil.copy2(src / extra, dst / extra)
        man2 = pd.read_csv(dst / "manifest.csv", encoding="utf-8")
        man2["group_v11"] = man2["file"].map(dict(zip(files, [f"g{c:06d}" for c in comp])))
        man2.to_csv(dst / "manifest.csv", index=False, encoding="utf-8")
        print(f"written {dst}")


if __name__ == "__main__":
    main()
