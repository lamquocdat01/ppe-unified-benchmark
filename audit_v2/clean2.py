"""T4(2): "clean2" re-split of the leaky source datasets B and C for the E2 comparison.

The split procedure is E2's, unchanged (notebooks/E2_LeakyVsClean_YOLO26s.ipynb, cell 6): sorted group
keys, random.Random(42).shuffle, then groups are filled into train / valid / test until 70 % / 90 % of
the image count is reached. Only the group key changes:
  clean   (E2)  Roboflow source stem (basename before ".rf.")
  clean2        exact (.rf stem + MD5) U 2a (any pair cos >= 0.98, or flip-invariant pHash = 0) U 2b (k = 2)
A clean2 group is keyed by its smallest E2 key, so the key order does not depend on feature-cache order.

Arms (decision (a), 21-10-2026):
  original              the authors' Roboflow split
  clean_exact           E2 key, E2 procedure (the published E2 "clean" arm)
  clean_exact_balanced  E2 key, group-aware class-balanced split (stratify_bg, seed 42, 70/20/10)
  clean2                clean2 key, stratify_bg
  clean2_e2             clean2 key, E2 procedure (first attempt, collapsed; kept for the record)
stratify_bg runs unchanged except for the minimum-support table: the PPE-Unified floors (val 150 /
test 100 boxes) are infeasible for B (no-boots has 163 boxes in total), so B and C use req = {}
(no absolute floor; the +-5 pp class-ratio and background penalties still apply). The min-support floor is a
PPE-Unified property only; validator point 7 is N/A for every B/C arm (schemas set no min_support).

Usage:
  python -u -m audit_v2.clean2 [--names B C] [--materialize]
Outputs results/audit_v2/clean2/{summary.json, manifest_<name>.csv}; --materialize hard-links each split
to THS_DATASETS/_derived/ppe_unified/clean2/<name> (plus manifest.csv) for the validator.
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .audit import range_pairs
from .registry import DATASETS_ROOT, DERIVED, REGISTRY, RESULTS
from .resplit import near_dup_groups

SEED = 42
THETA, TAU, K = 0.98, 0, 2
TARGET = {"train": 0.7, "valid": 0.2, "test": 0.1}
SPLITS = ("train", "valid", "test")
COS_REPORT = (0.98, 0.95, 0.90)


def src_of(fname):
    """E2's group key, verbatim."""
    s = os.path.basename(fname)
    return s.split(".rf.")[0] if ".rf." in s else os.path.splitext(s)[0]


def e2_split(group_of):
    """E2 cell 6: {file: group key} -> {file: split}."""
    groups = {}
    for fn, g in group_of.items():
        groups.setdefault(g, []).append(fn)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    total, acc, out = len(group_of), 0, {}
    for k in keys:
        sp = "train" if acc < 0.7 * total else ("valid" if acc < 0.9 * total else "test")
        for fn in groups[k]:
            out[fn] = sp
            acc += 1
    return out


def balanced_split(rel, keys, boxes, names):
    """Group-aware class-balanced split (stratify_bg.split, seed 42) -> {rel: split} with E2 split names."""
    from . import stratify_bg
    groups, bpf = {}, {}
    for r, k, b in zip(rel, keys, boxes):
        g = groups.setdefault(k, {"files": [], "boxes": Counter(), "n_img": 0, "old_split": None})
        g["files"].append(r); g["n_img"] += 1; g["boxes"].update(b)
        bpf[r] = dict(b)
    for g in groups.values():
        g["boxes"] = dict(g["boxes"])
    res = stratify_bg.split(groups, bpf, names, seed=SEED, req={})
    alias = {"val": "valid"}
    return {r: alias.get(res["assign"][k], res["assign"][k]) for r, k in zip(rel, keys)}


def read_boxes(root, rel):
    lbl = root / rel.replace("/images/", "/labels/").rsplit(".", 1)[0]
    p = lbl.with_name(lbl.name + ".txt")
    if not p.exists():
        return Counter()
    return Counter(int(float(line.split()[0])) for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def split_stats(split, boxes, names):
    n = len(split)
    img = Counter(split)
    per_cls = {}
    worst = (0.0, None)
    for cid, cn in enumerate(names):
        tot = sum(b[cid] for b in boxes)
        pct = {s: (100.0 * sum(b[cid] for b, sp in zip(boxes, split) if sp == s) / tot if tot else 0.0) for s in SPLITS}
        per_cls[cn] = {s: round(v, 1) for s, v in pct.items()}
        for s in SPLITS:
            d = abs(pct[s] - 100 * TARGET[s])
            if d > worst[0]:
                worst = (d, f"{cn}@{s}")
    bg = Counter(sp for b, sp in zip(boxes, split) if not b)
    return {"images": {s: img[s] for s in SPLITS}, "image_pct": {s: round(100 * img[s] / n, 2) for s in SPLITS},
            "max_class_dev_pp": round(worst[0], 2), "worst_class": worst[1], "class_pct": per_cls,
            "background": {s: bg[s] for s in SPLITS}}


def cross_pairs(split_arr, E, cos_pairs):
    out = {kname: int((split_arr[e[0]] != split_arr[e[1]]).sum()) for kname, e in E.items()}
    for t, (a, b) in cos_pairs.items():
        out[f"cos>={t}"] = int((split_arr[a] != split_arr[b]).sum())
    return out


def run(name, materialize):
    root = DATASETS_ROOT / REGISTRY[name][0]
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    f, files, comp, E = near_dup_groups(name, THETA, TAU, K)
    rel = list(f["rel"])
    orig = np.array([{"val": "valid"}.get(s, s) for s in f["split"]])
    e2key = [src_of(r) for r in rel]
    comp_key = {}
    for c, k in zip(comp, e2key):
        comp_key[c] = min(comp_key.get(c, k), k)
    c2key = [comp_key[c] for c in comp]
    boxes = [read_boxes(root, r) for r in rel]
    split_of = {"clean_exact": e2_split(dict(zip(rel, e2key))),
                "clean_exact_balanced": balanced_split(rel, e2key, boxes, names),
                "clean2": balanced_split(rel, c2key, boxes, names),
                "clean2_e2": e2_split(dict(zip(rel, c2key)))}
    arms = {"original": orig} | {k: np.array([v[r] for r in rel]) for k, v in split_of.items()}
    key_of = {"original": e2key, "clean_exact": e2key, "clean_exact_balanced": e2key, "clean2": c2key,
              "clean2_e2": c2key}

    cos_pairs = {t: range_pairs(f["emb"], t) for t in COS_REPORT}
    res = {"dataset": name, "root": REGISTRY[name][0], "classes": names, "images": len(rel),
           "edges": {kname: int(len(e[0])) for kname, e in E.items()} | {f"cos>={t}": int(len(p[0])) for t, p in cos_pairs.items()},
           "arms": {}}
    for arm, sp in arms.items():
        sizes = Counter(key_of[arm])
        res["arms"][arm] = ({"groups": len(sizes), "largest_group": max(sizes.values())}
                            | split_stats(list(sp), boxes, names)
                            | {"cross_split_pairs": cross_pairs(sp, E, cos_pairs)})
    man = pd.DataFrame({"file": files, "rel": rel, "md5": f["md5"], "split_original": orig,
                        "group_clean": e2key, "split_clean_exact": arms["clean_exact"],
                        "split_clean_exact_balanced": arms["clean_exact_balanced"],
                        "group_clean2": c2key, "split_clean2": arms["clean2"], "split_clean2_e2": arms["clean2_e2"]})
    out = RESULTS / "clean2"
    out.mkdir(parents=True, exist_ok=True)
    man.to_csv(out / f"manifest_{name}.csv", index=False, encoding="utf-8")

    if materialize:
        res["materialized"] = {}
        for arm, gcol in (("clean2", "group_clean2"), ("clean_exact_balanced", "group_clean")):
            dst = DERIVED / arm / name
            if dst.exists():
                shutil.rmtree(dst)
            for s in SPLITS:
                (dst / s / "images").mkdir(parents=True)
                (dst / s / "labels").mkdir(parents=True)
            for r, s in zip(rel, arms[arm]):
                src_img = root / r
                os.link(src_img, dst / s / "images" / src_img.name)
                lbl = (root / r.replace("/images/", "/labels/")).with_suffix(".txt")
                if lbl.exists():
                    os.link(lbl, dst / s / "labels" / lbl.name)
            shutil.copy2(root / "data.yaml", dst / "data.yaml")
            man[["file", gcol, f"split_{arm}", "split_original"]].rename(columns={gcol: "group"}).to_csv(
                dst / "manifest.csv", index=False, encoding="utf-8")
            res["materialized"][arm] = str(dst.relative_to(DATASETS_ROOT))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["B", "C"])
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--paper", help="paper folder: write generated/numbers.json key 'clean2'")
    a = ap.parse_args()
    summary = {"procedure": {"e2": "E2 cell 6 (sorted keys, Random(42).shuffle, 70/20/10 by image count)",
                             "balanced": "audit_v2.stratify_bg.split, seed 42, 70/20/10, req={}"},
               "clean2_key": {"theta": THETA, "phash": TAU, "scene_k": K}, "datasets": {}}
    for n in a.names:
        r = run(n, a.materialize)
        summary["datasets"][n] = r
        print(json.dumps({n: {arm: {k: v[k] for k in ("groups", "largest_group", "images", "max_class_dev_pp", "worst_class",
                                                    "cross_split_pairs")}
                              for arm, v in r["arms"].items()}}), flush=True)
    (RESULTS / "clean2" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if a.paper:
        write_numbers(summary, a.paper)


def write_numbers(summary, paper):
    """numbers.json 'clean2': per arm without class tables, plus the E2 clean-exact residual (errata E9)."""
    num_path = Path(paper) / "generated" / "numbers.json"
    nums = json.loads(num_path.read_text(encoding="utf-8")) if num_path.exists() else {}
    infl = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "extended" / "e2_inflation.csv")
    infl = {r.dataset[0]: {"mAP50": r.inflation_mAP50, "mAP50_95": r.inflation_mAP50_95} for r in infl.itertuples()}
    nums["clean2"] = {
        "procedure": summary["procedure"], "clean2_key": summary["clean2_key"],
        "arms": {n: {arm: {k: v for k, v in st.items() if k != "class_pct"} for arm, st in d["arms"].items()}
                 for n, d in summary["datasets"].items()},
        "e2_clean_exact_residual": {n: {"cross_split_pairs": d["arms"]["clean_exact"]["cross_split_pairs"],
                                        "e2_inflation_v4": infl.get(n), "inflation_is": "lower bound"}
                                    for n, d in summary["datasets"].items()}}
    num_path.write_text(json.dumps(nums, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
