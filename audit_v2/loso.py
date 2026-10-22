"""T4(2)-C: leave-one-sequence-out folds for E2 source C (decision 22-10-2026).

C has no clean class-balanced 70/20/10 split: the 4 largest clean2 groups (exact U 2a U 2b components) are
video sequences holding 51 % of the images and most boxes of several classes, and every group must stay in one
split, so clean2 C misses the +-5 pp class-ratio target by 25.69 pp (audit_v2.clean2). This is reported as a
result. Instead of a 70/20/10 split:
  fold i: test  = sequence i (the i-th largest clean2 group)
          train = the other 3 sequences + every image outside the 4 sequences
  no validation split; training uses E2's fixed epoch count and the last checkpoint.
Each fold is checked for 0 train/test pairs at every grouping tier (exact, cos >= 0.98, pHash = 0, scene k = 2);
cos >= 0.95 / 0.90 are reported for information. Per-class box counts are reported per fold; a class with 0 test
boxes is N/A in that fold (it is not dropped from the dataset).

Usage:
  python -u -m audit_v2.loso [--paper <paper folder>]
Needs results/audit_v2/clean2/manifest_C.csv (python -m audit_v2.clean2). Writes
results/audit_v2/clean2/{loso_C.json, manifest_C_loso.csv}; --paper adds numbers.json key 'clean2_C_loso'.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .audit import range_pairs
from .clean2 import COS_REPORT, K, TAU, THETA, read_boxes
from .registry import DATASETS_ROOT, REGISTRY, RESULTS
from .resplit import near_dup_groups

NAME = "C"
N_SEQ = 4
E2_C_YOLO_MIN, E2_C_TRAIN_IMG = 142, 21641  # E2 YOLO26s on C (T4 GPU, 15 ep, with per-epoch val): minutes, train images


def family(fname):
    """File-name family of a sequence frame: Roboflow stem without the trailing frame number."""
    stem = fname.split(".rf.")[0]
    return re.sub(r"[_\-]?\d+(_(jpg|jpeg|png))?$", "", stem, flags=re.I) or stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", help="paper folder: write generated/numbers.json key 'clean2_C_loso'")
    a = ap.parse_args()
    root = DATASETS_ROOT / REGISTRY[NAME][0]
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    out = RESULTS / "clean2"
    man = pd.read_csv(out / f"manifest_{NAME}.csv")
    f, files, comp, E = near_dup_groups(NAME, THETA, TAU, K)
    assert list(f["rel"]) == list(man["rel"]), "feature cache order differs from manifest_C.csv"
    rel = list(man["rel"])
    n = len(rel)
    boxes = [read_boxes(root, r) for r in rel]
    tot_box = Counter()
    for b in boxes:
        tot_box.update(b)

    sizes = man["group_clean2"].value_counts()
    seqs = list(sizes.index[:N_SEQ])
    seq_of = man["group_clean2"].map({g: i for i, g in enumerate(seqs)}).fillna(-1).astype(int).to_numpy()
    cos_pairs = {t: range_pairs(f["emb"], t) for t in COS_REPORT}

    seq_info = []
    for i, g in enumerate(seqs):
        m = seq_of == i
        bx = Counter()
        for b, mm in zip(boxes, m):
            if mm:
                bx.update(b)
        fam = Counter(family(fn) for fn in man.loc[m, "file"])
        seq_info.append({"fold": i, "group": g, "images": int(m.sum()), "image_pct": round(100 * m.sum() / n, 2),
                         "families_top3": dict(fam.most_common(3)), "n_families": len(fam),
                         "box_share_pct": {cn: round(100 * bx[c] / tot_box[c], 1) if tot_box[c] else None
                                           for c, cn in enumerate(names)}})
    in_seq = seq_of >= 0

    folds, gpu = [], []
    for i in range(N_SEQ):
        sp = np.where(seq_of == i, "test", "train")
        cross = {k: int((sp[e[0]] != sp[e[1]]).sum()) for k, e in E.items()}
        cross |= {f"cos>={t}": int((sp[p[0]] != sp[p[1]]).sum()) for t, p in cos_pairs.items()}
        per_cls = {}
        for c, cn in enumerate(names):
            tr = sum(b[c] for b, s in zip(boxes, sp) if s == "train")
            te = sum(b[c] for b, s in zip(boxes, sp) if s == "test")
            per_cls[cn] = {"train": tr, "test": te, "eval": "N/A" if te == 0 or tr == 0 else "ok"}
        n_tr = int((sp == "train").sum())
        minutes = E2_C_YOLO_MIN * n_tr / E2_C_TRAIN_IMG
        gpu.append(minutes)
        folds.append({"fold": i, "test_group": seqs[i], "images": {"train": n_tr, "test": int((sp == "test").sum())},
                      "cross_split_pairs": cross,
                      "clean": all(cross[k] == 0 for k in E) and cross["cos>=0.98"] == 0,
                      "boxes": per_cls, "na_classes": [cn for cn, v in per_cls.items() if v["eval"] == "N/A"],
                      "yolo26s_gpu_min_est": round(minutes)})
        man[f"split_loso_f{i}"] = sp

    c2 = json.loads((out / "summary.json").read_text(encoding="utf-8"))["datasets"][NAME]["arms"]["clean2"]
    infeasible = {"n_sequences": N_SEQ, "sequences_image_pct": round(100 * in_seq.sum() / n, 2),
                  "outside_sequences_image_pct": round(100 * (~in_seq).sum() / n, 2),
                  "sequence_box_share_pct": {s["group"]: s["box_share_pct"] for s in seq_info},
                  "clean2_max_class_dev_pp": c2["max_class_dev_pp"], "clean2_worst_class": c2["worst_class"],
                  "target_pp": 5.0}
    res = {"dataset": NAME, "images": n, "classes": names, "clean2_key": {"theta": THETA, "phash": TAU, "scene_k": K},
           "protocol": "leave-one-sequence-out over the 4 largest clean2 groups; train = other 3 + all images outside "
                       "the 4; no val split; E2 fixed epochs (15), last checkpoint; YOLO26s x seed 42",
           "sequences": seq_info, "folds": folds, "infeasible_balanced_split": infeasible,
           "gpu_estimate": {"basis": f"E2 YOLO26s C {E2_C_YOLO_MIN} min for {E2_C_TRAIN_IMG} train images on T4, "
                                     "scaled by train images (no per-epoch val: conservative)",
                            "yolo26s_total_h": round(sum(gpu) / 60, 1)}}
    (out / "loso_C.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    man[["file", "rel", "md5", "group_clean2"] + [f"split_loso_f{i}" for i in range(N_SEQ)]].to_csv(
        out / "manifest_C_loso.csv", index=False, encoding="utf-8")
    for fo in folds:
        print(json.dumps({k: fo[k] for k in ("fold", "test_group", "images", "clean", "na_classes", "yolo26s_gpu_min_est")}),
              flush=True)
        print("   cross:", fo["cross_split_pairs"])
    print("infeasible:", json.dumps(infeasible))
    if a.paper:
        p = Path(a.paper) / "generated" / "numbers.json"
        nums = json.loads(p.read_text(encoding="utf-8"))
        nums["clean2_C_loso"] = {k: v for k, v in res.items() if k != "classes"}
        nums.setdefault("errata_v4", {})["E9_C_no_clean_balanced_split"] = infeasible
        p.write_text(json.dumps(nums, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
