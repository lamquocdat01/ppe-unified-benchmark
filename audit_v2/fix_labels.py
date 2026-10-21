"""Build PPE_Unified_v1fix: the v1 split with the approved label corrections (decision 1, 21-10-2026).

v1fix is the corrected pool; v1.1 is re-split from it (audit_v2.resplit --src ppeu_fix), so both
variants share identical images and labels. Images are hardlinks of v1; every label file is a
fresh copy so that a correction never touches v1. Corrections are logged to FIXES.csv
(file, line, action, before, after) and in ERRATA_v4.md (E8).

  clip      every box extending beyond the image by more than the validator tolerance is clipped
            to [0, 1] (xyxy clip, back to xywh) - b_Video3_34, b_Video4_53, b_image_1700
  a_003385  the no_helmet box overlapping a helmet box (IoU >= 0.8) is removed
  a_002686  --a002686 drop_image | drop_line:<n> | keep

Usage: python -u -m audit_v2.fix_labels --a002686 drop_line:4
"""
import argparse
import csv
import os
import shutil

import pandas as pd

from .registry import DATASETS_ROOT, REGISTRY

SRC = DATASETS_ROOT / REGISTRY["ppeu"][0]
DST = DATASETS_ROOT / "PPE_SH17" / "PPE_Unified_v1fix"
TOL = 0.005
HELMET, NO_HELMET = 0, 1


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def clip_line(line):
    c, *v = line.split()
    xc, yc, w, h = map(float, v)
    over = max(0.0, w / 2 - xc, xc + w / 2 - 1, h / 2 - yc, yc + h / 2 - 1)
    if over <= TOL:
        return None
    x0, y0 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
    x1, y1 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
    return f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a002686", required=True, help="drop_image | drop_line:<n> | keep")
    a = ap.parse_args()
    if DST.exists():
        shutil.rmtree(DST)
    man = pd.read_csv(SRC / "manifest.csv", encoding="utf-8")
    log = []
    drop_files = set()
    if a.a002686 == "drop_image":
        drop_files.add("a_002686_jpg.jpg")
        log.append(("a_002686_jpg.jpg", "", "drop_image", "image + label", "removed"))
    for split in ("train", "val", "test"):
        (DST / split / "images").mkdir(parents=True)
        (DST / split / "labels").mkdir(parents=True)
        for img in sorted((SRC / split / "images").iterdir()):
            if img.name in drop_files:
                continue
            os.link(img, DST / split / "images" / img.name)
            lp = SRC / split / "labels" / (img.stem + ".txt")
            if not lp.exists():
                continue
            lines = [l.rstrip("\n") for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
            out = []
            boxes = [(int(l.split()[0]), tuple(map(float, l.split()[1:5]))) for l in lines]
            for k, l in enumerate(lines, 1):
                new = clip_line(l)
                if new is not None:
                    log.append((img.name, k, "clip_to_image", l, new)); l = new
                cls, box = boxes[k - 1]
                if img.stem == "a_003385_jpg" and cls == NO_HELMET and any(
                        c == HELMET and iou(box, b) >= 0.8 for c, b in boxes):
                    log.append((img.name, k, "drop_box", l, "")); continue
                if img.stem == "a_002686_jpg" and a.a002686 == f"drop_line:{k}":
                    log.append((img.name, k, "drop_box", l, "")); continue
                out.append(l)
            (DST / split / "labels" / lp.name).write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    man[~man["file"].isin(drop_files)].to_csv(DST / "manifest.csv", index=False, encoding="utf-8")
    y = (SRC / "data.yaml").read_text(encoding="utf-8").splitlines()
    y[0] = "# Safety-PPE Unified v1-fix (v1 split, corrected labels) - 12 lop"
    (DST / "data.yaml").write_text("\n".join(y) + "\n", encoding="utf-8")
    with open(DST / "FIXES.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "line", "action", "before", "after"]); w.writerows(log)
    for r in log:
        print(r)
    print(f"written {DST} ({len(log)} corrections)")


if __name__ == "__main__":
    main()
