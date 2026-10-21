"""T3 dataset 2: VIRAT ground video (IARPA DIVA annotations, KPF) -> COCO, official clip-level split.

Frames are sampled every STEP frames (1 fps at 30 fps) from the 8 annotated clips; boxes come from
geom.yml at that frame, classes from types.yml. The split is the official DIVA one by clip
(train / validate -> train / val). Files are named <clip>_f<frame>.jpg so the schema can group by clip
and the scene level (2b) can link neighbouring samples (k in frame units = 2 * STEP).

Output: THS_DATASETS/_derived/generality/virat_diva_1fps/{train,val}/{images/, annotations.json}
Usage: python -u generality/prepare_virat.py
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(os.environ.get("THS_DATASETS", r"D:\Datasets"))
SRC = ROOT / "VIRAT"
DST = ROOT / "_derived" / "generality" / "virat_diva_1fps"
STEP = 30
SPLITS = {"train": "train", "validate": "val"}
CLASSES = ["Person", "Vehicle", "Bike", "Push_Pulled_Object", "Prop", "Door", "Other"]
_GEOM = re.compile(r"id1:\s*(\d+),\s*id0:\s*\d+,\s*ts0:\s*(\d+),.*?g0:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
_TYPE = re.compile(r"id1:\s*(\d+)\s*,\s*cset3:\s*\{\s*([A-Za-z_]+)")


def main():
    for src_split, split in SPLITS.items():
        out = DST / split / "images"
        out.mkdir(parents=True, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
        for geom in sorted((SRC / "annotations_diva" / src_split).glob("*.geom.yml")):
            clip = geom.name.replace(".geom.yml", "")
            types = {int(a): b for a, b in _TYPE.findall((geom.parent / f"{clip}.types.yml").read_text())}
            boxes = defaultdict(list)
            with open(geom, encoding="utf-8") as fh:
                for line in fh:
                    m = _GEOM.search(line)
                    if m:
                        tid, fr, x0, y0, x1, y1 = map(int, m.groups())
                        if fr % STEP == 0:
                            boxes[fr].append((types.get(tid, "Other"), x0, y0, x1, y1))
            cap = cv2.VideoCapture(str(SRC / "videos" / f"{clip}.mp4"))
            fr = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if fr % STEP == 0:
                    ok, img = cap.retrieve()
                    if not ok:
                        break
                    h, w = img.shape[:2]
                    name = f"{clip}_f{fr:06d}.jpg"
                    cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    iid = len(coco["images"]) + 1
                    coco["images"].append({"id": iid, "file_name": f"images/{name}", "width": w, "height": h})
                    for lab, x0, y0, x1, y1 in boxes.get(fr, []):
                        lab = lab if lab in CLASSES else "Other"
                        coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": iid,
                                                    "category_id": CLASSES.index(lab) + 1,
                                                    "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                                                    "iscrowd": 0})
                fr += 1
            cap.release()
            print(f"  {split} {clip}: {fr} frames", flush=True)
        (DST / split / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
        print(split, len(coco["images"]), "images", len(coco["annotations"]), "boxes", flush=True)
    (DST / "SOURCE.md").write_text(
        "Derived from VIRAT Video Dataset 2.0 ground clips with IARPA DIVA KPF annotations "
        "(gitlab.kitware.com/viratdata/viratannotations); licence: VIRAT release terms. Frames sampled every "
        f"{STEP} frames; official DIVA train/validate split by clip. Local validation only, not redistributed.\n")


if __name__ == "__main__":
    main()
