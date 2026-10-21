"""Format loaders -> one uniform record per image.

Record: dict(split, path, stem, boxes=[(cls_id, xc, yc, w, h)], parse_errors=[...], label_path)
Coordinates are normalised to [0, 1] (YOLO convention); COCO boxes are converted.
"""
import json
import os
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dataset_root(schema):
    base = Path(os.environ.get("THS_DATASETS", r"D:\Datasets"))
    return base / schema["root"]


def load_yolo(schema):
    root = dataset_root(schema)
    recs, orphans = [], {"images_without_label": [], "labels_without_image": []}
    for split, sub in schema["splits"].items():
        img_dir = root / sub / "images"
        lbl_dir = root / sub / "labels"
        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS} if img_dir.is_dir() else {}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}
        orphans["images_without_label"] += [f"{split}/{images[s].name}" for s in sorted(set(images) - set(labels))]
        orphans["labels_without_image"] += [f"{split}/{labels[s].name}" for s in sorted(set(labels) - set(images))]
        for stem, ip in sorted(images.items()):
            boxes, errs = [], []
            lp = labels.get(stem)
            if lp is not None:
                for n, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
                    parts = line.split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        # polygons (len > 5) are converted to their bounding box; anything else is an error
                        if len(parts) > 5 and len(parts) % 2 == 1:
                            try:
                                xs = [float(v) for v in parts[1::2]]; ys = [float(v) for v in parts[2::2]]
                                boxes.append((parts[0], (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                                              max(xs) - min(xs), max(ys) - min(ys)))
                                continue
                            except ValueError:
                                pass
                        errs.append((n, line.strip()))
                        continue
                    try:
                        boxes.append((parts[0], *map(float, parts[1:])))
                    except ValueError:
                        errs.append((n, line.strip()))
            recs.append({"split": split, "path": ip, "stem": stem, "boxes": boxes,
                         "parse_errors": errs, "label_path": lp})
    return recs, orphans


def load_coco(schema):
    """schema['splits'][split] = path of the COCO json relative to root; images relative to json dir."""
    root = dataset_root(schema)
    recs, orphans = [], {"images_without_label": [], "labels_without_image": []}
    names = schema["classes"]
    for split, ann_rel in schema["splits"].items():
        ann = root / ann_rel
        d = json.loads(ann.read_text(encoding="utf-8"))
        cat_name = {c["id"]: c["name"] for c in d["categories"]}
        by_img = {}
        for a in d.get("annotations", []):
            by_img.setdefault(a["image_id"], []).append(a)
        for im in d["images"]:
            p = ann.parent / im["file_name"]
            W, H = im["width"], im["height"]
            boxes, errs = [], []
            for a in by_img.get(im["id"], []):
                nm = cat_name.get(a["category_id"])
                cid = str(names.index(nm)) if nm in names else f"unknown:{nm}"
                x, y, w, h = a["bbox"]
                boxes.append((cid, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
            if not p.exists():
                orphans["labels_without_image"].append(f"{split}/{im['file_name']}")
                continue
            recs.append({"split": split, "path": p, "stem": p.stem, "boxes": boxes,
                         "parse_errors": errs, "label_path": ann, "declared_size": (W, H)})
    return recs, orphans


LOADERS = {"yolo": load_yolo, "coco": load_coco}
