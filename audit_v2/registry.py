"""Dataset registry and file enumeration for the two-tier leakage audit.

Every dataset path is resolved under the THS_DATASETS environment variable
(default D:\\Datasets); nothing dataset-specific is hard-coded beyond the
relative layout of each export.
"""
import os
import re
from pathlib import Path

DATASETS_ROOT = Path(os.environ.get("THS_DATASETS", r"D:\Datasets"))
DERIVED = DATASETS_ROOT / "_derived" / "ppe_unified"
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "audit_v2"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"valid": "val", "validation": "val"}

# name -> (root relative to THS_DATASETS, layout)
#   layout "yolo_splits": <root>/<split>/images/*
#   layout "sh17": <root>/images/* with train_files.txt / val_files.txt
REGISTRY = {
    "ppeu": ("PPE_SH17/PPE_Unified_v1_resplit", "yolo_splits"),
    "ppeu11": ("PPE_SH17/PPE_Unified_v1.1", "yolo_splits"),
    "ppeu_fix": ("PPE_SH17/PPE_Unified_v1fix", "yolo_splits"),
    "A": ("PPE_SH17/my_dataset_yolo", "yolo_splits"),
    "B": ("PPE_SH17/PPE detection.v1i.yolov8.zip", "yolo_splits"),
    "C": ("PPE_SH17/ppe-detection-l80fg-v5", "yolo_splits"),
    "sh17": ("PPE_SH17/SH17", "sh17"),
}

_RF = re.compile(r"\.rf\.[0-9a-f]+$", re.I)


def source_id(fname: str) -> str:
    """Roboflow export names are <stem>_<ext>.rf.<hash>.<ext>; the stem identifies the source photo."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    return _RF.sub("", stem)


def has_rf(fname: str) -> bool:
    return bool(_RF.search(os.path.splitext(os.path.basename(fname))[0]))


def register(name: str, root: str, layout: str = "yolo_splits"):
    """Add a dataset at runtime (used by the literature audit, T2)."""
    REGISTRY[name] = (root, layout)


def list_images(name: str):
    """Return a sorted list of (relative path, split) for one dataset."""
    rel_root, layout = REGISTRY[name]
    root = DATASETS_ROOT / rel_root
    items = []
    if layout == "yolo_splits":
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            img_dir = d / "images"
            if not img_dir.is_dir():
                continue
            split = SPLIT_ALIASES.get(d.name.lower(), d.name.lower())
            for p in img_dir.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    items.append((p.relative_to(root).as_posix(), split))
    elif layout == "sh17":
        split_of = {}
        for sp in ("train", "val"):
            f = root / f"{sp}_files.txt"
            if f.exists():
                for line in f.read_text(encoding="utf-8").split():
                    split_of[os.path.basename(line.strip())] = sp
        for p in (root / "images").iterdir():
            if p.suffix.lower() in IMG_EXTS:
                items.append((p.relative_to(root).as_posix(), split_of.get(p.name, "unsplit")))
    else:
        raise ValueError(layout)
    items.sort()
    return root, items
