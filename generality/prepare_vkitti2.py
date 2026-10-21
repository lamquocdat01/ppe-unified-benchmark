"""T3 dataset 1: Virtual KITTI 2 (clone, Camera_0, all 5 scenes) -> YOLO layout, naive random frame split.

The split mimics common practice for video-derived detection sets: frames shuffled and cut 70/20/10
(seed 42) with no grouping. The validator is then run unchanged (schema_vkitti2.yaml).
RGB frames are read from the kept tar in place (only the needed members are extracted).

Output: THS_DATASETS/_derived/generality/vkitti2_clone_cam0/{train,val,test}/{images,labels}, data.yaml
Usage: python -u generality/prepare_vkitti2.py
"""
import os
import random
import tarfile
from pathlib import Path

ROOT = Path(os.environ.get("THS_DATASETS", r"D:\Datasets"))
SRC = ROOT / "VKITTI2"
DST = ROOT / "_derived" / "generality" / "vkitti2_clone_cam0"
SCENES = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
CLASSES = ["Car", "Van", "Truck"]
W, H = 1242, 375
SPLIT = (0.7, 0.2, 0.1)
SEED = 42


def main():
    labels = {}
    for sc in SCENES:
        d = SRC / "x" / sc / "clone"
        cls_of = {}
        for line in (d / "info.txt").read_text().splitlines()[1:]:
            tid, lab = line.split()[:2]
            cls_of[int(tid)] = lab
        for line in (d / "bbox.txt").read_text().splitlines()[1:]:
            fr, cam, tid, l, r, t, b = line.split()[:7]
            if cam != "0":
                continue
            name = f"{sc}_clone_cam0_{int(fr):05d}"
            labels.setdefault(name, [])
            lab = cls_of.get(int(tid))
            if lab not in CLASSES:
                continue
            l, r, t, b = map(float, (l, r, t, b))
            labels[name].append(f"{CLASSES.index(lab)} {(l + r) / 2 / W:.6f} {(t + b) / 2 / H:.6f} "
                                f"{(r - l) / W:.6f} {(b - t) / H:.6f}")
    tmp = DST / "_all"
    tmp.mkdir(parents=True, exist_ok=True)
    want = {}
    for sc in SCENES:
        want_prefix = f"{sc}/clone/frames/rgb/Camera_0/"
        want[want_prefix] = sc
    n = 0
    with tarfile.open(SRC / "vkitti_2.0.3_rgb.tar") as tf:
        for m in tf:
            p = m.name.replace("\\", "/")
            for pre, sc in want.items():
                if pre in p and p.endswith(".jpg"):
                    fr = int(Path(p).stem.split("_")[-1])
                    name = f"{sc}_clone_cam0_{fr:05d}"
                    (tmp / f"{name}.jpg").write_bytes(tf.extractfile(m).read())
                    labels.setdefault(name, [])
                    n += 1
    names = sorted(p.stem for p in tmp.glob("*.jpg"))
    rng = random.Random(SEED)
    rng.shuffle(names)
    cut1, cut2 = int(SPLIT[0] * len(names)), int((SPLIT[0] + SPLIT[1]) * len(names))
    for k, name in enumerate(names):
        sp = "train" if k < cut1 else ("val" if k < cut2 else "test")
        (DST / sp / "images").mkdir(parents=True, exist_ok=True)
        (DST / sp / "labels").mkdir(parents=True, exist_ok=True)
        os.replace(tmp / f"{name}.jpg", DST / sp / "images" / f"{name}.jpg")
        (DST / sp / "labels" / f"{name}.txt").write_text("\n".join(labels.get(name, [])) + ("\n" if labels.get(name) else ""))
    tmp.rmdir()
    (DST / "data.yaml").write_text("train: train/images\nval: val/images\ntest: test/images\nnames:\n"
                                   + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (DST / "SOURCE.md").write_text(
        "Derived from Virtual KITTI 2 v2.0.3 (CC BY-NC-SA 3.0, Naver Corporation): clone variation, Camera_0, "
        f"scenes {', '.join(SCENES)}; 2D boxes from bbox.txt, classes from info.txt. Naive random frame split "
        f"70/20/10 (seed {SEED}). Local validation only, not redistributed.\n")
    print(f"{n} frames -> {DST}")


if __name__ == "__main__":
    main()
