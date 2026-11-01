"""E1 per-class AP@50 (YOLO26s, RT-DETR-L) on the PPE-Unified v1 test split, extracted from the Kaggle logs
of notebooks 01 and 03 -> results/extended/e1_perclass_ap50.csv (P6, 01-11-2026, C6).

Sources (Kaggle output, kept outside the repo; SHA-256 recorded in the CSV header comment):
  --yolo-log   notebook01_v2_log.txt            line "mAP50 từng lớp: {...}" + the Ultralytics test table
  --detr-log   <notebook 03 output>/__output__.json   the same dict printed by notebook 03 (RT-DETR-L)
Box counts per class are the 'Instances' column of the Ultralytics test table in the YOLO log
(the test split is the same for both detectors).

Usage: python scripts/extract_e1_perclass.py --yolo-log <file> --detr-log <file>
"""
import argparse
import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["helmet", "no_helmet", "vest", "no_vest", "gloves", "no_gloves", "goggles", "no_goggles",
           "boots", "no_boots", "mask", "no_mask"]
DICT = re.compile(r"\{'helmet': np\.float64\([0-9.]+\).*?'no_mask': np\.float64\([0-9.]+\)\}")


def ap_dict(text, tag):
    found = DICT.findall(text)
    assert found, f"no per-class mAP50 dict in {tag}"
    d = {k: float(v) for k, v in re.findall(r"'(\w+)': np\.float64\(([0-9.]+)\)", found[-1])}
    assert set(d) == set(CLASSES), d.keys()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-log", required=True)
    ap.add_argument("--detr-log", required=True)
    a = ap.parse_args()
    ty = Path(a.yolo_log).read_text(encoding="utf-8", errors="replace")
    td = Path(a.detr_log).read_text(encoding="utf-8", errors="replace")
    y, r = ap_dict(ty, "yolo log"), ap_dict(td, "detr log")
    # Ultralytics test table directly preceding the dict: "<class> <images> <instances> P R mAP50 mAP50-95"
    tail = ty[: ty.index("mAP50 từng lớp")]
    tail = tail[tail.rindex("Class     Images  Instances"):]
    inst = {m.group(1): int(m.group(3)) for m in re.finditer(r"\s(\w+)\s+(\d+)\s+(\d+)\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+", tail)
            if m.group(1) in CLASSES}
    assert set(inst) == set(CLASSES), inst
    # consistency: the dict and the table agree for YOLO (table prints 3 significant digits)
    for m in re.finditer(r"\s(\w+)\s+\d+\s+\d+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+", tail):
        if m.group(1) in CLASSES:
            assert abs(float(m.group(2)) - y[m.group(1)]) < 6e-4, (m.group(1), m.group(2), y[m.group(1)])
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out = REPO / "results" / "extended" / "e1_perclass_ap50.csv"
    lines = [f"# source yolo: {Path(a.yolo_log).name} sha256={h(a.yolo_log)}",
             f"# source rtdetr: {Path(a.detr_log).parent.name}/{Path(a.detr_log).name} sha256={h(a.detr_log)}",
             "class,boxes_test,ap50_yolo26s,ap50_rtdetr_l"]
    lines += [f"{c},{inst[c]},{y[c]:.4f},{r[c]:.4f}" for c in CLASSES]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
