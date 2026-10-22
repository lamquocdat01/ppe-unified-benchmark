"""Generate the T4(1) Kaggle notebooks: PPE-Unified v1-fix vs v1.1, YOLO26s + RT-DETR-L x 2 seeds (8 runs;
3 seeds planned, cut to 2 on 22-10-2026 to fit T4(1) + T4(2)-B into one weekly Kaggle quota).

Both variants share the same images and corrected labels (dataset ppe-unified-v1fix-v11, folders v1fix/ and
v1.1/); only the split differs, so the gap measures split leakage alone.

Training settings are the E1 settings selected by HPO in notebooks 01/03 (see results/summary_0*.csv):
  YOLO26s  : 30 ep, imgsz 640, batch 16, optimizer auto (MuSGD), lr0 0.01, patience 10
  RT-DETR-L: 30 ep, imgsz 640, batch 8,  optimizer AdamW,       lr0 1e-4, patience 10
  repeat-factor oversampling of rare classes on train only (t = 0.15, cap 4), as in E1.
One notebook per model so each fits Kaggle's 12-hour session; both resume from their own
previous output (state.json + runs/*/last.pt).

Usage: python kaggle/make_t4_notebooks.py <out_dir>   (KAGGLE_USERNAME env fills kernel-metadata ids)
"""
import json
import os
import sys
from pathlib import Path

CONFIG = '''# ==================== CONFIG (T4-1: PPE-Unified v1-fix vs v1.1) ====================
MODEL      = "{model}"           # this notebook trains one model; the other notebook trains the other
SEEDS      = [0, 1]
VARIANTS   = ["v1fix", "v1.1"]
QUICK_TEST = False                # True: 2 epochs on 10% of the data, pipeline smoke test only
TIME_BUDGET_H = 11.5              # do not START a run that would end after this (previous run as estimate; limit 12 h)
RFS_T, REPEAT_CAP = 0.15, 4       # E1 repeat-factor oversampling, train split only
HP = {{
    "yolo26s":  dict(weights="yolo26s.pt",  epochs=30, imgsz=640, batch=16, optimizer="auto",  lr0=0.01, patience=10),
    "rtdetr-l": dict(weights="rtdetr-l.pt", epochs=30, imgsz=640, batch=8,  optimizer="AdamW", lr0=1e-4, patience=10),
}}[MODEL]

import os, glob, json, shutil, time
T_START = time.time()
WORK = "/kaggle/working"; os.makedirs(f"{{WORK}}/results", exist_ok=True)
STATE_PATH = f"{{WORK}}/state.json"
# resume: state.json + runs/ from THIS notebook's previous output, if it was added as input
for _st in glob.glob("/kaggle/input/**/state.json", recursive=True):
    try: _s = json.load(open(_st))
    except Exception: continue
    if any(k.startswith(f"t4_{{MODEL}}_") for k in _s):
        shutil.copy(_st, STATE_PATH)
        _runs = os.path.join(os.path.dirname(_st), "runs")
        if os.path.isdir(_runs): shutil.copytree(_runs, f"{{WORK}}/runs", dirs_exist_ok=True)
        print("CHECKPOINT LOADED FROM:", _st); break
STATE = json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else {{}}
def save_state(): json.dump(STATE, open(STATE_PATH, "w"), indent=1, default=str)
def done(k): return k in STATE
def mark(k, v): STATE[k] = v; save_state()
print(f"[T4-1] {{MODEL}} | runs done:", [k for k in STATE] or "(none yet)")
'''

SETUP = '''%pip install -q ultralytics
import torch
assert torch.cuda.is_available(), "Enable a GPU accelerator (T4 x2 or P100)!"
print("GPU:", torch.cuda.get_device_name(0))
'''

DATA = '''import yaml, pandas as pd
from pathlib import Path
from collections import Counter
from math import sqrt

def find_variants():
    """Folders v1fix/ and v1.1/ of the input dataset ppe-unified-v1fix-v11 (checked by content:
    both carry FIXES.csv; only v1.1 has the manifest column group_v11)."""
    found = {}
    for y in glob.glob("/kaggle/input/**/data.yaml", recursive=True):
        root = os.path.dirname(y)
        if not all(os.path.isdir(f"{root}/{s}/images") for s in ("train", "val", "test")):
            continue
        if not os.path.exists(f"{root}/FIXES.csv"):
            continue  # an older uncorrected v1 upload, not part of this comparison
        cols = pd.read_csv(f"{root}/manifest.csv", nrows=1).columns
        found["v1.1" if "group_v11" in cols else "v1fix"] = root
    return found

SRC = find_variants()
print("dataset inputs:", SRC)
assert set(VARIANTS) <= set(SRC), "Add the dataset ppe-unified-v1fix-v11 as input (folders v1fix/ and v1.1/)"

def prepare(variant):
    """Copy one variant to /kaggle/working (YOLO writes label caches) and apply E1 RFS on train."""
    src, dst = SRC[variant], f"{WORK}/data_{variant}"
    if os.path.exists(f"{dst}/data.yaml"):
        return f"{dst}/data.yaml"
    names = yaml.safe_load(open(f"{src}/data.yaml"))["names"]
    if isinstance(names, dict): names = [names[k] for k in sorted(names)]
    for sp in ("train", "val", "test"):
        os.makedirs(f"{dst}/{sp}/images"); os.makedirs(f"{dst}/{sp}/labels")
        files = sorted(os.listdir(f"{src}/{sp}/images"))
        if QUICK_TEST: files = files[: max(50, len(files) // 10)]
        for fn in files:
            shutil.copy(f"{src}/{sp}/images/{fn}", f"{dst}/{sp}/images/{fn}")
            lp = f"{src}/{sp}/labels/{Path(fn).stem}.txt"
            if os.path.exists(lp): shutil.copy(lp, f"{dst}/{sp}/labels/")
    img_classes = {}
    for lf in os.listdir(f"{dst}/train/labels"):
        img_classes[lf] = Counter(int(l.split()[0]) for l in open(f"{dst}/train/labels/{lf}") if l.strip())
    n = len(os.listdir(f"{dst}/train/images"))
    f_c = {c: sum(1 for s in img_classes.values() if c in s) / max(1, n) for c in range(len(names))}
    r_c = {c: max(1.0, sqrt(RFS_T / max(f, 1e-6))) for c, f in f_c.items()}
    added = 0
    for lf, present in img_classes.items():
        if not present: continue
        r = min(REPEAT_CAP, max(r_c[c] for c in present))
        if r < 1.5: continue
        stem = Path(lf).stem
        img = glob.glob(f"{dst}/train/images/{stem}.*")[0]
        for k in range(int(round(r)) - 1):
            shutil.copy(img, f"{dst}/train/images/{stem}_rfs{k}{Path(img).suffix}")
            shutil.copy(f"{dst}/train/labels/{lf}", f"{dst}/train/labels/{stem}_rfs{k}.txt")
            added += 1
    with open(f"{dst}/data.yaml", "w") as f:
        f.write(f"path: {dst}\\ntrain: train/images\\nval: val/images\\ntest: test/images\\nnames:\\n"
                + "".join(f"  {i}: {nm}\\n" for i, nm in enumerate(names)))
    print(f"{variant}: train {n} (+{added} RFS copies)")
    return f"{dst}/data.yaml"
'''

TRAIN = '''from ultralytics import YOLO, RTDETR
import gc
Net = RTDETR if MODEL.startswith("rtdetr") else YOLO
CLASSES = None
prev_h = 0.0
for variant in VARIANTS:
    dyaml = None
    for sd in SEEDS:
        key = f"t4_{MODEL}_{variant}_s{sd}"
        if done(key): print("skip:", STATE[key]); continue
        if (time.time() - T_START) / 3600 + prev_h > TIME_BUDGET_H:
            print(f"time budget reached - stop; re-run this notebook with its output added as input to resume"); break
        dyaml = dyaml or prepare(variant)
        name = key; last = f"{WORK}/runs/{name}/weights/last.pt"; t0 = time.time()
        if os.path.exists(last):
            print("MID-RUN RESUME:", last); m = Net(last); m.train(resume=True)
        else:
            m = Net(HP["weights"])
            m.train(data=dyaml, epochs=2 if QUICK_TEST else HP["epochs"], imgsz=HP["imgsz"], batch=HP["batch"],
                    optimizer=HP["optimizer"], lr0=HP["lr0"], patience=HP["patience"], seed=sd, deterministic=True,
                    project=f"{WORK}/runs", name=name, exist_ok=True, plots=False, verbose=False)
        best = f"{WORK}/runs/{name}/weights/best.pt"
        m = Net(best if os.path.exists(best) else last)
        v = m.val(data=dyaml, split="val", verbose=False)
        t = m.val(data=dyaml, split="test", verbose=False)
        CLASSES = CLASSES or [t.names[i] for i in sorted(t.names)]
        row = {"model": MODEL, "variant": variant, "seed": sd, "epochs_cfg": HP["epochs"],
               "mAP50_val": round(v.box.map50, 4), "mAP50_95_val": round(v.box.map, 4),
               "mAP50_test": round(t.box.map50, 4), "mAP50_95_test": round(t.box.map, 4),
               "P_test": round(t.box.mp, 4), "R_test": round(t.box.mr, 4),
               "ap50_test_per_class": {t.names[int(c)]: round(float(a), 4) for c, a in zip(t.box.ap_class_index, t.box.ap50)},
               "train_min": round((time.time() - t0) / 60, 1)}
        mark(key, row); print(row); prev_h = (time.time() - t0) / 3600
        del m; gc.collect(); torch.cuda.empty_cache()
'''

SUMMARY = '''rows = [v for k, v in STATE.items() if k.startswith(f"t4_{MODEL}_")]
df = pd.DataFrame(rows)
if len(df):
    df.drop(columns=["ap50_test_per_class"]).to_csv(f"{WORK}/results/t4_{MODEL}_runs.csv", index=False)
    agg = df.groupby("variant")[["mAP50_val", "mAP50_test", "mAP50_95_test"]].agg(["mean", "std", "count"]).round(4)
    display(agg); agg.to_csv(f"{WORK}/results/t4_{MODEL}_summary.csv")
    if {"v1fix", "v1.1"} <= set(df.variant):
        g = df.groupby("variant")
        infl = {m: round(float(g[m].mean()["v1fix"] - g[m].mean()["v1.1"]), 4) for m in ("mAP50_test", "mAP50_95_test")}
        print("inflation v1fix - v1.1 (mean over seeds):", infl)
        json.dump(infl, open(f"{WORK}/results/t4_{MODEL}_inflation.json", "w"), indent=1)
# keep only results/, state.json and the last.pt/best.pt needed to resume; drop copied data
for item in os.listdir(WORK):
    if item not in {"results", "state.json", "runs"}:
        p = os.path.join(WORK, item); shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
print("output:", sorted(os.listdir(WORK)))
'''


def notebook(model):
    cells = [("markdown", f"# T4-1 — PPE-Unified v1 vs v1.1 — {model} × seeds 0,1\n"
                          "E1 hyper-parameters (HPO-selected). Add the dataset ppe-unified-v1fix-v11 as input; to resume, add this "
                          "notebook's previous output as input and run all again."),
             ("code", CONFIG.format(model=model)), ("code", SETUP), ("code", DATA), ("code", TRAIN), ("code", SUMMARY)]
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    out = Path(sys.argv[1])
    user = os.environ.get("KAGGLE_USERNAME", "<KAGGLE_USERNAME>")
    for model, slug in (("yolo26s", "t4-ppeu-v1fix-vs-v11-yolo26s"), ("rtdetr-l", "t4-ppeu-v1fix-vs-v11-rtdetr-l")):
        d = out / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{slug}.ipynb").write_text(json.dumps(notebook(model), indent=1), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{user}/ppe-unified-v1fix-v11"],
            "competition_sources": [], "kernel_sources": []}, indent=1), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
