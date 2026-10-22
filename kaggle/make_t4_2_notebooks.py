"""Generate the T4(2) Kaggle notebooks (decision 22-10-2026), E2 settings. Queue after T4(1).

T4(2)-B (8 runs): B x {clean2, clean_exact_balanced} x {YOLO26s, RT-DETR-L} x seeds {42, 0}.
  clean_exact_balanced uses E2's group key (source stem) with the same class-balanced algorithm as clean2, so
  clean2 - clean_exact_balanced isolates the near-duplicate groups from the balancing. Seed 42 matches E2/E2b.
T4(2)-C (4 runs): C leave-one-sequence-out (audit_v2.loso), folds 0-3 x YOLO26s x seed 42; RT-DETR dropped.
  No validation split: E2's fixed epoch count, the last checkpoint is evaluated on the held-out sequence.
  C has no clean class-balanced 70/20/10 split (audit_v2.clean2: 25.69 pp); that is reported as a result.

Training settings are E2's, unchanged (notebooks/E2_LeakyVsClean_YOLO26s.ipynb, E2b_LeakyVsClean_RTDETR.ipynb):
  YOLO26s  : 15 ep, imgsz 640, batch 16, default optimizer
  RT-DETR-L: 15 ep, imgsz 640, batch 8, AdamW, lr0 1e-4
  no repeat-factor oversampling (E2 had none).
Images are downloaded from Roboflow inside the notebook (as in E2; key from Kaggle Secrets, ROBOFLOW_API_KEY);
the splits come from the small input dataset ppe-e2-clean2-manifests (file, md5, split per arm; C: one column per
LOSO fold). Files are matched by name, then by MD5; the notebook stops if any image is unmatched.

Usage: python kaggle/make_t4_2_notebooks.py <out_dir> [<manifest_dir>]
"""
import json
import sys
from pathlib import Path

import pandas as pd

CONFIG = '''# ==================== CONFIG (T4-2{part}: E2 source {part}) ====================
MODEL = "{model}"
SEEDS = {seeds}
RUNS  = {runs}                    # (dataset, arm)
QUICK_TEST = False                # True: 2 epochs, fraction 0.15, pipeline smoke test only
TIME_BUDGET_H = 11.5              # do not START a run that would end after this (previous run as estimate; limit 12 h)
HP = {{
    "yolo26s":  dict(weights="yolo26s.pt",  epochs=15, imgsz=640, batch=16),
    "rtdetr-l": dict(weights="rtdetr-l.pt", epochs=15, imgsz=640, batch=8, optimizer="AdamW", lr0=1e-4),
}}[MODEL]

import os, glob, json, shutil, time, hashlib
T_START = time.time()
WORK = "/kaggle/working"; os.makedirs(f"{{WORK}}/results", exist_ok=True)
STATE_PATH = f"{{WORK}}/state.json"
for _st in glob.glob("/kaggle/input/**/state.json", recursive=True):
    try: _s = json.load(open(_st))
    except Exception: continue
    if any(k.startswith(f"t42{part}_{{MODEL}}_") for k in _s):
        shutil.copy(_st, STATE_PATH)
        _runs = os.path.join(os.path.dirname(_st), "runs")
        if os.path.isdir(_runs): shutil.copytree(_runs, f"{{WORK}}/runs", dirs_exist_ok=True)
        print("CHECKPOINT LOADED FROM:", _st); break
STATE = json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else {{}}
def save_state(): json.dump(STATE, open(STATE_PATH, "w"), indent=1, default=str)
def done(k): return k in STATE
def mark(k, v): STATE[k] = v; save_state()
print(f"[T4-2{part}] {{MODEL}} | runs done:", list(STATE) or "(none yet)")
TAG = "t42{part}"
'''

SETUP = '''%pip install -q ultralytics roboflow
import torch
assert torch.cuda.is_available(), "Enable a GPU accelerator (T4 x2)!"
print("GPU:", torch.cuda.get_device_name(0))
from kaggle_secrets import UserSecretsClient
ROBOFLOW_API_KEY = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
assert ROBOFLOW_API_KEY, "Add the Kaggle secret ROBOFLOW_API_KEY (Add-ons > Secrets)"
'''

DATA = '''import yaml, pandas as pd
from roboflow import Roboflow
SOURCES = {"B": ("testcasque", "ppe-detection-qlq3d", 1),          # E2 B_testcasque
           "C": ("mohamed-traore-2ekkp", "ppe-detection-l80fg", 5)}  # E2 C_mohamed
MAN = {}
for d in {d for d, _ in RUNS}:
    hits = glob.glob(f"/kaggle/input/**/manifest_{d}.csv", recursive=True)
    assert hits, "Add the dataset ppe-e2-clean2-manifests as input"
    MAN[d] = pd.read_csv(hits[0])

def download(d):
    out = f"{WORK}/raw_{d}"
    if not os.path.exists(f"{out}/data.yaml"):
        ws, proj, ver = SOURCES[d]
        Roboflow(api_key=ROBOFLOW_API_KEY).workspace(ws).project(proj).version(ver).download("yolov8", location=out)
    return out

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def prepare(d, arm):
    """Re-split the downloaded export by the manifest (name match, MD5 fallback); hard-link into data_<d>_<arm>."""
    dst = f"{WORK}/data_{d}_{arm}"
    if os.path.exists(f"{dst}/data.yaml"): return f"{dst}/data.yaml"
    raw = download(d)
    man = MAN[d]; col = f"split_{arm}"
    by_name = dict(zip(man["file"], man[col]))
    by_md5 = dict(zip(man["md5"], man[col]))
    imgs = sorted(glob.glob(f"{raw}/*/images/*"))
    split_of, via_md5, miss = {}, 0, []
    splits = sorted(set(man[col]))
    for p in imgs:
        s = by_name.get(os.path.basename(p))
        if s is None:
            s = by_md5.get(md5(p)); via_md5 += s is not None
        if s is None: miss.append(p)
        else: split_of[p] = s
    print(f"{d}/{arm}: {len(imgs)} downloaded, {len(man)} in manifest, {via_md5} matched by MD5, {len(miss)} unmatched")
    assert not miss and len(imgs) == len(man), "download does not match the manifest - stop (see counts above)"
    for sp in splits:
        os.makedirs(f"{dst}/{sp}/images"); os.makedirs(f"{dst}/{sp}/labels")
    for p, sp in split_of.items():
        os.link(p, f"{dst}/{sp}/images/{os.path.basename(p)}")
        lp = p.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(lp): os.link(lp, f"{dst}/{sp}/labels/{os.path.basename(lp)}")
    names = yaml.safe_load(open(f"{raw}/data.yaml"))["names"]
    # LOSO folds have no validation split: "val" points at the held-out sequence and is used only for the final
    # evaluation of the last checkpoint (train(val=False)), never for checkpoint selection.
    val = "valid/images" if "valid" in splits else "test/images"
    yaml.safe_dump({"path": dst, "train": "train/images", "val": val, "test": "test/images",
                    "names": names, "nc": len(names)}, open(f"{dst}/data.yaml", "w"))
    print("  ", {sp: len(os.listdir(f"{dst}/{sp}/images")) for sp in splits})
    return f"{dst}/data.yaml"
'''

TRAIN = '''from ultralytics import YOLO, RTDETR
import gc
Net = RTDETR if MODEL.startswith("rtdetr") else YOLO
extra = {k: HP[k] for k in ("optimizer", "lr0") if k in HP}
prev_h = 0.0
for d, arm in RUNS:
    dyaml = None
    for sd in SEEDS:
        key = f"{TAG}_{MODEL}_{d}_{arm}_s{sd}"
        if done(key): print("skip:", STATE[key]); continue
        if (time.time() - T_START) / 3600 + prev_h > TIME_BUDGET_H:
            print("time budget reached - stop; re-run with this notebook's output added as input to resume"); break
        dyaml = dyaml or prepare(d, arm)
        has_val = yaml.safe_load(open(dyaml))["val"] == "valid/images"
        last = f"{WORK}/runs/{key}/weights/last.pt"; t0 = time.time()
        if os.path.exists(last):
            print("MID-RUN RESUME:", last); m = Net(last); m.train(resume=True)
        else:
            m = Net(HP["weights"])
            m.train(data=dyaml, epochs=2 if QUICK_TEST else HP["epochs"], imgsz=HP["imgsz"], batch=HP["batch"],
                    seed=sd, fraction=0.15 if QUICK_TEST else 1.0, val=has_val, **extra,
                    project=f"{WORK}/runs", name=key, exist_ok=True, plots=False, verbose=False)
        if not has_val:
            del m; m = Net(last)          # fixed epochs: evaluate the last checkpoint, not best.pt
        v = m.val(data=dyaml, split="val", verbose=False) if has_val else None
        t = m.val(data=dyaml, split="test", verbose=False)
        ap = {t.names[int(c)]: round(float(a), 4) for c, a in zip(t.box.ap_class_index, t.box.ap50)}
        row = {"model": MODEL, "dataset": d, "arm": arm, "seed": sd, "epochs": HP["epochs"],
               "mAP50_val": round(v.box.map50, 4) if v else None, "mAP50_test": round(t.box.map50, 4),
               "mAP50_95_test": round(t.box.map, 4),   # mean over classes with test boxes only
               "ap50_test_per_class": {n: ap.get(n, "N/A") for n in t.names.values()},
               "min": round((time.time() - t0) / 60, 1)}
        mark(key, row); print(row); prev_h = (time.time() - t0) / 3600
        del m; gc.collect(); torch.cuda.empty_cache()
'''

SUMMARY = '''rows = [v for k, v in STATE.items() if k.startswith(f"{TAG}_{MODEL}_")]
df = pd.DataFrame(rows)
if len(df):
    df.drop(columns=["ap50_test_per_class"]).to_csv(f"{WORK}/results/{TAG}_{MODEL}_runs.csv", index=False)
    pd.DataFrame([{"dataset": r["dataset"], "arm": r["arm"], "seed": r["seed"], **r["ap50_test_per_class"]} for r in rows]
                 ).to_csv(f"{WORK}/results/{TAG}_{MODEL}_ap50_per_class.csv", index=False)
    agg = df.groupby(["dataset", "arm"])[["mAP50_val", "mAP50_test", "mAP50_95_test"]].agg(["mean", "std", "count"]).round(4)
    display(agg); agg.to_csv(f"{WORK}/results/{TAG}_{MODEL}_summary.csv")
for item in os.listdir(WORK):
    if item not in {"results", "state.json", "runs"}:
        p = os.path.join(WORK, item); shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
print("output:", sorted(os.listdir(WORK)))
'''

LOSO = [("C", f"loso_f{i}") for i in range(4)]
# (part, model, seeds, runs), in queue order
PLAN = [("B", "yolo26s", [42, 0], [("B", "clean2"), ("B", "clean_exact_balanced")]),
        ("B", "rtdetr-l", [42, 0], [("B", "clean2"), ("B", "clean_exact_balanced")]),
        ("C", "yolo26s", [42], LOSO)]
MAN_COLS = {"B": ["split_clean2", "split_clean_exact_balanced"], "C": [f"split_{a}" for _, a in LOSO]}
MAN_SRC = {"B": "manifest_B.csv", "C": "manifest_C_loso.csv"}  # audit_v2.clean2 / audit_v2.loso


def notebook(part, model, seeds, runs):
    what = ("clean2 vs clean_exact_balanced" if part == "B" else
            "leave-one-sequence-out (4 folds, no val split, fixed epochs, last checkpoint)")
    cells = [("markdown", f"# T4-2{part} — E2 source {part}: {what} — {model} × seeds {', '.join(map(str, seeds))}\n"
                          "E2 settings. Inputs: dataset ppe-e2-clean2-manifests; secret ROBOFLOW_API_KEY; Internet ON. "
                          "To resume, add this notebook's previous output as input and run all again."),
             ("code", CONFIG.format(part=part, model=model, seeds=seeds, runs=runs)), ("code", SETUP), ("code", DATA),
             ("code", TRAIN), ("code", SUMMARY)]
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    out = Path(sys.argv[1])
    for part, model, seeds, runs in PLAN:
        slug = f"t4-2{part.lower()}-e2-{model}"
        d = out / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{slug}.ipynb").write_text(json.dumps(notebook(part, model, seeds, runs), indent=1), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"<KAGGLE_USERNAME>/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": ["<KAGGLE_USERNAME>/ppe-e2-clean2-manifests"],
            "competition_sources": [], "kernel_sources": []}, indent=1), encoding="utf-8")
    if len(sys.argv) > 2:
        # manifest dataset: only the columns the notebooks need
        src = Path(__file__).resolve().parents[1] / "results" / "audit_v2" / "clean2"
        mdir = Path(sys.argv[2]); mdir.mkdir(parents=True, exist_ok=True)
        for n, cols in MAN_COLS.items():
            m = pd.read_csv(src / MAN_SRC[n])
            m[["file", "md5"] + cols].to_csv(mdir / f"manifest_{n}.csv", index=False)
        (mdir / "dataset-metadata.json").write_text(json.dumps({
            "title": "ppe-e2-clean2-manifests", "id": "<KAGGLE_USERNAME>/ppe-e2-clean2-manifests",
            "licenses": [{"name": "CC-BY-4.0"}]}, indent=1), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
