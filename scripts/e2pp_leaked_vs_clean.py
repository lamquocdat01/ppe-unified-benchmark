"""E2'' (inference only): the same v1-fix models on leaked versus clean test images of the v1 split (24-10-2026).

Test split = PPE-Unified v1-fix test (v1 split, corrected labels). Every test image is assigned by its canonical
v1.1 group (group_v11 = connected component of exact U 2a U 2b at the operating point, 1,770 groups):
  leaked = its group also has images in v1-fix train or val;  clean = all other test images.
With --edges, a second split of the test set: copy-leaked = the image itself has an exact or copy-level (2a)
edge to a train/val image (audit_v2 edge list, FAISS top-5); not copy-leaked = the rest. Group leakage chains
whole video sequences through 2b and covers most of the v1 test split, so the direct split is the sharper test.
Each model is validated once on the test split by Ultralytics itself (conf 0.001, IoU 0.7, 640 px); a recording
validator keeps every image's matched predictions, so 'full' is exactly model.val and the subsets reuse the
same matches (ap_per_class). A leaked/clean pair is compared on the classes with >= --min-gt boxes in both
subsets. 95 % CIs: 1,000 bootstrap resamples over images, per subset, redrawn when a resample lacks a class of the
compared set (count recorded); the leaked-minus-clean difference resamples both subsets independently. Stops if free RAM < --min-free-gb.

Usage:
  python -u scripts/e2pp_leaked_vs_clean.py --v1fix <THS_DATASETS>/PPE_SH17/PPE_Unified_v1fix
      --v11 <THS_DATASETS>/PPE_SH17/PPE_Unified_v1.1 --model yolo26s_s0=<best.pt> [--model ...]
      --cache <dir> --out results/e2pp/e2pp.json [--paper <paper folder>] [--max-hours-rtdetr 1]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


def yaml_names(p):
    import yaml
    nm = yaml.safe_load(open(p, encoding="utf-8"))["names"]
    return [nm[k] for k in sorted(nm)] if isinstance(nm, dict) else list(nm)


def free_gb():
    return psutil.virtual_memory().available / 2**30


class _Abort(Exception):
    pass


def predict(weights, files, cache, data_yaml, max_hours=None, threads=4):
    """Per-image (tp[N,10], conf[N], pred_cls[N], gt_cls[M]) in the order of `files`, recorded inside Ultralytics'
    own test-split validation (so the full-set metric is exactly model.val); cached as npz."""
    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / (Path(weights).parent.parent.name + ".npz")
    if cf.exists():
        z = np.load(cf, allow_pickle=True)
        return list(z["per_image"]), json.loads(str(z["val"]))
    import torch
    torch.set_num_threads(threads)
    from ultralytics import RTDETR, YOLO
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.models.yolo.detect import DetectionValidator
    is_detr = "rtdetr" in str(weights)
    rec, t0 = {}, time.time()

    class Rec(RTDETRValidator if is_detr else DetectionValidator):
        def update_metrics(self, preds, batch):
            for si, pred in enumerate(preds):
                pb, pn = self._prepare_batch(si, batch), self._prepare_pred(pred)
                tp = self._process_batch(pn, pb)["tp"]
                no = pn["cls"].shape[0] == 0
                rec[Path(pb["im_file"]).name] = (
                    np.asarray(tp), np.zeros(0) if no else pn["conf"].cpu().numpy(),
                    np.zeros(0, int) if no else pn["cls"].cpu().numpy().astype(int), pb["cls"].cpu().numpy().astype(int))
            n = len(rec)
            if n and n % 64 == 0:
                if free_gb() < ARGS.min_free_gb:
                    raise SystemExit(f"free RAM {free_gb():.1f} GB < {ARGS.min_free_gb} GB: stop")
                if n == 64:
                    est = (time.time() - t0) / n * len(files) / 3600
                    print(f"  {Path(weights).parent.parent.name}: est. {est:.2f} h", flush=True)
                    if max_hours and est > max_hours:
                        raise _Abort()
            super().update_metrics(preds, batch)

    net = (RTDETR if is_detr else YOLO)(weights)
    try:
        m = net.val(data=str(data_yaml), split="test", imgsz=640, batch=8, conf=0.001, iou=0.7, device="cpu",
                    plots=False, verbose=False, validator=Rec, project=str(cache / "val"), name="run", exist_ok=True)
    except _Abort:
        return None, None
    val = {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)}
    out = [rec[f.name] for f in files]
    arr = np.empty(len(out), dtype=object)
    arr[:] = out
    np.savez_compressed(cf, per_image=arr, val=json.dumps(val))
    print(f"  {Path(weights).parent.parent.name}: {len(files)} images in {(time.time() - t0) / 60:.1f} min; val {val}",
          flush=True)
    return out, val


def metric(per_image, idx, classes=None):
    """mAP50, mAP50-95 over `classes` (None: every class with ground truth, as Ultralytics val)."""
    from ultralytics.utils.metrics import ap_per_class
    tp = np.concatenate([per_image[i][0] for i in idx])
    conf = np.concatenate([per_image[i][1] for i in idx])
    pcl = np.concatenate([per_image[i][2] for i in idx])
    gcl = np.concatenate([per_image[i][3] for i in idx])
    if not len(tp):
        return 0.0, 0.0
    r = ap_per_class(tp, conf, pcl, gcl)
    ap, cls = r[5], r[6]
    if classes is not None:
        ap = ap[np.isin(cls, list(classes))]
    return float(ap[:, 0].mean()), float(ap.mean())


def gt_counts(per_image, idx):
    return np.bincount(np.concatenate([per_image[i][3] for i in idx]).astype(int), minlength=12)


def _ci(b):
    return [round(float(x), 4) for x in np.percentile(b, [2.5, 97.5])]


def boot(per_image, idx, cl, n_boot, rng):
    """Bootstrap over images, conditional on every class of `cl` having ground truth in the resample (a resample
    that drops a class would average over fewer classes); returns draws and the number of redrawn resamples."""
    out, redraw = [], 0
    while len(out) < n_boot:
        s = rng.choice(idx, len(idx))
        if (gt_counts(per_image, s)[cl] == 0).any():
            redraw += 1
            continue
        out.append(metric(per_image, s, cl))
    return np.array(out), redraw


def summarise(per_image, idx_sets, n_boot, rng, min_gt):
    """full: all classes with test ground truth (= val). Each leaked/clean pair: the classes with >= min_gt boxes in
    BOTH subsets, so both means cover the same classes; bootstrap over images within each subset (conditional, see
    boot), the difference resampled independently."""
    res = {}
    cl_full = [int(c) for c in np.where(gt_counts(per_image, idx_sets["full"]) > 0)[0]]
    b, rd = boot(per_image, idx_sets["full"], cl_full, n_boot, rng)
    m = metric(per_image, idx_sets["full"])
    res["full"] = {"mAP50": round(m[0], 4), "mAP50_95": round(m[1], 4), "ci95_mAP50": _ci(b[:, 0]),
                   "ci95_mAP50_95": _ci(b[:, 1]), "classes": cl_full, "redrawn": rd}
    for lk, ck, key in (("leaked", "clean", "leaked_minus_clean"),
                        ("direct_leaked", "direct_clean", "direct_leaked_minus_clean")):
        if lk not in idx_sets:
            continue
        cl = [int(c) for c in np.where((gt_counts(per_image, idx_sets[lk]) >= min_gt)
                                       & (gt_counts(per_image, idx_sets[ck]) >= min_gt))[0]]
        boots = {}
        for s in (lk, ck):
            idx = idx_sets[s]
            boots[s], rd = boot(per_image, idx, cl, n_boot, rng)
            m = metric(per_image, idx, cl)
            res[s] = {"mAP50": round(m[0], 4), "mAP50_95": round(m[1], 4), "classes": cl, "redrawn": rd,
                      "ci95_mAP50": _ci(boots[s][:, 0]), "ci95_mAP50_95": _ci(boots[s][:, 1])}
        d = boots[lk] - boots[ck]
        res[key] = {"classes": cl,
                    "mAP50": round(res[lk]["mAP50"] - res[ck]["mAP50"], 4),
                    "mAP50_95": round(res[lk]["mAP50_95"] - res[ck]["mAP50_95"], 4),
                    "ci95_mAP50": _ci(d[:, 0]), "ci95_mAP50_95": _ci(d[:, 1])}
    return res


def table(R, path):
    lab = {"full": "full test", "leaked": "group-leaked", "clean": "group-clean",
           "direct_leaked": "copy-leaked", "direct_clean": "not copy-leaked"}
    L = [r"\begin{table}[tp]", r"\centering\footnotesize",
         r"\caption{E2$''$: the same v1-fix models on subsets of the v1 test split. Group-leaked: the image's "
         r"near-duplicate group (exact $\cup$ 2a $\cup$ 2b, operating point) also has train or validation images. "
         r"Copy-leaked: the image itself has an exact or copy-level (2a) partner in train or validation. "
         r"Full test: all classes (equals Ultralytics val); each leaked/clean pair: mean over the classes with at "
         r"least " + str(R["min_gt_per_class"]) + r" boxes in both subsets. "
         r"95\% bootstrap CI over images (" + format(R["n_boot"], ",").replace(",", "{,}") + r" resamples); "
         r"$\Delta$ = leaked $-$ clean, in points.}",
         r"\label{tab:e2pp}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrcc}", r"\toprule",
         r"Model & Test subset & Images & Boxes & mAP@50 [95\% CI] & mAP@50--95 [95\% CI] \\", r"\midrule"]
    sep = lambda s: s.replace(",", "{,}")
    f = lambda v, ci: f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    fd = lambda v, ci: f"${100 * v:+.1f}$ [${100 * ci[0]:+.1f}$, ${100 * ci[1]:+.1f}$]"
    S = R["subsets"]
    blocks = [("full",), ("leaked", "clean", "leaked_minus_clean")]
    if "direct_leaked" in S:
        blocks.append(("direct_leaked", "direct_clean", "direct_leaked_minus_clean"))
    for mi, (m, r) in enumerate(R["models"].items()):
        if mi:
            L.append(r"\midrule")
        first = True
        for blk in blocks:
            for k in blk:
                name = m.replace("_", " ") if first else ""
                first = False
                if k.endswith("minus_clean"):
                    dd = r[k]
                    L.append(f"{name} & $\\Delta$ & & & {fd(dd['mAP50'], dd['ci95_mAP50'])} & "
                             f"{fd(dd['mAP50_95'], dd['ci95_mAP50_95'])} \\\\")
                else:
                    L.append(f"{name} & {lab[k]} & {sep(format(S[k]['images'], ','))} & {sep(format(S[k]['boxes'], ','))} & "
                             f"{f(r[k]['mAP50'], r[k]['ci95_mAP50'])} & {f(r[k]['mAP50_95'], r[k]['ci95_mAP50_95'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1fix", required=True)
    ap.add_argument("--v11", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=path/to/best.pt")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--max-hours-rtdetr", type=float, default=1.0)
    ap.add_argument("--edges", help="audit_v2 edges_cross_split_ppeu_fix.csv: adds the direct copy-level subsets")
    ap.add_argument("--theta", type=float, default=0.98)
    ap.add_argument("--tau", type=int, default=0)
    ap.add_argument("--min-gt", type=int, default=10, help="class kept in a leaked/clean comparison if it has at "
                    "least this many boxes in both subsets")
    ap.add_argument("--table-only", action="store_true", help="rewrite tab_e2pp.tex from --out without re-running")
    ARGS = ap.parse_args()
    if ARGS.table_only:
        table(json.loads(Path(ARGS.out).read_text(encoding="utf-8")), Path(ARGS.paper) / "generated" / "tab_e2pp.tex")
        return
    v1fix, v11 = Path(ARGS.v1fix), Path(ARGS.v11)
    split = pd.read_csv(v1fix / "manifest.csv").set_index("file")["split"]
    group = pd.read_csv(v11 / "manifest.csv").set_index("file")["group_v11"]
    nontest_groups = set(group[split[split != "test"].index])
    files = sorted((v1fix / "test" / "images").iterdir())
    leaked = np.array([group[f.name] in nontest_groups for f in files])
    idx = {"full": np.arange(len(files)), "leaked": np.where(leaked)[0], "clean": np.where(~leaked)[0]}
    if ARGS.edges:
        # direct: the test image itself has an exact or copy-level (2a) edge to a train/val image
        e = pd.read_csv(ARGS.edges)
        e = e[e.split_pair.str.contains("test")]
        copy = (e.in_knn & (e.cos >= ARGS.theta)) | (e.ph <= ARGS.tau) | e.exact
        ends = set(e.loc[copy, "a"]) | set(e.loc[copy, "b"])
        direct = np.array([f"test/images/{f.name}" in ends for f in files])
        idx["direct_leaked"], idx["direct_clean"] = np.where(direct)[0], np.where(~direct)[0]
    cls_count = []
    for f in files:
        lf = v1fix / "test" / "labels" / (f.stem + ".txt")
        cs = [int(l.split()[0]) for l in open(lf) if l.strip()] if lf.exists() else []
        cls_count.append(np.bincount(cs, minlength=12))
    cls_count = np.array(cls_count)
    R = {"protocol": __doc__.split("\n\n")[1].strip(), "n_boot": ARGS.boot, "min_gt_per_class": ARGS.min_gt,
         "subsets": {k: {"images": int(len(v)), "boxes": int(cls_count[v].sum()),
                         "boxes_per_class": cls_count[v].sum(0).tolist()} for k, v in idx.items()},
         "models": {}, "skipped": {}}
    print(json.dumps({k: (v["images"], v["boxes"]) for k, v in R["subsets"].items()}), flush=True)
    cache = Path(ARGS.cache)
    cache.mkdir(parents=True, exist_ok=True)
    data_yaml = cache / "v1fix_test.yaml"
    names = yaml_names(v1fix / "data.yaml")
    data_yaml.write_text(f"path: {v1fix.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                         f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    rng = np.random.default_rng(0)
    for spec in ARGS.model:
        name, w = spec.split("=", 1)
        per, val = predict(w, files, cache, data_yaml, ARGS.max_hours_rtdetr if "rtdetr" in w else None, ARGS.threads)
        if per is None:
            R["skipped"][name] = f"CPU estimate > {ARGS.max_hours_rtdetr} h"
            print(name, "skipped", flush=True)
            continue
        R["models"][name] = {"ultralytics_val_full": val, **summarise(per, idx, ARGS.boot, rng, ARGS.min_gt)}
        print(name, json.dumps(R["models"][name]), flush=True)
    Path(ARGS.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ARGS.out).write_text(json.dumps(R, indent=1), encoding="utf-8")
    if ARGS.paper:
        p = Path(ARGS.paper) / "generated" / "numbers.json"
        J = json.loads(p.read_text(encoding="utf-8"))
        J["e2pp_leaked_vs_clean"] = R
        p.write_text(json.dumps(J, indent=1), encoding="utf-8")
        table(R, Path(ARGS.paper) / "generated" / "tab_e2pp.tex")


if __name__ == "__main__":
    main()
