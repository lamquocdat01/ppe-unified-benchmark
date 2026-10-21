"""Schema-driven eight-point dataset validation (dataset-agnostic; nothing PPE-specific in code).

 1 class IDs integer and in range; declared names agree with the schema
 2 box coordinates normalised, non-empty and inside the image; labels parse; image/label pairing
 3 compliance-violation pair consistency: label map (members exist, distinct, disjoint), both members
   present in every split, and no object labelled as both members of a pair (IoU >= threshold)
 4 uniform image size (N/A when the schema declares none)
 5 zero cross-split leakage: group key, MD5, copy level 2a (any pair with cosine >= theta, or pHash <= tau)
   and scene level 2b (same file-name family, frame numbers within +-k)
 6 zero corrupt/unreadable images (every image fully decoded)
 7 minimum per-class support in every evaluated split
 8 background-image share per split within tolerance of the split ratio (N/A without background images)

Each check returns PASS / FAIL / N/A with machine-readable detail.
"""
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps

from audit_v2.features import MAX_SIDE, trim_border

from .loaders import LOADERS, dataset_root

REPO = Path(__file__).resolve().parents[1]
CALIBRATION = REPO / "results" / "audit_v2" / "calibration.json"


def _status(ok, na=False):
    return "N/A" if na else ("PASS" if ok else "FAIL")


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def scan_images(recs, want_hash):
    """Single pass over every image: full decode, size, MD5, optional flip-invariant pHash/dHash."""
    import imagehash

    def one(r):
        out = {"size": None, "error": None, "md5": None, "hashes": None}
        try:
            raw = r["path"].read_bytes()
            out["md5"] = hashlib.md5(raw).hexdigest()
            with Image.open(r["path"]) as im:
                im.verify()
            with Image.open(r["path"]) as im:
                im.load()  # full decode
                out["size"] = im.size
                if want_hash:  # same preprocessing as the calibrated audit (audit_v2.features)
                    g = ImageOps.exif_transpose(im).convert("RGB")
                    g.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                    g = trim_border(g)
                    m = ImageOps.mirror(g)
                    out["hashes"] = tuple(np.uint64(int(str(f(x)), 16)) for f, x in
                                          ((imagehash.phash, g), (imagehash.phash, m),
                                           (imagehash.dhash, g), (imagehash.dhash, m)))
        except Exception as e:  # noqa: BLE001 - any decode failure counts as corrupt
            out["error"] = repr(e)[:200]
        return out

    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(one, recs))


def group_keys(schema, recs):
    g = schema.get("group") or {"source": "none"}
    src = g.get("source", "none")
    if src == "manifest":
        man = pd.read_csv(dataset_root(schema) / g["manifest"], encoding="utf-8")
        lut = dict(zip(man[g.get("file_column", "file")], man[g["column"]]))
        return [lut.get(r["path"].name, f"__missing__{r['path'].name}") for r in recs]
    if src == "filename":
        rx = re.compile(g["strip_regex"], re.I)
        return [rx.sub("", r["stem"]) for r in recs]
    return [f"{r['split']}/{r['stem']}" for r in recs]  # no grouping information: every image its own group


def perceptual_config(schema):
    """Copy-level (2a) and scene-level (2b) settings: schema `leakage` overrides calibration.json."""
    lk = dict(schema.get("leakage") or {})
    op = {}
    if CALIBRATION.exists():
        op = json.loads(CALIBRATION.read_text(encoding="utf-8")).get("operating_point", {})
    return {"cos": lk.get("cos", op.get("cos")), "phash": lk.get("phash", op.get("phash")),
            "topk": int(lk.get("topk", op.get("topk", 5))), "scene_k": int(lk.get("scene_k", op.get("scene_k", 0))),
            "feature_cache": lk.get("feature_cache")}


def embeddings(recs, scans, cache_dir):
    """DINOv2 embeddings, looked up by MD5 in an existing feature cache; missing ones are computed."""
    from audit_v2 import features as F
    emb = [None] * len(recs)
    if cache_dir:
        root = Path(os.environ.get("THS_DATASETS", r"D:\Datasets")) / cache_dir
        want = {s["md5"]: k for k, s in enumerate(scans) if s["md5"]}
        for p in sorted(root.glob("*/part_*.npz")):
            d = np.load(p)
            for m, e in zip(d["md5"], d["emb"]):
                k = want.get(str(m))
                if k is not None and emb[k] is None:
                    emb[k] = e
    miss = [k for k, e in enumerate(emb) if e is None and scans[k]["error"] is None]
    if miss:
        import torch
        model, tf = F.build_model()
        with torch.inference_mode():
            for b0 in range(0, len(miss), F.BATCH):
                ks = miss[b0:b0 + F.BATCH]
                x = torch.stack([tf(F.load_one(recs[k]["path"])[2]) for k in ks])
                out = torch.nn.functional.normalize(model(x), dim=1).numpy()
                for k, e in zip(ks, out):
                    emb[k] = e
    return emb, len(miss)


def cross_split_perceptual(recs, scans, cfg):
    """Cross-split pairs of the copy level (any pair with cos >= theta, or pHash <= tau) and the scene level."""
    from audit_v2.audit import hamming_pairs, range_pairs, scene_edges
    ok = [i for i, s in enumerate(scans) if s["hashes"] is not None]
    split = np.array([recs[i]["split"] for i in ok])
    out = {"copy": set(), "scene": set(), "embeddings_computed": 0}
    if cfg["phash"] is not None:
        H = np.array([scans[i]["hashes"] for i in ok], dtype=np.uint64)
        a, b = hamming_pairs(H, H, int(cfg["phash"]), -1, same=True)
        m = split[a] != split[b]
        out["copy"].update(zip(a[m].tolist(), b[m].tolist()))
    if cfg["cos"] is not None:
        emb, out["embeddings_computed"] = embeddings([recs[i] for i in ok], [scans[i] for i in ok], cfg["feature_cache"])
        E = np.stack(emb).astype(np.float32)
        # exhaustive on purpose, NOT top-k: a top-k list misses pairs inside dense clusters of near-identical
        # frames, so a split could pass point 5 while still holding cross-split pairs with cos >= theta
        a, b = range_pairs(E, float(cfg["cos"]))
        m = split[a] != split[b]
        out["copy"].update(zip(a[m].tolist(), b[m].tolist()))
    if cfg["scene_k"] > 0:
        a, b = scene_edges([recs[i]["path"].name for i in ok], cfg["scene_k"])
        m = split[a] != split[b]
        out["scene"].update(zip(a[m].tolist(), b[m].tolist()))
    out["copy"] = sorted((ok[x], ok[y]) for x, y in out["copy"])
    out["scene"] = sorted((ok[x], ok[y]) for x, y in out["scene"])
    return out


def validate(schema_path):
    schema = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
    names = schema["classes"]
    nc = len(names)
    recs, orphans = LOADERS[schema.get("format", "yolo")](schema)
    perc = schema.get("leakage", {}).get("perceptual", "hash")
    scans = scan_images(recs, want_hash=(perc != "none"))
    splits = list(schema["splits"])
    report = {"schema": str(schema_path), "dataset": schema["name"], "images": len(recs),
              "images_per_split": dict(Counter(r["split"] for r in recs)), "checks": {}}

    # 1 class IDs ------------------------------------------------------------------
    bad_cls = []
    cls_of = []  # per record: list of (cid, xc, yc, w, h) with valid integer cid
    for r in recs:
        good = []
        for b in r["boxes"]:
            tok = b[0]
            try:
                v = float(tok)
                cid = int(v) if v == int(v) else None
            except ValueError:
                cid = None
            if cid is None or not 0 <= cid < nc:
                bad_cls.append(f"{r['split']}/{r['stem']}:{tok}")
            else:
                good.append((cid, *b[1:]))
        cls_of.append(good)
    declared = None
    for fn in ("data.yaml", "data.yml"):
        p = dataset_root(schema) / fn
        if p.exists():
            y = yaml.safe_load(p.read_text(encoding="utf-8"))
            nm = y.get("names")
            declared = [nm[k] for k in sorted(nm)] if isinstance(nm, dict) else nm
    names_ok = declared is None or list(declared) == list(names)
    report["checks"]["1_class_ids"] = {"status": _status(not bad_cls and names_ok),
                                       "boxes": sum(len(c) for c in cls_of) + len(bad_cls),
                                       "invalid_ids": len(bad_cls), "examples": bad_cls[:10],
                                       "declared_names_match_schema": names_ok if declared is not None else "no data.yaml"}

    # 2 boxes + parsing + pairing ----------------------------------------------------
    tol = float(schema.get("box_edge_tolerance", 0.0))
    bad_box, edge = [], []
    for r, boxes in zip(recs, cls_of):
        for cid, xc, yc, w, h in boxes:
            if not (0 < xc <= 1 and 0 < yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad_box.append(f"{r['split']}/{r['stem']}")
                continue
            over = max(0.0, w / 2 - xc, xc + w / 2 - 1, h / 2 - yc, yc + h / 2 - 1)
            if over > tol:
                bad_box.append(f"{r['split']}/{r['stem']} (outside by {over:.4f})")
            elif over > 1e-6:
                edge.append(over)
    parse_err = [f"{r['split']}/{r['stem']}:{n}" for r in recs for n, _ in r["parse_errors"]]
    need_pair = schema.get("require_label_file", True)
    pair_bad = orphans["labels_without_image"] + (orphans["images_without_label"] if need_pair else [])
    report["checks"]["2_boxes"] = {
        "status": _status(not bad_box and not parse_err and not pair_bad),
        "invalid_boxes": len(bad_box), "examples": bad_box[:10], "unparseable_lines": len(parse_err),
        "images_without_label": len(orphans["images_without_label"]),
        "labels_without_image": len(orphans["labels_without_image"]),
        "boxes_touching_edge_within_tolerance": len(edge), "edge_tolerance": tol}

    # 3 compliance-violation pairs ----------------------------------------------------
    pairs_ = schema.get("pairs") or []
    if not pairs_:
        report["checks"]["3_pair_consistency"] = {"status": "N/A", "reason": "schema declares no compliance-violation pairs"}
    else:
        problems = []
        members = [m for p in pairs_ for m in p]
        for p in pairs_:
            if len(p) != 2 or p[0] == p[1]:
                problems.append(f"malformed pair {p}")
            for m in p:
                if m not in names:
                    problems.append(f"{m} not in classes")
        dup = [m for m, c in Counter(members).items() if c > 1]
        if dup:
            problems.append(f"class in more than one pair: {dup}")
        box_cnt = defaultdict(Counter)
        for r, boxes in zip(recs, cls_of):
            for b in boxes:
                box_cnt[r["split"]][b[0]] += 1
        missing = [f"{m}@{s}" for p in pairs_ for m in p if m in names
                   for s in splits if box_cnt[s][names.index(m)] == 0]
        thr = float(schema.get("pair_conflict_iou", 0.8))
        conflicts = []
        for r, boxes in zip(recs, cls_of):
            for a, b in pairs_:
                if a not in names or b not in names:
                    continue
                ia, ib = names.index(a), names.index(b)
                A = [x[1:] for x in boxes if x[0] == ia]
                B = [x[1:] for x in boxes if x[0] == ib]
                for x in A:
                    for y in B:
                        if _iou(x, y) >= thr:
                            conflicts.append(f"{r['split']}/{r['stem']} {a}/{b}")
        report["checks"]["3_pair_consistency"] = {
            "status": _status(not problems and not missing and not conflicts),
            "pairs": pairs_, "label_map_problems": problems,
            "pair_members_absent_from_split": missing,
            "same_object_conflicts": len(conflicts), "conflict_iou": thr, "examples": conflicts[:10]}

    # 4 image size --------------------------------------------------------------------
    sizes = Counter(s["size"] for s in scans if s["size"])
    want = schema.get("image_size")
    if want is None:
        report["checks"]["4_image_size"] = {"status": "N/A", "reason": "schema declares no uniform size",
                                            "distinct_sizes": len(sizes), "most_common": [list(map(str, sizes.most_common(3)))]}
    else:
        bad = sum(c for s, c in sizes.items() if tuple(s) != tuple(want))
        report["checks"]["4_image_size"] = {"status": _status(bad == 0), "expected": want, "mismatched": bad}

    # 5 leakage -------------------------------------------------------------------------
    gk = group_keys(schema, recs)
    gs = defaultdict(set)
    for k, r in zip(gk, recs):
        gs[k].add(r["split"])
    grp_leak = sum(1 for v in gs.values() if len(v) > 1)
    ms = defaultdict(set)
    for s, r in zip(scans, recs):
        if s["md5"]:
            ms[s["md5"]].add(r["split"])
    md5_leak = sum(1 for v in ms.values() if len(v) > 1)
    pp, cfg = None, None
    if perc != "none":
        cfg = perceptual_config(schema)
        pp = cross_split_perceptual(recs, scans, cfg)
    ex = lambda L: [f"{recs[a]['split']}/{recs[a]['stem']} ~ {recs[b]['split']}/{recs[b]['stem']}" for a, b in L[:10]]
    report["checks"]["5_leakage"] = {
        "status": _status(grp_leak == 0 and md5_leak == 0 and (pp is None or (not pp["copy"] and not pp["scene"]))),
        "group_source": (schema.get("group") or {}).get("source", "none"),
        "groups": len(gs), "groups_in_multiple_splits": grp_leak, "md5_in_multiple_splits": md5_leak,
        "perceptual_tier": perc, "perceptual_config": {k: v for k, v in (cfg or {}).items() if k != "feature_cache"},
        "copy_level_cross_split_pairs": None if pp is None else len(pp["copy"]),
        "scene_level_cross_split_pairs": None if pp is None else len(pp["scene"]),
        "embeddings_computed_not_cached": None if pp is None else pp["embeddings_computed"],
        "copy_examples": [] if pp is None else ex(pp["copy"]),
        "scene_examples": [] if pp is None else ex(pp["scene"])}

    # 6 corrupt ---------------------------------------------------------------------------
    errs = [f"{r['split']}/{r['stem']}: {s['error']}" for s, r in zip(scans, recs) if s["error"]]
    report["checks"]["6_corrupt"] = {"status": _status(not errs), "decoded": len(scans) - len(errs),
                                     "corrupt": len(errs), "examples": errs[:10]}

    # 7 min support ----------------------------------------------------------------------
    ms_cfg = schema.get("min_support") or {}
    if not ms_cfg:
        report["checks"]["7_min_support"] = {"status": "N/A", "reason": "schema declares no minimum support"}
    else:
        over = schema.get("min_support_overrides") or {}
        exempt = set(schema.get("exempt_classes") or [])
        cnt = defaultdict(Counter)
        for r, boxes in zip(recs, cls_of):
            for b in boxes:
                cnt[r["split"]][b[0]] += 1
        viol = []
        for s, need in ms_cfg.items():
            for i, n in enumerate(names):
                if n in exempt:
                    continue
                req = (over.get(s) or {}).get(n, need)
                if cnt[s][i] < req:
                    viol.append(f"{n}@{s}: {cnt[s][i]} < {req}")
        report["checks"]["7_min_support"] = {"status": _status(not viol), "violations": viol,
                                             "exempt": sorted(exempt), "min_support": ms_cfg}

    # 8 background distribution -------------------------------------------------------------
    target = schema.get("target_ratio")
    bg = Counter(r["split"] for r, boxes in zip(recs, cls_of) if not boxes and not r["parse_errors"])
    nbg = sum(bg.values())
    if not nbg or not target:
        report["checks"]["8_background"] = {"status": "N/A",
                                            "reason": "no background images" if not nbg else "schema declares no split ratio",
                                            "background_images": nbg}
    else:
        tolpp = float(schema.get("background_tolerance_pp", 2.0))
        dev = {s: 100.0 * bg[s] / nbg - 100.0 * target[s] for s in splits}
        worst = max(abs(v) for v in dev.values())
        report["checks"]["8_background"] = {"status": _status(worst <= tolpp), "background_images": nbg,
                                            "per_split": dict(bg), "max_deviation_pp": round(worst, 3),
                                            "tolerance_pp": tolpp}

    st = [c["status"] for c in report["checks"].values()]
    report["summary"] = {"PASS": st.count("PASS"), "FAIL": st.count("FAIL"), "N/A": st.count("N/A")}
    return report
