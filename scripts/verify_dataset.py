# -*- coding: utf-8 -*-
"""
verify_dataset.py - VIEC 1 + VIEC 2 + VIEC 4

VIEC 1: Kiem chung dong nhat class ID / toa do / cap anh-label / kich thuoc anh / chong leakage.
VIEC 2: Bang phan bo box & anh theo lop x split, ty le split thuc te, canh bao lech > +-5%.
VIEC 4: Bao cao tom tat cuoi cung (PASS/FAIL + bang phan bo + canh banh con lai).

Chi dung thu vien chuan + Pillow + pandas. Seed 42. Moi I/O dung encoding="utf-8".

Cach dung:
    python verify_dataset.py                       # kiem tra PPE_Unified_v1
    python verify_dataset.py PPE_Unified_v1_resplit
"""

import sys
import os
import random
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image

SEED = 42
random.seed(SEED)

SPLITS = ["train", "val", "test"]
TARGET_RATIO = {"train": 0.70, "val": 0.20, "test": 0.10}
RATIO_TOLERANCE = 5.0          # +-5 diem phan tram
EXPECTED_SIZE = (640, 640)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Rang buoc toi thieu cua VIEC 3 (dung lai de bao cao o VIEC 4)
MIN_TEST_DEFAULT = 100
MIN_VAL_DEFAULT = 150
MIN_TEST_SPECIAL = {"no_vest": 100, "mask": 110}
EXEMPT_CLASSES = {"no_boots"}   # tong chi 37 box -> mien rang buoc

# ----------------------------------------------------------------------------- utils


def _stdout_utf8():
    """Tranh UnicodeEncodeError khi console Windows dung cp1258/cp437."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def banner(title, ch="="):
    print("\n" + ch * 78)
    print(title)
    print(ch * 78)


def load_class_names(root: Path):
    """Doc `names:` trong data.yaml bang parser toi gian (khong can pyyaml)."""
    yml = root / "data.yaml"
    names = {}
    if not yml.exists():
        raise FileNotFoundError(f"Khong tim thay {yml}")
    in_names = False
    with open(yml, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.strip().startswith("names:"):
                in_names = True
                continue
            if in_names:
                if not line.startswith((" ", "\t")):      # het block names
                    break
                body = line.strip()
                if ":" not in body:
                    continue
                k, v = body.split(":", 1)
                try:
                    idx = int(k.strip())
                except ValueError:
                    continue
                names[idx] = v.strip().strip("'\"")
    if not names:
        raise ValueError("Khong doc duoc `names:` tu data.yaml")
    return [names[i] for i in sorted(names)]


def read_label_file(path: Path):
    """Tra ve (rows, parse_errors). rows = list[(cls_str, xc, yc, bw, bh, raw_line, lineno)]."""
    rows, errs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                errs.append((lineno, line, "so truong < 5"))
                continue
            rows.append((parts[0], parts[1:5], line, lineno))
    return rows, errs


def is_int_token(tok: str):
    """'3' -> True ; '3.0' -> True (gia tri nguyen) ; '3.5'/'abc' -> False."""
    try:
        v = float(tok)
    except ValueError:
        return False, None
    if v != int(v):
        return False, None
    return True, int(v)


# ----------------------------------------------------------------------------- scan


def scan_dataset(root: Path, n_classes: int, check_image_size=True, verbose=True):
    """Quet toan bo label + image. Tra ve dict ket qua tho."""
    res = {
        "size_checked": check_image_size,
        "bad_class": [],        # (split, file, lineno, token, ly do)
        "bad_box": [],          # (split, file, lineno, ly do, raw)
        "parse_err": [],
        "orphan_images": [],    # anh khong co label
        "orphan_labels": [],    # label khong co anh
        "bad_size": [],         # (split, file, (w,h))
        "unreadable": [],
        "box_counts": defaultdict(lambda: defaultdict(int)),   # [split][cls] = n box
        "img_counts": defaultdict(lambda: defaultdict(int)),   # [split][cls] = n anh chua lop
        "n_images": defaultdict(int),
        "n_labels": defaultdict(int),
        "n_boxes": defaultdict(int),
        "n_background": defaultdict(int),
        "file_classes": {},     # basename -> set(cls)  (dung cho resplit)
        "file_boxes": {},       # basename -> {cls: n}
        "out_of_frame": 0,      # thong tin them: box tran khoi khung
    }

    for split in SPLITS:
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            print(f"  [!] Thieu thu muc cho split '{split}': {img_dir} / {lbl_dir}")
            continue

        images = {p.stem: p for p in sorted(img_dir.iterdir())
                  if p.is_file() and p.suffix.lower() in IMG_EXTS}
        labels = {p.stem: p for p in sorted(lbl_dir.iterdir())
                  if p.is_file() and p.suffix.lower() == ".txt"}

        res["n_images"][split] = len(images)
        res["n_labels"][split] = len(labels)

        for stem in sorted(set(images) - set(labels)):
            res["orphan_images"].append((split, images[stem].name))
        for stem in sorted(set(labels) - set(images)):
            res["orphan_labels"].append((split, labels[stem].name))

        if verbose:
            print(f"  quet split '{split}': {len(images)} anh / {len(labels)} label ...")

        # --- labels
        for stem, lp in labels.items():
            rows, errs = read_label_file(lp)
            for lineno, line, why in errs:
                res["parse_err"].append((split, lp.name, lineno, why, line))

            per_cls = defaultdict(int)
            for cls_tok, coord_toks, raw, lineno in rows:
                ok_int, cid = is_int_token(cls_tok)
                if not ok_int:
                    res["bad_class"].append((split, lp.name, lineno, cls_tok,
                                             "khong phai so nguyen"))
                    continue
                if not (0 <= cid < n_classes):
                    res["bad_class"].append((split, lp.name, lineno, cls_tok,
                                             f"ngoai khoang [0,{n_classes - 1}]"))
                    continue

                try:
                    xc, yc, bw, bh = (float(t) for t in coord_toks)
                except ValueError:
                    res["bad_box"].append((split, lp.name, lineno,
                                           "toa do khong phai so", raw))
                    continue

                reasons = []
                for nm, v in (("xc", xc), ("yc", yc), ("bw", bw), ("bh", bh)):
                    if not (0.0 < v <= 1.0):
                        reasons.append(f"{nm}={v:g} ngoai (0,1]")
                if bw * bh <= 0:
                    reasons.append("dien tich bw*bh <= 0")
                if reasons:
                    res["bad_box"].append((split, lp.name, lineno,
                                           "; ".join(reasons), raw))
                    continue

                # thong tin them (khong tinh FAIL): box vuot ra ngoai khung anh
                if (xc - bw / 2 < -1e-6 or xc + bw / 2 > 1 + 1e-6
                        or yc - bh / 2 < -1e-6 or yc + bh / 2 > 1 + 1e-6):
                    res["out_of_frame"] += 1

                per_cls[cid] += 1
                res["box_counts"][split][cid] += 1
                res["n_boxes"][split] += 1

            for cid in per_cls:
                res["img_counts"][split][cid] += 1
            if not per_cls:
                res["n_background"][split] += 1

            res["file_classes"][stem] = set(per_cls)
            res["file_boxes"][stem] = dict(per_cls)

        # --- image sizes
        if check_image_size:
            for stem, ip in images.items():
                try:
                    with Image.open(ip) as im:
                        wh = im.size
                except Exception as e:
                    res["unreadable"].append((split, ip.name, repr(e)))
                    continue
                if wh != EXPECTED_SIZE:
                    res["bad_size"].append((split, ip.name, wh))

    return res


def check_leakage(root: Path, res):
    """Doc manifest.csv, kiem tra 1 source_image_id khong nam o 2 split."""
    out = {"ok": True, "leaks": [], "manifest": None, "mismatch": [], "missing": []}
    mp = root / "manifest.csv"
    if not mp.exists():
        out["ok"] = False
        out["error"] = f"Khong tim thay {mp}"
        return out

    man = pd.read_csv(mp, encoding="utf-8")
    out["manifest"] = man
    need = {"file", "split", "source_image_id"}
    if not need.issubset(man.columns):
        out["ok"] = False
        out["error"] = f"manifest.csv thieu cot {need - set(man.columns)}"
        return out

    g = man.groupby("source_image_id")["split"].nunique()
    bad = g[g > 1]
    for sid in bad.index:
        sub = man[man.source_image_id == sid]
        out["leaks"].append((sid, sorted(sub.split.unique()),
                             sorted(sub.file.tolist())[:5]))
    out["ok"] = len(out["leaks"]) == 0

    # doi chieu manifest voi vi tri file thuc te
    actual = {}
    for split in SPLITS:
        d = root / split / "images"
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    actual[p.name] = split
    for fn, sp in zip(man["file"], man["split"]):
        a = actual.get(fn)
        if a is None:
            out["missing"].append(fn)
        elif a != sp:
            out["mismatch"].append((fn, sp, a))
    return out


# ----------------------------------------------------------------------------- report


def print_verification(res, leak, class_names, root):
    n_classes = len(class_names)
    banner(f"VIEC 1 - KIEM CHUNG DATASET: {root}")

    checks = []

    # 1. class id
    n_bad_cls = len(res["bad_class"]) + len(res["parse_err"])
    print(f"\n[1] Class ID nguyen trong [0,{n_classes - 1}]")
    print(f"    Tong box hop le da doc : {sum(res['n_boxes'].values()):,}")
    print(f"    Dong loi class/parse   : {n_bad_cls}")
    if n_bad_cls:
        for rec in (res["parse_err"] + res["bad_class"])[:20]:
            print(f"      - {rec}")
        if n_bad_cls > 20:
            print(f"      ... con {n_bad_cls - 20} dong khac")
    checks.append(("Class ID trong [0,11]", n_bad_cls == 0))

    # 2. toa do
    print("\n[2] Toa do box: xc,yc,bw,bh trong (0,1] va bw*bh > 0")
    print(f"    So box loi             : {len(res['bad_box'])}")
    if res["bad_box"]:
        for rec in res["bad_box"][:20]:
            print(f"      - {rec}")
        if len(res["bad_box"]) > 20:
            print(f"      ... con {len(res['bad_box']) - 20} box khac")
    if res["out_of_frame"]:
        print(f"    (thong tin) box tran mep khung: {res['out_of_frame']:,} "
              f"- hop le voi YOLO, khong tinh FAIL")
    checks.append(("Toa do box hop le", len(res["bad_box"]) == 0))

    # 3. 1-1
    n_orph = len(res["orphan_images"]) + len(res["orphan_labels"])
    print("\n[3] Ghep cap 1-1 anh <-> label")
    for s in SPLITS:
        print(f"    {s:<6} anh={res['n_images'][s]:>6,}  label={res['n_labels'][s]:>6,}")
    print(f"    Anh khong co label     : {len(res['orphan_images'])}")
    print(f"    Label khong co anh     : {len(res['orphan_labels'])}")
    for rec in (res["orphan_images"] + res["orphan_labels"])[:20]:
        print(f"      - {rec}")
    checks.append(("Ghep cap 1-1 anh/label", n_orph == 0))

    # 4. kich thuoc
    total_img = sum(res["n_images"].values())
    n_bad_sz = len(res["bad_size"]) + len(res["unreadable"])
    print(f"\n[4] Kich thuoc anh = {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")
    if not res.get("size_checked", True):
        # KHONG duoc bao PASS khi chua he mo anh nao -> se gay hieu nham nghiem trong
        print("    BO QUA (--no-size-check) - khong mo anh nao, KHONG ket luan PASS.")
        checks.append(("100% anh 640x640", None))
    else:
        print(f"    Da kiem tra            : {total_img:,} anh")
        print(f"    Sai kich thuoc         : {len(res['bad_size'])}")
        print(f"    Khong doc duoc         : {len(res['unreadable'])}")
        for rec in (res["bad_size"] + res["unreadable"])[:20]:
            print(f"      - {rec}")
        if n_bad_sz > 20:
            print(f"      ... con {n_bad_sz - 20} anh khac")
        checks.append(("100% anh 640x640", n_bad_sz == 0))

    # 5. leakage
    print("\n[5] Chong leakage theo source_image_id (manifest.csv)")
    if "error" in leak:
        print(f"    LOI: {leak['error']}")
    else:
        man = leak["manifest"]
        print(f"    Dong manifest          : {len(man):,}")
        print(f"    So source_image_id     : {man.source_image_id.nunique():,}")
        print(f"    ID nam o >1 split      : {len(leak['leaks'])}")
        for rec in leak["leaks"][:20]:
            print(f"      - {rec}")
        print(f"    File manifest lech vi tri thuc te : {len(leak['mismatch'])}")
        for rec in leak["mismatch"][:10]:
            print(f"      - {rec}")
        print(f"    File trong manifest khong ton tai  : {len(leak['missing'])}")
        for rec in leak["missing"][:10]:
            print(f"      - {rec}")
    ok_leak = (leak.get("ok", False) and not leak.get("mismatch")
               and not leak.get("missing"))
    checks.append(("Khong leakage giua cac split", ok_leak))

    # --- xuat DANH SACH DAY DU cac vi pham ra CSV (console chi in 20 dong dau)
    dumps = [
        ("size_violations.csv", res["bad_size"] + [(s, f, "unreadable")
                                                   for s, f, _ in res["unreadable"]],
         ["split", "file", "size"]),
        ("class_id_violations.csv", res["bad_class"],
         ["split", "file", "line", "token", "reason"]),
        ("box_violations.csv", res["bad_box"],
         ["split", "file", "line", "reason", "raw"]),
        ("orphan_files.csv",
         [("image_khong_co_label", s, f) for s, f in res["orphan_images"]]
         + [("label_khong_co_anh", s, f) for s, f in res["orphan_labels"]],
         ["loai", "split", "file"]),
    ]
    for fname, rows, cols in dumps:
        p = root / fname
        if rows:
            pd.DataFrame(rows, columns=cols).to_csv(p, index=False, encoding="utf-8")
            print(f"\n  Danh sach day du ({len(rows):,} dong) -> {p}")
        elif p.exists():
            p.unlink()          # khong con vi pham -> xoa file cu cho khoi hieu nham

    print("\n--- KET QUA VIEC 1 ---")
    for name, ok in checks:
        print(f"  [{'SKIP' if ok is None else ('PASS' if ok else 'FAIL')}] {name}")
    return checks


def build_distribution(res, class_names):
    """Tra ve DataFrame phan bo (VIEC 2)."""
    rows = []
    for cid, cname in enumerate(class_names):
        b = {s: res["box_counts"][s].get(cid, 0) for s in SPLITS}
        i = {s: res["img_counts"][s].get(cid, 0) for s in SPLITS}
        tb = sum(b.values())
        ti = sum(i.values())
        row = {"class_id": cid, "class_name": cname}
        for s in SPLITS:
            row[f"boxes_{s}"] = b[s]
        row["boxes_total"] = tb
        for s in SPLITS:
            row[f"images_{s}"] = i[s]
        row["images_total"] = ti
        for s in SPLITS:
            row[f"pct_{s}"] = round(100.0 * b[s] / tb, 2) if tb else 0.0
        devs = {}
        for s in SPLITS:
            pct = 100.0 * b[s] / tb if tb else 0.0
            devs[s] = pct - TARGET_RATIO[s] * 100.0
            row[f"dev_{s}"] = round(devs[s], 2)
        worst = max(abs(v) for v in devs.values()) if tb else 0.0
        row["max_abs_dev"] = round(worst, 2)
        row["flag_ratio"] = "LECH" if worst > RATIO_TOLERANCE else "OK"
        rows.append(row)

    df = pd.DataFrame(rows)

    tot = {"class_id": "", "class_name": "TOTAL"}
    for s in SPLITS:
        tot[f"boxes_{s}"] = int(df[f"boxes_{s}"].sum())
    tot["boxes_total"] = int(df["boxes_total"].sum())
    for s in SPLITS:
        tot[f"images_{s}"] = res["n_images"][s]
    tot["images_total"] = sum(res["n_images"].values())
    gt = tot["boxes_total"]
    for s in SPLITS:
        tot[f"pct_{s}"] = round(100.0 * tot[f"boxes_{s}"] / gt, 2) if gt else 0.0
        tot[f"dev_{s}"] = round(tot[f"pct_{s}"] - TARGET_RATIO[s] * 100.0, 2)
    tot["max_abs_dev"] = round(max(abs(tot[f"dev_{s}"]) for s in SPLITS), 2)
    tot["flag_ratio"] = "LECH" if tot["max_abs_dev"] > RATIO_TOLERANCE else "OK"

    df_out = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    return df, df_out


def md_table(rows, header, aligns=None):
    if aligns is None:
        aligns = ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def print_distribution(df, res, root, write_csv=True):
    banner("VIEC 2 - BANG PHAN BO")

    # (a) box
    print("\n(a) So BOX moi lop theo split (kem % trong tong so box cua lop)\n")
    rows = []
    for _, r in df.iterrows():
        rows.append([
            int(r.class_id), r.class_name,
            f"{int(r.boxes_train):,}", f"{int(r.boxes_val):,}",
            f"{int(r.boxes_test):,}", f"{int(r.boxes_total):,}",
            f"{r.pct_train:.1f}%", f"{r.pct_val:.1f}%", f"{r.pct_test:.1f}%",
        ])
    tb = int(df.boxes_total.sum())
    rows.append(["", "**TOTAL**",
                 f"**{int(df.boxes_train.sum()):,}**",
                 f"**{int(df.boxes_val.sum()):,}**",
                 f"**{int(df.boxes_test.sum()):,}**",
                 f"**{tb:,}**",
                 f"**{100 * df.boxes_train.sum() / tb:.1f}%**",
                 f"**{100 * df.boxes_val.sum() / tb:.1f}%**",
                 f"**{100 * df.boxes_test.sum() / tb:.1f}%**"])
    print(md_table(rows,
                   ["id", "lop", "train", "val", "test", "tong",
                    "%train", "%val", "%test"],
                   ["---:", "---", "---:", "---:", "---:", "---:",
                    "---:", "---:", "---:"]))

    # (b) anh
    print("\n(b) So ANH chua moi lop theo split\n")
    rows = []
    for _, r in df.iterrows():
        rows.append([int(r.class_id), r.class_name,
                     f"{int(r.images_train):,}", f"{int(r.images_val):,}",
                     f"{int(r.images_test):,}", f"{int(r.images_total):,}"])
    rows.append(["", "**anh trong split**",
                 f"**{res['n_images']['train']:,}**",
                 f"**{res['n_images']['val']:,}**",
                 f"**{res['n_images']['test']:,}**",
                 f"**{sum(res['n_images'].values()):,}**"])
    rows.append(["", "anh background (0 box)",
                 f"{res['n_background']['train']:,}",
                 f"{res['n_background']['val']:,}",
                 f"{res['n_background']['test']:,}",
                 f"{sum(res['n_background'].values()):,}"])
    print(md_table(rows, ["id", "lop", "train", "val", "test", "tong"],
                   ["---:", "---", "---:", "---:", "---:", "---:"]))

    # (c) do lech so voi 70/20/10
    print("\n(c) Ty le split thuc te vs muc tieu 70/20/10 (danh dau lech > +-5 diem %)\n")
    rows = []
    n_lech = 0
    for _, r in df.iterrows():
        mark = "**LECH**" if r.flag_ratio == "LECH" else "OK"
        if r.flag_ratio == "LECH":
            n_lech += 1
        rows.append([int(r.class_id), r.class_name,
                     f"{r.pct_train:.1f}/{r.pct_val:.1f}/{r.pct_test:.1f}",
                     f"{r.dev_train:+.1f}", f"{r.dev_val:+.1f}",
                     f"{r.dev_test:+.1f}", f"{r.max_abs_dev:.1f}", mark])
    print(md_table(rows,
                   ["id", "lop", "train/val/test %", "d_train", "d_val",
                    "d_test", "max|d|", "danh gia"],
                   ["---:", "---", ":---:", "---:", "---:", "---:", "---:", ":---:"]))
    print(f"\n  => {n_lech}/{len(df)} lop lech qua +-{RATIO_TOLERANCE:.0f} diem phan tram.")

    if write_csv:
        out = root / "distribution_report.csv"
        _, df_full = build_distribution_from_df(df, res)
        df_full.to_csv(out, index=False, encoding="utf-8")
        print(f"\n  Da ghi: {out}")
    return n_lech


def build_distribution_from_df(df, res):
    """Ghep dong TOTAL vao df de xuat CSV."""
    tot = {"class_id": "", "class_name": "TOTAL"}
    for s in SPLITS:
        tot[f"boxes_{s}"] = int(df[f"boxes_{s}"].sum())
    tot["boxes_total"] = int(df["boxes_total"].sum())
    for s in SPLITS:
        tot[f"images_{s}"] = res["n_images"][s]
    tot["images_total"] = sum(res["n_images"].values())
    gt = tot["boxes_total"]
    for s in SPLITS:
        tot[f"pct_{s}"] = round(100.0 * tot[f"boxes_{s}"] / gt, 2) if gt else 0.0
        tot[f"dev_{s}"] = round(tot[f"pct_{s}"] - TARGET_RATIO[s] * 100.0, 2)
    tot["max_abs_dev"] = round(max(abs(tot[f"dev_{s}"]) for s in SPLITS), 2)
    tot["flag_ratio"] = "LECH" if tot["max_abs_dev"] > RATIO_TOLERANCE else "OK"
    tot = {k: tot.get(k, "") for k in df.columns}
    return df, pd.concat([df, pd.DataFrame([tot])], ignore_index=True)


def min_requirements(class_names):
    """Tra ve {class_name: (min_test, min_val)} theo rang buoc VIEC 3."""
    req = {}
    for c in class_names:
        if c in EXEMPT_CLASSES:
            continue
        req[c] = (MIN_TEST_SPECIAL.get(c, MIN_TEST_DEFAULT), MIN_VAL_DEFAULT)
    return req


def check_min_constraints(df, class_names):
    """Kiem tra rang buoc toi thieu box o test/val. Tra ve (ok, list vi pham)."""
    req = min_requirements(class_names)
    viol = []
    for _, r in df.iterrows():
        c = r.class_name
        if c not in req:
            continue
        mt, mv = req[c]
        if int(r.boxes_test) < mt:
            viol.append((c, "test", int(r.boxes_test), mt))
        if int(r.boxes_val) < mv:
            viol.append((c, "val", int(r.boxes_val), mv))
    return len(viol) == 0, viol


def print_final_summary(root, checks, df, res, n_lech, class_names):
    banner("VIEC 4 - BAO CAO TOM TAT CUOI CUNG", "#")
    print(f"Dataset: {root}")
    print(f"Tong: {sum(res['n_images'].values()):,} anh / "
          f"{sum(res['n_boxes'].values()):,} box / {len(class_names)} lop")

    print("\n[A] KET QUA KIEM TRA")
    all_ok = True
    for name, ok in checks:
        all_ok = all_ok and (ok is True)     # SKIP (None) khong duoc tinh la dat
        print(f"  [{'SKIP' if ok is None else ('PASS' if ok else 'FAIL')}] {name}")

    ok_min, viol = check_min_constraints(df, class_names)
    print(f"  [{'PASS' if ok_min else 'FAIL'}] Rang buoc toi thieu "
          f"(test>={MIN_TEST_DEFAULT} box, val>={MIN_VAL_DEFAULT} box; "
          f"no_vest test>=100, mask test>=110; mien tru no_boots)")
    for c, s, got, need in viol:
        print(f"        - {c}: {s}={got} < {need}")
    print(f"  [{'PASS' if n_lech == 0 else 'WARN'}] Ty le split moi lop trong "
          f"+-{RATIO_TOLERANCE:.0f} diem % so voi 70/20/10 ({n_lech} lop lech)")

    print("\n[B] BANG PHAN BO CUOI CUNG (box)\n")
    rows = []
    for _, r in df.iterrows():
        rows.append([int(r.class_id), r.class_name,
                     f"{int(r.boxes_train):,}", f"{int(r.boxes_val):,}",
                     f"{int(r.boxes_test):,}", f"{int(r.boxes_total):,}",
                     f"{r.pct_train:.1f}/{r.pct_val:.1f}/{r.pct_test:.1f}"])
    tb = int(df.boxes_total.sum())
    rows.append(["", "TOTAL", f"{int(df.boxes_train.sum()):,}",
                 f"{int(df.boxes_val.sum()):,}", f"{int(df.boxes_test.sum()):,}",
                 f"{tb:,}",
                 f"{100 * df.boxes_train.sum() / tb:.1f}/"
                 f"{100 * df.boxes_val.sum() / tb:.1f}/"
                 f"{100 * df.boxes_test.sum() / tb:.1f}"])
    print(md_table(rows, ["id", "lop", "train", "val", "test", "tong", "%t/v/te"],
                   ["---:", "---", "---:", "---:", "---:", "---:", ":---:"]))

    print("\n[C] CANH BAO CON LAI")
    warns = []
    for _, r in df.iterrows():
        if r.class_name in EXEMPT_CLASSES:
            warns.append(
                f"{r.class_name}: chi {int(r.boxes_total)} box toan dataset "
                f"({int(r.boxes_train)}/{int(r.boxes_val)}/{int(r.boxes_test)}). "
                f"GIU trong dataset nhung PHAI LOAI khoi metric tong hop "
                f"(mAP50-95 trung binh) khi danh gia - chi bao cao rieng, "
                f"coi la 'khong du du lieu ket luan'.")
    thin = df[(df.boxes_total < 1500) & (~df.class_name.isin(EXEMPT_CLASSES))]
    if len(thin):
        warns.append("Lop it du lieu (<1500 box) - can class-weighted loss / "
                     "oversampling va bao cao per-class: "
                     + ", ".join(f"{r.class_name}({int(r.boxes_total)})"
                                 for _, r in thin.iterrows()))
    imb = df.boxes_total.max() / max(1, df[df.boxes_total > 0].boxes_total.min())
    warns.append(f"Mat can bang lop cuc dai: ty le lop nhieu nhat/it nhat = {imb:.0f}x "
                 f"-> dung mosaic + copy-paste augment va theo doi per-class recall.")
    warns.append("vest/no_vest va boots/no_boots chi den tu 1 nguon (domain bias) - "
                 "ket qua tren 2 cap nay khong suy rong ra domain khac.")
    warns.append("Near-duplicate xuyen nguon chi duoc loai khi trung byte (MD5); "
                 "van co the con anh gan giong nhau giua cac split.")
    if res["out_of_frame"]:
        warns.append(f"{res['out_of_frame']:,} box co canh tran ra ngoai khung anh "
                     f"(hop le voi YOLO sau letterbox, khong can sua).")
    for w in warns:
        print(f"  - {w}")

    print("\n[D] KET LUAN")
    verdict = "PASS TOAN BO" if (all_ok and ok_min) else "CON MUC CHUA DAT"
    print(f"  ==> {verdict}")
    return all_ok and ok_min


# ----------------------------------------------------------------------------- main


def run(root: Path, check_image_size=True, write_csv=True, final_summary=True,
        verbose=True):
    class_names = load_class_names(root)
    if verbose:
        print(f"Class ({len(class_names)}): {', '.join(class_names)}")
    res = scan_dataset(root, len(class_names), check_image_size, verbose)
    leak = check_leakage(root, res)
    checks = print_verification(res, leak, class_names, root)
    df, _ = build_distribution(res, class_names)
    n_lech = print_distribution(df, res, root, write_csv)
    ok = None
    if final_summary:
        ok = print_final_summary(root, checks, df, res, n_lech, class_names)
    return {"res": res, "leak": leak, "checks": checks, "df": df,
            "n_lech": n_lech, "class_names": class_names, "all_pass": ok}


def main():
    _stdout_utf8()
    ap = argparse.ArgumentParser(description="Kiem chung + thong ke dataset YOLO PPE")
    ap.add_argument("dataset", nargs="?", default="PPE_Unified_v1",
                    help="thu muc dataset (mac dinh: PPE_Unified_v1)")
    ap.add_argument("--no-size-check", action="store_true",
                    help="bo qua kiem tra kich thuoc anh (nhanh hon)")
    ap.add_argument("--no-csv", action="store_true",
                    help="khong ghi distribution_report.csv")
    a = ap.parse_args()

    root = Path(a.dataset)
    if not root.is_absolute():
        root = Path(os.path.join(os.environ.get("THS_DATASETS", r"D:\Datasets"), "PPE_SH17")) / a.dataset  # data dir (moved 28-09-2026)
    if not root.is_dir():
        print(f"LOI: khong tim thay dataset {root}")
        sys.exit(2)

    out = run(root, check_image_size=not a.no_size_check, write_csv=not a.no_csv)
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
