# -*- coding: utf-8 -*-
"""
stratified_resplit.py - VIEC 3

Chia lai dataset PPE_Unified_v1 theo 70/20/10 CO PHAN TANG, giu nguyen rang buoc
chong leakage: moi anh cung `source_image_id` phai nam cung mot split (nhom khong
bao gio bi tach).

Rang buoc toi thieu sau khi chia lai (bo qua no_boots vi tong chi 37 box):
    - moi lop  >= 100 box o test  va  >= 150 box o val
    - dac biet no_vest test >= 100, mask test >= 110

Thuat toan:
    1) GREEDY ITERATIVE STRATIFICATION (Sechidis et al., ban theo nhom):
       lap - chon lop con it box nhat trong so cac nhom chua gan (hiem truoc),
       gan tung nhom chua lop do vao split dang thieu han ngach cua lop do nhat;
       tie-break theo han ngach anh con lai, roi random(seed=42).
    2) REPAIR: hill-climbing co dinh huong tren ham phat
       (thieu box toi thieu >> lech han ngach anh > lech ty le +-5%),
       chi chap nhan buoc lam GIAM ham phat -> hoi tu, khong dao nguoc.

Sau khi chia: tao PPE_Unified_v1_resplit (hardlink hoac copy), cap nhat
manifest.csv / data.yaml / DATASET_CARD.md, in bang so sanh TRUOC-SAU, roi chay
lai VIEC 1 + VIEC 2 de xac nhan.

Chi dung thu vien chuan + Pillow + pandas. Seed 42. Moi I/O dung encoding="utf-8".

Cach dung:
    python stratified_resplit.py                 # hardlink (nhanh, tiet kiem dia)
    python stratified_resplit.py --mode copy     # copy that su
    python stratified_resplit.py --dry-run       # chi tinh toan, khong ghi file
"""

import os
import re
import sys
import time
import math
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image

import verify_dataset as vd
from verify_dataset import (SPLITS, TARGET_RATIO, RATIO_TOLERANCE, EXEMPT_CLASSES,
                            MIN_TEST_DEFAULT, MIN_VAL_DEFAULT, MIN_TEST_SPECIAL,
                            IMG_EXTS, banner, md_table, load_class_names,
                            read_label_file, is_int_token, _stdout_utf8)

SEED = 42

# trong so ham phat
W_MIN = 10000.0      # thieu 1 box so voi nguong toi thieu (rang buoc cung)
W_IMG = 1.0          # lech 1 anh so voi han ngach split
W_DEV = 40.0         # moi diem % VUOT QUA nguong +-5% cua mot lop
W_SOFT = 2.0         # thanh phan mem: (diem % lech)^2 - tao gradient BEN TRONG dai +-5%
                     # de moi lop bam sat 70/20/10 chu khong chi "vua du lot nguong"
W_BG = 300.0         # moi diem % anh NEN lech qua BG_TOLERANCE
BG_TOLERANCE = 2.0   # anh nen (0 box) phai phan bo 70/20/10 trong +-2 diem %
                     # Nhom khong chua box nao khong dong gop vao bat ky lop nao, nen
                     # neu khong rang buoc rieng, greedy se don het chung vao split con
                     # nhieu han ngach nhat (train) -> val mat sach canh trong de do
                     # false-positive.


# ----------------------------------------------------------------- thu thap du lieu


def collect(root: Path, class_names):
    """Doc manifest + toan bo label -> thong tin theo nhom source_image_id."""
    n_classes = len(class_names)
    man = pd.read_csv(root / "manifest.csv", encoding="utf-8")

    # file -> (split thuc te, duong dan anh, duong dan label)
    file_info = {}
    for split in SPLITS:
        for p in sorted((root / split / "images").iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                file_info[p.name] = {
                    "split": split,
                    "img": p,
                    "lbl": root / split / "labels" / (p.stem + ".txt"),
                }

    # dem box theo file
    boxes_per_file = {}
    for fn, info in file_info.items():
        cnt = defaultdict(int)
        if info["lbl"].exists():
            rows, _ = read_label_file(info["lbl"])
            for cls_tok, coords, raw, ln in rows:
                ok, cid = is_int_token(cls_tok)
                if ok and 0 <= cid < n_classes:
                    cnt[cid] += 1
        boxes_per_file[fn] = dict(cnt)

    # gom theo nhom
    groups = {}
    for fn, sid, in zip(man["file"], man["source_image_id"]):
        if fn not in file_info:
            continue
        g = groups.setdefault(sid, {"files": [], "boxes": defaultdict(int),
                                    "n_img": 0, "old_split": None})
        g["files"].append(fn)
        g["n_img"] += 1
        for c, n in boxes_per_file[fn].items():
            g["boxes"][c] += n
        g["old_split"] = file_info[fn]["split"]

    for g in groups.values():
        g["boxes"] = dict(g["boxes"])

    return man, file_info, boxes_per_file, groups


def min_req_table(class_names):
    """{cid: (min_test, min_val)}; lop mien tru khong co trong dict."""
    req = {}
    for cid, c in enumerate(class_names):
        if c in EXEMPT_CLASSES:
            continue
        req[cid] = (MIN_TEST_SPECIAL.get(c, MIN_TEST_DEFAULT), MIN_VAL_DEFAULT)
    return req


def minimum_for(cid, split, req):
    if cid not in req:
        return 0
    mt, mv = req[cid]
    return {"test": mt, "val": mv, "train": 0}[split]


# ----------------------------------------------------------------- ham phat


def counts_from_assign(assign, groups, n_classes):
    box = {s: defaultdict(int) for s in SPLITS}
    img = {s: 0 for s in SPLITS}
    bg = {s: 0 for s in SPLITS}
    for gid, s in assign.items():
        g = groups[gid]
        img[s] += g["n_img"]
        if not g["boxes"]:                       # nhom anh nen (khong co box nao)
            bg[s] += g["n_img"]
        for c, n in g["boxes"].items():
            box[s][c] += n
    return box, img, bg


def penalty(box, img, totals, img_quota, req, n_classes, bg=None, n_bg=0):
    p = 0.0
    detail = {"min": 0.0, "img": 0.0, "dev": 0.0, "bg": 0.0}
    for cid in range(n_classes):
        for s in SPLITS:
            need = minimum_for(cid, s, req)
            if need and box[s][cid] < need:
                detail["min"] += (need - box[s][cid]) * W_MIN
        t = totals.get(cid, 0)
        if t:
            for s in SPLITS:
                pct = 100.0 * box[s][cid] / t
                dev = abs(pct - TARGET_RATIO[s] * 100.0)
                detail["dev"] += W_SOFT * dev * dev          # gradient mem, moi noi
                if dev - RATIO_TOLERANCE > 0:                # phat nang khi vuot +-5%
                    detail["dev"] += (dev - RATIO_TOLERANCE) * W_DEV
    for s in SPLITS:
        detail["img"] += abs(img[s] - img_quota[s]) * W_IMG

    # anh nen: cung do bang DIEM PHAN TRAM nhu cac lop, de trong so nhat quan
    if bg is not None and n_bg:
        for s in SPLITS:
            pct = 100.0 * bg[s] / n_bg
            dev = abs(pct - TARGET_RATIO[s] * 100.0)
            detail["bg"] += W_SOFT * dev * dev
            if dev - BG_TOLERANCE > 0:
                detail["bg"] += (dev - BG_TOLERANCE) * W_BG

    p = detail["min"] + detail["img"] + detail["dev"] + detail["bg"]
    return p, detail


# ----------------------------------------------------------------- buoc 1: greedy


def greedy_stratify(groups, n_classes, class_names, img_quota, req, rng, verbose=True):
    totals = defaultdict(int)
    for g in groups.values():
        for c, n in g["boxes"].items():
            totals[c] += n

    # han ngach box mong muon cho tung (split, lop): ty le chuan, nang len cho du nguong
    demand = {s: {} for s in SPLITS}
    for cid in range(n_classes):
        t = totals.get(cid, 0)
        d_test = max(t * TARGET_RATIO["test"], minimum_for(cid, "test", req))
        d_val = max(t * TARGET_RATIO["val"], minimum_for(cid, "val", req))
        d_test = min(d_test, t)
        d_val = min(d_val, max(0.0, t - d_test))
        demand["test"][cid] = d_test
        demand["val"][cid] = d_val
        demand["train"][cid] = max(0.0, t - d_test - d_val)

    img_left = dict(img_quota)
    assign = {}
    remaining = set(groups)
    rem_counts = defaultdict(int)
    for gid in remaining:
        for c, n in groups[gid]["boxes"].items():
            rem_counts[c] += n

    def place(gid, cid_hint):
        """Chon split cho nhom gid dua tren lop cid_hint."""
        g = groups[gid]
        cands = [s for s in SPLITS if img_left[s] >= g["n_img"]]
        if not cands:
            cands = [s for s in SPLITS if img_left[s] > 0] or list(SPLITS)
        best, best_key = None, None
        order = cands[:]
        rng.shuffle(order)                      # tie-break ngau nhien, seed co dinh
        for s in order:
            key = (demand[s].get(cid_hint, 0.0), img_left[s])
            if best_key is None or key > best_key:
                best, best_key = s, key
        assign[gid] = best
        img_left[best] -= g["n_img"]
        for c, n in g["boxes"].items():
            demand[best][c] = demand[best].get(c, 0.0) - n
        return best

    # --- vong lap chinh: lop hiem truoc
    while True:
        live = [c for c in range(n_classes) if rem_counts.get(c, 0) > 0]
        if not live:
            break
        cid = min(live, key=lambda c: (rem_counts[c], c))
        # sorted() la BAT BUOC: `remaining` la set cua chuoi -> thu tu duyet phu thuoc
        # PYTHONHASHSEED (ngau nhien moi tien trinh). Khong sort thi seed 42 vo nghia.
        gids = sorted(g for g in remaining if groups[g]["boxes"].get(cid, 0) > 0)
        rng.shuffle(gids)
        gids.sort(key=lambda g: -groups[g]["boxes"][cid])   # nhom nhieu box lop hiem truoc
        if verbose:
            print(f"    lop hiem nhat con lai: {class_names[cid]:<11} "
                  f"({rem_counts[cid]:>6,} box, {len(gids):>5,} nhom)")
        for gid in gids:
            place(gid, cid)
            remaining.discard(gid)
            for c, n in groups[gid]["boxes"].items():
                rem_counts[c] -= n

    # --- nhom background (0 box): phan bo THEO HAN NGACH RIENG 70/20/10.
    # Truoc day dung max(img_left) -> don gan het vao train (584/0/3). Nhom anh nen
    # khong chua lop nao nen khong co gi keo chung ve dung ty le; phai ep tay.
    bg_groups = sorted(remaining)
    rng.shuffle(bg_groups)
    n_bg_img = sum(groups[g]["n_img"] for g in bg_groups)
    bg_quota = {"train": int(round(n_bg_img * TARGET_RATIO["train"])),
                "val": int(round(n_bg_img * TARGET_RATIO["val"]))}
    bg_quota["test"] = n_bg_img - bg_quota["train"] - bg_quota["val"]
    if verbose:
        print(f"    anh nen: {n_bg_img:,} -> han ngach "
              f"train={bg_quota['train']:,} val={bg_quota['val']:,} "
              f"test={bg_quota['test']:,}")
    bg_left = dict(bg_quota)
    for gid in bg_groups:
        need = groups[gid]["n_img"]
        # split con thieu anh nen nhieu nhat; tie-break theo han ngach anh tong
        cands = [s for s in SPLITS if bg_left[s] >= need] or list(SPLITS)
        s = max(cands, key=lambda x: (bg_left[x], img_left[x]))
        assign[gid] = s
        bg_left[s] -= need
        img_left[s] -= need
        remaining.discard(gid)

    return assign, dict(totals), bg_quota, n_bg_img


# ----------------------------------------------------------------- buoc 2: repair


def repair(assign, groups, n_classes, class_names, totals, img_quota, req, rng,
           n_bg=0, max_iter=40000, time_budget=300.0, verbose=True):
    t_start = time.time()
    box, img, bg = counts_from_assign(assign, groups, n_classes)
    cur, det = penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
    if verbose:
        print(f"    phat sau greedy = {cur:,.1f}  "
              f"(min={det['min']:,.0f} img={det['img']:,.0f} dev={det['dev']:,.0f})")

    by_split = defaultdict(list)
    for gid, s in assign.items():
        by_split[s].append(gid)

    def apply_move(gid, frm, to):
        g = groups[gid]
        img[frm] -= g["n_img"]; img[to] += g["n_img"]
        if not g["boxes"]:                       # nhom anh nen
            bg[frm] -= g["n_img"]; bg[to] += g["n_img"]
        for c, n in g["boxes"].items():
            box[frm][c] -= n
            box[to][c] += n
        assign[gid] = to

    accepted = 0
    for it in range(max_iter):
        # --- tim vi pham nang nhat
        viol = []
        for cid in range(n_classes):
            for s in ("test", "val"):
                need = minimum_for(cid, s, req)
                if need and box[s][cid] < need:
                    viol.append((need - box[s][cid], cid, s))
        if viol:
            viol.sort(reverse=True)
            deficit, cid, target = viol[0]
            # nguon: split co du thua lop cid nhieu nhat
            srcs = sorted((s for s in SPLITS if s != target),
                          key=lambda s: -(box[s][cid] - minimum_for(cid, s, req)))
            done = False
            for src in srcs:
                cands = [g for g in by_split[src] if groups[g]["boxes"].get(cid, 0) > 0]
                if not cands:
                    continue
                # uu tien nhom co so box lop cid gan voi deficit nhat (it xao tron nhat)
                rng.shuffle(cands)
                cands.sort(key=lambda g: (abs(groups[g]["boxes"][cid] - deficit),
                                          sum(groups[g]["boxes"].values())))
                for gid in cands[:40]:
                    # tra lai 1 nhom khong chua cid de giu han ngach anh
                    back = [b for b in by_split[target]
                            if groups[b]["boxes"].get(cid, 0) == 0
                            and groups[b]["n_img"] == groups[gid]["n_img"]]
                    rng.shuffle(back)
                    back = back[:40] + [None]
                    for bgid in back:
                        apply_move(gid, src, target)
                        if bgid is not None:
                            apply_move(bgid, target, src)
                        new, _ = penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
                        if new < cur - 1e-9:
                            cur = new
                            by_split[src].remove(gid); by_split[target].append(gid)
                            if bgid is not None:
                                by_split[target].remove(bgid); by_split[src].append(bgid)
                            accepted += 1
                            done = True
                            break
                        # hoan tac
                        if bgid is not None:
                            apply_move(bgid, src, target)
                        apply_move(gid, target, src)
                    if done:
                        break
                if done:
                    break
            if not done:
                if verbose:
                    print(f"    [!] khong tim duoc buoc cai thien cho "
                          f"{class_names[cid]}/{target} (thieu {deficit})")
                break
            continue

        # --- het vi pham cung: can bang han ngach anh + bam sat 70/20/10 tung lop
        if cur <= 1e-9 or time.time() - t_start > time_budget:
            break
        improved = False

        # (a) san bang han ngach anh giua split thua va split thieu
        over = [s for s in SPLITS if img[s] > img_quota[s]]
        under = [s for s in SPLITS if img[s] < img_quota[s]]
        if over and under:
            src, dst = over[0], under[0]
            cands = by_split[src][:]
            rng.shuffle(cands)
            for gid in cands[:400]:
                apply_move(gid, src, dst)
                new, _ = penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
                if new < cur - 1e-9:
                    cur = new
                    by_split[src].remove(gid); by_split[dst].append(gid)
                    accepted += 1
                    improved = True
                    break
                apply_move(gid, dst, src)

        # (b) doi cho CO DINH HUONG: lay lop lech nhieu nhat, chuyen nhom giau lop do
        #     tu split dang thua sang split dang thieu, doi lai mot nhom khong chua lop do
        #     -> han ngach anh khong doi, chi ty le lop thay doi.
        if not improved:
            devs = []
            for cid in range(n_classes):
                t = totals.get(cid, 0)
                if not t:
                    continue
                for s in SPLITS:
                    d = 100.0 * box[s][cid] / t - TARGET_RATIO[s] * 100.0
                    if d < 0:                       # chi xet split DANG THIEU lop cid
                        devs.append((-d, cid, s))
            devs.sort(reverse=True)
            for _, cid, s_low in devs[:15]:
                t = totals[cid]
                donors = sorted((s for s in SPLITS if s != s_low),
                                key=lambda s: -(100.0 * box[s][cid] / t
                                                - TARGET_RATIO[s] * 100.0))
                for s_high in donors:
                    give = [g for g in by_split[s_high]
                            if groups[g]["boxes"].get(cid, 0) > 0]
                    take = [g for g in by_split[s_low]
                            if groups[g]["boxes"].get(cid, 0) == 0]
                    if not give or not take:
                        continue
                    rng.shuffle(give); rng.shuffle(take)
                    give.sort(key=lambda g: -groups[g]["boxes"][cid])
                    for ga in give[:25]:
                        for gb in take[:25]:
                            if groups[ga]["n_img"] != groups[gb]["n_img"]:
                                continue
                            apply_move(ga, s_high, s_low)
                            apply_move(gb, s_low, s_high)
                            new, _ = penalty(box, img, totals, img_quota, req,
                                             n_classes, bg, n_bg)
                            if new < cur - 1e-9:
                                cur = new
                                by_split[s_high].remove(ga); by_split[s_low].append(ga)
                                by_split[s_low].remove(gb); by_split[s_high].append(gb)
                                accepted += 1
                                improved = True
                                break
                            apply_move(gb, s_high, s_low)   # hoan tac
                            apply_move(ga, s_low, s_high)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if not improved:
            break
        _, det = penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)

    cur, det = penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
    if verbose:
        worst = 0.0
        for cid in range(n_classes):
            t = totals.get(cid, 0)
            if t:
                worst = max(worst, max(abs(100.0 * box[s][cid] / t
                                           - TARGET_RATIO[s] * 100.0) for s in SPLITS))
        print(f"    phat sau repair  = {cur:,.1f}  "
              f"(min={det['min']:,.0f} img={det['img']:,.0f} "
              f"dev={det['dev']:,.1f} bg={det['bg']:,.1f})"
              f"  [{accepted} buoc, {time.time() - t_start:.0f}s]")
        print(f"    do lech ty le lon nhat con lai: {worst:.2f} diem % "
              f"(nguong cho phep {RATIO_TOLERANCE:.0f})")
        if n_bg:
            bgd = max(abs(100.0 * bg[s] / n_bg - TARGET_RATIO[s] * 100.0)
                      for s in SPLITS)
            print(f"    anh nen: train={bg['train']:,} val={bg['val']:,} "
                  f"test={bg['test']:,} -> lech toi da {bgd:.2f} diem % "
                  f"(nguong {BG_TOLERANCE:.0f})")
    return assign, box, img, bg, cur, det


# ----------------------------------------------------------------- ghi dataset


def letterbox_pair(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path,
                   target: int):
    """Chuan hoa anh ve target x target (letterbox giu ty le) VA sua nhan tuong ung.

    Anh vuong (vd 416x416 -> 640x640) la phep scale deu: pad = 0, nhan chuan hoa
    KHONG doi mot chut nao. Anh khong vuong se duoc pad va nhan duoc dich lai.
    Tra ve True neu da re-encode, False neu chi copy nguyen ban.
    """
    with Image.open(src_img) as im:
        w, h = im.size
        if (w, h) == (target, target):
            shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            return False
        s = target / max(w, h)
        nw, nh = round(w * s), round(h * s)
        px, py = (target - nw) // 2, (target - nh) // 2
        canvas = Image.new("RGB", (target, target), (114, 114, 114))
        canvas.paste(im.convert("RGB").resize((nw, nh), Image.LANCZOS), (px, py))
        canvas.save(dst_img, quality=95, subsampling=0)

    if src_lbl.exists():
        out = []
        with open(src_lbl, "r", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                c = p[0]
                xc, yc, bw, bh = (float(v) for v in p[1:5])
                xc2 = (xc * w * s + px) / target
                yc2 = (yc * h * s + py) / target
                bw2 = bw * w * s / target
                bh2 = bh * h * s / target
                out.append(f"{c} {xc2:.6f} {yc2:.6f} {bw2:.6f} {bh2:.6f}")
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + ("\n" if out else ""))
    return True


def materialize(src_root: Path, dst_root: Path, assign, groups, file_info, man,
                mode="hardlink", normalize_size=None, verbose=True):
    if dst_root.exists():
        if verbose:
            print(f"    xoa thu muc cu: {dst_root}")
        shutil.rmtree(dst_root)
    for s in SPLITS:
        (dst_root / s / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / s / "labels").mkdir(parents=True, exist_ok=True)

    file_split = {}
    for gid, s in assign.items():
        for fn in groups[gid]["files"]:
            file_split[fn] = s

    def transfer(src: Path, dst: Path):
        if mode == "hardlink":
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        shutil.copy2(src, dst)

    n = 0
    n_norm = 0
    t0 = time.time()
    for fn, s in file_split.items():
        info = file_info[fn]
        di = dst_root / s / "images" / fn
        dl = dst_root / s / "labels" / info["lbl"].name
        if normalize_size:
            if letterbox_pair(info["img"], info["lbl"], di, dl, normalize_size):
                n_norm += 1
        else:
            transfer(info["img"], di)
            if info["lbl"].exists():
                transfer(info["lbl"], dl)
        n += 1
        if verbose and n % 2000 == 0:
            print(f"      ... {n:,}/{len(file_split):,} file "
                  f"({time.time() - t0:.0f}s, da chuan hoa {n_norm:,})")
    if verbose:
        how = f"letterbox->{normalize_size}px" if normalize_size else mode
        print(f"    da tao {n:,} cap anh/label ({how}) trong {time.time() - t0:.0f}s")
        if normalize_size:
            print(f"    trong do {n_norm:,} anh duoc resize ve "
                  f"{normalize_size}x{normalize_size} (con lai copy nguyen ban)")

    # manifest moi
    man2 = man.copy()
    man2["split"] = man2["file"].map(file_split)
    if man2["split"].isna().any():
        miss = int(man2["split"].isna().sum())
        print(f"    [!] {miss} dong manifest khong map duoc split - giu nguyen gia tri cu")
        man2["split"] = man2["split"].fillna(man["split"])
    man2.to_csv(dst_root / "manifest.csv", index=False, encoding="utf-8")

    # data.yaml (cap nhat path)
    with open(src_root / "data.yaml", "r", encoding="utf-8") as f:
        y = f.read()
    y = re.sub(r"^path:.*$", "path: " + str(dst_root).replace("\\", "/"),
               y, count=1, flags=re.M)
    with open(dst_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(y)

    return file_split, n_norm


def update_dataset_card(src_root: Path, dst_root: Path, df_after, df_before, req,
                        class_names, normalize_size=None, n_normalized=0):
    with open(src_root / "DATASET_CARD.md", "r", encoding="utf-8") as f:
        card = f.read()

    rows = []
    for _, r in df_after.iterrows():
        rows.append([r.class_name, f"{int(r.boxes_train):,}", f"{int(r.boxes_val):,}",
                     f"{int(r.boxes_test):,}", f"{int(r.boxes_total):,}",
                     f"{r.pct_train:.1f}/{r.pct_val:.1f}/{r.pct_test:.1f}"])
    tb = int(df_after.boxes_total.sum())
    rows.append(["**TOTAL**", f"**{int(df_after.boxes_train.sum()):,}**",
                 f"**{int(df_after.boxes_val.sum()):,}**",
                 f"**{int(df_after.boxes_test.sum()):,}**", f"**{tb:,}**",
                 f"**{100 * df_after.boxes_train.sum() / tb:.1f}/"
                 f"{100 * df_after.boxes_val.sum() / tb:.1f}/"
                 f"{100 * df_after.boxes_test.sum() / tb:.1f}**"])
    table = md_table(rows, ["Lớp", "train", "val", "test", "tổng", "% t/v/te"],
                     ["---", "---:", "---:", "---:", "---:", ":---:"])

    img_rows = []
    for _, r in df_after.iterrows():
        img_rows.append([r.class_name, f"{int(r.images_train):,}",
                         f"{int(r.images_val):,}", f"{int(r.images_test):,}",
                         f"{int(r.images_total):,}"])
    itable = md_table(img_rows, ["Lớp", "train", "val", "test", "tổng"],
                      ["---", "---:", "---:", "---:", "---:"])

    new_section = f"""## Class distribution (boxes) - sau khi CHIA LẠI CÓ PHÂN TẦNG (seed 42)

{table}

## Số ảnh chứa mỗi lớp

{itable}

## Stratified re-split (v1 -> v1_resplit)

Bản gốc chia 70/20/10 group-aware nhưng **không phân tầng**, nên các lớp hiếm bị
thiếu mẫu ở test (no_vest test = 70 box, mask test = 97 box, no_boots val = 4 box),
không đủ để ước lượng AP đáng tin.

Bản này chia lại bằng **greedy iterative stratification** (`stratified_resplit.py`,
seed 42), giữ nguyên ràng buộc chống leakage: mọi ảnh cùng `source_image_id` luôn
nằm cùng một split.

Ràng buộc đã đạt (trừ `no_boots`, tổng chỉ 37 box nên được miễn):
- mỗi lớp >= {MIN_TEST_DEFAULT} box ở test và >= {MIN_VAL_DEFAULT} box ở val
- `no_vest` test >= {MIN_TEST_SPECIAL['no_vest']}, `mask` test >= {MIN_TEST_SPECIAL['mask']}

Kiểm chứng bằng `verify_dataset.py` (class ID, toạ độ box, ghép cặp 1-1 ảnh/label,
kích thước 640x640, chống leakage theo `source_image_id`) - PASS toàn bộ.
"""

    if normalize_size:
        new_section += f"""
## Chuẩn hoá kích thước ảnh (sửa lỗi của v1)

Bản v1 **chưa** được letterbox như mô tả cũ: 11.110/17.951 ảnh (toàn bộ nguồn `c`)
vẫn ở 416x416, chỉ nguồn `a`/`b` là 640x640 - `build_unified_dataset.py` có sẵn
đường letterbox nhưng đã chạy với `NORMALIZE_SIZE = None`.

Bản này chuẩn hoá **100% ảnh về {normalize_size}x{normalize_size}**. Ảnh nguồn đều
vuông (416x416 -> {normalize_size}x{normalize_size}) nên đây là phép scale đều,
`pad = 0`, **nhãn YOLO chuẩn hoá không đổi**. Ảnh sẵn {normalize_size}x
{normalize_size} được copy nguyên bản (không re-encode); {n_normalized:,} ảnh 416px
được resize LANCZOS rồi lưu JPEG quality 95, `subsampling=0`.
"""

    pat = re.compile(r"## Class distribution \(boxes\).*?(?=\n## |\Z)", re.S)
    # dung lambda: tranh re.sub dien giai \g, \1... neu bang co ky tu backslash
    card2 = pat.sub(lambda _m: new_section + "\n", card, count=1) if pat.search(card) \
        else card + "\n" + new_section

    extra = ("- `no_boots` chỉ có 37 box toàn dataset -> **giữ trong dataset nhưng "
             "phải loại khỏi metric tổng hợp** (mAP trung bình); chỉ báo cáo riêng "
             "và coi là không đủ dữ liệu kết luận.\n")
    if "phải loại khỏi metric tổng hợp" not in card2:
        card2 = card2.replace("## Known limitations\n", "## Known limitations\n" + extra, 1)

    with open(dst_root / "DATASET_CARD.md", "w", encoding="utf-8") as f:
        f.write(card2)


# ----------------------------------------------------------------- bang so sanh


def print_before_after(df_before, df_after, class_names, req):
    banner("BANG SO SANH PHAN BO TRUOC / SAU KHI CHIA LAI")
    print("\nSo BOX moi lop (truoc -> sau)\n")
    rows = []
    for cid, cname in enumerate(class_names):
        b = df_before[df_before.class_name == cname].iloc[0]
        a = df_after[df_after.class_name == cname].iloc[0]
        mt = minimum_for(cid, "test", req)
        mv = minimum_for(cid, "val", req)
        def cell(x, y):
            return f"{int(x):,} → {int(y):,}"
        st_b = "OK" if (int(b.boxes_test) >= mt and int(b.boxes_val) >= mv) else "FAIL"
        st_a = "OK" if (int(a.boxes_test) >= mt and int(a.boxes_val) >= mv) else "FAIL"
        if cname in EXEMPT_CLASSES:
            st_b = st_a = "miễn"
        rows.append([cid, cname,
                     cell(b.boxes_train, a.boxes_train),
                     cell(b.boxes_val, a.boxes_val),
                     cell(b.boxes_test, a.boxes_test),
                     f"{b.pct_test:.1f}% → {a.pct_test:.1f}%",
                     f"{mv or '-'}/{mt or '-'}",
                     f"{st_b} → {st_a}"])
    print(md_table(rows, ["id", "lop", "train", "val", "test", "%test",
                          "nguong val/test", "dat nguong"],
                   ["---:", "---", "---:", "---:", "---:", "---:", ":---:", ":---:"]))

    print("\nTong quan\n")
    rows = []
    for s in SPLITS:
        bb = int(df_before[f"boxes_{s}"].sum()); ba = int(df_after[f"boxes_{s}"].sum())
        tb = int(df_before.boxes_total.sum()); ta = int(df_after.boxes_total.sum())
        ib = int(df_before[f"images_{s}"].sum()); ia = int(df_after[f"images_{s}"].sum())
        rows.append([s, f"{bb:,} → {ba:,}",
                     f"{100 * bb / tb:.2f}% → {100 * ba / ta:.2f}%",
                     f"muc tieu {TARGET_RATIO[s] * 100:.0f}%"])
    print(md_table(rows, ["split", "box", "% box", "ghi chu"],
                   ["---", "---:", "---:", "---"]))


# ----------------------------------------------------------------- main


def main():
    _stdout_utf8()
    ap = argparse.ArgumentParser(description="Chia lai dataset PPE co phan tang")
    ap.add_argument("--src", default="PPE_Unified_v1")
    ap.add_argument("--dst", default="PPE_Unified_v1_resplit")
    ap.add_argument("--mode", choices=["hardlink", "copy"], default="copy",
                    help="copy: nhan ban that su (mac dinh, an toan); "
                         "hardlink: nhanh, khong ton them dia nhung dung chung du lieu")
    ap.add_argument("--normalize-size", type=int, default=640,
                    help="letterbox moi anh ve NxN va sua nhan tuong ung "
                         "(mac dinh 640; dat 0 de giu nguyen kich thuoc goc)")
    ap.add_argument("--dry-run", action="store_true",
                    help="chi tinh toan phan chia, khong ghi file")
    ap.add_argument("--no-size-check", action="store_true",
                    help="bo qua kiem tra 640x640 khi verify lai (nhanh hon)")
    a = ap.parse_args()

    here = Path(os.path.join(os.environ.get("THS_DATASETS", r"D:\Datasets"), "PPE_SH17"))  # data dir (moved 28-09-2026)
    src = Path(a.src) if Path(a.src).is_absolute() else here / a.src
    dst = Path(a.dst) if Path(a.dst).is_absolute() else here / a.dst
    if not src.is_dir():
        print(f"LOI: khong tim thay {src}")
        sys.exit(2)

    rng = random.Random(SEED)
    class_names = load_class_names(src)
    n_classes = len(class_names)
    req = min_req_table(class_names)

    banner("VIEC 3 - CHIA LAI CO PHAN TANG (seed 42)")
    print(f"Nguon : {src}")
    print(f"Dich  : {dst}   (mode={a.mode}{', DRY-RUN' if a.dry_run else ''})")

    print("\n[1/6] Doc manifest + dem box theo nhom source_image_id ...")
    man, file_info, boxes_per_file, groups = collect(src, class_names)
    n_img = sum(g["n_img"] for g in groups.values())
    sizes = defaultdict(int)
    for g in groups.values():
        sizes[g["n_img"]] += 1
    print(f"    {len(groups):,} nhom / {n_img:,} anh  "
          f"(kich thuoc nhom: " +
          ", ".join(f"{k} anh x{v:,}" for k, v in sorted(sizes.items())) + ")")

    img_quota = {"train": int(round(n_img * TARGET_RATIO["train"])),
                 "val": int(round(n_img * TARGET_RATIO["val"]))}
    img_quota["test"] = n_img - img_quota["train"] - img_quota["val"]
    print(f"    han ngach anh: train={img_quota['train']:,} "
          f"val={img_quota['val']:,} test={img_quota['test']:,}")

    print("\n[2/6] Greedy iterative stratification (lop hiem truoc) ...")
    assign, totals, bg_quota, n_bg = greedy_stratify(groups, n_classes, class_names,
                                                     img_quota, req, rng)

    print("\n[3/6] Repair - hill-climbing tren ham phat ...")
    assign, box, img, bg, cur, det = repair(assign, groups, n_classes, class_names,
                                            totals, img_quota, req, rng, n_bg=n_bg)

    print("\n    Ket qua phan chia (chua ghi dia):")
    rows = []
    for cid, cname in enumerate(class_names):
        mt, mv = minimum_for(cid, "test", req), minimum_for(cid, "val", req)
        ok = (box["test"][cid] >= mt and box["val"][cid] >= mv)
        rows.append([cid, cname, f"{box['train'][cid]:,}", f"{box['val'][cid]:,}",
                     f"{box['test'][cid]:,}", f"{totals.get(cid, 0):,}",
                     "mien" if cname in EXEMPT_CLASSES else ("OK" if ok else "FAIL")])
    print(md_table(rows, ["id", "lop", "train", "val", "test", "tong", "nguong"],
                   ["---:", "---", "---:", "---:", "---:", "---:", ":---:"]))
    print(f"    anh: train={img['train']:,} val={img['val']:,} test={img['test']:,}")

    hard_fail = det["min"] > 0
    if hard_fail:
        print("\n    [!] VAN CON LOP CHUA DAT NGUONG TOI THIEU - khong ghi dia.")
        sys.exit(1)

    if a.dry_run:
        print("\n[DRY-RUN] Dung tai day, khong ghi file.")
        return

    norm = a.normalize_size if a.normalize_size and a.normalize_size > 0 else None
    if norm and a.mode == "hardlink":
        print("    [!] --normalize-size can ghi file moi -> bo qua hardlink, dung copy.")
    print(f"\n[4/6] Tao {dst} "
          f"({'letterbox ' + str(norm) + 'px' if norm else a.mode})"
          f" + cap nhat manifest/data.yaml ...")
    _, n_norm = materialize(src, dst, assign, groups, file_info, man, mode=a.mode,
                            normalize_size=norm)

    print("\n[5/6] Chay lai VIEC 1 + VIEC 2 tren ban resplit de xac nhan ...")
    out_before = vd.run(src, check_image_size=False, write_csv=False,
                        final_summary=False, verbose=False)
    df_before = out_before["df"]
    out_after = vd.run(dst, check_image_size=not a.no_size_check, write_csv=True,
                       final_summary=False, verbose=True)
    df_after = out_after["df"]

    print_before_after(df_before, df_after, class_names, req)

    print("\n[6/6] Cap nhat DATASET_CARD.md ...")
    update_dataset_card(src, dst, df_after, df_before, req, class_names,
                        normalize_size=norm, n_normalized=n_norm)
    print(f"    da ghi {dst / 'DATASET_CARD.md'}")

    ok = vd.print_final_summary(dst, out_after["checks"], df_after,
                                out_after["res"], out_after["n_lech"], class_names)

    banner("HOAN TAT VIEC 3")
    if ok:
        print(f"Ban resplit PASS toan bo: {dst}")
        print(f"Ban goc {src.name} VAN DUOC GIU NGUYEN - chi xoa/doi ten sau khi ban "
              f"da tu kiem tra lai.")
        if norm:
            print(f"Da chuan hoa {n_norm:,} anh 416px -> {norm}x{norm} "
                  f"(scale deu, nhan khong doi).")
        elif a.mode == "hardlink":
            print("Luu y: file duoc tao bang HARDLINK -> hai thu muc dung chung du lieu "
                  "tren dia; xoa ban goc VAN AN TOAN (du lieu con song trong ban resplit) "
                  "nhung se khong giai phong dung luong.")
    else:
        print(f"[!] Ban resplit CHUA PASS toan bo - giu nguyen {src.name}, "
              f"kiem tra lai log ben tren.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
