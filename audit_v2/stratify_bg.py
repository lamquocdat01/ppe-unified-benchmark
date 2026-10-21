"""Background-aware extension of the v1 group-aware stratified split (decision 3, 21-10-2026).

The v1 algorithm (stratified_resplit.py: greedy iterative stratification + repair, seed 42)
counts background images only through groups that contain no box at all. With v1 groups
(one image each) that is every background image; with near-duplicate groups (v1.1) many
background frames sit inside mixed groups and the algorithm cannot see them.

Changes, marked BG-EXT below (everything else is the v1 code, unchanged):
  1. every group carries n_bg = number of its images with no box; background counts and
     moves use n_bg instead of "group has no box";
  2. the background quota of the pure-background phase is the total-background quota minus
     the background already placed through mixed groups;
  3. after repair, a background-balancing phase swaps/moves groups while the penalty (the v1
     penalty, unchanged weights) decreases; it only runs when the background deviation
     exceeds the v1 tolerance (2 pp).
With singleton groups n_bg is 0 or n_img, nothing is placed through mixed groups and the
v1 split already meets the tolerance, so the extension reproduces v1 exactly (verified by
compat_check(): 100 % identical assignment on PPE_Unified_v1).
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

from .registry import DATASETS_ROOT
from .resplit import ALGO_DIR

sys.path.insert(0, str(ALGO_DIR))
import stratified_resplit as sr  # noqa: E402

SPLITS, TARGET_RATIO = sr.SPLITS, sr.TARGET_RATIO


def add_bg(groups, boxes_per_file):
    """BG-EXT 1: per-group background image count."""
    for g in groups.values():
        g["n_bg"] = sum(1 for fn in g["files"] if not boxes_per_file.get(fn))
    return groups


def counts_from_assign(assign, groups, n_classes):
    box = {s: defaultdict(int) for s in SPLITS}
    img = {s: 0 for s in SPLITS}
    bg = {s: 0 for s in SPLITS}
    for gid, s in assign.items():
        g = groups[gid]
        img[s] += g["n_img"]
        bg[s] += g["n_bg"]  # BG-EXT 1
        for c, n in g["boxes"].items():
            box[s][c] += n
    return box, img, bg


def greedy_stratify(groups, n_classes, class_names, img_quota, req, rng):
    """v1 greedy_stratify; only the background phase differs (BG-EXT 2)."""
    totals = defaultdict(int)
    for g in groups.values():
        for c, n in g["boxes"].items():
            totals[c] += n
    demand = {s: {} for s in SPLITS}
    for cid in range(n_classes):
        t = totals.get(cid, 0)
        d_test = max(t * TARGET_RATIO["test"], sr.minimum_for(cid, "test", req))
        d_val = max(t * TARGET_RATIO["val"], sr.minimum_for(cid, "val", req))
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
        g = groups[gid]
        cands = [s for s in SPLITS if img_left[s] >= g["n_img"]]
        if not cands:
            cands = [s for s in SPLITS if img_left[s] > 0] or list(SPLITS)
        best, best_key = None, None
        order = cands[:]
        rng.shuffle(order)
        for s in order:
            key = (demand[s].get(cid_hint, 0.0), img_left[s])
            if best_key is None or key > best_key:
                best, best_key = s, key
        assign[gid] = best
        img_left[best] -= g["n_img"]
        for c, n in g["boxes"].items():
            demand[best][c] = demand[best].get(c, 0.0) - n
        return best

    while True:
        live = [c for c in range(n_classes) if rem_counts.get(c, 0) > 0]
        if not live:
            break
        cid = min(live, key=lambda c: (rem_counts[c], c))
        gids = sorted(g for g in remaining if groups[g]["boxes"].get(cid, 0) > 0)
        rng.shuffle(gids)
        gids.sort(key=lambda g: -groups[g]["boxes"][cid])
        for gid in gids:
            place(gid, cid)
            remaining.discard(gid)
            for c, n in groups[gid]["boxes"].items():
                rem_counts[c] -= n

    bg_groups = sorted(remaining)
    rng.shuffle(bg_groups)
    # BG-EXT 2: quota over ALL background images, minus those already placed inside mixed groups
    n_bg_total = sum(g["n_bg"] for g in groups.values())
    placed = {s: 0 for s in SPLITS}
    for gid, s in assign.items():
        placed[s] += groups[gid]["n_bg"]
    bg_quota = {"train": int(round(n_bg_total * TARGET_RATIO["train"])),
                "val": int(round(n_bg_total * TARGET_RATIO["val"]))}
    bg_quota["test"] = n_bg_total - bg_quota["train"] - bg_quota["val"]
    bg_left = {s: bg_quota[s] - placed[s] for s in SPLITS}
    for gid in bg_groups:
        need = groups[gid]["n_img"]
        cands = [s for s in SPLITS if bg_left[s] >= need] or list(SPLITS)
        s = max(cands, key=lambda x: (bg_left[x], img_left[x]))
        assign[gid] = s
        bg_left[s] -= need
        img_left[s] -= need
        remaining.discard(gid)
    return assign, dict(totals), bg_quota, n_bg_total


def _penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg):
    return sr.penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)


def repair(assign, groups, n_classes, class_names, totals, img_quota, req, rng, n_bg=0,
           max_iter=40000, time_budget=300.0):
    """v1 repair with group background counted by n_bg (BG-EXT 1); moves and acceptance unchanged."""
    t_start = time.time()
    box, img, bg = counts_from_assign(assign, groups, n_classes)
    cur, det = _penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
    by_split = defaultdict(list)
    for gid, s in assign.items():
        by_split[s].append(gid)

    def apply_move(gid, frm, to):
        g = groups[gid]
        img[frm] -= g["n_img"]; img[to] += g["n_img"]
        bg[frm] -= g["n_bg"]; bg[to] += g["n_bg"]  # BG-EXT 1
        for c, n in g["boxes"].items():
            box[frm][c] -= n
            box[to][c] += n
        assign[gid] = to

    for it in range(max_iter):
        viol = []
        for cid in range(n_classes):
            for s in ("test", "val"):
                need = sr.minimum_for(cid, s, req)
                if need and box[s][cid] < need:
                    viol.append((need - box[s][cid], cid, s))
        if viol:
            viol.sort(reverse=True)
            deficit, cid, target = viol[0]
            srcs = sorted((s for s in SPLITS if s != target),
                          key=lambda s: -(box[s][cid] - sr.minimum_for(cid, s, req)))
            done = False
            for src in srcs:
                cands = [g for g in by_split[src] if groups[g]["boxes"].get(cid, 0) > 0]
                if not cands:
                    continue
                rng.shuffle(cands)
                cands.sort(key=lambda g: (abs(groups[g]["boxes"][cid] - deficit), sum(groups[g]["boxes"].values())))
                for gid in cands[:40]:
                    back = [b for b in by_split[target]
                            if groups[b]["boxes"].get(cid, 0) == 0 and groups[b]["n_img"] == groups[gid]["n_img"]]
                    rng.shuffle(back)
                    back = back[:40] + [None]
                    for bgid in back:
                        apply_move(gid, src, target)
                        if bgid is not None:
                            apply_move(bgid, target, src)
                        new, _ = _penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
                        if new < cur - 1e-9:
                            cur = new
                            by_split[src].remove(gid); by_split[target].append(gid)
                            if bgid is not None:
                                by_split[target].remove(bgid); by_split[src].append(bgid)
                            done = True
                            break
                        if bgid is not None:
                            apply_move(bgid, src, target)
                        apply_move(gid, target, src)
                    if done:
                        break
                if done:
                    break
            if not done:
                break
            continue
        if cur <= 1e-9 or time.time() - t_start > time_budget:
            break
        improved = False
        over = [s for s in SPLITS if img[s] > img_quota[s]]
        under = [s for s in SPLITS if img[s] < img_quota[s]]
        if over and under:
            src, dst = over[0], under[0]
            cands = by_split[src][:]
            rng.shuffle(cands)
            for gid in cands[:400]:
                apply_move(gid, src, dst)
                new, _ = _penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
                if new < cur - 1e-9:
                    cur = new
                    by_split[src].remove(gid); by_split[dst].append(gid)
                    improved = True
                    break
                apply_move(gid, dst, src)
        if not improved:
            devs = []
            for cid in range(n_classes):
                t = totals.get(cid, 0)
                if not t:
                    continue
                for s in SPLITS:
                    d = 100.0 * box[s][cid] / t - TARGET_RATIO[s] * 100.0
                    if d < 0:
                        devs.append((-d, cid, s))
            devs.sort(reverse=True)
            for _, cid, s_low in devs[:15]:
                t = totals[cid]
                donors = sorted((s for s in SPLITS if s != s_low),
                                key=lambda s: -(100.0 * box[s][cid] / t - TARGET_RATIO[s] * 100.0))
                for s_high in donors:
                    give = [g for g in by_split[s_high] if groups[g]["boxes"].get(cid, 0) > 0]
                    take = [g for g in by_split[s_low] if groups[g]["boxes"].get(cid, 0) == 0]
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
                            new, _ = _penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
                            if new < cur - 1e-9:
                                cur = new
                                by_split[s_high].remove(ga); by_split[s_low].append(ga)
                                by_split[s_low].remove(gb); by_split[s_high].append(gb)
                                improved = True
                                break
                            apply_move(gb, s_high, s_low)
                            apply_move(ga, s_low, s_high)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    cur, det = _penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
    return assign, box, img, bg, cur, det, by_split


def bg_dev(bg, n_bg):
    return max(abs(100.0 * bg[s] / n_bg - TARGET_RATIO[s] * 100.0) for s in SPLITS) if n_bg else 0.0


def bg_balance(assign, groups, n_classes, totals, img_quota, req, rng, n_bg, by_split, max_rounds=5000):
    """BG-EXT 3: move or swap groups between an over- and an under-represented split (background
    share) while the v1 penalty decreases. Skipped when the deviation is within tolerance."""
    box, img, bg = counts_from_assign(assign, groups, n_classes)
    steps = 0
    if bg_dev(bg, n_bg) <= sr.BG_TOLERANCE:
        return assign, steps
    cur, _ = _penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)

    def move(gid, frm, to):
        g = groups[gid]
        img[frm] -= g["n_img"]; img[to] += g["n_img"]
        bg[frm] -= g["n_bg"]; bg[to] += g["n_bg"]
        for c, n in g["boxes"].items():
            box[frm][c] -= n; box[to][c] += n
        assign[gid] = to

    for _ in range(max_rounds):
        if bg_dev(bg, n_bg) <= sr.BG_TOLERANCE:
            break
        share = {s: 100.0 * bg[s] / n_bg - TARGET_RATIO[s] * 100.0 for s in SPLITS}
        s_hi = max(SPLITS, key=lambda s: share[s]); s_lo = min(SPLITS, key=lambda s: share[s])
        give = sorted(g for g in by_split[s_hi] if groups[g]["n_bg"] > 0)
        take = sorted(by_split[s_lo])
        rng.shuffle(give); rng.shuffle(take)
        give.sort(key=lambda g: -groups[g]["n_bg"] / groups[g]["n_img"])  # background-rich first
        take.sort(key=lambda g: groups[g]["n_bg"] / groups[g]["n_img"])   # background-poor first
        best = None
        for ga in give[:60]:
            for gb in [None] + take[:60]:
                move(ga, s_hi, s_lo)
                if gb is not None:
                    move(gb, s_lo, s_hi)
                new, _ = _penalty(box, img, totals, img_quota, req, n_classes, bg, n_bg)
                if new < cur - 1e-9 and (best is None or new < best[0]):
                    best = (new, ga, gb)
                if gb is not None:
                    move(gb, s_hi, s_lo)
                move(ga, s_lo, s_hi)
        if best is None:
            break
        cur, ga, gb = best
        move(ga, s_hi, s_lo); by_split[s_hi].remove(ga); by_split[s_lo].append(ga)
        if gb is not None:
            move(gb, s_lo, s_hi); by_split[s_lo].remove(gb); by_split[s_hi].append(gb)
        steps += 1
    return assign, steps


def split(groups, boxes_per_file, class_names, seed=None):
    """Full extended algorithm: greedy -> repair -> background balancing. Deterministic (seed 42)."""
    import random
    add_bg(groups, boxes_per_file)
    nc = len(class_names)
    req = sr.min_req_table(class_names)
    n_img = sum(g["n_img"] for g in groups.values())
    quota = {"train": int(round(n_img * TARGET_RATIO["train"])), "val": int(round(n_img * TARGET_RATIO["val"]))}
    quota["test"] = n_img - quota["train"] - quota["val"]
    rng = random.Random(sr.SEED if seed is None else seed)
    assign, totals, _, n_bg = greedy_stratify(groups, nc, class_names, quota, req, rng)
    assign, box, img, bg, cur, det, by_split = repair(assign, groups, nc, class_names, totals, quota, req, rng, n_bg=n_bg)
    assign, steps = bg_balance(assign, groups, nc, totals, quota, req, rng, n_bg, by_split)
    box, img, bg = counts_from_assign(assign, groups, nc)
    cur, det = _penalty(box, img, totals, quota, req, nc, bg, n_bg)
    return {"assign": assign, "box": box, "img": img, "bg": bg, "penalty": cur, "detail": det, "quota": quota,
            "totals": totals, "n_bg": n_bg, "bg_balance_steps": steps, "bg_dev_pp": round(bg_dev(bg, n_bg), 3)}


def compat_check(src=DATASETS_ROOT / "PPE_SH17" / "PPE_Unified_v1", ref=DATASETS_ROOT / "PPE_SH17" / "PPE_Unified_v1_resplit"):
    """Run v1's own input (one group per source image) through the extended algorithm and compare
    with the published v1 split."""
    import pandas as pd
    src = Path(src)
    cn = sr.load_class_names(src)
    man, fi, bpf, groups = sr.collect(src, cn)
    r = split(groups, bpf, cn)
    fs = {fn: r["assign"][gid] for gid, g in groups.items() for fn in g["files"]}
    v1 = pd.read_csv(Path(ref) / "manifest.csv", encoding="utf-8")
    same = (v1["file"].map(fs) == v1["split"])
    return {"images": int(len(v1)), "identical": int(same.sum()), "identical_pct": round(100.0 * same.mean(), 4),
            "bg_balance_steps": r["bg_balance_steps"], "bg_dev_pp": r["bg_dev_pp"],
            "groups": len(groups), "max_group_size": max(g["n_img"] for g in groups.values())}


if __name__ == "__main__":
    import json
    print(json.dumps(compat_check(), indent=1))
