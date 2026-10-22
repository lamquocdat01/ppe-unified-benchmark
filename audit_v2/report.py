"""Figures, LaTeX tables, numbers.json, validation sheets for the T1 audit and v1.1.

Usage: python -u -m audit_v2.report --paper "<Q1 ESWA Submission dir>"
Writes results/audit_v2/{fig_calibration.pdf, fig_cos_distribution.pdf, tab_audit_v2.tex,
contact_ppeu_cross_split.png} and <paper>/generated/{numbers.json, same figures/table}.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import features
from .registry import RESULTS, has_rf, source_id
from .sheet import contact_sheet

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1f1f1e", "#8a897f"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 1.5})


def fig_calibration(cal):
    """(a) cosine and (b) pHash on cross-source negatives (C calibration, B holdout);
    (c) cosine on held-out B with same-set negatives, the calibration behind the operating point."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    op = cal["operating_point"]
    panels = (("cos", "cosine θ (DINOv2)", "(a) cross-source negatives"),
              ("ph", "pHash Hamming τ (flip-invariant)", "(b) cross-source negatives"))
    for ax, (k, xl, title) in zip(axes, panels):
        for s, ls in ((cal["calibration_set"], "-"), (cal["holdout_set"], "--")):
            rows = cal["sets"][s]["curves"][k]
            t = [r["t"] for r in rows]
            ax.plot(t, [r["precision"] for r in rows], ls, color=BLUE, label=f"precision ({s})")
            ax.plot(t, [r["recall"] for r in rows], ls, color=ORANGE, label=f"recall ({s})")
        thr = (cal.get("thresholds") or {}).get(k)
        if thr:
            ax.axvline(thr["threshold"], color=MUTED, lw=0.8, ls="-.")
        ax.set_xlim(*((0.3, 1.0) if k == "cos" else (0, 24)))
        ax.set_xlabel(xl); ax.set_title(title, fontsize=7.5, color=INK)
    ws = cal["within_set_holdout"]["curves"]["cos"]
    t = [r["t"] for r in ws]
    axes[2].plot(t, [r["precision"] for r in ws], "--", color=BLUE, label="precision (B)")
    axes[2].plot(t, [r["recall"] for r in ws], "--", color=ORANGE, label="recall (B)")
    axes[2].axvline(op["cos"], color=INK, lw=0.9, ls="-.")
    p = cal["within_set_holdout"]["at_operating_point"]
    axes[2].annotate(f"θ={op['cos']}: P {p['precision']:.3f}, R {p['recall']:.2f}", (op["cos"], 0.5),
                     xytext=(-4, 0), textcoords="offset points", ha="right", fontsize=6.5, color=INK)
    axes[2].set_xlim(0.7, 1.0); axes[2].set_xlabel("cosine θ (DINOv2)")
    axes[2].set_title("(c) held-out B, same-set negatives", fontsize=7.5, color=INK)
    for ax in axes:
        ax.axhline(cal["target_precision"], color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(0, 1.02); ax.grid(axis="y", color="#e6e5df", lw=0.5)
    axes[0].set_ylabel("value on sampled pairs")
    axes[0].legend(loc="lower left", fontsize=6)
    fig.savefig(RESULTS / "fig_calibration.pdf")
    plt.close(fig)


def fig_cos(cal):
    d = np.load(RESULTS / "calibration_distances.npz")
    sets = (cal["calibration_set"], cal["holdout_set"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), constrained_layout=True, sharey=False)
    bins = np.linspace(-0.1, 1.0, 111)
    for ax, s in zip(axes, sets):
        for key, col, lab in (("pos", BLUE, "positive: same .rf group (clean groups)"),
                              ("neg", ORANGE, "negative: random cross-source pair")):
            v = d[f"{s}_{key}_cos"]
            if not len(v):
                continue
            ax.hist(v, bins=bins, density=True, histtype="step", color=col, lw=1.5, label=f"{lab} (n={len(v):,})")
        thr = (cal.get("thresholds") or {}).get("cos")
        if thr:
            ax.axvline(thr["threshold"], color=INK, lw=0.8, ls="-.")
        ax.set_yscale("log"); ax.set_xlabel("cosine (DINOv2 ViT-S/14)"); ax.set_title(f"set {s}", fontsize=8, color=INK)
    axes[0].set_ylabel("density (log)")
    axes[0].legend(loc="upper left", fontsize=6)
    fig.savefig(RESULTS / "fig_cos_distribution.pdf")
    plt.close(fig)


def errata_numbers():
    """E1-E4 recomputed from the feature cache (file names, splits, MD5)."""
    out = {}
    for n in ("A", "B", "C", "sh17"):
        f = features.load(n)
        sp = f["split"]
        o = {"images": int(len(sp)), "splits": {s: int((sp == s).sum()) for s in sorted(set(sp.tolist()))},
             "md5_unique": int(len(np.unique(f["md5"])))}
        if all(has_rf(r) for r in f["rel"]):
            sid = np.array([source_id(r) for r in f["rel"]])
            df = pd.DataFrame({"g": sid, "s": sp, "m": f["md5"]})
            gs = df.groupby("g")["s"].agg(set)
            o["groups"] = int(len(gs))
            mult = df.groupby("g").size().value_counts().sort_index()
            o["multiplicity"] = {int(k): int(v) for k, v in mult.items()}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                o[f"groups_{a}_{b}"] = int(gs.apply(lambda x: a in x and b in x).sum())
            o["groups_multi_split"] = int((gs.apply(len) > 1).sum())
            o["groups_all3"] = int((gs.apply(len) == 3).sum())
            ms = df.groupby("m")["s"].nunique()
            o["md5_groups_cross_split"] = int((ms > 1).sum())
        out[n] = o
    return out


def pick_examples(name, k=10, thr=0.98):
    e = pd.read_csv(RESULTS / f"edges_cross_split_{name}.csv")
    e = e[e.in_knn & (e.cos >= thr)].copy()
    fam = lambda r: re.sub(r"\d+", "#", r.split("/")[-1])[:14]
    e["fam"] = e.a.map(fam) + e.split_pair
    # spread over cosine range, split pairs and file families instead of the top-10 of one video
    e = e.sample(frac=1, random_state=0).drop_duplicates("fam")
    q = np.quantile(e.cos, np.linspace(0, 1, k))
    rows, used = [], set()
    for t in q:
        r = e.loc[(e.cos - t).abs().sort_values().index]
        r = r[~r.index.isin(used)].iloc[0]
        used.add(r.name)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("cos", ascending=False)


def N(x):
    return f"{x:,}".replace(",", "{,}")


LABEL = {"ppeu": "PPE-U v1", "ppeu_fix": "PPE-U v1-fix", "ppeu11": "PPE-U v1.1", "sh17": "SH17", "A": "A", "B": "B", "C": "C"}


def tex_audit(aud):
    pts = list(aud["points"])
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{$\theta={aud['points'][p]['cos']}$}}" for p in pts)
    L = [r"\begin{table*}[t]", r"\centering\footnotesize",
         r"\caption{Two-level cross-split audit. Exact: same source ID or MD5. Copy level (2a): cosine "
         rf"$\ge\theta$ among FAISS top-{aud['topk']} neighbours or pHash $\le {aud['points'][pts[0]]['ph']}$ "
         r"(flip-invariant); $\theta=0.98$ is the operating point, 0.95 and 0.90 are sensitivity. "
         rf"Scene level (2b): same file-name sequence, frame numbers within $\pm{aud['scene_k']}$, no threshold. "
         r"Pairs are counted across splits; +2a = 2a pairs not caught by the exact tier; "
         r"groups = connected components of exact $\cup$ 2a $\cup$ 2b spanning more than one split (images).}",
         r"\label{tab:audit_v2}",
         r"\begin{tabular}{ll r " + "rr " * len(pts) + "rr}", r"\toprule",
         r" & & & " + head + r" & \multicolumn{2}{c}{2b scene} \\",
         " ".join(rf"\cmidrule(lr){{{4 + 2 * k}-{5 + 2 * k}}}" for k in range(len(pts) + 1)),
         r"Dataset & Split pair & Exact & " + " & ".join(["2a & +2a"] * len(pts)) + r" & 2b & 2b-only \\",
         r"\midrule"]
    for n, d in aud["within"].items():
        P = d["points"]
        for sp in P[pts[0]]["per_split_pair"]:
            r0 = P[pts[0]]["per_split_pair"][sp]
            cells = [N(r0["exact"])]
            for p in pts:
                r = P[p]["per_split_pair"][sp]
                cells += [N(r["copy_2a"]), N(r["copy_only"])]
            cells += [N(r0["scene_2b"]), N(P[pts[-1]]["per_split_pair"][sp]["scene_only"])]
            L.append(f"{LABEL.get(n, n)} & {sp} & " + " & ".join(cells) + r" \\")
        g = " / ".join(f"{N(P[p]['groups_all']['spanning'])} ({N(P[p]['groups_all']['images_in_spanning'])})" for p in pts)
        L.append(rf"\multicolumn{{{5 + 2 * len(pts)}}}{{r}}{{\scriptsize groups spanning splits, exact only: "
                 rf"{N(P[pts[0]]['groups_exact']['spanning'])} ({N(P[pts[0]]['groups_exact']['images_in_spanning'])} img); "
                 rf"exact$\cup$2a$\cup$2b at $\theta$ = {', '.join(str(aud['points'][p]['cos']) for p in pts)}: {g}}} \\")
        L.append(r"\midrule")
    for spec, d in aud["cross"].items():
        x, y = spec.split("=")
        name = f"{LABEL.get(x.split(':')[0], x)}-{x.split(':')[1]} vs {LABEL.get(y.split(':')[0], y)}-{y.split(':')[1]}"
        cells = [N(d["points"][pts[0]]["exact"])]
        for p in pts:
            cells += [N(d["points"][p]["copy_2a"]), N(d["points"][p]["copy_only"])]
        cells += ["--", "--"]
        L.append(f"\\multicolumn{{2}}{{l}}{{{name}}} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def tex_sensitivity(cal):
    rows = {t["tag"]: t for t in cal["threshold_sensitivity"]}
    order = [("literal", r"cross-source calibration, P$\ge$0.99"),
             ("fpr1e-7", r"FPR $\le 10^{-7}$, exhaustive cross-source"),
             ("t095", "sensitivity"), ("v11_final", r"\textbf{operating point (v1.1)}")]
    L = [r"\begin{table}[t]", r"\centering\footnotesize",
         r"\caption{Threshold sensitivity of the PPE-Unified v1.1 re-split. Groups = connected components of exact "
         r"$\cup$ 2a($\theta,\tau$) $\cup$ 2b($k=2$); split by the group-aware stratified algorithm of v1 with "
         r"background-aware counting (identical to v1 on singleton groups; seed 42, 70/20/10), on the corrected labels. "
         r"Max.\ dev.\ = largest deviation of a class's box share from its target split share "
         r"(v1: 0.07 pp). $\theta=0.98$ is the precision-conservative copy-level operating point; scene-level duplicates "
         r"are handled by 2b, not by a threshold.}",
         r"\label{tab:threshold_sensitivity}",
         r"\begin{tabular}{lrrrrrl}", r"\toprule",
         r"Rule & $\theta$ & $\tau$ & Groups & Largest (\%) & Train/val/test & Max.\ dev.\ (pp) \\", r"\midrule"]
    for tag, desc in order:
        t = rows.get(tag)
        if not t:
            continue
        im = t["images_v11"]
        fail = " (min-support FAIL)" if t["min_support_violations"] else ""
        L.append(f"{desc} & {t['theta']} & {t['tau_phash']} & {N(t['groups_v11'])} & {N(t['largest_groups'][0])} "
                 f"({t['largest_pct']}) & {N(im['train'])}/{N(im['val'])}/{N(im['test'])} & "
                 f"{t['max_class_ratio_dev_pp']['v11']}{fail} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def validation_sheets(n=300, lo=0.95, hi=0.98, per_sheet=10):
    """Random cross-split pairs of PPE-U v1 in the boundary band, for manual labelling (validation)."""
    e = pd.read_csv(RESULTS / "edges_cross_split_ppeu.csv")
    band = e[e.in_knn & (e.cos >= lo) & (e.cos <= hi)]
    pick = band.sample(n=min(n, len(band)), random_state=42).reset_index(drop=True)
    out = RESULTS / "validation_300"
    out.mkdir(exist_ok=True)
    pick.insert(0, "pair_id", [f"v{k:03d}" for k in range(len(pick))])
    pick["label"] = ""  # to fill by hand: dup_copy / dup_scene / not_dup
    pick["note"] = ""
    pick[["pair_id", "split_pair", "a", "b", "cos", "ph", "dh", "exact", "scene", "label", "note"]].to_csv(
        out / "validation_300.csv", index=False)
    for s0 in range(0, len(pick), per_sheet):
        rows = [dict(a_set="ppeu", a_rel=r.a, b_set="ppeu", b_rel=r.b,
                     label=f"{r.pair_id}  {r.split_pair}  cos={r.cos:.4f} ph={r.ph} dh={r.dh}")
                for r in pick.iloc[s0:s0 + per_sheet].itertuples()]
        contact_sheet(rows, out / f"sheet_{s0 // per_sheet + 1:02d}.png", cell=300,
                      title=f"PPE-U v1 validation pairs {s0:03d}-{s0 + len(rows) - 1:03d} (cos in [{lo}, {hi}])")
    (out / "README.txt").write_text(
        f"{len(pick)} cross-split pairs of PPE-Unified v1 drawn at random (seed 42) from the {len(band)} FAISS top-5 pairs "
        f"with cosine in [{lo}, {hi}].\nLabel each row of validation_300.csv: dup_copy (same photo, edited), dup_scene "
        f"(same camera/scene, different moment), not_dup.\nThis is a validation set for the copy-level operating "
        f"point, not used for calibration.\n", encoding="utf-8")
    return len(band), len(pick)


def residual(aud):
    """Cross-split pairs left in v1.1 at each cosine level, and the same counts in v1."""
    out = {}
    for n in ("ppeu", "ppeu_fix", "ppeu11"):
        if n in aud["within"]:
            out[n] = {p: {k: v["cross_split"][k] for k in ("exact", "copy_2a", "cos_tier", "phash_tier", "scene_2b", "union_all")}
                      for p, v in aud["within"][n]["points"].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    cal = json.loads((RESULTS / "calibration.json").read_text(encoding="utf-8"))
    aud = json.loads((RESULTS / "audit_v2.json").read_text(encoding="utf-8"))
    fig_calibration(cal)
    fig_cos(cal)
    (RESULTS / "tab_audit_v2.tex").write_text(tex_audit(aud), encoding="utf-8")
    (RESULTS / "tab_threshold_sensitivity.tex").write_text(tex_sensitivity(cal), encoding="utf-8")
    band, npick = validation_sheets()

    gen = Path(a.paper) / "generated"
    gen.mkdir(exist_ok=True)
    num_path = gen / "numbers.json"
    nums = json.loads(num_path.read_text(encoding="utf-8")) if num_path.exists() else {}
    nums["errata_v4"] = errata_numbers()
    if "clean2" in nums:  # E9, written by audit_v2.clean2 --paper
        nums["errata_v4"]["E9_e2_clean_exact_residual"] = nums["clean2"]["e2_clean_exact_residual"]
    if "clean2_C_loso" in nums:  # E9 (C), written by audit_v2.loso --paper
        nums["errata_v4"]["E9_C_no_clean_balanced_split"] = nums["clean2_C_loso"]["infeasible_balanced_split"]
    v11 = json.loads((RESULTS / "resplit_v11" / "resplit_v11_final.json").read_text(encoding="utf-8"))
    v11.pop("class_ratio_pct", None)
    nums["audit_v2"] = {
        "calibration_cross_source": {"thresholds_at_p99": cal["thresholds"], "copy_level_rule": cal["copy_level_rule"],
                                     "sets": {s: {k: v[k] for k in ("positives", "negatives")} for s, v in cal["sets"].items()}},
        "calibration_within_set_holdout_B": cal["within_set_holdout"]["at_operating_point"],
        "operating_point": cal["operating_point"],
        "threshold_sensitivity": cal["threshold_sensitivity"],
        "within": {n: {p: {k: v[k] for k in ("thresholds", "cross_split", "groups_exact", "groups_exact_2a",
                                                 "groups_exact_2b", "groups_all")}
                       for p, v in d["points"].items()} for n, d in aud["within"].items()},
        "cross": {s: d["points"] for s, d in aud["cross"].items()},
        "validation_300": {"band": [0.95, 0.98], "pairs_in_band": band, "sampled": npick}}
    nums["ppe_unified_v11"] = {"resplit": v11, "residual_cross_split": residual(aud)}
    num_path.write_text(json.dumps(nums, indent=1), encoding="utf-8")
    for fn in ("fig_calibration.pdf", "fig_cos_distribution.pdf", "tab_audit_v2.tex", "tab_threshold_sensitivity.tex",
               "calibration.json"):
        shutil.copy2(RESULTS / fn, gen / fn)
    print(json.dumps(nums["ppe_unified_v11"]["residual_cross_split"], indent=1))


if __name__ == "__main__":
    main()
