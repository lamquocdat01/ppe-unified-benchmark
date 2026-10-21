"""tab_generality.tex + summary.json: eight-point validation of four datasets with ONE unchanged validator.

Rows are the eight checks; columns PPE-Unified v1.1, v1-fix, VKITTI2 (naive frame split) and VIRAT-DIVA
(official clip split). Each cell is the status plus the decisive number; N/A cells name what the schema
does not declare.
Usage: python generality/make_table.py "<Q1 ESWA Submission dir>"
"""
import json
import shutil
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results" / "generality"
COLS = [("validator_ppe_unified_v11.json", "PPE-U v1.1"), ("validator_ppe_unified_v1fix.json", "PPE-U v1-fix"),
        ("validator_vkitti2.json", "VKITTI2$^a$"), ("validator_virat.json", "VIRAT-DIVA$^b$")]
ROWS = [("1_class_ids", "1 Class IDs"), ("2_boxes", "2 Boxes / pairing"), ("3_pair_consistency", "3 Compl.--viol.\\ pairs"),
        ("4_image_size", "4 Uniform size"), ("5_leakage", "5 Leakage (ID, MD5, 2a, 2b)"), ("6_corrupt", "6 Corrupt images"),
        ("7_min_support", "7 Min.\\ support"), ("8_background", "8 Background share")]


def N(x):
    return f"{x:,}".replace(",", "{,}")


def cell(k, c):
    st = c["status"]
    if st == "N/A":
        why = {"3_pair_consistency": "no pairs", "4_image_size": "mixed sizes", "8_background": c.get("reason", "")}.get(k, "")
        why = {"no background images": "no bg.\\ images", "schema declares no split ratio": "no ratio"}.get(why, why)
        return f"N/A ({why})"
    if k == "2_boxes":
        d = c["invalid_boxes"] + c["unparseable_lines"] + c["images_without_label"] + c["labels_without_image"]
        return f"{st} ({d} bad)" if d else st
    if k == "3_pair_consistency":
        return f"{st} ({c['same_object_conflicts']} confl.)" if c["same_object_conflicts"] else st
    if k == "5_leakage":
        g, a, b = c["groups_in_multiple_splits"] + c["md5_in_multiple_splits"], c["copy_level_cross_split_pairs"], c["scene_level_cross_split_pairs"]
        return f"{st} ({N(g)}/{N(a)}/{N(b)})"
    if k == "7_min_support":
        return f"{st} ({len(c['violations'])} viol.)" if c["violations"] else st
    if k == "8_background":
        return f"{st} ({c['max_deviation_pp']:.1f} pp)"
    if k == "6_corrupt":
        return f"{st} ({N(c['decoded'])} dec.)"
    return st


def main():
    reps = [json.loads((RES / f).read_text(encoding="utf-8")) for f, _ in COLS]
    L = [r"\begin{table*}[t]", r"\centering\footnotesize",
         r"\caption{Eight-point validation with one unchanged, schema-driven validator on PPE and non-PPE data. "
         r"Leakage cell: (groups or MD5 across splits / copy-level 2a pairs, any cosine $\ge0.98$ or pHash $=0$ / "
         r"scene-level 2b pairs). N/A = the schema declares nothing to check. "
         r"$^a$Virtual KITTI~2 clone, Camera~0, 5 scenes, frames shuffled 70/20/10 (common practice); "
         r"$^b$VIRAT ground video, IARPA DIVA boxes, 1 frame/s, official train/validate split by clip.}",
         r"\label{tab:generality}",
         r"\begin{tabular}{l" + "l" * len(COLS) + "}", r"\toprule",
         "Check & " + " & ".join(n for _, n in COLS) + r" \\",
         "Images & " + " & ".join(N(r["images"]) for r in reps) + r" \\", r"\midrule"]
    for k, lab in ROWS:
        L.append(f"{lab} & " + " & ".join(cell(k, r["checks"][k]) for r in reps) + r" \\")
    L += [r"\midrule", "PASS / FAIL / N/A & " + " & ".join(
        f"{r['summary']['PASS']} / {r['summary']['FAIL']} / {r['summary']['N/A']}" for r in reps) + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (RES / "tab_generality.tex").write_text("\n".join(L), encoding="utf-8")
    summary = {n: {"images": r["images"], "summary": r["summary"], "status": {k: r["checks"][k]["status"] for k, _ in ROWS},
                   "leakage": {x: r["checks"]["5_leakage"][x] for x in ("groups_in_multiple_splits", "md5_in_multiple_splits",
                                                                         "copy_level_cross_split_pairs", "scene_level_cross_split_pairs")}}
               for (_, n), r in zip(COLS, reps)}
    (RES / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if len(sys.argv) > 1:
        gen = Path(sys.argv[1]) / "generated"
        shutil.copy2(RES / "tab_generality.tex", gen / "tab_generality.tex")
        num = json.loads((gen / "numbers.json").read_text(encoding="utf-8"))
        num["generality"] = summary
        (gen / "numbers.json").write_text(json.dumps(num, indent=1), encoding="utf-8")
    print("\n".join(L[9:-3]))


if __name__ == "__main__":
    main()
