"""numbers.json -> numbers.tex (LaTeX macros) for manuscript v5 (P5, 24-10-2026).

Every quantitative statement in the v5 body uses a macro from numbers.tex; nothing is typed by hand.
Two steps:
  1. --import-v4: copy the v4 experiment results (E1-E6, published in v4.0) from the result CSVs into
     numbers.json key 'v4_results' (source file recorded per entry). E2b (RT-DETR) lives outside the repo
     (Kaggle output), hence --e2b.
  2. write numbers.tex. Values of key 't4_1_partial' (status PARTIAL) and every T4 cell that has no result yet
     are wrapped in \\TBD{...} (bold red), so they cannot be mistaken for final numbers.

Usage:
  python -u scripts/make_numbers_tex.py --paper <Q1 ESWA Submission> --out <latex_source>/numbers.tex
         [--import-v4 --e2b <dir with e2b_runs.csv, e2b_inflation.csv>]
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def import_v4(e2b_dir):
    r, ext = REPO / "results", REPO / "results" / "extended"
    det = {row["model"].split(" (")[0]: row for row in read_csv(r / "final_detection_compare.csv")}
    cls = {row["model"].split(" (")[0]: row for row in read_csv(r / "final_classification_compare.csv")}
    e2 = {(x["dataset"][0], x["arm"]): x for x in read_csv(ext / "e2_runs.csv")}
    e2b = {(x["dataset"][0], x["arm"]): x for x in read_csv(Path(e2b_dir) / "e2b_runs.csv")}
    e3 = {x["model"]: x for x in read_csv(ext / "e3_sh17.csv")}
    e4 = json.loads((ext / "e4_stats.json").read_text(encoding="utf-8"))
    e5 = {x["arm"]: x for x in read_csv(ext / "e5_ablation_rfs.csv")}
    e6 = {x["arm"]: x for x in read_csv(ext / "e6_resolution.csv")}
    f = float
    out = {
        "_source": {"E1": "results/final_{detection,classification}_compare.csv",
                    "E2": "results/extended/e2_runs.csv", "E2b": str(Path(e2b_dir) / "e2b_runs.csv"),
                    "E3": "results/extended/e3_sh17.csv", "E4": "results/extended/e4_stats.json",
                    "E5": "results/extended/e5_ablation_rfs.csv", "E6": "results/extended/e6_resolution.csv",
                    "note": "PPE-Unified v1 (E1, E3-E6) and original vs exact-clean splits of B/C (E2); published in v4.0"},
        "E1_det": {m: {"mAP50": f(d["mAP50_test"]), "mAP50_95": f(d["mAP50_95_test"]), "ms": f(d["infer_ms"]),
                       "P": f(d["precision_test"]) if d["precision_test"] else None,
                       "R": f(d["recall_test"]) if d["recall_test"] else None} for m, d in det.items()},
        "E1_cls": {m: {"acc": f(d["accuracy"]), "macro_f1": f(d["macro_f1"]), "worst_f1": f(d["worst_class_f1"])}
                   for m, d in cls.items()},
        "E2": {},
        "E3": {m: {"sh17": f(d["mAP50_4cls_SH17"]), "in_dist": f(d["mAP50_4cls_inDist"]),
                   **{c: f(d[f"AP50_{c}"]) for c in ("helmet", "vest", "gloves", "mask")}} for m, d in e3.items()},
        "E4": e4,
        "E5": {a: {"mAP50": f(d["mAP50_test"]), "mAP50_95": f(d["mAP50_95_test"])} for a, d in e5.items()},
        "E6": {a: {"mAP50": f(d["mAP50_test"]), "mAP50_95": f(d["mAP50_95_test"])} for a, d in e6.items()},
    }
    v11 = json.loads((r / "generality" / "validator_ppe_unified_v11.json").read_text(encoding="utf-8"))
    out["_source"]["v11_boxes"] = "results/generality/validator_ppe_unified_v11.json (check 1)"
    out["v11_boxes"] = v11["checks"]["1_class_ids"]["boxes"]
    for model, runs in (("YOLO26s", e2), ("RT-DETR-L", e2b)):
        for ds in ("B", "C"):
            lk, cl = runs[(ds, "leaky")], runs[(ds, "clean")]
            out["E2"][f"{model}_{ds}"] = {
                "leaky_mAP50": f(lk["mAP50_test"]), "clean_mAP50": f(cl["mAP50_test"]),
                "leaky_mAP50_95": f(lk["mAP50_95_test"]), "clean_mAP50_95": f(cl["mAP50_95_test"]),
                "infl_mAP50": round(f(lk["mAP50_test"]) - f(cl["mAP50_test"]), 4),
                "infl_mAP50_95": round(f(lk["mAP50_95_test"]) - f(cl["mAP50_95_test"]), 4)}
    return out


# ---------------------------------------------------------------- formatting
def n(x):
    """integer with LaTeX thousands separator"""
    s = f"{int(round(x)):,}"
    return s.replace(",", "{,}")


def d(x, k=3):
    return f"{x:.{k}f}"


def pts(x, k=1, sign=True):
    """mAP difference -> points"""
    v = 100 * x
    return (f"+{v:.{k}f}" if v >= 0 else f"{v:.{k}f}") if sign else f"{v:.{k}f}"


def pct(a, b, k=1):
    return f"{100 * a / b:.{k}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--import-v4", action="store_true")
    ap.add_argument("--e2b")
    a = ap.parse_args()
    p = Path(a.paper) / "generated" / "numbers.json"
    J = json.loads(p.read_text(encoding="utf-8"))
    if a.import_v4:
        J["v4_results"] = import_v4(a.e2b)
        p.write_text(json.dumps(J, indent=1), encoding="utf-8")

    M = {}  # macro name -> value (LaTeX)
    E, A = J["errata_v4"], J["audit_v2"]
    W = A["within"]

    # --- source datasets (errata E1-E4)
    for k, ds in (("A", "A"), ("B", "B"), ("C", "C"), ("SH", "sh17")):
        M[f"img{k}"] = n(E[ds]["images"])
    for k in ("B", "C"):
        e = E[k]
        M[f"groups{k}"] = n(e["groups"])
        M[f"multMax{k}"] = n(max(int(m) for m in e["multiplicity"]))
        M[f"tv{k}"], M[f"tt{k}"], M[f"vt{k}"] = n(e["groups_train_val"]), n(e["groups_train_test"]), n(e["groups_val_test"])
        M[f"multi{k}"] = n(e["groups_multi_split"])
    M["mdfiveCrossB"] = n(E["B"]["md5_groups_cross_split"])
    M["mdfiveUniqueB"] = n(E["B"]["md5_unique"])

    # --- two-tier audit at the operating point and sensitivity
    for key, ds in (("A", "A"), ("B", "B"), ("C", "C"), ("SH", "sh17"), ("V", "ppeu"), ("Vel", "ppeu11")):
        for tag, t in (("", "t098"), ("NF", "t095"), ("Ninety", "t090")):
            c = W[ds][t]["cross_split"]
            M[f"exact{key}{tag}"] = n(c["exact"])
            M[f"copy{key}{tag}"] = n(c["copy_2a"])
            M[f"scene{key}{tag}"] = n(c["scene_2b"])
            M[f"leakImgCopy{key}{tag}"] = n(c["leaked_images_2a"])
            M[f"leakImgAll{key}{tag}"] = n(c["leaked_images_all"])
            g = W[ds][t]["groups_all"]
            M[f"spanGroups{key}{tag}"], M[f"spanImg{key}{tag}"] = n(g["spanning"]), n(g["images_in_spanning"])
            g2 = W[ds][t]["groups_exact_2a"]
            M[f"spanGroupsCopy{key}{tag}"], M[f"spanImgCopy{key}{tag}"] = n(g2["spanning"]), n(g2["images_in_spanning"])
        c = W[ds]["t098"]["cross_split"]
        M[f"extraCopy{key}"] = n(c["union_exact_2a"] - c["exact"])
        if c["exact"]:
            M[f"extraCopyPct{key}"] = pct(c["union_exact_2a"] - c["exact"], c["exact"], 0)
            M[f"extraCopyX{key}"] = f"{c['union_exact_2a'] / c['exact']:.1f}"
    M["imgV"] = n(W["ppeu"]["t098"]["groups_exact"]["groups"])
    X = A["cross"]
    M["crossVSH"] = n(X["ppeu:train=sh17:val"]["t098"]["union_all"])
    M["crossVelSHNinety"] = n(X["ppeu11:train=sh17:val"]["t090"]["union_all"])

    # --- calibration
    cs = A["calibration_cross_source"]
    cos = cs["thresholds_at_p99"]["cos"]
    M["calCrossTheta"] = d(cos["threshold"], 3)
    M["calCrossP"] = d(cos["calibration"]["precision"], 3)
    M["calCrossHoldP"] = d(cos["holdout"]["precision"], 3)
    M["calCrossTau"] = n(cs["thresholds_at_p99"]["ph"]["threshold"])
    M["calRuleHoldP"] = d(cs["copy_level_rule"]["holdout"]["precision"], 3)
    M["calPosC"] = n(cs["sets"]["C"]["positives"]["positive_pairs"])
    M["calPosB"] = n(cs["sets"]["B"]["positives"]["positive_pairs"])
    M["calPosBall"] = n(cs["sets"]["B"]["positives"]["positive_pairs_all"])
    M["calDropGroupsB"] = n(cs["sets"]["B"]["positives"]["groups_dropped_cos_lt_0.5"])
    M["calDropImgB"] = n(cs["sets"]["B"]["positives"]["images_in_dropped_groups"])
    M["calNegPairs"] = n(sum(v["pairs"] for s in cs["sets"].values() for v in s["negatives"].values()))
    h = A["calibration_within_set_holdout_B"]
    M["opTheta"] = d(A["operating_point"]["cos"], 2)
    M["opTau"] = n(A["operating_point"]["phash"])
    M["opK"] = n(A["operating_point"]["scene_k"])
    M["opTopk"] = n(A["operating_point"]["topk"])
    M["holdP"], M["holdR"] = d(h["precision"], 3), d(h["recall"], 2)
    M["holdTP"], M["holdFP"] = n(h["tp"]), n(h["fp"])
    M["valBandPairs"] = n(A["validation_300"]["pairs_in_band"])
    ts = {t["tag"]: t for t in A["threshold_sensitivity"]}
    M["tsLitGroups"] = n(ts["literal"]["groups_v11"])
    M["tsLitLargestPct"] = d(ts["literal"]["largest_pct"], 2)
    M["tsFprTheta"] = d(ts["fpr1e-7"]["theta"], 4)
    M["tsFprTau"] = n(ts["fpr1e-7"]["tau_phash"])
    M["tsFprGroups"] = n(ts["fpr1e-7"]["groups_v11"])
    M["tsFprDev"] = d(ts["fpr1e-7"]["max_class_ratio_dev_pp"]["v11"], 2)
    M["tsNFDev"] = d(ts["t095"]["max_class_ratio_dev_pp"]["v11"], 2)
    M["tsNFGroups"] = n(ts["t095"]["groups_v11"])

    # --- PPE-Unified v1.1
    R = J["ppe_unified_v11"]["resplit"]
    M["velGroups"] = n(R["groups_v11"])
    M["vOneGroups"] = n(R["groups_v1"])
    M["velLargest"] = n(R["largest_groups"][0])
    M["velLargestPct"] = d(R["largest_pct"], 1)
    for s in ("train", "val", "test"):
        M[f"vel{s.capitalize()}"] = n(R["images_v11"][s])
        M[f"velBg{s.capitalize()}"] = n(R["background_per_split"][s])
    M["velDev"] = d(R["max_class_ratio_dev_pp"]["v11"], 2)
    M["vOneDev"] = d(R["max_class_ratio_dev_pp"]["v1"], 2)
    M["velEdgesCos"], M["velEdgesPh"], M["velEdgesScene"] = (n(R["edges"]["copy_cos"]), n(R["edges"]["copy_phash"]),
                                                            n(R["edges"]["scene"]))
    cc = J["ppe_unified_v11"]["compat_check_v1"]
    M["compatPct"] = d(cc["identical_pct"], 0)
    M["labelFixes"] = n(len(J["ppe_unified_v11"]["label_fixes"]))
    M["labelFixClip"] = n(sum(1 for x in J["ppe_unified_v11"]["label_fixes"] if x["action"] == "clip_to_image"))
    M["labelFixDrop"] = n(sum(1 for x in J["ppe_unified_v11"]["label_fixes"] if x["action"] != "clip_to_image"))
    G = J["generality"]
    vf = G["PPE-U v1-fix"]["leakage"]
    M["valVfixCopy"], M["valVfixScene"] = n(vf["copy_level_cross_split_pairs"]), n(vf["scene_level_cross_split_pairs"])
    for k, name in (("Vk", "VKITTI2$^a$"), ("Vi", "VIRAT-DIVA$^b$")):
        g = G[name]
        M[f"img{k}"] = n(g["images"])
        M[f"pass{k}"], M[f"fail{k}"], M[f"na{k}"] = (n(g["summary"]["PASS"]), n(g["summary"]["FAIL"]),
                                                     n(g["summary"]["N/A"]))
        M[f"copy{k}Val"] = n(g["leakage"]["copy_level_cross_split_pairs"])
        M[f"scene{k}Val"] = n(g["leakage"]["scene_level_cross_split_pairs"])

    # --- B / C arms (clean2)
    arms = J["clean2"]["arms"]
    for ds in ("B", "C"):
        for ak, arm in (("Orig", "original"), ("Ex", "clean_exact"), ("ExBal", "clean_exact_balanced"), ("Two", "clean2")):
            v = arms[ds][arm]
            cp = v["cross_split_pairs"]
            pre = f"arm{ds}{ak}"
            M[pre + "Groups"], M[pre + "Largest"] = n(v["groups"]), n(v["largest_group"])
            M[pre + "Dev"] = d(v["max_class_dev_pp"], 2)
            M[pre + "Exact"], M[pre + "Cos"], M[pre + "Ph"], M[pre + "Scene"] = (n(cp["exact"]), n(cp["cos>=0.98"]),
                                                                              n(cp["copy_phash"]), n(cp["scene"]))
            M[pre + "CosNF"], M[pre + "CosNinety"] = n(cp["cos>=0.95"]), n(cp["cos>=0.9"])
            M[pre + "Test"] = n(v["images"]["test"])
    inf = E["E9_C_no_clean_balanced_split"]
    M["seqN"] = n(inf["n_sequences"])
    M["seqPct"] = d(inf["sequences_image_pct"], 2)
    sh = inf["sequence_box_share_pct"]
    M["seqPPnoglove"], M["seqPPglove"] = d(sh["PP02img1000_jpg"]["no_glove"], 1), d(sh["PP02img1000_jpg"]["glove"], 1)
    M["seqKpGoggles"] = d(sh["kp1000_jpg"]["goggles"], 1)
    M["seqPackNoGoggles"] = d(sh["packing1000_jpg"]["no_goggles"], 1)
    M["seqDev"] = d(inf["clean2_max_class_dev_pp"], 2)
    M["seqTarget"] = d(inf["target_pp"], 0)
    L = J["clean2_C_loso"]
    for f_ in L["folds"]:
        k = "ABCD"[f_["fold"]]
        M[f"loso{k}Test"] = n(f_["images"]["test"])
        M[f"loso{k}Train"] = n(f_["images"]["train"])
        M[f"loso{k}Eval"] = ", ".join(c.replace("_", r"\_") for c, b in f_["boxes"].items() if b["eval"] == "ok")
        M[f"loso{k}CosNF"] = n(f_["cross_split_pairs"]["cos>=0.95"])
        M[f"loso{k}Seq"] = f_["test_group"].split("1000")[0].split("0--")[0].replace("_", r"\_")
    M["losoGPUh"] = d(L["gpu_estimate"]["yolo26s_total_h"], 1)

    # --- v4 results (E1-E6)
    V = J["v4_results"]
    e1 = V["E1_det"]
    for k, m in (("Yolo", "YOLO26s"), ("Detr", "RT-DETR-L"), ("Frcnn", "Faster R-CNN")):
        M[f"eOne{k}"] = d(e1[m]["mAP50"], 3)
        M[f"eOne{k}Hi"] = d(e1[m]["mAP50_95"], 3)
        M[f"eOne{k}Ms"] = d(e1[m]["ms"], 1)
    M["eOneSpeedup"] = f"{e1['RT-DETR-L']['ms'] / e1['YOLO26s']['ms']:.1f}"
    c1 = V["E1_cls"]
    for k, m in (("VitB", "ViT-B/16"), ("VitS", "ViT-S/16"), ("Rn", "ResNet50"), ("Vgg", "VGG16"), ("Gru", "GRU-256x1")):
        M[f"eOne{k}Acc"], M[f"eOne{k}Fone"] = d(c1[m]["acc"], 3), d(c1[m]["macro_f1"], 3)
        M[f"eOne{k}Worst"] = d(c1[m]["worst_f1"], 3)
    e2 = V["E2"]
    for k, m in (("Yolo", "YOLO26s"), ("Detr", "RT-DETR-L")):
        for ds in ("B", "C"):
            x = e2[f"{m}_{ds}"]
            M[f"eTwo{k}{ds}Leaky"], M[f"eTwo{k}{ds}Clean"] = d(x["leaky_mAP50"], 4), d(x["clean_mAP50"], 4)
            M[f"eTwo{k}{ds}Infl"], M[f"eTwo{k}{ds}InflHi"] = pts(x["infl_mAP50"], 2), pts(x["infl_mAP50_95"], 2)
    infl = [e2[k]["infl_mAP50"] for k in e2]
    M["eTwoMin"], M["eTwoMax"] = pts(min(infl), 1), pts(max(infl), 1)
    e3 = V["E3"]
    for k, m in (("Yolo", "YOLO26s"), ("Detr", "RT-DETR-L")):
        M[f"eThree{k}"], M[f"eThree{k}In"] = d(e3[m]["sh17"], 3), d(e3[m]["in_dist"], 3)
        M[f"eThree{k}Drop"] = d(100 * (1 - e3[m]["sh17"] / e3[m]["in_dist"]), 1)
        M[f"eThree{k}Helmet"], M[f"eThree{k}Vest"] = d(e3[m]["helmet"], 3), d(e3[m]["vest"], 3)
    M["velBoxes"] = n(V["v11_boxes"])
    x = e2["YOLO26s_C"]
    M["eTwoHiddenErr"] = pct(x["infl_mAP50"], 1 - x["clean_mAP50"], 0)
    M["eFourMean"], M["eFourStd"] = d(V["E4"]["mAP50_mean"], 3), d(V["E4"]["mAP50_std"], 3)
    M["eFourN"] = n(V["E4"]["n_seeds"])
    M["eFiveOn"], M["eFiveOff"] = d(V["E5"]["rfs_on"]["mAP50"], 4), d(V["E5"]["rfs_off"]["mAP50"], 4)
    M["eSixHi"], M["eSixLo"] = d(V["E6"]["res640"]["mAP50"], 4), d(V["E6"]["res416"]["mAP50"], 4)
    M["eSixGain"] = pts(V["E6"]["res640"]["mAP50"] - V["E6"]["res416"]["mAP50"], 2)

    # --- E2'': same v1-fix models, leaked vs clean test images (inference only)
    P2 = J.get("e2pp_leaked_vs_clean")
    if P2:
        sub = {"full": "Full", "leaked": "Grp", "clean": "GrpClean", "direct_leaked": "Cp", "direct_clean": "CpClean"}
        S2 = P2["subsets"]
        for k, s in S2.items():
            M[f"epp{sub[k]}Img"], M[f"epp{sub[k]}Box"] = n(s["images"]), n(s["boxes"])
        M["eppGrpPct"] = pct(S2["leaked"]["images"], S2["full"]["images"], 1)
        M["eppCpPct"] = pct(S2["direct_leaked"]["images"], S2["full"]["images"], 1)
        M["eppBoot"], M["eppMinGt"] = n(P2["n_boot"]), n(P2["min_gt_per_class"])
        for mk, name in (("yolo26s_s0", "YoloA"), ("yolo26s_s1", "YoloB"), ("rtdetr-l_s0", "Detr")):
            r = P2["models"].get(mk)
            if not r:
                continue
            M[f"epp{name}Val"] = d(r["ultralytics_val_full"]["mAP50"], 3)
            for k, s in sub.items():
                M[f"epp{name}{s}"], M[f"epp{name}{s}Hi"] = d(r[k]["mAP50"], 3), d(r[k]["mAP50_95"], 3)
            for dk, s in (("leaked_minus_clean", "Grp"), ("direct_leaked_minus_clean", "Cp")):
                dd = r[dk]
                M[f"epp{name}{s}Delta"] = pts(dd["mAP50"], 1)
                M[f"epp{name}{s}DeltaLo"], M[f"epp{name}{s}DeltaUp"] = (pts(dd["ci95_mAP50"][0], 1),
                                                                        pts(dd["ci95_mAP50"][1], 1))
                M[f"epp{name}{s}DeltaHi"] = pts(dd["mAP50_95"], 1)
                M[f"epp{name}{s}DeltaHiLo"], M[f"epp{name}{s}DeltaHiUp"] = (pts(dd["ci95_mAP50_95"][0], 1),
                                                                            pts(dd["ci95_mAP50_95"][1], 1))
                M[f"epp{s}NCls"] = n(len(dd["classes"]))
        for dk, s in (("leaked_minus_clean", "Grp"), ("direct_leaked_minus_clean", "Cp")):
            v = [r[dk]["mAP50"] for r in P2["models"].values()]
            M[f"epp{s}DeltaMin"], M[f"epp{s}DeltaMax"] = pts(min(v), 1), pts(max(v), 1)
        M["eppNModels"] = n(len(P2["models"]))

    # --- T4 (PARTIAL or pending): always wrapped in \TBD
    T = J.get("t4_1_partial", {})
    tbd = {}
    if T:
        y, r = T["yolo26s"], T["rtdetr-l"]
        tbd["tFourYoloFix"] = d(y["per_variant"]["v1fix"]["mAP50_test"]["mean"], 3)
        tbd["tFourYoloFixN"] = n(y["per_variant"]["v1fix"]["n"])
        tbd["tFourYoloVel"] = d(y["per_variant"]["v1.1"]["mAP50_test"]["mean"], 3) if y["per_variant"]["v1.1"]["n"] else "pending"
        tbd["tFourYoloVelN"] = n(y["per_variant"]["v1.1"]["n"])
        yi = y["inflation_v1fix_minus_v11"]
        tbd["tFourYoloInfl"] = pts(yi["mAP50_test"], 1) if yi else "pending"
        tbd["tFourYoloInflHi"] = pts(yi["mAP50_95_test"], 1) if yi else "pending"
        tbd["tFourYoloFixHi"] = d(y["per_variant"]["v1fix"]["mAP50_95_test"]["mean"], 3)
        tbd["tFourYoloVelHi"] = d(y["per_variant"]["v1.1"]["mAP50_95_test"]["mean"], 3) if y["per_variant"]["v1.1"]["n"] else "pending"
        tbd["tFourDetrFix"] = d(r["per_variant"]["v1fix"]["mAP50_test"]["mean"], 3)
        tbd["tFourDetrFixHi"] = d(r["per_variant"]["v1fix"]["mAP50_95_test"]["mean"], 3)
        for k in ("tFourDetrVel", "tFourDetrVelHi", "tFourDetrInfl", "tFourDetrInflHi"):
            tbd[k] = "pending"
    # T4(2)-B arms (kernels t4-2b-e2-*): final when the model's runs are complete, else \TBD
    TB = J.get("t4_2B", {})
    for model, key in (("Yolo", "yolo26s"), ("Detr", "rtdetr-l")):
        res = TB.get(key, {})
        tgt = M if res.get("status") == "complete" else tbd
        for arm, ak in (("ExBal", "clean_exact_balanced"), ("Two", "clean2")):
            s = res.get("arms", {}).get(ak)
            tgt[f"tTwoB{arm}{model}"] = d(s["mAP50"]["mean"], 4) if s else "pending"
            tgt[f"tTwoB{arm}{model}Hi"] = d(s["mAP50_95"]["mean"], 4) if s else "pending"
        if res.get("arms"):
            orig = V["E2"][f"{'YOLO26s' if key == 'yolo26s' else 'RT-DETR-L'}_B"]["leaky_mAP50"]
            ex = V["E2"][f"{'YOLO26s' if key == 'yolo26s' else 'RT-DETR-L'}_B"]["clean_mAP50"]
            bal, two = res["arms"]["clean_exact_balanced"]["mAP50"]["mean"], res["arms"]["clean2"]["mAP50"]["mean"]
            tgt[f"tTwoBInfl{model}"] = pts(orig - two, 2)
            tgt[f"tTwoBStepBal{model}"] = pts(ex - bal, 2)   # clean-exact -> balanced
            tgt[f"tTwoBStepTwo{model}"] = pts(bal - two, 2)  # balanced -> clean2 (perceptual grouping)
            tgt[f"tTwoBSeedSd{model}"] = d(max(res["arms"][k]["mAP50"]["std"] for k in res["arms"]), 4)
        else:
            for k in ("Infl", "StepBal", "StepTwo", "SeedSd"):
                tgt[f"tTwoB{k}{model}"] = "pending"
    # T4(2)-C LOSO folds: no result yet
    for k in "ABCD":
        tbd[f"loso{k}Map"] = "pending"
    tbd["inflRange"] = "+a to +b"
    VR = J.get("validator_runtime")
    if VR and VR.get("exit_code") in (0, 1) and not VR.get("other_python_processes"):
        M["validatorRuntime"] = f"{VR['wall_min']:.0f}~min"   # exit 1 = a check FAILed, still a complete run
        M["validatorImages"] = n(VR["images"])
    else:
        tbd["validatorRuntime"] = "runtime"

    lines = ["% Generated by scripts/make_numbers_tex.py from generated/numbers.json -- do not edit by hand.",
             r"\providecommand{\TBD}[1]{\textcolor{red}{\textbf{#1}}}"]
    for k, v in sorted(M.items()):
        assert k.isalpha(), k
        lines.append(rf"\newcommand{{\n{k}}}{{{v}}}")
    for k, v in sorted(tbd.items()):
        assert k.isalpha(), k
        lines.append(rf"\newcommand{{\n{k}}}{{\TBD{{{v}}}}}")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(M)} macros + {len(tbd)} TBD macros -> {a.out}")


if __name__ == "__main__":
    main()
