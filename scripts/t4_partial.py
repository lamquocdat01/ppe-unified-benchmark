"""T4(1) partial results -> numbers.json key 't4_1_partial' (decision 24-10-2026).

Reads results/t4/<model>/state.json (Kaggle output of t4-ppeu-v1fix-vs-v11-<model>, first session).
At 24-10: YOLO26s v1fix seeds 0,1 + v1.1 seed 0; RT-DETR-L v1fix seed 0 only. Labelled PARTIAL:
the inflation v1fix - v1.1 uses unequal seed counts and is NOT in ERRATA E7 until T4(1) is complete
(YOLO v1.1_s1 resume; RT-DETR v1.1_s0 in week 2, RT-DETR cut to 1 seed).

Usage:
  python -u scripts/t4_partial.py --paper <paper folder>
"""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

METRICS = ("mAP50_val", "mAP50_95_val", "mAP50_test", "mAP50_95_test", "P_test", "R_test")
PLANNED = {"yolo26s": {"v1fix": [0, 1], "v1.1": [0, 1]}, "rtdetr-l": {"v1fix": [0], "v1.1": [0]}}


def summarise(state, model):
    runs = {k: v for k, v in state.items() if k.startswith(f"t4_{model}_")}
    per_var = {}
    for var in ("v1fix", "v1.1"):
        rows = [r for r in runs.values() if r["variant"] == var]
        if not rows:
            per_var[var] = {"seeds": [], "n": 0}
            continue
        s = {"seeds": sorted(r["seed"] for r in rows), "n": len(rows)}
        for m in METRICS:
            xs = [r[m] for r in rows]
            s[m] = {"mean": round(mean(xs), 4), "std": round(stdev(xs), 4) if len(xs) > 1 else None}
        s["train_min_mean"] = round(mean(r["train_min"] for r in rows), 1)
        per_var[var] = s
    infl = None
    if per_var["v1fix"]["n"] and per_var["v1.1"]["n"]:
        infl = {m: round(per_var["v1fix"][m]["mean"] - per_var["v1.1"][m]["mean"], 4)
                for m in ("mAP50_test", "mAP50_95_test")}
    missing = [f"t4_{model}_{v}_s{sd}" for v, sds in PLANNED[model].items() for sd in sds
               if f"t4_{model}_{v}_s{sd}" not in runs]
    return {"runs": runs, "per_variant": per_var, "inflation_v1fix_minus_v11": infl, "missing_runs": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True, help="paper folder: write generated/numbers.json key 't4_1_partial'")
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results" / "t4"))
    a = ap.parse_args()
    out = {"status": "PARTIAL",
           "note": "unequal seed counts; not in ERRATA E7 until T4(1) complete (decision 24-10-2026)",
           "planned": PLANNED}
    for model in PLANNED:
        st = Path(a.results) / model / "state.json"
        out[model] = summarise(json.loads(st.read_text(encoding="utf-8")), model)
        print(model, json.dumps({k: out[model][k] for k in ("inflation_v1fix_minus_v11", "missing_runs")}))
    p = Path(a.paper) / "generated" / "numbers.json"
    nums = json.loads(p.read_text(encoding="utf-8"))
    nums["t4_1_partial"] = out
    p.write_text(json.dumps(nums, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
