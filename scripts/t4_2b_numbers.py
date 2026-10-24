"""T4(2)-B results -> numbers.json key 't4_2B' (24-10-2026).

Reads results/t4_2/<model>/results/t42B_<model>_runs.csv (Kaggle output of t4-2b-e2-<model>): source B,
arms clean_exact_balanced and clean2, seeds 42 and 0, E2 settings (15 epochs, 640). A model whose output is
missing is recorded as pending. The original and clean_exact arms are the v4 E2 runs (v4_results.E2, seed 42).

Usage:
  python -u scripts/t4_2b_numbers.py --paper <paper folder>
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

RES = Path(__file__).resolve().parents[1] / "results" / "t4_2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    out = {"note": "source B; arms clean_exact_balanced, clean2; seeds 42, 0; E2 settings (15 ep, 640)"}
    for model in ("yolo26s", "rtdetr-l"):
        f = RES / model / "results" / f"t42B_{model}_runs.csv"
        if not f.exists():
            out[model] = {"status": "pending"}
            continue
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        arms = {}
        for arm in ("clean_exact_balanced", "clean2"):
            r = [x for x in rows if x["arm"] == arm]
            s = {"seeds": [int(x["seed"]) for x in r], "n": len(r),
                 "runs": {x["seed"]: {"mAP50": float(x["mAP50_test"]), "mAP50_95": float(x["mAP50_95_test"])} for x in r}}
            for m, col in (("mAP50", "mAP50_test"), ("mAP50_95", "mAP50_95_test")):
                xs = [float(x[col]) for x in r]
                s[m] = {"mean": round(mean(xs), 4), "std": round(stdev(xs), 4) if len(xs) > 1 else None}
            arms[arm] = s
        out[model] = {"status": "complete" if all(arms[k]["n"] == 2 for k in arms) else "partial",
                      "source": str(f.relative_to(RES.parents[1])), "arms": arms}
    p = Path(a.paper) / "generated" / "numbers.json"
    nums = json.loads(p.read_text(encoding="utf-8"))
    nums["t4_2B"] = out
    p.write_text(json.dumps(nums, indent=1), encoding="utf-8")
    print(json.dumps({k: v.get("status") if isinstance(v, dict) else v for k, v in out.items()}))


if __name__ == "__main__":
    main()
