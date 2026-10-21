"""One-shot build of PPE_Unified_v1fix + PPE_Unified_v1.1 from the corrected pool, then audit and validate.

Steps: fix_labels -> feature caches (re-keyed from the v1 cache; images are hardlinks, bytes identical)
-> resplit (background-aware algorithm, theta 0.98 / pHash 0 / scene k 2) -> SHA256SUMS + dataset cards
-> audit (2a top-5, 2b, residual 0.95/0.90) -> validator on both variants.

Usage: python -u -m audit_v2.pipeline_v11 --a002686 drop_line:4
"""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import features
from .registry import DATASETS_ROOT, DERIVED, REGISTRY, RESULTS, list_images

REPO = Path(__file__).resolve().parents[1]


def run(*args):
    print(">>", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-u", *args], cwd=REPO)
    return r.returncode


def cache_from_v1(name):
    """Feature cache for a variant whose files are hardlinks of v1 (verified on 100 random MD5s)."""
    f = features.load("ppeu")
    by = {r.split("/")[-1]: k for k, r in enumerate(f["rel"])}
    root, items = list_images(name)
    ix = np.array([by[r.split("/")[-1]] for r, _ in items])
    out = DERIVED / "features" / name
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("part_*.npz"):
        p.unlink()
    (out / "index.json").write_text(json.dumps({"root": str(root), "n": len(items), "model": features.MODEL,
                                                "derived_from": "ppeu (hardlinked images, identical bytes)"}))
    np.savez(out / "part_0000.npz", rel=np.array([r for r, _ in items]), split=np.array([s for _, s in items]),
             md5=f["md5"][ix], hashes=f["hashes"][ix], size=f["size"][ix], emb=f["emb"][ix])
    for k in random.Random(0).sample(range(len(items)), min(100, len(items))):
        assert hashlib.md5((root / items[k][0]).read_bytes()).hexdigest() == f["md5"][ix[k]], items[k]


def sha256sums(root):
    rows = [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size)
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.csv"]
    with open(root / "SHA256SUMS.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["path", "sha256", "bytes"]); w.writerows(rows)
    return len(rows), hashlib.sha256((root / "SHA256SUMS.csv").read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a002686", required=True)
    a = ap.parse_args()
    assert run("-m", "audit_v2.fix_labels", "--a002686", a.a002686) == 0
    cache_from_v1("ppeu_fix")
    assert run("-m", "audit_v2.resplit", "--name", "ppeu_fix", "--theta", "0.98", "--tau", "0", "--k", "2",
               "--tag", "v11_final", "--write", "PPE_SH17/PPE_Unified_v1.1") == 0
    cache_from_v1("ppeu11")
    sums = {}
    for name in ("ppeu_fix", "ppeu11"):
        root = DATASETS_ROOT / REGISTRY[name][0]
        y = (root / "data.yaml").read_text(encoding="utf-8").splitlines()
        y[0] = {"ppeu_fix": "# Safety-PPE Unified v1-fix (v1 split, corrected labels) - 12 lop",
                "ppeu11": "# Safety-PPE Unified v1.1 (near-duplicate-aware group re-split, corrected labels) - 12 lop"}[name]
        (root / "data.yaml").write_text("\n".join(y) + "\n", encoding="utf-8")
        sums[name] = sha256sums(root)  # card is written after, by build_cards(); sums recomputed there
    assert run("-m", "audit_v2.audit", "ppeu_fix", "ppeu11", "--cross", "ppeu11:train=sh17:val",
               "--out", "audit_v11_final.json") == 0
    rc = {}
    for schema, tag in (("validator/schema_ppe_unified_v11.yaml", "v11"), ("validator/schema_ppe_unified_v1fix.yaml", "v1fix")):
        rc[tag] = run("-m", "validator", schema, "--out", f"results/generality/validator_ppe_unified_{tag}.json")
    (RESULTS / "pipeline_v11.json").write_text(json.dumps({"a002686": a.a002686, "sha256sums_pre_card": sums,
                                                          "validator_exit": rc}, indent=1), encoding="utf-8")
    print("validator exit codes:", rc)


if __name__ == "__main__":
    main()
