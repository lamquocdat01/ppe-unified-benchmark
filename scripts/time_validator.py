"""Wall-clock time of the eight-point validator on one dataset -> numbers.json key 'validator_runtime' (24-10-2026).

Runs `python -m validator <schema>` once, as released (embeddings read from the feature cache when present; the
report states how many had to be computed). Records wall time, image count, embeddings computed and the machine.

Usage:
  python -u scripts/time_validator.py --schema validator/schema_ppe_unified_v11.yaml --paper <paper folder>
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--wait-for", help="first wait until no process command line contains this text")
    a = ap.parse_args()
    def waiting():
        cmds = [" ".join(p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"]) if p.pid != os.getpid()]
        return any(a.wait_for in c and "time_validator" not in c for c in cmds)

    while a.wait_for and waiting():
        time.sleep(120)
    out = Path(tempfile.gettempdir()) / "validator_timing_report.json"
    others = [p.info["cmdline"] for p in psutil.process_iter(["name", "cmdline"])
              if p.info["name"] and "python" in p.info["name"].lower() and p.pid != os.getpid()
              and p.info["cmdline"] and "time_validator" not in " ".join(p.info["cmdline"])]
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "validator", a.schema, "--out", str(out)], cwd=REPO)
    wall = time.time() - t0
    rep = json.loads(out.read_text(encoding="utf-8"))
    res = {"schema": a.schema, "dataset": rep["dataset"], "images": rep["images"], "summary": rep["summary"],
           "exit_code": r.returncode, "wall_s": round(wall, 1), "wall_min": round(wall / 60, 2),
           "embeddings_computed": rep["checks"]["5_leakage"].get("embeddings_computed_not_cached"),
           "machine": f"{platform.processor()} / {psutil.cpu_count(logical=True)} threads / "
                      f"{round(psutil.virtual_memory().total / 2**30)} GB RAM, CPU only",
           "other_python_processes": len(others), "date": time.strftime("%Y-%m-%d %H:%M")}
    p = Path(a.paper) / "generated" / "numbers.json"
    J = json.loads(p.read_text(encoding="utf-8"))
    J["validator_runtime"] = res
    p.write_text(json.dumps(J, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
