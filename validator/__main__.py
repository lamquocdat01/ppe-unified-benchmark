"""python -m validator <schema.yaml> [--out report.json]"""
import argparse
import json
import sys
from pathlib import Path

from .checks import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schema")
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = validate(a.schema)
    for k, c in rep["checks"].items():
        print(f"  [{c['status']:>4}] {k}")
    print(f"  {rep['dataset']}: {rep['summary']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    sys.exit(1 if rep["summary"]["FAIL"] else 0)


if __name__ == "__main__":
    main()
