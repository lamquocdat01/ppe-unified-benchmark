"""SHA256SUMS of every tracked file at a commit, hashed from the committed blobs (the bytes that
raw.githubusercontent.com serves), plus an anonymous fetch-and-compare check (C7).

Usage:
  python scripts/sha256sums.py write [--rev HEAD]        # -> SHA256SUMS (excludes itself)
  python scripts/sha256sums.py verify --tag v5.0-eswa    # fetch each file without login, compare
"""
import argparse
import hashlib
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OWNER_REPO = "lamquocdat01/ppe-unified-benchmark"


def git(*a, binary=False):
    out = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout
    return out if binary else out.decode("utf-8")


def write(rev):
    lines = []
    for row in git("ls-tree", "-r", "--full-tree", rev).splitlines():
        meta, path = row.split("\t", 1)
        mode, typ, sha = meta.split()
        if typ != "blob" or path == "SHA256SUMS":
            continue
        data = git("cat-file", "blob", sha, binary=True)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {path}")
    (REPO / "SHA256SUMS").write_bytes(("\n".join(sorted(lines, key=lambda x: x[66:])) + "\n").encode("utf-8"))
    print(f"{len(lines)} files -> SHA256SUMS (from {rev})")


def verify(tag):
    base = f"https://raw.githubusercontent.com/{OWNER_REPO}/{tag}/"
    sums = urllib.request.urlopen(base + "SHA256SUMS", timeout=60).read().decode("utf-8")
    bad, n = [], 0
    for line in sums.splitlines():
        h, path = line.split("  ", 1)
        url = base + urllib.parse.quote(path)
        req = urllib.request.Request(url, headers={"User-Agent": "c7-check (noreply@example.org)"})
        data = urllib.request.urlopen(req, timeout=120).read()
        n += 1
        if hashlib.sha256(data).hexdigest() != h:
            bad.append(path)
    print(f"C7 anonymous fetch @ {tag}: {n} files, {len(bad)} hash mismatches {bad[:10]}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "verify"])
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--tag")
    a = ap.parse_args()
    if a.cmd == "write":
        write(a.rev)
    else:
        sys.exit(verify(a.tag))
