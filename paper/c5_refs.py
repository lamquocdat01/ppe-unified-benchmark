"""C5 - every bibliography entry must carry a resolvable identifier (DOI or URL).

1. Report entries without doi/url.
2. Verify proposed identifiers before writing them:
   - Crossref DOIs  : GET api.crossref.org/works/<doi>, title similarity >= 0.9 and year +-1
   - DataCite DOIs  : 10.48550 (arXiv) via api.datacite.org/dois/<doi>, same checks
   - URLs           : HTTP 200 and the title words appear in the page
3. Write references_v5.bib with verified identifiers only; log everything to c5_log.json.

User-Agent carries noreply@example.org (no personal e-mail in API parameters).
Usage: python c5_refs.py [--apply]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import bibtexparser
import requests
from bibtexparser.bwriter import BibTexWriter
from rapidfuzz import fuzz

HERE = Path(__file__).resolve().parent
BIB = HERE / "references_v5.bib"
UA = {"User-Agent": "ppe-refcheck/0.1 (mailto:noreply@example.org)"}

# key -> ("doi", value) | ("url", value)
PROPOSED = {
    "ahmad2024sh17": ("doi", "10.48550/arXiv.2407.04590"),
    "lin2014coco": ("doi", "10.1007/978-3-319-10602-1_48"),
    "gupta2019lvis": ("doi", "10.1109/CVPR.2019.00550"),
    "deng2009imagenet": ("doi", "10.1109/CVPR.2009.5206848"),
    "ren2015faster": ("url", "https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html"),
    "redmon2016yolo": ("doi", "10.1109/CVPR.2016.91"),
    "ultralytics_yolo26": ("url", "https://docs.ultralytics.com/models/yolo26/"),
    "carion2020detr": ("doi", "10.1007/978-3-030-58452-8_13"),
    "simonyan2015vgg": ("doi", "10.48550/arXiv.1409.1556"),
    "he2016resnet": ("doi", "10.1109/CVPR.2016.90"),
    "dosovitskiy2021vit": ("url", "https://arxiv.org/abs/2010.11929"),  # OpenReview blocks bots
    "cho2014gru": ("doi", "10.3115/v1/D14-1179"),
    "kingma2015adam": ("doi", "10.48550/arXiv.1412.6980"),
    "loshchilov2019adamw": ("url", "https://arxiv.org/abs/1711.05101"),
    "northcutt2021labelerrors": ("doi", "10.48550/arXiv.2103.14749"),
    "recht2019imagenet": ("url", "https://proceedings.mlr.press/v97/recht19a.html"),
    "koh2021wilds": ("url", "https://proceedings.mlr.press/v139/koh21a.html"),
    "touvron2021deit": ("url", "https://proceedings.mlr.press/v139/touvron21a.html"),
}

# new references (plan section 6 + A1 dossier); added only if the DOI resolves and the title matches
NEW = [
    ("barz2020cifair", "article", "10.3390/jimaging6060041",
     "Do We Train on Test Data? Purging CIFAR of Near-Duplicates", 2020),
    ("oquab2024dinov2", "article", "10.48550/arXiv.2304.07193",
     "DINOv2: Learning Robust Visual Features without Supervision", 2023),
    ("radford2021clip", "inproceedings", "10.48550/arXiv.2103.00020",
     "Learning Transferable Visual Models From Natural Language Supervision", 2021),
    ("schuhmann2022laion", "inproceedings", "10.48550/arXiv.2210.08402",
     "LAION-5B: An open large-scale dataset for training next generation image-text models", 2022),
    ("farid2021perceptual", "article", "10.54501/jots.v1i1.24",
     "An Overview of Perceptual Hashing", 2021),
    ("dlede2024leakage", "misc", "10.48550/arXiv.2410.23312",
     "Improving Image Data Leakage Detection in Automotive Software", 2024),
    ("bouke2023leakage", "article", "10.1016/j.eswa.2023.120715",
     "An empirical study of pattern leakage impact during data preprocessing on machine learning-based intrusion detection models reliability", 2023),
    ("le2027behavioural", "article", "10.1016/j.eswa.2026.134262",
     "Behavioural leakage and the illusion of actionability: A deployment-honest evaluation framework for educational early warning systems", 2027),
    ("kim2023unknownbench", "article", "10.1016/j.eswa.2023.120461",
     "A unified benchmark for the unknown detection capability of deep neural networks", 2023),
    ("pisu2024ppe", "article", "10.1016/j.eswa.2023.122285",
     "Enhancing workplace safety: A flexible approach for personal protective equipment monitoring", 2024),
]


def norm(s):
    s = re.sub(r"[{}\\$]", "", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def get(url, **kw):
    for _ in range(4):
        r = requests.get(url, headers=UA, timeout=60, **kw)
        if r.status_code == 429:
            time.sleep(60)
            continue
        return r
    return r


def resolve_doi(doi):
    """Return (title, year, metadata dict) or None."""
    if doi.lower().startswith("10.48550/"):
        r = get(f"https://api.datacite.org/dois/{doi}")
        if r.status_code != 200:
            return None
        a = r.json()["data"]["attributes"]
        creators = [{"family": c.get("familyName", c.get("name", "")), "given": c.get("givenName", "")}
                    for c in a.get("creators", [])]
        return a["titles"][0]["title"], int(a["publicationYear"]), {"author": creators, "publisher": "arXiv"}
    r = get(f"https://api.crossref.org/works/{doi}")
    if r.status_code != 200:
        return None
    m = r.json()["message"]
    y = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    return m["title"][0], int(y), m


def check_url(url, title):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (refcheck; noreply@example.org)"},
                         timeout=60, allow_redirects=True)
    except requests.RequestException as e:
        return False, str(e)
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    words = [w for w in re.findall(r"[a-z]{4,}", norm(title))][:6]
    hit = sum(w in r.text.lower() for w in words)
    return hit >= max(1, len(words) - 1), f"HTTP 200, title words {hit}/{len(words)}"


def bib_authors(meta):
    names = []
    for a in meta.get("author", [])[:50]:
        fam, giv = a.get("family", ""), a.get("given", "")
        names.append(f"{fam}, {giv}".strip(", ") if fam else a.get("name", ""))
    return " and ".join(n for n in names if n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    db = bibtexparser.load(open(BIB, encoding="utf-8"))
    by_id = {e["ID"]: e for e in db.entries}
    log = {"existing": [], "proposed": [], "new": []}
    missing = [e["ID"] for e in db.entries if "doi" not in e and "url" not in e]
    log["missing_before"] = missing
    print(f"{len(db.entries)} entries, {len(missing)} without doi/url")
    for e in db.entries:  # re-verify DOIs already present
        if "doi" in e:
            res = resolve_doi(e["doi"])
            sim = fuzz.token_sort_ratio(norm(res[0]), norm(e.get("title"))) / 100 if res else 0
            # Crossref sometimes stores the title without its subtitle: accept a prefix match
            # when the year also agrees
            prefix = bool(res) and norm(e.get("title")).startswith(norm(res[0])) and str(res[1]) == e.get("year")
            ok = bool(res) and (sim >= 0.9 or prefix)
            log["existing"].append({"id": e["ID"], "doi": e["doi"], "resolves": bool(res), "sim": round(sim, 3),
                                    "prefix_match": prefix, "ok": ok})
            if not ok:
                print(f"  [existing?] {e['ID']} {e['doi']} sim={sim:.2f}")
            time.sleep(0.3)
    for key, (kind, val) in PROPOSED.items():
        e = by_id[key]
        if kind == "doi":
            res = resolve_doi(val)
            sim = fuzz.token_sort_ratio(norm(res[0]), norm(e["title"])) / 100 if res else 0
            yr_ok = bool(res) and abs(res[1] - int(e["year"])) <= 1
            ok = bool(res) and sim >= 0.9 and yr_ok
            info = f"sim={sim:.2f} year={'ok' if yr_ok else (res[1] if res else '-')}"
        else:
            ok, info = check_url(val, e["title"])
        log["proposed"].append({"id": key, kind: val, "ok": ok, "info": info})
        print(f"  [{'OK' if ok else 'FAIL'}] {key}: {kind} {val} ({info})")
        if ok and a.apply:
            e[kind] = val
        time.sleep(0.5)
    for key, typ, doi, title, year in NEW:
        res = resolve_doi(doi)
        sim = fuzz.token_sort_ratio(norm(res[0]), norm(title)) / 100 if res else 0
        ok = bool(res) and sim >= 0.9 and abs(res[1] - year) <= 1
        log["new"].append({"id": key, "doi": doi, "ok": ok, "sim": round(sim, 3), "year": res[1] if res else None})
        print(f"  [{'OK' if ok else 'FAIL'}] NEW {key}: {doi} sim={sim:.2f} year={res[1] if res else '-'}")
        if ok and a.apply and key not in by_id:
            meta = res[2]
            ent = {"ENTRYTYPE": typ, "ID": key, "title": "{" + res[0] + "}", "year": str(res[1]),
                   "doi": doi, "author": bib_authors(meta)}
            if meta.get("container-title"):
                ent["journal" if typ == "article" else "booktitle"] = meta["container-title"][0]
            for f in ("volume", "issue"):
                if meta.get(f):
                    ent["number" if f == "issue" else f] = str(meta[f])
            if meta.get("article-number"):
                ent["pages"] = str(meta["article-number"])
            elif meta.get("page"):
                ent["pages"] = meta["page"].replace("-", "--")
            if doi.lower().startswith("10.48550/"):
                ent["ENTRYTYPE"] = "misc"
                ent["howpublished"] = "arXiv preprint arXiv:" + doi.split("arXiv.")[1]
            db.entries.append(ent)
        time.sleep(0.5)
    if a.apply:
        w = BibTexWriter(); w.indent = "  "; w.order_entries_by = None
        BIB.write_text(w.write(db), encoding="utf-8")
    log["missing_after"] = [e["ID"] for e in db.entries if "doi" not in e and "url" not in e]
    (HERE / "c5_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"missing after: {len(log['missing_after'])}  {log['missing_after']}")
    bad = [x["id"] for x in log["existing"] if not x["ok"]]
    print(f"existing DOIs not verified: {len(bad)} {bad}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
