"""Top-level dataset card (README.md), ERRATA.md and SHA256SUMS.txt for the Kaggle dataset
ppe-unified-v1fix-v11 (P6, 01-11-2026). Every number comes from paper/generated/numbers.json.
The per-variant files (v1.1/, v1fix/: SHA256SUMS.csv, DATASET_CARD.md, FIXES.csv, manifest.csv) are
left untouched, because the SHA-256 of each SHA256SUMS.csv is recorded in numbers.json.

Usage: python scripts/make_dataset_card.py <numbers.json> <release folder>
"""
import hashlib
import json
import sys
from pathlib import Path

J = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
R = J["ppe_unified_v11"]["resplit"]
V = J["ppe_unified_v11"]["validator"]
W = J["audit_v2"]["within"]["ppeu"]["t098"]["cross_split"]
fixes = J["ppe_unified_v11"]["label_fixes"]
boxes = J["v4_results"]["v11_boxes"]
sha = J["ppe_unified_v11"]["sha256_of_SHA256SUMS"]
img = R["images_v11"]
n_img = sum(img.values())
f = lambda x: f"{int(x):,}"

card = f"""# PPE-Unified v1-fix and v1.1

Personal protective equipment (PPE) detection benchmark, 12 classes forming six compliance-violation
pairs (helmet/no_helmet, vest/no_vest, gloves/no_gloves, goggles/no_goggles, boots/no_boots,
mask/no_mask), YOLO format, images letterboxed to 640x640.

Two folders hold **the same {f(n_img)} images and the same corrected labels** ({f(boxes)} boxes); they
differ only in the split:

| Folder | Split | Use |
|---|---|---|
| `v1.1/` | near-duplicate-aware group split: {f(R['groups_v11'])} groups, train {f(img['train'])} / val {f(img['val'])} / test {f(img['test'])} | **use this** |
| `v1fix/` | the split of PPE-Unified v1, with the same label corrections | comparison with v1 only; it leaks (see ERRATA.md) |

## How v1.1 is split
Images are grouped by the connected components of three link types: tier 1, same source identifier or
MD5; tier 2a, copy-level near-duplicates (DINOv2 ViT-S/14 cosine >= 0.98 over all pairs, or
perceptual hash distance 0); tier 2b, adjacent video frames of one file-name sequence (frame numbers
within 2). Whole groups are assigned to train/val/test (70/20/10) by iterative stratification with
background balancing (seed 42). Result: zero cross-split pairs at every tier; the eight-point
validator passes {V['v11']['summary']['PASS']}/8 on v1.1 and {V['v1fix']['summary']['PASS']}/8 on v1-fix
(leakage check fails). Grouping by all tiers raises the largest class-share deviation from its split
target to {R['max_class_ratio_dev_pp']['v11']:.2f} percentage points (v1: {R['max_class_ratio_dev_pp']['v1']:.2f}).

## Files
- `FOLDER/{{train,val,test}}/{{images,labels}}/` YOLO images and labels; `FOLDER/data.yaml`.
- `FOLDER/manifest.csv` one row per image: file, split, source, source image identifier, number of
  boxes; in `v1.1/` also the group key `group_v11`.
- `FOLDER/FIXES.csv` the {len(fixes)} label corrections applied to both folders.
- `FOLDER/SHA256SUMS.csv` SHA-256 of every other file in the folder.
  SHA-256 of `v1.1/SHA256SUMS.csv`: `{sha['PPE_Unified_v1.1']}`;
  of `v1fix/SHA256SUMS.csv`: `{sha['PPE_Unified_v1fix']}`.
- `FOLDER/DATASET_CARD.md` build notes (Vietnamese) written when the folder was built; this README
  is the English card.
- `ERRATA.md` corrections to PPE-Unified v1 and to the earlier description of it.
- `SHA256SUMS.txt` SHA-256 of the top-level files.

## Sources and licence
Merged from three Roboflow Universe exports, each released under CC BY 4.0:
- Source A: Hard Hat Detection v2, https://universe.roboflow.com/computer-vision-filqz/hard-hat-detection-62rrp/dataset/2
- Source B: PPE detection v1 (testcasque), https://universe.roboflow.com/testcasque/ppe-detection-qlq3d/dataset/1
- Source C: PPE Detection v5 (mohamed-traore), https://universe.roboflow.com/mohamed-traore-2ekkp/ppe-detection-l80fg/dataset/5

This compilation is released under **CC BY 4.0**; attribute the three sources above and this dataset.
SH17 is not included (its licence does not allow redistribution).

## Code
Audit, re-split, validator and training notebooks: https://github.com/lamquocdat01/ppe-unified-benchmark
(tag `v5.0-eswa`).
"""

errata = f"""# ERRATA: PPE-Unified v1

1. **v1 is not leakage-free.** v1 was split by source identifier and MD5 and passes those checks, but
   at the perceptual tier its split holds {f(W['copy_2a'])} copy-level and {f(W['scene_2b'])} sequence-level
   (adjacent video frame) cross-split pairs (FAISS top-5 audit at cosine 0.98); the exhaustive
   copy-level count of the validator on v1-fix is {f(V['v1fix']['leakage']['copy_level_cross_split_pairs'])} pairs.
   Earlier descriptions of v1 as "leakage-free" or "split so that leakage is eliminated" are withdrawn.
   Scores on the v1 test split overstate performance on unseen scenes.
2. **Label corrections.** {len(fixes)} label errors are corrected in both folders (`FIXES.csv`): boxes
   extending beyond the image are clipped, and duplicate contradictory boxes (helmet and no_helmet on
   the same head) are removed.
3. **Replacement.** v1.1 (folder `v1.1/`) replaces v1. Folder `v1fix/` keeps the v1 split with the
   corrected labels, only so that v1 results can be compared with v1.1 on identical images.
4. The "clean" re-splits of sources B and C used in the earlier study to measure inflation were grouped
   by identifier only and still leak at the perceptual tier; the inflation figures reported with them
   are lower bounds.
"""

(out / "README.md").write_text(card, encoding="utf-8", newline="\n")
(out / "ERRATA.md").write_text(errata, encoding="utf-8", newline="\n")
files = ["README.md", "ERRATA.md", "v1.1/SHA256SUMS.csv", "v1fix/SHA256SUMS.csv"]
lines = [f"{hashlib.sha256((out / p).read_bytes()).hexdigest()}  {p}" for p in files]
(out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))
