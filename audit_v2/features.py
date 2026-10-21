"""Per-image features for the two-tier audit, cached under THS_DATASETS/_derived.

Exact tier : MD5 of the file bytes.
Perceptual : 64-bit pHash and dHash of the image and of its horizontal mirror
             (Roboflow augmentation flips 50% of copies, plain hashes are not
             flip-invariant), plus an L2-normalised DINOv2 ViT-S/14 CLS embedding.
Uniform letterbox borders are trimmed before hashing/embedding so that a
letterboxed copy matches its unpadded original.

Usage:  python -m audit_v2.features C B ppeu sh17 A
Resumable (chunked .npz parts); pauses when free RAM < 6 GB and stops after 10 min.
"""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import imagehash
import numpy as np
import psutil
from PIL import Image, ImageOps

from .registry import DERIVED, list_images

MODEL = "vit_small_patch14_dinov2.lvd142m"
CHUNK = 2000
BATCH = 32
MIN_FREE_GB = 6.0
MAX_SIDE = 1024
FEAT_DIR = DERIVED / "features"


def trim_border(img: Image.Image, tol: float = 3.0) -> Image.Image:
    """Crop rows/columns at the edges that are (near-)constant, e.g. letterbox padding."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = a.shape
    rows = a.std(axis=1) < tol
    cols = a.std(axis=0) < tol
    top = np.argmax(~rows) if not rows.all() else 0
    bot = h - np.argmax(~rows[::-1]) if not rows.all() else h
    left = np.argmax(~cols) if not cols.all() else 0
    right = w - np.argmax(~cols[::-1]) if not cols.all() else w
    if (bot - top) < 0.3 * h or (right - left) < 0.3 * w:
        return img  # mostly flat image: keep as is
    if top or left or bot < h or right < w:
        return img.crop((left, top, right, bot))
    return img


def _hash64(h) -> np.uint64:
    return np.uint64(int(str(h), 16))


def load_one(path):
    raw = open(path, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    with Image.open(path) as im:
        orig_size = im.size
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG DCT-domain downscale: SH17 ships up to 8k px
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    im = trim_border(im)
    mir = ImageOps.mirror(im)
    hashes = (_hash64(imagehash.phash(im)), _hash64(imagehash.phash(mir)),
              _hash64(imagehash.dhash(im)), _hash64(imagehash.dhash(mir)))
    # squash (not centre-crop) to the ViT input so no border content is lost; keeps chunk RAM small
    return md5, hashes, im.resize((224, 224), Image.BICUBIC), orig_size


def wait_for_ram():
    for _ in range(10):
        if psutil.virtual_memory().available / 2**30 >= MIN_FREE_GB:
            return
        print(f"  [ram] free < {MIN_FREE_GB} GB, waiting 60 s", flush=True)
        time.sleep(60)
    raise SystemExit("free RAM stayed below threshold for 10 min - stopping (resumable)")


def build_model():
    import timm
    import torch
    torch.set_num_threads(4)
    m = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=224).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    cfg.update(input_size=(3, 224, 224), crop_pct=1.0)
    return m, timm.data.create_transform(**cfg)


def extract(name, model, tf):
    import torch
    root, items = list_images(name)
    out_dir = FEAT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps({"root": str(root), "n": len(items), "model": MODEL}))
    pool = ThreadPoolExecutor(3)
    t0 = time.time()
    for c0 in range(0, len(items), CHUNK):
        part = out_dir / f"part_{c0 // CHUNK:04d}.npz"
        if part.exists():
            continue
        wait_for_ram()
        chunk = items[c0:c0 + CHUNK]
        loaded = list(pool.map(lambda it: load_one(root / it[0]), chunk))
        embs = []
        with torch.inference_mode():
            for b0 in range(0, len(loaded), BATCH):
                x = torch.stack([tf(r[2]) for r in loaded[b0:b0 + BATCH]])
                e = model(x)
                embs.append(torch.nn.functional.normalize(e, dim=1).numpy().astype(np.float32))
        np.savez(part,
                 rel=np.array([it[0] for it in chunk]), split=np.array([it[1] for it in chunk]),
                 md5=np.array([r[0] for r in loaded]),
                 hashes=np.array([r[1] for r in loaded], dtype=np.uint64),
                 size=np.array([r[3] for r in loaded], dtype=np.int32),
                 emb=np.concatenate(embs))
        done = min(c0 + CHUNK, len(items))
        print(f"  {name}: {done}/{len(items)}  ({time.time() - t0:.0f} s this run)", flush=True)
        t0 = time.time()
    pool.shutdown()


def load(name):
    """Concatenate cached parts -> dict of arrays (hashes columns: ph, ph_mirror, dh, dh_mirror)."""
    parts = sorted((FEAT_DIR / name).glob("part_*.npz"))
    if not parts:
        raise FileNotFoundError(f"no features for {name}")
    d = [np.load(p) for p in parts]
    return {k: np.concatenate([x[k] for x in d]) for k in d[0].files}


if __name__ == "__main__":
    names = sys.argv[1:] or ["C", "B", "ppeu", "sh17", "A"]
    model, tf = build_model()
    for n in names:
        print(f"[features] {n}", flush=True)
        extract(n, model, tf)
    print("[features] done", flush=True)
