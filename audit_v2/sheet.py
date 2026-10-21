"""Side-by-side contact sheets of image pairs (for manual inspection of audit hits)."""
import json

from PIL import Image, ImageDraw

from .registry import DERIVED


def _root(name):
    return json.loads((DERIVED / "features" / name / "index.json").read_text())["root"]


def contact_sheet(rows, out_path, cell=256, title=None):
    """rows: list of dict(a_set, a_rel, b_set, b_rel, label). Writes one PNG, one pair per row."""
    pad, head = 6, 18
    W = 2 * cell + 3 * pad
    H = len(rows) * (cell + head + pad) + pad + (22 if title else 0)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    y = pad
    if title:
        d.text((pad, y), title, fill="black")
        y += 22
    for r in rows:
        d.text((pad, y), r["label"], fill="black")
        for k, side in enumerate(("a", "b")):
            with Image.open(f"{_root(r[side + '_set'])}/{r[side + '_rel']}") as im:
                im = im.convert("RGB")
                im.thumbnail((cell, cell))
                sheet.paste(im, (pad + k * (cell + pad), y + head))
        y += cell + head + pad
    sheet.save(out_path)
    return out_path
