"""Random full characters from Liberated Pixel Cup layers (walk sheets, 9 x 4 frames of 64 px).

Usage as a module: compose(root, rng) -> RGBA sheet (256 x 576) or None.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

ORDER = ["feet", "legs", "torso", "head", "hair"]
TYPES = {"male": ("male", "adult"), "female": ("female", "thin", "adult")}


def index(root: Path) -> dict:
    root = Path(root)
    files = [p for p in root.rglob("*.png") if p.name == "walk.png" or p.parent.name == "walk"]
    idx = {t: {k: [] for k in ORDER} for t in TYPES}
    for p in files:
        rel = p.relative_to(root).as_posix()
        top = rel.split("/")[0]
        cat = {"feet": "feet", "legs": "legs", "torso": "torso", "head": "head", "hair": "hair"}.get(top)
        if cat is None or "/accessory/" in rel or (cat == "torso" and "/clothes/" not in rel and "/jacket/" not in rel and "/armour/" not in rel):
            continue
        if cat == "head" and "/heads/" not in rel:
            continue
        parts = rel.split("/")
        for t, keys in TYPES.items():
            if any(k in parts for k in keys):
                idx[t][cat].append(p)
    return idx


def _load(p):
    im = Image.open(p).convert("RGBA")
    return im if im.size == (576, 256) else None


def compose(root: Path, idx: dict, rng: random.Random):
    t = rng.choice(list(TYPES))
    body = _load(Path(root) / "body" / "bodies" / t / "walk.png")
    if body is None:
        return None
    sheet = body.copy()
    for cat in ORDER:
        pool = idx[t][cat]
        if not pool or (cat != "head" and rng.random() < 0.15):
            continue
        layer = _load(rng.choice(pool))
        if layer is not None:
            sheet.alpha_composite(layer)
    return np.array(sheet)


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1])
    idx = index(root)
    print({t: {k: len(v) for k, v in d.items()} for t, d in idx.items()})
    rng = random.Random(1)
    out = Image.new("RGBA", (576, 256 * 3), (60, 60, 70, 255))
    for i in range(3):
        s = compose(root, idx, rng)
        out.alpha_composite(Image.fromarray(s), (0, 256 * i))
    out.save(sys.argv[2])
