"""Cut a sprite sheet into separate sprites.

Background = the most common colours (each covering at least `bg_share` of the sheet:
the sheet colour and the cell colour on ripped sheets), or real transparency. Every
connected blob of other pixels (8-neighbours, small gaps bridged) becomes one sprite.
Exact duplicates are dropped. Tiny blobs (labels, sparks) are skipped by size.

Usage: python -m sprite_rigger cut <sheet.png> <out_dir> [--min-h 12] [--max-h 200]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def background_mask(a: np.ndarray, bg_share: float = 0.08) -> np.ndarray:
    alpha = a[..., 3] > 0
    if (~alpha).mean() > 0.3:
        return ~alpha
    flat = a.reshape(-1, 4)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    bg = cols[counts / len(flat) >= bg_share]
    m = ~alpha
    for c in bg:
        m |= np.all(a == c, axis=-1)
    return m


def _components(fg: np.ndarray, bridge: int = 1):
    """Label 8-connected blobs; pixels up to `bridge` px apart join the same blob."""
    from collections import deque
    h, w = fg.shape
    grow = fg.copy()
    for _ in range(bridge):
        p = np.pad(grow, 1)
        grow = grow | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    lab = np.zeros((h, w), np.int32)
    n = 0
    boxes = []
    for y0, x0 in zip(*np.nonzero(grow)):
        if lab[y0, x0]:
            continue
        n += 1
        lab[y0, x0] = n
        q = deque([(y0, x0)])
        ys, xs = [y0], [x0]
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and grow[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = n
                        q.append((ny, nx))
                        ys.append(ny)
                        xs.append(nx)
        boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return lab, boxes


def pixel_scale(a: np.ndarray) -> int:
    """Detect sprites drawn at 2x/3x (every pixel a k x k block); returns k."""
    for k in (4, 3, 2):
        h, w = a.shape[:2]
        hh, ww = h - h % k, w - w % k
        b = a[:hh, :ww]
        blocks = b.reshape(hh // k, k, ww // k, k, 4)
        if (blocks == blocks[:, :1, :, :1]).all():
            return k
    return 1


def cut(sheet: str | Path, out_dir: str | Path, min_h: int = 12, max_h: int = 200, min_w: int = 6) -> list[dict]:
    a = np.array(Image.open(sheet).convert("RGBA"))
    bgm = background_mask(a)
    fg = ~bgm
    lab, boxes = _components(fg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    items = []
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        w, h = x1 - x0, y1 - y0
        if not (min_h <= h <= max_h and w >= min_w):
            continue
        crop = a[y0:y1, x0:x1].copy()
        crop[(lab[y0:y1, x0:x1] != i) | bgm[y0:y1, x0:x1]] = 0
        k = pixel_scale(crop)
        if k > 1:
            crop = crop[::k, ::k]
        digest = hashlib.sha1(crop.tobytes()).hexdigest()[:12]
        if digest in seen:
            continue
        seen.add(digest)
        name = f"s{len(items):04d}.png"
        Image.fromarray(crop, "RGBA").save(out / name)
        cols = {tuple(p[:3]) for p in crop.reshape(-1, 4) if p[3]}
        items.append({"file": name, "x": int(x0), "y": int(y0), "w": int(crop.shape[1]), "h": int(crop.shape[0]),
                      "scale": k, "colours": len(cols)})
    (out / "index.json").write_text(json.dumps(items, indent=1))
    return items


def contact_sheet(items: list[dict], src_dir: str | Path, cols: int = 16, scale: int = 2) -> Image.Image:
    """All cut sprites in a numbered grid, for picking references by eye."""
    from .viz import _font
    src = Path(src_dir)
    cw = max(i["w"] for i in items) * scale + 6
    ch = max(i["h"] for i in items) * scale + 16
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * cw, rows * ch), (40, 40, 48, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    f = _font(11)
    for n, it in enumerate(items):
        im = Image.open(src / it["file"]).convert("RGBA")
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        x, y = (n % cols) * cw, (n // cols) * ch
        sheet.alpha_composite(im, (x + 3, y + 14))
        d.text((x + 3, y + 1), str(n), font=f, fill=(255, 255, 255, 255))
    return sheet
