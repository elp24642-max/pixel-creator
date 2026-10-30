"""Pictures for people and vision models: numbered regions, rigs with joints, frame sheets."""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def distinct(n: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def regions_overlay(img: np.ndarray, masks: dict, scale: int = 12) -> Image.Image:
    """Sprite with each region tinted and numbered, on a grid every 8 px."""
    h, w = img.shape[:2]
    base = Image.fromarray(img, "RGBA")
    bg = Image.new("RGBA", (w, h), (40, 40, 48, 255))
    bg.alpha_composite(base)
    big = bg.resize((w * scale, h * scale), Image.NEAREST)
    tint = np.zeros((h, w, 4), dtype=np.uint8)
    cols = distinct(len(masks))
    for k, (i, m) in enumerate(masks.items()):
        tint[m] = (*cols[k], 110)
    big.alpha_composite(Image.fromarray(tint, "RGBA").resize(big.size, Image.NEAREST))
    d = ImageDraw.Draw(big)
    for x in range(0, w + 1, 8):
        d.line([(x * scale, 0), (x * scale, h * scale)], fill=(255, 255, 255, 60))
    for y in range(0, h + 1, 8):
        d.line([(0, y * scale), (w * scale, y * scale)], fill=(255, 255, 255, 60))
    f = _font(max(10, scale))
    for k, (i, m) in enumerate(masks.items()):
        ys, xs = np.nonzero(m)
        # Label on the region pixel nearest to its centre so it lands on the part.
        cy, cx = ys.mean(), xs.mean()
        j = ((ys - cy) ** 2 + (xs - cx) ** 2).argmin()
        px, py = (xs[j] + 0.5) * scale, (ys[j] + 0.5) * scale
        d.text((px, py), str(i), font=f, fill=(255, 255, 255, 255), anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return big
