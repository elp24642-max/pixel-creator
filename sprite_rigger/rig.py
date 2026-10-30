"""Rig model and frame renderer.

A rig view is one drawn pose (front, back, side) split into named parts. Each part is a
set of segmentation regions with a pivot (joint), a parent and a draw order. A pose
moves parts by whole pixels or turns them around their pivot; children follow parents.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .palette import colours, luminance, outline_colour, snap
from .rotsprite import rotate
from .segment import segment


@dataclass
class Part:
    name: str
    mask: np.ndarray            # HxW bool, pixels of this part in the rest pose
    pivot: tuple[float, float]  # joint position, pixel coords (x, y)
    parent: str | None = None
    z: float = 0.0
    connect: bool = True        # fill the gap to the parent when the joint stretches
    image: np.ndarray = field(default=None, repr=False)  # HxWx4, only this part's pixels
    # Alternative drawings of the same part (bent knee, fist, raised foot), same canvas
    # size as the view image; a motion picks one per frame with the "variant" channel.
    variants: dict = field(default_factory=dict, repr=False)


@dataclass
class View:
    name: str
    image: np.ndarray
    parts: dict[str, Part]
    palette: list[tuple[int, int, int]]
    labels: np.ndarray = field(default=None, repr=False)

    @property
    def size(self):
        return self.image.shape[1], self.image.shape[0]


def _shade(img: np.ndarray, mask: np.ndarray, palette, steps: int, outline=None) -> np.ndarray:
    """Move each fill colour `steps` places along the palette (negative = darker).

    The outline colour stays as it is and fills never turn into it, otherwise a far
    limb in a 4-colour palette becomes a black blob.
    """
    pal = [c for c in sorted(palette, key=luminance) if c != outline]
    idx = {c: i for i, c in enumerate(pal)}
    out = img.copy()
    for y, x in zip(*np.nonzero(mask)):
        c = tuple(int(v) for v in img[y, x, :3])
        if c in idx:
            out[y, x, :3] = pal[max(0, min(len(pal) - 1, idx[c] + steps))]
    return out


def build_view(name: str, image: np.ndarray, spec: dict) -> View:
    seg = segment(image, **spec.get("segment", {}))
    masks = seg["masks"]
    palette = colours(image)
    parts: dict[str, Part] = {}
    for pname, p in spec["parts"].items():
        if "clone_of" in p:
            continue
        m = np.zeros(image.shape[:2], dtype=bool)
        for r in p.get("regions", []):
            m |= masks[int(r)]
        for x, y in p.get("add_pixels", []):
            m[y, x] = True
        for x, y in p.get("remove_pixels", []):
            m[y, x] = False
        parts[pname] = Part(pname, m, tuple(p["pivot"]), p.get("parent"), p.get("z", 0), p.get("connect", True))
    for pname, p in spec["parts"].items():
        if "clone_of" not in p:
            continue
        src = parts[p["clone_of"]]
        dx, dy = p.get("offset", [0, 0])
        m = np.roll(np.roll(src.mask, dy, 0), dx, 1)
        piv = p.get("pivot", [src.pivot[0] + dx, src.pivot[1] + dy])
        part = Part(pname, m, tuple(piv), p.get("parent", src.parent), p.get("z", src.z - 0.5), p.get("connect", True))
        shifted = np.roll(np.roll(image, dy, 0), dx, 1)
        shaded = _shade(shifted, m, palette, p.get("shade", -1), outline_colour(image))
        part.image = np.where(m[..., None], shaded, 0).astype(np.uint8)
        parts[pname] = part
    for part in parts.values():
        if part.image is None:
            part.image = np.where(part.mask[..., None], image, 0).astype(np.uint8)
    for pname, p in spec["parts"].items():
        for vname, v in p.get("variants", {}).items():
            vimg = np.array(Image.open(Path(spec.get("_dir", ".")) / v["image"]).convert("RGBA"))
            if vimg.shape != image.shape:
                raise ValueError(f"variant {pname}/{vname}: size {vimg.shape[1::-1]} differs from the view {image.shape[1::-1]}")
            parts[pname].variants[vname] = snap(vimg, palette)
    for part in parts.values():
        if part.parent is not None and part.parent not in parts:
            raise ValueError(f"part {part.name}: unknown parent {part.parent}")
    return View(name, image, parts, palette, seg["labels"])


def load_rig(path: str | Path) -> dict[str, View]:
    path = Path(path)
    spec = json.loads(path.read_text())
    views = {}
    for vname, v in spec["views"].items():
        img = np.array(Image.open(path.parent / v["image"]).convert("RGBA"))
        views[vname] = build_view(vname, img, {**v, "_dir": str(path.parent)})
    return views


# ---- transforms: rigid (angle, tx, ty) mapping p -> R(angle) p + t ----

def _rot(a, x, y):
    c, s = math.cos(a), math.sin(a)
    return c * x - s * y, s * x + c * y


def _compose(outer, inner):
    a1, tx1, ty1 = outer
    a2, tx2, ty2 = inner
    rx, ry = _rot(a1, tx2, ty2)
    return a1 + a2, rx + tx1, ry + ty1


def _apply(T, x, y):
    a, tx, ty = T
    rx, ry = _rot(a, x, y)
    return rx + tx, ry + ty


def world_transforms(view: View, pose: dict) -> dict:
    out = {}

    def get(name):
        if name in out:
            return out[name]
        part = view.parts[name]
        p = pose.get(name, {})
        a = math.radians(p.get("rot", 0.0))
        cx, cy = part.pivot
        rx, ry = _rot(a, cx, cy)
        local = (a, cx - rx + p.get("dx", 0.0), cy - ry + p.get("dy", 0.0))
        T = local if part.parent is None else _compose(get(part.parent), local)
        out[name] = T
        return T

    for n in view.parts:
        get(n)
    return out


def _paste(canvas, img, ox, oy):
    h, w = img.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(W, ox + w), min(H, oy + h)
    if x0 >= x1 or y0 >= y1:
        return
    src = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    m = src[..., 3] > 0
    canvas[y0:y1, x0:x1][m] = src[m]


def _joint_pixels(view: View, part: Part):
    """Pixels of `part` that touch its parent at rest: the seam that may need stretching."""
    if part.parent is None or not part.connect:
        return []
    pm = view.parts[part.parent].mask
    nb = np.zeros_like(pm)
    nb[1:] |= pm[:-1]; nb[:-1] |= pm[1:]; nb[:, 1:] |= pm[:, :-1]; nb[:, :-1] |= pm[:, 1:]
    seam = part.mask & (nb | pm)
    return [(int(x), int(y), part.image[y, x].copy()) for y, x in zip(*np.nonzero(seam))]


def render(view: View, pose: dict, margin: int = 4, with_touched: bool = False):
    """Draw one frame. The canvas is the source size plus `margin` on every side.

    with_touched=True also returns a mask of pixels that are not a plain shifted copy
    of the source drawing (turned parts, stretched seams, swapped variants): the places
    where cleanup and the refiner model are allowed to work.
    """
    w, h = view.size
    canvas = np.zeros((h + 2 * margin, w + 2 * margin, 4), dtype=np.uint8)
    touched = np.zeros(canvas.shape[:2], dtype=bool)
    T = world_transforms(view, pose)
    if not hasattr(view, "_seams"):
        view._seams = {n: _joint_pixels(view, p) for n, p in view.parts.items()}
    for name in sorted(view.parts, key=lambda n: view.parts[n].z):
        part = view.parts[name]
        Tc = T[name]
        # Stretch seams: smear seam pixels from where the child is to where the parent holds it.
        if view._seams[name]:
            Tp = T[part.parent]
            for x, y, col in view._seams[name]:
                ax, ay = _apply(Tc, x, y)
                bx, by = _apply(Tp, x, y)
                ax, ay, bx, by = round(ax), round(ay), round(bx), round(by)
                n = max(abs(bx - ax), abs(by - ay))
                for i in range(1, n + 1):
                    px = ax + round((bx - ax) * i / n) + margin
                    py = ay + round((by - ay) * i / n) + margin
                    if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0] and canvas[py, px, 3] == 0:
                        canvas[py, px] = col
                        touched[py, px] = True
        a = Tc[0]
        cx, cy = part.pivot
        wx, wy = _apply(Tc, cx, cy)
        variant = pose.get(name, {}).get("variant")
        src = part.image
        if variant:
            if variant not in part.variants:
                raise ValueError(f"part {name} has no variant {variant!r}")
            src = part.variants[variant]
        if abs(math.degrees(a)) < 0.5:
            ox, oy, img = round(wx - cx) + margin, round(wy - cy) + margin, src
        else:
            img, (px, py) = rotate(src, math.degrees(a), (cx, cy))
            ox, oy = round(wx - px) + margin, round(wy - py) + margin
        _paste(canvas, img, ox, oy)
        if variant or abs(math.degrees(a)) >= 0.5:
            _paste_mask(touched, img[..., 3] > 0, ox, oy)
    out = snap(canvas, view.palette)
    return (out, touched) if with_touched else out


def _paste_mask(dst, m, ox, oy):
    h, w = m.shape
    H, W = dst.shape
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
    if x0 < x1 and y0 < y1:
        dst[y0:y1, x0:x1] |= m[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
