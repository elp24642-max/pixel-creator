"""Sprite recipes: pixel art written as data, for people and AI agents.

A recipe is plain JSON (nothing in it is executed, so it is safe to take from any model):

{
  "palette": {"k": "#081820", "d": "#346856", "m": "#88c070", "l": "#e0f8d0"},
  "ramps": ["kdml"],                         # dark -> light, for shade/light
  "sprites": {
    "_eye":  {"grid": ["lk", "kk"]},          # "_" prefix = part, not exported
    "hero_front": {"w": 32, "h": 40, "mirror": "x", "outline": "k", "items": [
        {"grid": [".....kkk", "....kmmm"], "x": 0, "y": 2},
        {"rect": [6, 20, 10, 8, "d"]},
        {"stamp": ["_eye", 12, 9]}
    ]},
    "hero_red": {"base": "hero_front", "swap": {"m": "r"}}
  },
  "poses": {"front": "hero_front", "back": "hero_back", "side": "hero_side"}
}

Items run in order. Then: mirror -> swap -> flip -> outline -> crop. The canvas never grows:
leave a free 1 px border for the outline.

Colours: a palette key, "#rrggbb", "." (leave as is) or "_" (erase).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


class RecipeError(ValueError):
    """Raised with the sprite and item that failed, so a model can fix exactly that."""


# ------------------------------------------------------------------ colours

def _hex(s: str):
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) not in (6, 8):
        raise RecipeError(f"bad colour {s!r}")
    v = [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]
    return tuple(v[:3]) + ((v[3],) if len(v) == 4 else (255,))


class Palette:
    def __init__(self, keys: dict, ramps: list[str] | None = None):
        self.keys = {}
        for k, v in keys.items():
            if len(k) != 1 or k in "._ #":
                raise RecipeError(f"palette key {k!r} must be one character, not . _ # or space")
            self.keys[k] = _hex(v)
        self.ramps = ramps or []
        for r in self.ramps:
            for k in r:
                if k not in self.keys:
                    raise RecipeError(f"ramp {r!r} uses unknown key {k!r}")

    def colour(self, c):
        """RGBA tuple, or None for '.' (skip), or (0,0,0,0) for '_' (erase)."""
        if c == ".":
            return None
        if c == "_":
            return (0, 0, 0, 0)
        if isinstance(c, str) and c.startswith("#"):
            return _hex(c)
        if c in self.keys:
            return self.keys[c]
        near = [k for k in self.keys if k.lower() == str(c).lower()]
        hint = f" (did you mean {near[0]!r}?)" if near else ""
        raise RecipeError(f"unknown colour {c!r}{hint}; palette keys: {''.join(self.keys)}")

    def key_of(self, rgba):
        for k, v in self.keys.items():
            if v == tuple(int(x) for x in rgba):
                return k
        return None

    def step(self, rgba, steps):
        """Move a colour along its ramp (negative = darker)."""
        k = self.key_of(rgba)
        if k is None:
            return rgba
        for r in self.ramps:
            if k in r:
                i = max(0, min(len(r) - 1, r.index(k) + steps))
                return self.keys[r[i]]
        return rgba


# ------------------------------------------------------------------ canvas ops

class Canvas:
    def __init__(self, w: int, h: int, pal: Palette):
        self.w, self.h, self.pal = w, h, pal
        self.a = np.zeros((h, w, 4), np.uint8)

    def put(self, x, y, c):
        col = self.pal.colour(c) if not isinstance(c, tuple) else c
        if col is None:
            return
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.a[y, x] = col

    def grid(self, rows, x=0, y=0):
        if isinstance(rows, str):
            rows = rows.split("\n")
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch not in ". ":
                    self.put(x + i, y + j, ch)

    def rect(self, x, y, w, h, c):
        for j in range(int(h)):
            for i in range(int(w)):
                self.put(x + i, y + j, c)

    def box(self, x, y, w, h, c):
        for i in range(int(w)):
            self.put(x + i, y, c)
            self.put(x + i, y + h - 1, c)
        for j in range(int(h)):
            self.put(x, y + j, c)
            self.put(x + w - 1, y + j, c)

    def line(self, x0, y0, x1, y1, c, width=1):
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        r = (width - 1) / 2
        for t in range(int(n) + 1):
            x = x0 + (x1 - x0) * t / n
            y = y0 + (y1 - y0) * t / n
            if width <= 1:
                self.put(round(x), round(y), c)
            else:
                for j in range(-math.ceil(r), math.ceil(r) + 1):
                    for i in range(-math.ceil(r), math.ceil(r) + 1):
                        if i * i + j * j <= r * r + 0.5:
                            self.put(round(x) + i, round(y) + j, c)

    def ellipse(self, cx, cy, rx, ry, c, ring=False):
        rx, ry = max(rx, 0.5), max(ry, 0.5)
        inside = set()
        for y in range(int(cy - ry) - 1, int(cy + ry) + 2):
            for x in range(int(cx - rx) - 1, int(cx + rx) + 2):
                if ((x - cx) / (rx + 0.5)) ** 2 + ((y - cy) / (ry + 0.5)) ** 2 <= 1:
                    inside.add((x, y))
        for (x, y) in inside:
            if not ring or any((x + dx, y + dy) not in inside for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                self.put(x, y, c)

    def poly(self, pts, c):
        xy = list(zip(pts[0::2], pts[1::2]))
        if len(xy) < 3:
            raise RecipeError("poly needs at least 3 points")
        ys = [p[1] for p in xy]
        for y in range(int(min(ys)), int(math.ceil(max(ys)))):
            yc = y + 0.5
            xs = []
            for (x0, y0), (x1, y1) in zip(xy, xy[1:] + xy[:1]):
                if (y0 <= yc < y1) or (y1 <= yc < y0):
                    xs.append(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for a, b in zip(xs[0::2], xs[1::2]):
                for x in range(int(math.ceil(a - 0.5)), int(math.floor(b - 0.5)) + 1):
                    self.put(x, y, c)

    def fill(self, x, y, c):
        x, y = int(x), int(y)
        if not (0 <= x < self.w and 0 <= y < self.h):
            return
        target = self.a[y, x].copy()
        col = self.pal.colour(c)
        if col is None or tuple(target) == tuple(col):
            return
        stack = [(x, y)]
        while stack:
            px, py = stack.pop()
            if 0 <= px < self.w and 0 <= py < self.h and (self.a[py, px] == target).all():
                self.a[py, px] = col
                stack += [(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]

    def dither(self, x, y, w, h, c1, c2, density=0.5):
        bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16
        for j in range(int(h)):
            for i in range(int(w)):
                self.put(x + i, y + j, c2 if bayer[(y + j) % 4, (x + i) % 4] < density else c1)

    def region(self, where):
        m = np.zeros((self.h, self.w), bool)
        if where is None:
            m[:] = True
        elif len(where) == 3:
            cx, cy, r = where
            yy, xx = np.mgrid[0:self.h, 0:self.w]
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        elif len(where) == 4:
            x, y, w, h = where
            m[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = True
        else:
            tmp = Canvas(self.w, self.h, Palette({"x": "#ffffff"}))
            tmp.poly(where, "x")
            m = tmp.a[..., 3] > 0
        return m

    def tone(self, where, steps):
        m = self.region(where) & (self.a[..., 3] > 0)
        for y, x in zip(*np.nonzero(m)):
            self.a[y, x] = self.pal.step(self.a[y, x], steps)


# ------------------------------------------------------------------ recipe

OPS = ("grid", "px", "rect", "box", "line", "circ", "ring", "ellipse", "poly", "fill", "dither",
       "shade", "light", "stamp", "recolor")


class Recipe:
    def __init__(self, data: dict):
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text())
        self.data = data
        self.pal = Palette(data.get("palette", {}), data.get("ramps"))
        self.defs = data.get("sprites", {})
        if not self.defs:
            raise RecipeError("recipe has no sprites")
        self._cache: dict[str, np.ndarray] = {}
        self._stack: list[str] = []

    def names(self):
        return [n for n in self.defs if not n.startswith("_")]

    def image(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        if name not in self.defs:
            near = [n for n in self.defs if n.lower().startswith(name.lower()[:3])]
            raise RecipeError(f"no sprite {name!r}" + (f" (did you mean {near[0]!r}?)" if near else ""))
        if name in self._stack:
            raise RecipeError(f"sprite {name!r} uses itself: {' -> '.join(self._stack + [name])}")
        self._stack.append(name)
        try:
            img = self._build(name, self.defs[name])
        finally:
            self._stack.pop()
        self._cache[name] = img
        return img

    def _build(self, name, d):
        if "base" in d:
            base = self.image(d["base"])
            h, w = base.shape[:2]
            c = Canvas(d.get("w", w), d.get("h", h), self.pal)
            c.a[:h, :w] = base[:c.h, :c.w]
        else:
            if "w" in d and "h" in d:
                c = Canvas(int(d["w"]), int(d["h"]), self.pal)
            elif "grid" in d:
                rows = d["grid"] if isinstance(d["grid"], list) else d["grid"].split("\n")
                half = max(len(r) for r in rows)
                w = {"x": half * 2, "xc": half * 2 - 1}.get(d.get("mirror"), half)
                c = Canvas(w, len(rows), self.pal)
            else:
                raise RecipeError(f"sprite {name!r}: give w and h, a grid, or a base")
        if "grid" in d:
            c.grid(d["grid"])
        for i, it in enumerate(d.get("items", [])):
            try:
                self._op(c, it)
            except RecipeError as e:
                raise RecipeError(f"sprite {name!r}, item {i} {json.dumps(it)[:80]}: {e}") from None
            except (TypeError, ValueError, IndexError, KeyError) as e:
                raise RecipeError(f"sprite {name!r}, item {i} {json.dumps(it)[:80]}: bad arguments ({e})") from None
        a = c.a
        m = d.get("mirror")
        if m in ("x", "xc"):
            w = a.shape[1]
            half = w // 2
            for x in range(half):
                src = a[:, x]
                dst = w - 1 - x
                keep = src[:, 3] > 0
                a[keep, dst] = src[keep]
        if m in ("y",):
            h = a.shape[0]
            for y in range(h // 2):
                keep = a[y, :, 3] > 0
                a[h - 1 - y, keep] = a[y, keep]
        if "swap" in d:
            for k_from, k_to in d["swap"].items():
                src = np.array(self.pal.colour(k_from))
                dst = np.array(self.pal.colour(k_to))
                sel = np.all(a == src, axis=-1)
                a[sel] = dst
        fl = d.get("flip", "")
        if "x" in fl:
            a = a[:, ::-1].copy()
        if "y" in fl:
            a = a[::-1].copy()
        if d.get("outline"):
            col = np.array(self.pal.colour(d["outline"]), np.uint8)
            op = a[..., 3] > 0
            p = np.pad(op, 1)
            nb = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
            if d.get("outline8"):
                nb |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
            a[nb & ~op] = col
        if d.get("crop"):
            ys, xs = np.nonzero(a[..., 3])
            if len(ys):
                a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        return a

    def _op(self, c: Canvas, it: dict):
        keys = [k for k in it if k in OPS]
        if len(keys) != 1:
            raise RecipeError(f"each item needs exactly one op from {', '.join(OPS)}")
        op = keys[0]
        v = it[op]
        if op == "grid":
            c.grid(v, it.get("x", 0), it.get("y", 0))
        elif op == "px":
            c.put(*v)
        elif op == "rect":
            c.rect(*v)
        elif op == "box":
            c.box(*v)
        elif op == "line":
            c.line(*v)
        elif op == "circ":
            cx, cy, r, col = v
            c.ellipse(cx, cy, r, r, col)
        elif op == "ring":
            cx, cy, r, col = v
            c.ellipse(cx, cy, r, r, col, ring=True)
        elif op == "ellipse":
            c.ellipse(*v)
        elif op == "poly":
            pts, col = v
            c.poly(pts, col)
        elif op == "fill":
            c.fill(*v)
        elif op == "dither":
            c.dither(*v)
        elif op in ("shade", "light"):
            where, steps = (v + [1])[:2] if isinstance(v, list) and v and isinstance(v[0], list) else (v, 1)
            c.tone(where, -steps if op == "shade" else steps)
        elif op == "recolor":
            mapping, where = (v + [None])[:2] if isinstance(v, list) else (v, None)
            m = c.region(where)
            for k_from, k_to in mapping.items():
                sel = m & np.all(c.a == np.array(self.pal.colour(k_from)), axis=-1)
                c.a[sel] = self.pal.colour(k_to)
        elif op == "stamp":
            name, x, y = v[0], int(v[1]), int(v[2])
            opts = v[3] if len(v) > 3 else {}
            img = self.image(name)
            if "x" in opts.get("flip", ""):
                img = img[:, ::-1]
            if "y" in opts.get("flip", ""):
                img = img[::-1]
            h, w = img.shape[:2]
            for j in range(h):
                for i in range(w):
                    if img[j, i, 3]:
                        c.put(x + i, y + j, tuple(int(t) for t in img[j, i]))

    # ---------------------------------------------------------------- outputs

    def ascii(self, name: str) -> str:
        a = self.image(name)
        h, w = a.shape[:2]
        tens = "    " + "".join(str((x // 10) % 10) if x % 10 == 0 else " " for x in range(w))
        ones = "    " + "".join(str(x % 10) for x in range(w))
        rows = [tens, ones]
        for y in range(h):
            row = ""
            for x in range(w):
                p = a[y, x]
                row += "." if p[3] == 0 else (self.pal.key_of(p) or "?")
            rows.append(f"{y:3d} {row}")
        return "\n".join(rows)


def preview(img: np.ndarray, target: int = 512) -> Image.Image:
    """Upscaled view for eyes and vision models: checkerboard behind, grid when big enough."""
    h, w = img.shape[:2]
    s = max(1, target // max(w, h))
    big = Image.fromarray(img, "RGBA").resize((w * s, h * s), Image.NEAREST)
    t = max(s, 8)
    yy, xx = np.mgrid[0:big.height, 0:big.width]
    chk = ((xx // t + yy // t) % 2).astype(bool)
    arr = np.where(chk[..., None], np.array([58, 62, 72, 255], np.uint8), np.array([44, 47, 55, 255], np.uint8))
    bg = Image.fromarray(arr.astype(np.uint8), "RGBA")
    bg.alpha_composite(big)
    if s >= 6:
        arr = np.array(bg)
        for x in range(0, w * s, s):
            arr[:, x] = (arr[:, x] * 0.75).astype(np.uint8) if (x // s) % 8 else (arr[:, x] * 0.45).astype(np.uint8)
        for y in range(0, h * s, s):
            arr[y] = (arr[y] * 0.75).astype(np.uint8) if (y // s) % 8 else (arr[y] * 0.45).astype(np.uint8)
        arr[..., 3] = 255
        bg = Image.fromarray(arr, "RGBA")
    return bg


def sheet_preview(images: dict, target: int = 384) -> Image.Image:
    """Several sprites side by side, each previewed at the same scale."""
    hs = max(a.shape[0] for a in images.values())
    ws = max(a.shape[1] for a in images.values())
    s = max(1, target // max(ws, hs))
    tiles = [preview(np.pad(a, ((hs - a.shape[0], 0), (0, ws - a.shape[1]), (0, 0))), s * max(ws, hs))
             for a in images.values()]
    out = Image.new("RGBA", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), tiles[0].height), (30, 32, 38, 255))
    x = 0
    for t in tiles:
        out.alpha_composite(t, (x, 0))
        x += t.width + 8
    return out


def build(recipe_path: str | Path, out_dir: str | Path | None = None, only: list[str] | None = None) -> dict:
    """Render a recipe file: <out>/<name>.png, <out>/_preview/<name>.png, poses copied as <view>.png."""
    rp = Path(recipe_path)
    r = Recipe(json.loads(rp.read_text()))
    out = Path(out_dir) if out_dir else rp.parent / "out"
    (out / "_preview").mkdir(parents=True, exist_ok=True)
    done, errors = [], {}
    for n in r.names():
        if only and n not in only:
            continue
        try:
            img = r.image(n)
        except RecipeError as e:
            errors[n] = str(e)
            continue
        Image.fromarray(img, "RGBA").save(out / f"{n}.png")
        preview(img).save(out / "_preview" / f"{n}.png")
        done.append(n)
    poses = r.data.get("poses", {})
    if poses and not only:
        imgs = {}
        for view, n in poses.items():
            if n in errors:
                continue
            imgs[view] = r.image(n)
        sizes = {a.shape for a in imgs.values()}
        if len(sizes) > 1:
            errors["poses"] = f"all poses must be the same size, got {sorted(s[:2] for s in sizes)}"
        else:
            for view, a in imgs.items():
                Image.fromarray(a, "RGBA").save(out / f"{view}.png")
            if imgs:
                sheet_preview(imgs).save(out / "_preview" / "_poses.png")
    return {"built": done, "errors": errors, "out": str(out)}
