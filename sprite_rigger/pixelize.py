"""Turn an AI-generated "pixel art" picture into real pixel art.

Image models paint pixel art at some enlarged, slightly wobbly grid, with JPEG noise,
soft edges and dozens of near-identical colours. This module:

1. finds the grid size (the size of one "pixel" in the picture),
2. samples every cell by its most common colour,
3. removes the flat background (magenta by default),
4. reduces to a small palette (or maps to the game's palette),
5. splits the picture into separate figures (turnaround sheets),
6. optionally rescales a figure to a target height and repairs the outline.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .cleanup import cleanup
from .palette import colours, luminance


def _period(signal: np.ndarray, lo: int = 3, hi: int = 40) -> float:
    s = signal - signal.mean()
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    ac /= ac[0] + 1e-9
    best, best_v = lo, -1
    for p in range(lo, min(hi, len(ac) - 1)):
        # Peaks of the autocorrelation at the grid period and its multiples.
        v = ac[p]
        if v > best_v and ac[p] >= ac[p - 1] and ac[p] >= ac[p + 1]:
            best, best_v = p, v
    # Refine to sub-pixel with the 2nd and 3rd multiples (grids are rarely exact).
    ests = [best]
    for k in (2, 3, 4):
        c = int(round(best * k))
        win = range(max(1, c - 2), min(len(ac) - 1, c + 3))
        if win:
            m = max(win, key=lambda i: ac[i])
            ests.append(m / k)
    return float(np.median(ests))


def grid_size(img: np.ndarray) -> tuple[float, float]:
    g = img[..., :3].astype(np.float32).mean(-1)
    gx = np.abs(np.diff(g, axis=1)).sum(0)
    gy = np.abs(np.diff(g, axis=0)).sum(1)
    return _period(gx), _period(gy)


def _phase(signal: np.ndarray, period: float) -> float:
    """Offset of the grid lines: where edges line up best."""
    best, best_v = 0.0, -1
    for off in np.arange(0, period, 0.5):
        idx = np.round(np.arange(off, len(signal) - 1, period)).astype(int)
        v = signal[idx].sum()
        if v > best_v:
            best, best_v = off, v
    return best


def sample_grid(img: np.ndarray, px: float, py: float) -> np.ndarray:
    """One output pixel per grid cell: the most common colour in the cell's middle."""
    g = img[..., :3].astype(np.float32).mean(-1)
    ox = _phase(np.abs(np.diff(g, axis=1)).sum(0), px)
    oy = _phase(np.abs(np.diff(g, axis=0)).sum(1), py)
    H, W = img.shape[:2]
    xs = np.arange(ox, W - 1, px)
    ys = np.arange(oy, H - 1, py)
    out = np.zeros((len(ys) - 1, len(xs) - 1, 4), np.uint8)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            x0, x1 = int(xs[i] + px * 0.25), int(xs[i + 1] - px * 0.25)
            y0, y1 = int(ys[j] + py * 0.25), int(ys[j + 1] - py * 0.25)
            cell = img[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].reshape(-1, img.shape[2])
            q = (cell[:, :3] // 8).astype(np.int32)  # tolerate JPEG noise when voting
            keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
            vals, counts = np.unique(keys, return_counts=True)
            win = vals[counts.argmax()]
            pick = cell[keys == win][:, :3].mean(0)
            out[j, i, :3] = pick.round().astype(np.uint8)
            out[j, i, 3] = 255
    return out


def remove_background(img: np.ndarray, colour=None, tol: float = 60) -> np.ndarray:
    out = img.copy()
    if colour is None:  # the most common colour along the border
        border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])[:, :3]
        vals, counts = np.unique(border // 16, axis=0, return_counts=True)
        colour = (vals[counts.argmax()] * 16 + 8)
    d = np.sqrt(((img[..., :3].astype(np.float32) - np.array(colour, np.float32)) ** 2).sum(-1))
    out[d < tol] = 0
    return out


def reduce_palette(img: np.ndarray, n: int | None = None, palette=None) -> np.ndarray:
    """Map opaque pixels to `palette` (list of RGB) or to the best n colours."""
    out = img.copy()
    m = out[..., 3] > 0
    px = out[m][:, :3]
    if palette is None:
        q = Image.fromarray(px.reshape(1, -1, 3), "RGB").quantize(colors=n or 8, method=Image.Quantize.MEDIANCUT,
                                                                    dither=Image.Dither.NONE)
        px2 = np.array(q.convert("RGB")).reshape(-1, 3)
    else:
        pal = np.array(palette, np.int32)
        # Match by brightness rank so a dark/light design keeps its values in a 4-colour palette.
        d = ((px[:, None, :].astype(np.int32) - pal[None]) ** 2).sum(-1)
        px2 = pal[d.argmin(1)].astype(np.uint8)
    out[m, :3] = px2
    return out


def split_figures(img: np.ndarray, min_px: int = 30) -> list[np.ndarray]:
    """Separate figures standing side by side (columns of empty space between them)."""
    op = img[..., 3] > 0
    cols = op.any(0)
    figs, start = [], None
    for x, c in enumerate(list(cols) + [False]):
        if c and start is None:
            start = x
        elif not c and start is not None:
            part = img[:, start:x]
            if (part[..., 3] > 0).sum() >= min_px:
                figs.append(part)
            start = None
    out = []
    for f in figs:
        ys = np.nonzero(f[..., 3].any(1))[0]
        out.append(f[ys.min():ys.max() + 1])
    return out


def close_outline(img: np.ndarray, outline=None) -> np.ndarray:
    """Make sure every silhouette edge pixel is the darkest colour (1 px closed outline)."""
    out = img.copy()
    op = out[..., 3] > 0
    if not op.any():
        return out
    dark = outline if outline is not None else colours(out)[0]
    p = np.pad(op, 1)
    edge = op & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    out[edge, :3] = dark
    return out


def fit_height(img: np.ndarray, height: int) -> np.ndarray:
    """Shrink a figure to `height` px. Each new pixel takes the most common colour of the
    block it covers (a vote, not an average), so no new colours appear and thin dark
    lines survive better than with nearest neighbour."""
    h, w = img.shape[:2]
    nw = max(1, round(w * height / h))
    out = np.zeros((height, nw, 4), np.uint8)
    for j in range(height):
        y0, y1 = int(j * h / height), max(int((j + 1) * h / height), int(j * h / height) + 1)
        for i in range(nw):
            x0, x1 = int(i * w / nw), max(int((i + 1) * w / nw), int(i * w / nw) + 1)
            block = img[y0:y1, x0:x1].reshape(-1, 4)
            op = block[block[:, 3] > 0]
            if len(op) * 2 < len(block):
                continue
            # Darkest colours get a bonus: outlines and eyes are thin but matter most.
            keys, inv, counts = np.unique(op[:, :3], axis=0, return_inverse=True, return_counts=True)
            lum = keys @ np.array([0.299, 0.587, 0.114])
            score = counts * (1.0 + 0.6 * (lum < np.percentile(lum, 30) if len(lum) > 1 else 0))
            out[j, i, :3] = keys[score.argmax()]
            out[j, i, 3] = 255
    return out


def pixelize(path, n_colours: int = 8, palette=None, bg=None, height: int | None = None,
             canvas: tuple[int, int] | None = None):
    """Full pipeline. Returns (list of figure arrays, info dict)."""
    img = np.array(Image.open(path).convert("RGBA"))
    px, py = grid_size(img)
    small = sample_grid(img, px, py)
    small = remove_background(small, bg)
    small = reduce_palette(small, n_colours, palette)
    # Specks left from the background edge and from JPEG noise (after the palette is small).
    small = cleanup(small, small[..., 3] >= 0, closed_outline=False, passes=1)
    figs = split_figures(small)
    out = []
    for f in figs:
        if height:
            f = fit_height(f, height)
        f = close_outline(f)
        if canvas:
            cw, ch = canvas
            c = np.zeros((ch, cw, 4), np.uint8)
            fh, fw = f.shape[:2]
            ox, oy = (cw - fw) // 2, ch - fh - 1
            c[max(0, oy):oy + fh, max(0, ox):ox + fw] = f[max(0, -oy):, max(0, -ox):][:ch - max(0, oy), :cw - max(0, ox)]
            f = c
        out.append(f)
    return out, {"grid": [round(px, 2), round(py, 2)], "native_size": list(small.shape[1::-1]),
                 "figures": [list(f.shape[1::-1]) for f in out]}
