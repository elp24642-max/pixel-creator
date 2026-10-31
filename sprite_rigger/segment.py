"""Split a sprite into small regions that a person or a model can group into body parts.

Pixel art draws parts as fills enclosed by an outline colour, so the fills give the
regions and every outline pixel goes with the fills it touches. Outline pixels shared
by two regions are kept by both: when the parts move apart, each keeps a closed edge.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .palette import outline_colour

N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
N8 = N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _fill_regions(fill: np.ndarray, colour: np.ndarray | None = None) -> np.ndarray:
    """4-connected labelling of the fill pixels. 0 = not a fill pixel.

    With `colour` (HxW ints), only pixels of the same colour join, so parts drawn as
    touching flat colours (no line between them) still come apart."""
    h, w = fill.shape
    lab = np.zeros((h, w), dtype=np.int32)
    n = 0
    for y in range(h):
        for x in range(w):
            if fill[y, x] and not lab[y, x]:
                n += 1
                lab[y, x] = n
                q = deque([(y, x)])
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in N4:
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w and fill[ny, nx] and not lab[ny, nx]
                                and (colour is None or colour[ny, nx] == colour[cy, cx])):
                            lab[ny, nx] = n
                            q.append((ny, nx))
    return lab


def _merge_tiny(lab: np.ndarray, min_px: int) -> np.ndarray:
    """Fold regions smaller than min_px into the neighbour they share the longest border with."""
    lab = lab.copy()
    h, w = lab.shape
    changed = True
    while changed:
        changed = False
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        for i, n in sorted(zip(ids, counts), key=lambda t: t[1]):
            if n >= min_px:
                break
            border = {}
            for y, x in zip(*np.nonzero(lab == i)):
                for dy, dx in N4:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and lab[ny, nx] not in (0, i):
                        border[lab[ny, nx]] = border.get(lab[ny, nx], 0) + 1
            if border:
                lab[lab == i] = max(border, key=border.get)
                changed = True
                break
    return lab


def segment(img: np.ndarray, min_size: int = 1, outline=None, min_regions: int = 8) -> dict:
    """Returns {'labels': HxW int array of fill regions, 'masks': {id: HxW bool}}.

    masks include the outline pixels each region owns. Regions smaller than
    min_size fill pixels are merged into the neighbour they share most outline with.
    """
    alpha = img[..., 3] > 0
    outline = tuple(outline) if outline is not None else outline_colour(img)
    is_line = alpha & np.all(img[..., :3] == np.array(outline), axis=-1)
    lab = _fill_regions(alpha & ~is_line)
    if lab.max() < min_regions:
        # Few closed shapes: the drawing separates parts by colour, not by lines.
        packed = (img[..., 0].astype(np.int64) << 16) | (img[..., 1].astype(np.int64) << 8) | img[..., 2]
        lab = _merge_tiny(_fill_regions(alpha & ~is_line, packed), 4)
        ids = np.unique(lab[lab > 0])
        remap = np.zeros(int(lab.max()) + 1, np.int32)
        remap[ids] = np.arange(1, len(ids) + 1)
        lab = remap[lab]
    h, w = lab.shape

    # Outline pixels: owned by every region they touch (8-neighbourhood).
    owners: dict[tuple[int, int], set[int]] = {}
    for y, x in zip(*np.nonzero(is_line)):
        s = set()
        for dy, dx in N8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and lab[ny, nx]:
                s.add(int(lab[ny, nx]))
        owners[(y, x)] = s

    # Outline pixels touching no fill (thick lines, solid dark details): nearest region(s) through the line.
    pending = [p for p, s in owners.items() if not s]
    while pending:
        nxt = []
        updates = {}
        for (y, x) in pending:
            s = set()
            for dy, dx in N8:
                q = (y + dy, x + dx)
                if q in owners and owners[q]:
                    s |= owners[q]
            if s:
                updates[(y, x)] = s
            else:
                nxt.append((y, x))
        if not updates:
            break  # an isolated blob of pure outline: becomes its own region below
        owners.update(updates)
        pending = nxt

    n = int(lab.max())
    for (y, x) in pending:  # isolated outline-only islands
        n += 1
        lab[y, x] = n
        owners[(y, x)] = {n}

    masks = {i: (lab == i) for i in range(1, n + 1)}
    for (y, x), s in owners.items():
        for i in s:
            masks[i][y, x] = True

    # Merge tiny regions into the neighbour they overlap most.
    sizes = {i: int((lab == i).sum()) for i in masks}
    for i in sorted(sizes, key=sizes.get):
        if sizes[i] >= min_size or i not in masks:
            continue
        best, best_ov = None, 0
        for j, m in masks.items():
            if j != i:
                ov = int((m & masks[i]).sum())
                if ov > best_ov:
                    best, best_ov = j, ov
        if best is not None:
            masks[best] |= masks.pop(i)
            lab[lab == i] = best
            sizes[best] += sizes.pop(i)

    ids = sorted(masks, key=lambda i: (np.nonzero(masks[i])[0].min(), np.nonzero(masks[i])[1].min()))
    remap = {old: k + 1 for k, old in enumerate(ids)}
    new_lab = np.zeros_like(lab)
    for old, new in remap.items():
        new_lab[lab == old] = new
    return {"labels": new_lab, "masks": {remap[o]: masks[o] for o in ids}}


def region_info(masks: dict) -> list[dict]:
    """Bounding box, centre and size of each region (what a model gets as text)."""
    out = []
    for i, m in masks.items():
        ys, xs = np.nonzero(m)
        out.append({
            "id": i,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centre": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
            "pixels": int(m.sum()),
        })
    return out
