"""Rule-based cleanup of composed frames, the way a pixel artist tidies a rough pose.

Works only inside `area` (pixels the engine turned, stretched or swapped, plus a 1 px
border), so untouched parts of the original drawing are never changed.
"""
from __future__ import annotations

import numpy as np

from .palette import outline_colour

N4 = ((0, -1), (1, 0), (0, 1), (-1, 0))  # up, right, down, left as (dx, dy)
N8 = N4 + ((1, -1), (1, 1), (-1, 1), (-1, -1))


def dilate(m: np.ndarray, r: int = 1) -> np.ndarray:
    out = m.copy()
    for _ in range(r):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    return out


def fully_outlined(img: np.ndarray, outline) -> bool:
    """True when the source silhouette is (almost) entirely drawn in the outline colour."""
    a = img[..., 3] > 0
    p = np.pad(a, 1)
    edge = a & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    if not edge.any():
        return False
    is_line = np.all(img[..., :3] == np.array(outline), axis=-1)
    return (is_line & edge).sum() / edge.sum() >= 0.9


def cleanup(img: np.ndarray, area: np.ndarray, outline=None, closed_outline: bool | None = None,
            passes: int = 2) -> np.ndarray:
    img = img.copy()
    outline = tuple(outline) if outline is not None else outline_colour(img)
    if closed_outline is None:
        closed_outline = fully_outlined(img, outline)
    area = dilate(area, 1)
    h, w = img.shape[:2]
    line = np.array(outline, dtype=np.uint8)

    def op(x, y):
        return 0 <= x < w and 0 <= y < h and img[y, x, 3] > 0

    def col(x, y):
        return tuple(int(v) for v in img[y, x, :3])

    for _ in range(passes):
        changed = False
        for y, x in zip(*np.nonzero(area)):
            x, y = int(x), int(y)
            o = op(x, y)
            n4 = [op(x + dx, y + dy) for dx, dy in N4]
            n8 = sum(op(x + dx, y + dy) for dx, dy in N8)
            if o and n8 <= 1:                                   # orphan pixel
                img[y, x] = 0
                changed = True
                continue
            if not o:
                if all(n4):                                     # pin hole inside the body
                    cs = [col(x + dx, y + dy) for dx, dy in N4]
                    img[y, x, :3] = max(set(cs), key=cs.count)
                    img[y, x, 3] = 255
                    changed = True
                continue
            c = col(x, y)
            is_line = c == outline
            if is_line:
                # Outer-corner double: two perpendicular outline neighbours, outside on the
                # other two sides, and the diagonal between the two neighbours is solid.
                for i in range(4):
                    a, b = N4[i], N4[(i + 1) % 4]
                    oa, ob = N4[(i + 2) % 4], N4[(i + 3) % 4]
                    if (op(x + a[0], y + a[1]) and op(x + b[0], y + b[1])
                            and col(x + a[0], y + a[1]) == outline and col(x + b[0], y + b[1]) == outline
                            and not op(x + oa[0], y + oa[1]) and not op(x + ob[0], y + ob[1])
                            and op(x + a[0] + b[0], y + a[1] + b[1])):
                        img[y, x] = 0
                        changed = True
                        break
                continue
            if closed_outline and not all(n4):                  # fill touching the outside
                img[y, x, :3] = line
                changed = True
                continue
            # Lone speck of a fill colour: no 8-neighbour shares it.
            same = sum(op(x + dx, y + dy) and col(x + dx, y + dy) == c for dx, dy in N8)
            if same == 0:
                cs = [col(x + dx, y + dy) for dx, dy in N8 if op(x + dx, y + dy) and col(x + dx, y + dy) != outline]
                if cs:
                    img[y, x, :3] = max(set(cs), key=cs.count)
                    changed = True
        if not changed:
            break
    return img
