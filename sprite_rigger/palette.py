"""Palette helpers: read the colours a sprite uses and keep output inside them."""
from __future__ import annotations

import numpy as np


def luminance(rgb) -> float:
    r, g, b = rgb[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def colours(img: np.ndarray) -> list[tuple[int, int, int]]:
    """Opaque colours of an RGBA array, darkest first."""
    opaque = img[img[..., 3] > 0][:, :3]
    uniq = {tuple(int(v) for v in c) for c in opaque}
    return sorted(uniq, key=luminance)


def outline_colour(img: np.ndarray) -> tuple[int, int, int]:
    """The colour drawn most often on the silhouette edge (usually the darkest one)."""
    alpha = img[..., 3] > 0
    pad = np.pad(alpha, 1)
    edge = alpha & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    cols, counts = np.unique(img[edge][:, :3], axis=0, return_counts=True)
    return tuple(int(v) for v in cols[counts.argmax()])


def snap(img: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    """Map every opaque pixel to the nearest palette colour; alpha becomes 0 or 255."""
    out = img.copy()
    pal = np.array(palette, dtype=np.int32)
    mask = out[..., 3] >= 128
    px = out[mask][:, :3].astype(np.int32)
    if len(px):
        d = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
        out[mask, :3] = pal[d.argmin(1)]
    out[..., 3] = np.where(mask, 255, 0)
    out[~mask, :3] = 0
    return out
