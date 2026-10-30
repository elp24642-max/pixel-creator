"""Pixel-art rotation (RotSprite idea): upscale with Scale2x three times, rotate with
nearest neighbour, sample back down. Edges stay crisp and no new colours appear."""
from __future__ import annotations

import math

import numpy as np


def _scale2x(a: np.ndarray) -> np.ndarray:
    """Scale2x/EPX on an array of packed uint32 pixels."""
    p = np.pad(a, 1, mode="edge")
    B, D, F, H = p[:-2, 1:-1], p[1:-1, :-2], p[1:-1, 2:], p[2:, 1:-1]
    E = a
    e0 = np.where((D == B) & (B != F) & (D != H), D, E)
    e1 = np.where((B == F) & (B != D) & (F != H), F, E)
    e2 = np.where((D == H) & (D != B) & (H != F), D, E)
    e3 = np.where((H == F) & (D != H) & (B != F), F, E)
    h, w = a.shape
    out = np.empty((h * 2, w * 2), dtype=a.dtype)
    out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2] = e0, e1, e2, e3
    return out


def _pack(img: np.ndarray) -> np.ndarray:
    img = img.copy()
    img[img[..., 3] == 0] = 0  # all transparent pixels equal, so Scale2x treats them as one colour
    return img.view(np.uint32)[..., 0]


def _unpack(a: np.ndarray) -> np.ndarray:
    return a[..., None].view(np.uint8).reshape(a.shape + (4,))


def rotate(img: np.ndarray, angle_deg: float, pivot: tuple[float, float], pad: int | None = None):
    """Rotate an RGBA sprite around pivot (x, y in pixel-centre coords).

    Returns (rotated RGBA array, new pivot position inside it). Positive angle turns
    clockwise on screen (y grows downward).
    """
    if abs(angle_deg) < 1e-6:
        return img, pivot
    h, w = img.shape[:2]
    if pad is None:
        pad = int(math.ceil(math.hypot(h, w) / 2)) + 1
    src = np.pad(img, ((pad, pad), (pad, pad), (0, 0)))
    px, py = pivot[0] + pad, pivot[1] + pad
    big = _pack(np.ascontiguousarray(src))
    for _ in range(3):
        big = _scale2x(big)
    S = 8
    H2, W2 = src.shape[:2]
    # For each output pixel centre, find the source point in the 8x image (inverse rotation).
    ys, xs = np.mgrid[0:H2, 0:W2]
    cx, cy = xs + 0.5 - (px + 0.5), ys + 0.5 - (py + 0.5)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    sx = ca * cx + sa * cy + (px + 0.5)
    sy = -sa * cx + ca * cy + (py + 0.5)
    bx = np.floor(sx * S).astype(int)
    by = np.floor(sy * S).astype(int)
    ok = (bx >= 0) & (by >= 0) & (bx < W2 * S) & (by < H2 * S)
    out = np.zeros((H2, W2), dtype=np.uint32)
    out[ok] = big[by[ok], bx[ok]]
    return _unpack(out).copy(), (px, py)
