"""The refiner: a small U-Net that redraws the rough parts of a composed frame.

Runs on plain numpy (no deep-learning framework needed), ~0.4 M parameters, a few tens
of milliseconds per frame on a laptop CPU. Weights are a single .npz file.

Inputs per pixel: the rough frame (RGBA), the rest pose drawn where the body is (RGBA),
and the "touched" mask (where the engine turned, stretched or swapped something). The
network outputs an embedding per pixel; the palette colours (and "transparent") get an
embedding from a tiny MLP, and each pixel picks the palette entry with the highest dot
product. So it works with any palette and can never invent an off-palette colour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cleanup import dilate

ENC = ("e1", "e2", "e3", "m")
DEC = ("d3", "d2", "d1")


def palette_slots(palette, k: int | None = None):
    """Slot 0 is transparent, then the palette colours. Returns (k, 4) floats in 0..1."""
    cols = [(0, 0, 0, 0)] + [(r, g, b, 255) for r, g, b in palette]
    if k is not None:
        cols = cols[:k] + [(0, 0, 0, 0)] * max(0, k - len(cols))
    return np.array(cols, dtype=np.float32) / 255.0


def to_index(img: np.ndarray, palette) -> np.ndarray:
    """Palette slot of every pixel (0 = transparent), nearest colour for opaque ones."""
    pal = np.array(palette, dtype=np.int32)
    idx = np.zeros(img.shape[:2], dtype=np.int32)
    m = img[..., 3] > 0
    px = img[m][:, :3].astype(np.int32)
    if len(px):
        idx[m] = ((px[:, None] - pal[None]) ** 2).sum(-1).argmin(1) + 1
    return idx


# ---------------------------------------------------------------- numpy layers

def _conv3(x, w, b):
    """x: (H, W, Cin), w: (3, 3, Cin, Cout)."""
    H, W, C = x.shape
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    cols = np.empty((H, W, 9, C), dtype=np.float32)
    k = 0
    for dy in range(3):
        for dx in range(3):
            cols[:, :, k] = p[dy:dy + H, dx:dx + W]
            k += 1
    return (cols.reshape(H * W, 9 * C) @ w.reshape(9 * C, -1)).reshape(H, W, -1) + b


def _block(x, P, name):
    x = np.maximum(_conv3(x, P[f"{name}.c1.w"], P[f"{name}.c1.b"]), 0)
    return np.maximum(_conv3(x, P[f"{name}.c2.w"], P[f"{name}.c2.b"]), 0)


def _pool(x):
    H, W, C = x.shape
    return x.reshape(H // 2, 2, W // 2, 2, C).max((1, 3))


def _up(x):
    return x.repeat(2, 0).repeat(2, 1)


def forward_np(P, rough, ref, touched, slots, rough_idx):
    """rough/ref: (H, W, 4) float 0..1, touched (H, W) float, slots (K, 4). -> logits (H, W, K)."""
    x = np.concatenate([rough, ref, touched[..., None]], -1).astype(np.float32)
    skips = []
    for i, n in enumerate(ENC):
        x = _block(x, P, n)
        if i < 3:
            skips.append(x)
            x = _pool(x)
    for n in DEC:
        x = np.concatenate([_up(x), skips.pop()], -1)
        x = _block(x, P, n)
    emb = x @ P["head.w"] + P["head.b"]                          # (H, W, D)
    h = np.maximum(slots @ P["pal.w1"] + P["pal.b1"], 0)
    pe = h @ P["pal.w2"] + P["pal.b2"]                           # (K, D)
    logits = emb @ pe.T
    K = slots.shape[0]
    logits = logits + P["copy"] * np.eye(K, dtype=np.float32)[rough_idx]
    return logits


class Refiner:
    def __init__(self, path: str | Path):
        d = np.load(path)
        self.P = {k: d[k].astype(np.float32) for k in d.files}

    def refine(self, frame: np.ndarray, touched: np.ndarray, reference: np.ndarray, palette,
               grow: int = 2) -> np.ndarray:
        """Redraw only inside `touched` (grown by `grow` px); keep every other pixel."""
        H, W = frame.shape[:2]
        ph, pw = (-H) % 8, (-W) % 8
        pad = ((0, ph), (0, pw), (0, 0))
        fr = np.pad(frame, pad)
        rf = np.pad(reference, pad)
        area = np.pad(dilate(touched, grow), ((0, ph), (0, pw)))
        slots = palette_slots(palette)
        ridx = to_index(fr, palette)
        logits = forward_np(self.P, fr / 255.0, rf / 255.0, area.astype(np.float32), slots, ridx)
        pick = logits.argmax(-1)
        out = fr.copy()
        cols = (slots * 255).round().astype(np.uint8)
        out[area] = cols[pick[area]]
        return out[:H, :W]
