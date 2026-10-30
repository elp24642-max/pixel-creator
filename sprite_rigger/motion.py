"""Motions: per-part curves over one cycle, sampled to any number of frames.

A motion is plain JSON, which is what a model writes:

    {"name": "walk", "loop": true,
     "tracks": {"leg_l": {"dy": [0, -1, -2, -1, 0, 0, 0, 0], "rot": [...]}, ...}}

Each curve is a list of evenly spaced keys over the cycle. Sampling to N frames
interpolates linearly (wrapping round for loops), so the same motion works at 4, 6, 8
or 12 frames. Offsets in dx/dy are rounded to whole pixels when drawn.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cleanup import cleanup
from .rig import View, render

CHANNELS = ("dx", "dy", "rot")


def sample(keys: list[float], n: int, loop: bool = True) -> list[float]:
    keys = [float(k) for k in keys]
    if len(keys) == 1:
        return keys * n
    m = len(keys)
    out = []
    for i in range(n):
        if loop:
            t = i * m / n
            k0 = int(np.floor(t)) % m
            k1 = (k0 + 1) % m
        else:
            t = i * (m - 1) / max(1, n - 1)
            k0 = min(int(np.floor(t)), m - 1)
            k1 = min(k0 + 1, m - 1)
        f = t - np.floor(t)
        out.append(keys[k0] * (1 - f) + keys[k1] * f)
    return out


def poses(motion: dict, frames: int, amplitude: float = 1.0) -> list[dict]:
    loop = motion.get("loop", True)
    out = [dict() for _ in range(frames)]
    for part, tr in motion.get("tracks", {}).items():
        if "variant" in tr:  # drawings switch, they do not blend: nearest key, no interpolation
            keys = tr["variant"]
            for i in range(frames):
                k = int(i * len(keys) / frames) if loop else round(i * (len(keys) - 1) / max(1, frames - 1))
                if keys[k]:
                    out[i].setdefault(part, {})["variant"] = keys[k]
        for ch in CHANNELS:
            if ch not in tr:
                continue
            vals = sample(tr[ch], frames, loop)
            for i, v in enumerate(vals):
                if ch in ("dx", "dy"):
                    v = round(v * amplitude)
                else:
                    v = v * amplitude
                out[i].setdefault(part, {})[ch] = v
    return out


def animate(view: View, motion: dict, frames: int, amplitude: float = 1.0, margin: int = 4,
            clean: bool = True, refiner=None, with_touched: bool = False):
    """Frames for one motion. clean runs the rule pass; refiner is an optional model
    object with .refine(frame, touched, reference) (see refiner.py)."""
    unknown = set(motion.get("tracks", {})) - set(view.parts)
    if unknown:
        raise ValueError(f"motion moves parts the rig does not have: {sorted(unknown)}")
    ref = np.pad(view.image, ((margin, margin), (margin, margin), (0, 0)))
    out, masks = [], []
    for p in poses(motion, frames, amplitude):
        f, touched = render(view, p, margin, with_touched=True)
        if refiner is not None and touched.any():
            f = refiner.refine(f, touched, ref, view.palette)
        if clean and touched.any():
            f = cleanup(f, touched)
        out.append(f)
        masks.append(touched)
    return (out, masks) if with_touched else out


def load_motion(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
