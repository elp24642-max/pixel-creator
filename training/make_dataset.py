"""Build refiner training pairs from real hand-drawn walk cycles.

For every walk frame we fake what the rig engine would produce: take the standing
frame, cut it into regions, move each region to where it best matches the real frame
(sometimes turning it a little, as the engine does), and paste them together. The model
learns to turn that rough frame into the frame the artist actually drew.

Usage: python training/make_dataset.py <lpc spritesheets dir> <out.npz> [samples] [workers]
"""
from __future__ import annotations

import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sprite_rigger.cleanup import dilate  # noqa: E402
from sprite_rigger.rotsprite import rotate  # noqa: E402
from sprite_rigger.segment import segment  # noqa: E402
from training.lpc_compose import compose, index  # noqa: E402

S = 64
MAX_SHIFT = 4


def recolour(sheet: np.ndarray, rng: random.Random, k: int) -> np.ndarray:
    """Hue shift plus palette reduction to k colours, the same palette for the whole sheet."""
    im = Image.fromarray(sheet, "RGBA")
    a = np.array(im)[..., 3]
    rgb = np.array(im.convert("RGB")).astype(np.float32) / 255
    dh = rng.random()
    # Hue rotation in YIQ space (fast, vectorised).
    yiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    th = dh * 2 * np.pi
    rot = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    m = np.linalg.inv(yiq) @ rot @ yiq
    rgb = np.clip(rgb @ m.T, 0, 1)
    q = Image.fromarray((rgb * 255).astype(np.uint8), "RGB").quantize(colors=k, method=Image.Quantize.MEDIANCUT,
                                                                       dither=Image.Dither.NONE)
    out = np.array(q.convert("RGBA"))
    out[..., 3] = np.where(a >= 128, 255, 0)
    out[a < 128] = 0
    return out


def rough_from(ref: np.ndarray, tgt: np.ndarray, rng: random.Random):
    seg = segment(ref)
    masks = seg["masks"]
    t_op = tgt[..., 3] > 0
    canvas = np.zeros_like(ref)
    shifts = {}
    rotated = set()
    items = sorted(masks.items(), key=lambda kv: -kv[1].sum())
    for rid, m in items:
        part = np.where(m[..., None], ref, 0).astype(np.uint8)
        if m.sum() >= 12 and rng.random() < 0.25:
            ys, xs = np.nonzero(m)
            ang = rng.uniform(-15, 15)
            img, (px, py) = rotate(part, ang, (float(xs.mean()), float(ys.min())))
            pad = int(round(px - xs.mean()))
            img = img[pad:pad + S, pad:pad + S]
            if img.shape[:2] == (S, S):
                part = img
                rotated.add(rid)
        pm = part[..., 3] > 0
        best, best_s = (0, 0), -1e9
        for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
            for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
                sp = np.roll(np.roll(part, dy, 0), dx, 1)
                sm = np.roll(np.roll(pm, dy, 0), dx, 1)
                eq = sm & t_op & np.all(sp[..., :3] == tgt[..., :3], axis=-1)
                s = eq.sum() - 0.5 * (sm & ~t_op).sum() - 0.05 * (abs(dx) + abs(dy))
                if s > best_s:
                    best, best_s = (dx, dy), s
        dx, dy = best
        sp = np.roll(np.roll(part, dy, 0), dx, 1)
        sm = sp[..., 3] > 0
        canvas[sm] = sp[sm]
        shifts[rid] = (best, sm)
    # Regions that moved differently from the bulk of the body are the "touched" area.
    vals = [s for s, _ in shifts.values()]
    base = max(set(vals), key=vals.count)
    touched = np.zeros(ref.shape[:2], dtype=bool)
    for rid, (s, sm) in shifts.items():
        if s != base or rid in rotated:
            touched |= sm
    # Plus a loose zone around every wrong pixel. Loose on purpose: at run time the mask
    # comes from the parts the rig moved, which only roughly covers what needs redrawing.
    diff = (canvas[..., 3] > 0) != (tgt[..., 3] > 0)
    diff |= (canvas[..., 3] > 0) & np.any(canvas[..., :3] != tgt[..., :3], axis=-1)
    touched |= dilate(diff, 3)
    return canvas, touched, base


def make_samples(args):
    root, seed, n = args
    rng = random.Random(seed)
    idx = index(root)
    out = {k: [] for k in ("rough", "ref", "touched", "target")}
    while len(out["rough"]) < n:
        sheet = compose(root, idx, rng)
        if sheet is None:
            continue
        sheet = recolour(sheet, rng, rng.choice([4, 4, 5, 6, 8, 12, 16]))
        for _ in range(6):
            row = rng.randrange(4)
            col = rng.randrange(1, 9)
            ref = sheet[row * S:(row + 1) * S, 0:S]
            tgt = sheet[row * S:(row + 1) * S, col * S:(col + 1) * S]
            if not (ref[..., 3] > 0).any() or not (tgt[..., 3] > 0).any():
                continue
            if rng.random() < 0.5:
                ref, tgt = ref[:, ::-1].copy(), tgt[:, ::-1].copy()
            rough, touched, base = rough_from(ref, tgt, rng)
            # The reference is shown where the engine would have it: moved with the body.
            ref_al = np.roll(np.roll(ref, base[1], 0), base[0], 1)
            out["rough"].append(rough)
            out["ref"].append(ref_al)
            out["touched"].append(touched)
            out["target"].append(tgt)
            if len(out["rough"]) >= n:
                break
    return {k: np.stack(v) for k, v in out.items()}


def main():
    root = Path(sys.argv[1])
    dst = sys.argv[2]
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    chunk = 200
    jobs = [(root, 1000 + i, min(chunk, total - i * chunk)) for i in range((total + chunk - 1) // chunk)]
    with Pool(workers) as p:
        parts = p.map(make_samples, jobs)
    data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    np.savez_compressed(dst, **data)
    print({k: v.shape for k, v in data.items()})


if __name__ == "__main__":
    main()
