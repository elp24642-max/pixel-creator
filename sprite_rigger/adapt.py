"""Tune the refiner to one character without drawing anything ("there and back").

The character's own poses are clean pixel art in its own style. We spoil them the way
the engine does - a part turned through a lossy path - while the answer is the same
part turned without loss (a quarter turn, a flip, or back to where it was). The model
learns to turn spoiled pixels into clean ones *in this character's style*.

The base model is never changed; the tuned copy is saved in the project folder.
Training needs JAX (pip install jax optax); everything else stays numpy.
"""
from __future__ import annotations

import random

import numpy as np

from .cleanup import dilate
from .rig import View, _paste
from .rotsprite import rotate

MARGIN = 4


def _parts(view: View):
    """(mask, pivot) for every part that has a parent; whole regions if there is no rig."""
    out = [(p.mask, p.pivot, p.image) for p in view.parts.values() if p.parent is not None]
    if not out and view.labels is not None:
        for i in range(1, int(view.labels.max()) + 1):
            m = view.labels == i
            if m.sum() >= 6:
                ys, xs = np.nonzero(m)
                out.append((m, (float(xs.mean()), float(ys.min())), np.where(m[..., None], view.image, 0).astype(np.uint8)))
    return out


def _place(canvas, img, pivot, angle, flip):
    """Draw a part image turned by `angle` around `pivot` (optionally mirrored about it)."""
    if flip:
        px = pivot[0]
        img = img[:, ::-1]
        w = img.shape[1]
        pivot = (w - 1 - px, pivot[1])
        shift = round(px - pivot[0])
    else:
        shift = 0
    r, (qx, qy) = rotate(img, angle, pivot) if abs(angle) > 1e-6 else (img, pivot)
    ox = round(pivot[0] - qx) + shift + MARGIN
    oy = round(pivot[1] - qy) + MARGIN
    _paste(canvas, r, ox, oy)
    m = np.zeros(canvas.shape[:2], bool)
    a = r[..., 3] > 0
    h, w = a.shape
    H, W = m.shape
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
    if x0 < x1 and y0 < y1:
        m[y0:y1, x0:x1] = a[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return m


def make_pairs(view: View, n: int, seed: int = 0):
    rng = random.Random(seed)
    parts = _parts(view)
    if not parts:
        raise ValueError(f"view {view.name}: nothing to train on")
    img = view.image
    H, W = img.shape[0] + 2 * MARGIN, img.shape[1] + 2 * MARGIN
    H8, W8 = H + (-H) % 8, W + (-W) % 8
    out = {k: np.zeros((n, H8, W8, 4), np.uint8) for k in ("rough", "ref", "target")}
    out["touched"] = np.zeros((n, H8, W8), bool)
    for s in range(n):
        k = 1 if rng.random() < 0.7 else 2
        chosen = rng.sample(parts, min(k, len(parts)))
        base = np.zeros((H8, W8, 4), np.uint8)
        base[MARGIN:MARGIN + img.shape[0], MARGIN:MARGIN + img.shape[1]] = img
        for m, _, _ in chosen:  # lift the chosen parts off the body
            mm = np.zeros((H8, W8), bool)
            mm[MARGIN:MARGIN + img.shape[0], MARGIN:MARGIN + img.shape[1]] = m
            base[mm] = 0
        rough, target = base.copy(), base.copy()
        touched = np.zeros((H8, W8), bool)
        back_home = True
        for m, pivot, pimg in chosen:
            T = rng.choice([0, 0, 0, 90, -90, 180])
            flip = rng.random() < 0.2
            if T or flip:
                back_home = False
            a1 = rng.uniform(-35, 35)
            # Rough: two lossy turns that add up to T. Target: T in one (lossless) step.
            tmp = np.zeros_like(rough)
            _place(tmp, pimg, pivot, a1, flip)
            mid = tmp[MARGIN:MARGIN + img.shape[0], MARGIN:MARGIN + img.shape[1]]
            touched |= _place(rough, mid, pivot, T - a1, False)
            touched |= _place(target, pimg, pivot, T, flip)
        out["rough"][s], out["target"][s] = rough, target
        out["touched"][s] = dilate(touched, 1)
        ref = np.zeros((H8, W8, 4), np.uint8)
        ref[MARGIN:MARGIN + img.shape[0], MARGIN:MARGIN + img.shape[1]] = img
        # When the answer equals the rest pose, hide the reference half the time so
        # the model cannot learn "just copy the reference".
        if back_home and rng.random() < 0.5:
            ref[:] = 0
        out["ref"][s] = ref
    return out


def adapt(views: dict, base_weights: str, out_path: str, samples_per_view: int = 600, steps: int = 400,
          lr: float = 5e-4, log=print) -> str:
    try:
        import jax.numpy as jnp

        from .learn import prepare_arrays, to_numpy, train_loop
    except ImportError as e:
        raise RuntimeError("Tuning needs JAX: run install_training.bat (Windows) or install_training.sh, then restart") from e
    sets = []
    for i, v in enumerate(views.values()):
        log(f"making pairs from {v.name}")
        sets.append(make_pairs(v, samples_per_view, seed=i))
    # Views can differ in size; pad all to the largest.
    H = max(s["rough"].shape[1] for s in sets)
    W = max(s["rough"].shape[2] for s in sets)

    def pad(a):
        p = [(0, 0), (0, H - a.shape[1]), (0, W - a.shape[2])] + [(0, 0)] * (a.ndim - 3)
        return np.pad(a, p)

    data = {k: np.concatenate([pad(s[k]) for s in sets]) for k in sets[0]}
    tr = prepare_arrays(data["rough"], data["ref"], data["target"], data["touched"])
    perm = np.random.default_rng(0).permutation(len(tr["rough"]))
    tr = {k: v[perm] for k, v in tr.items()}
    nval = min(64, len(perm) // 10)
    val = {k: v[:nval] for k, v in tr.items()}
    tr = {k: v[nval:] for k, v in tr.items()}
    base = np.load(base_weights)
    P = {k: jnp.array(base[k]) for k in base.files}
    P = train_loop(P, tr, steps, 16, lr, val, log=log, warm=False)
    np.savez(out_path, **to_numpy(P))
    return out_path
