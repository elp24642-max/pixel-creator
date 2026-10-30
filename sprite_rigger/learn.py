"""JAX side of the refiner: data packing, model, loss, training loop.

Only needed for training/adapting; running the refiner uses numpy (refiner.py).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .refiner import DEC, ENC

K = 24  # palette slots per sample (transparent + up to 23 colours)


# ------------------------------------------------------------------ data

def prepare(path):
    d = np.load(path)
    return prepare_arrays(d["rough"], d["ref"], d["target"], d["touched"])


def prepare_arrays(rough, ref, tgt, touched):
    """Turn RGBA samples into palette indices + per-sample palettes."""
    N, H, W = rough.shape[:3]
    slots = np.zeros((N, K, 4), np.float32)
    idx = {k: np.zeros((N, H, W), np.int32) for k in ("rough", "target")}
    keep = []
    for i in range(N):
        cols = {}
        for img in (tgt[i], ref[i], rough[i]):
            m = img[..., 3] > 0
            for c in np.unique(img[m][:, :3], axis=0):
                cols.setdefault(tuple(int(v) for v in c), len(cols) + 1)
        if len(cols) + 1 > K:
            continue
        keep.append(i)
        slots[i, 0] = 0
        for c, j in cols.items():
            slots[i, j] = (*[v / 255 for v in c], 1.0)
        for name, img in (("rough", rough[i]), ("target", tgt[i])):
            flat = img.reshape(-1, 4)
            ids = np.zeros(len(flat), np.int32)
            m = flat[:, 3] > 0
            ids[m] = [cols[tuple(int(v) for v in c)] for c in flat[m][:, :3]]
            idx[name][i] = ids.reshape(H, W)
    keep = np.array(keep)
    return {
        "rough": rough[keep].astype(np.float32) / 255,
        "ref": ref[keep].astype(np.float32) / 255,
        "touched": touched[keep].astype(np.float32),
        "slots": slots[keep],
        "rough_idx": idx["rough"][keep],
        "target_idx": idx["target"][keep],
    }


# ------------------------------------------------------------------ model

def init(rng, widths, emb):
    P = {}
    keys = iter(jax.random.split(rng, 64))

    def conv(name, cin, cout):
        P[f"{name}.w"] = jax.random.normal(next(keys), (3, 3, cin, cout)) * np.sqrt(2 / (9 * cin))
        P[f"{name}.b"] = jnp.zeros(cout)

    cin = 9
    for n, w in zip(ENC, widths):
        conv(f"{n}.c1", cin, w)
        conv(f"{n}.c2", w, w)
        cin = w
    for n, w, skip in zip(DEC, widths[2::-1], widths[2::-1]):
        conv(f"{n}.c1", cin + skip, w)
        conv(f"{n}.c2", w, w)
        cin = w
    P["head.w"] = jax.random.normal(next(keys), (cin, emb)) * np.sqrt(1 / cin)
    P["head.b"] = jnp.zeros(emb)
    P["pal.w1"] = jax.random.normal(next(keys), (4, 32)) * 0.5
    P["pal.b1"] = jnp.zeros(32)
    P["pal.w2"] = jax.random.normal(next(keys), (32, emb)) * np.sqrt(1 / 32)
    P["pal.b2"] = jnp.zeros(emb)
    P["copy"] = jnp.array(2.0)
    return P


def _conv(x, w, b):
    y = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + b


def _block(x, P, n):
    x = jax.nn.relu(_conv(x, P[f"{n}.c1.w"], P[f"{n}.c1.b"]))
    return jax.nn.relu(_conv(x, P[f"{n}.c2.w"], P[f"{n}.c2.b"]))


def forward(P, rough, ref, touched, slots, rough_idx):
    x = jnp.concatenate([rough, ref, touched[..., None]], -1)
    skips = []
    for i, n in enumerate(ENC):
        x = _block(x, P, n)
        if i < 3:
            skips.append(x)
            B, H, W, C = x.shape
            x = x.reshape(B, H // 2, 2, W // 2, 2, C).max((2, 4))
    for n in DEC:
        x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
        x = jnp.concatenate([x, skips.pop()], -1)
        x = _block(x, P, n)
    emb = x @ P["head.w"] + P["head.b"]                               # B H W D
    h = jax.nn.relu(slots @ P["pal.w1"] + P["pal.b1"])
    pe = h @ P["pal.w2"] + P["pal.b2"]                                # B K D
    logits = jnp.einsum("bhwd,bkd->bhwk", emb, pe)
    logits = logits + P["copy"] * jax.nn.one_hot(rough_idx, slots.shape[1])
    # Empty slots (padding) can never be chosen.
    valid = (jnp.asarray(slots).sum(-1) > 0).at[:, 0].set(True)
    return jnp.where(valid[:, None, None, :], logits, -1e9)


def loss_fn(P, b):
    logits = forward(P, b["rough"], b["ref"], b["touched"], b["slots"], b["rough_idx"])
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, b["target_idx"])
    w = 0.1 + 0.9 * b["touched"]
    return (ce * w).sum() / w.sum()


def accuracy(P, b):
    logits = forward(P, b["rough"], b["ref"], b["touched"], b["slots"], b["rough_idx"])
    pred = logits.argmax(-1)
    t = b["touched"] > 0
    ok = (pred == b["target_idx"])
    base = (b["rough_idx"] == b["target_idx"])
    return (ok & t).sum() / t.sum(), (base & t).sum() / t.sum()




def train_loop(P, tr, steps, batch=16, lr=2e-3, val=None, log=print, save=None, seed=0, warm=True):
    """Adam with a cosine one-cycle schedule. tr/val: dicts from prepare_arrays."""
    sched = optax.cosine_onecycle_schedule(steps, lr) if warm else optax.cosine_decay_schedule(lr, steps)
    opt = optax.adamw(sched, weight_decay=1e-4)
    st = opt.init(P)

    @jax.jit
    def step(P, st, b):
        l, g = jax.value_and_grad(loss_fn)(P, b)
        u, st = opt.update(g, st, P)
        return optax.apply_updates(P, u), st, l

    acc = jax.jit(accuracy)
    rng = np.random.default_rng(seed)
    n = len(tr["rough"])
    t0 = time.time()
    for i in range(1, steps + 1):
        j = rng.integers(0, n, batch)
        b = {k: v[j] for k, v in tr.items()}
        if rng.random() < 0.5:  # mirror
            b = {k: (v[:, :, ::-1] if v.ndim >= 3 and k != "slots" else v) for k, v in b.items()}
        P, st, l = step(P, st, b)
        if i % 100 == 0 or i == steps:
            msg = f"step {i}/{steps} loss {float(l):.4f}"
            if val is not None:
                vb = {k: v[:64] for k, v in val.items()}
                va, vbase = acc(P, vb)
                msg += f" val touched-acc {float(va):.3f} (rough as-is {float(vbase):.3f})"
            log(msg + f" {time.time() - t0:.0f}s")
        if save and (i % 1000 == 0 or i == steps):
            save(P)
    return P


def to_numpy(P):
    return {k: np.asarray(v) for k, v in P.items()}
