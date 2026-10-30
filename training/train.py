"""Train the refiner with JAX (CPU works; a GPU, e.g. free Colab, is much faster).

Usage: python training/train.py <dataset.npz> <out_weights.npz> [--steps 6000] [--batch 16]
       [--widths 24,48,72,96] [--emb 16] [--resume weights.npz]

The saved weights run without JAX through sprite_rigger.refiner (numpy only).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sprite_rigger.refiner import DEC, ENC  # noqa: E402

K = 24  # palette slots per sample (transparent + up to 23 colours)


# ------------------------------------------------------------------ data

def prepare(path):
    """Turn RGBA samples into palette indices + per-sample palettes."""
    d = np.load(path)
    rough, ref, tgt, touched = d["rough"], d["ref"], d["target"], d["touched"]
    N = len(rough)
    slots = np.zeros((N, K, 4), np.float32)
    idx = {k: np.zeros((N, 64, 64), np.int32) for k in ("rough", "target")}
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
            idx[name][i] = ids.reshape(64, 64)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--widths", default="24,48,72,96")
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--resume")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = prepare(a.data)
    N = len(data["rough"])
    nval = max(64, N // 20)
    val = {k: v[:nval] for k, v in data.items()}
    tr = {k: v[nval:] for k, v in data.items()}
    print(f"samples: train {N - nval}, val {nval}", flush=True)

    widths = [int(v) for v in a.widths.split(",")]
    P = init(jax.random.PRNGKey(a.seed), widths, a.emb)
    if a.resume:
        r = np.load(a.resume)
        P = {k: jnp.array(r[k]) for k in P}
    sched = optax.cosine_onecycle_schedule(a.steps, a.lr)
    opt = optax.adamw(sched, weight_decay=1e-4)
    st = opt.init(P)

    @jax.jit
    def step(P, st, b):
        l, g = jax.value_and_grad(loss_fn)(P, b)
        u, st = opt.update(g, st, P)
        return optax.apply_updates(P, u), st, l

    acc = jax.jit(accuracy)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    ntr = len(tr["rough"])
    for i in range(1, a.steps + 1):
        j = rng.integers(0, ntr, a.batch)
        b = {k: v[j] for k, v in tr.items()}
        if rng.random() < 0.5:  # mirror
            b = {k: (v[:, :, ::-1] if v.ndim >= 3 and k != "slots" else v) for k, v in b.items()}
        P, st, l = step(P, st, b)
        if i % 100 == 0 or i == a.steps:
            vb = {k: v[:64] for k, v in val.items()}
            va, vbase = acc(P, vb)
            print(f"step {i} loss {float(l):.4f} val touched-acc {float(va):.3f} (rough as-is {float(vbase):.3f}) "
                  f"{time.time() - t0:.0f}s", flush=True)
        if i % 1000 == 0 or i == a.steps:
            np.savez(a.out, **{k: np.asarray(v) for k, v in P.items()})


if __name__ == "__main__":
    main()
