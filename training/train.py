"""Train the refiner with JAX (CPU works; a GPU, e.g. free Colab, is much faster).

Usage: python training/train.py <dataset.npz> <out_weights.npz> [--steps 6000] [--batch 16]
       [--widths 24,48,72,96] [--emb 16] [--resume weights.npz]

The saved weights run without JAX through sprite_rigger.refiner (numpy only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sprite_rigger.learn import init, prepare, to_numpy, train_loop  # noqa: E402


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
    train_loop(P, tr, a.steps, a.batch, a.lr, val, log=lambda m: print(m, flush=True),
               save=lambda P: np.savez(a.out, **to_numpy(P)), seed=a.seed)


if __name__ == "__main__":
    main()
