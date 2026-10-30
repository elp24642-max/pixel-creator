"""Command line.

  python -m sprite_rigger regions <pose.png> <out_dir>
      Numbered region picture + regions.json: what you (or a model) group into parts.

  python -m sprite_rigger animate <rig.json> --out <dir> [--anim down:front:walk_front ...]
      [--frames 8] [--fps 8] [--mirror right:left] [--refiner weights.npz] [--no-clean]
      Each --anim is <row name>:<view>:<motion file or preset name>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .export import crop_box, save_gif, save_sheet, save_strip, to_cells, write_meta
from .motion import animate, load_motion
from .rig import load_rig
from .segment import region_info, segment
from .viz import regions_overlay

PRESETS = Path(__file__).parent / "presets"


def find_motion(name: str) -> dict:
    p = Path(name)
    if not p.exists():
        p = PRESETS / (name if name.endswith(".json") else name + ".json")
    return load_motion(p)


def cmd_regions(a):
    img = np.array(Image.open(a.image).convert("RGBA"))
    seg = segment(img)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    regions_overlay(img, seg["masks"], a.scale).save(out / "regions.png")
    (out / "regions.json").write_text(json.dumps(region_info(seg["masks"]), indent=1))
    print(f"{len(seg['masks'])} regions -> {out}")


def cmd_animate(a):
    views = load_rig(a.rig)
    refiner = None
    if a.refiner:
        from .refiner import Refiner
        refiner = Refiner(a.refiner)
    sets = {}
    for spec in a.anim:
        row, view, motion = spec.split(":", 2)
        sets[row] = animate(views[view], find_motion(motion), a.frames, a.amplitude,
                            clean=not a.no_clean, refiner=refiner)
    for m in a.mirror or []:
        src, dst = m.split(":")
        sets[dst] = [np.ascontiguousarray(f[:, ::-1]) for f in sets[src]]
    box = crop_box(list(sets.values()))
    rows = {k: to_cells(v, box) for k, v in sets.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.rig).parent.name
    for k, cells in rows.items():
        save_strip(cells, out / f"{name}_{k}_strip{len(cells)}.png")
        save_gif(cells, out / f"{name}_{k}_preview.gif", a.fps, a.preview_scale, (0x34, 0x68, 0x56, 255))
    save_sheet(rows, out / f"{name}_sheet.png")
    w, h = next(iter(rows.values()))[0].size
    write_meta(out / f"{name}.json", name, rows, a.fps, (w // 2, h - 2))
    print(f"{len(rows)} rows x {a.frames} frames of {w}x{h} -> {out}")


def main():
    ap = argparse.ArgumentParser(prog="sprite_rigger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("regions")
    r.add_argument("image")
    r.add_argument("out")
    r.add_argument("--scale", type=int, default=12)
    r.set_defaults(fn=cmd_regions)
    n = sub.add_parser("animate")
    n.add_argument("rig")
    n.add_argument("--out", required=True)
    n.add_argument("--anim", action="append", required=True)
    n.add_argument("--frames", type=int, default=8)
    n.add_argument("--fps", type=int, default=8)
    n.add_argument("--amplitude", type=float, default=1.0)
    n.add_argument("--mirror", action="append")
    n.add_argument("--refiner")
    n.add_argument("--no-clean", action="store_true")
    n.add_argument("--name")
    n.add_argument("--preview-scale", type=int, default=4)
    n.set_defaults(fn=cmd_animate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
