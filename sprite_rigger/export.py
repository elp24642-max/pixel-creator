"""Write frames out: GIF preview, horizontal strips, a sheet, and metadata for engines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def crop_box(frame_sets: list[list[np.ndarray]], pad: int = 1):
    """One box that holds every frame of every animation, so all cells line up."""
    boxes = []
    for frames in frame_sets:
        for f in frames:
            bb = Image.fromarray(f, "RGBA").getbbox()
            if bb:
                boxes.append(bb)
    x0 = min(b[0] for b in boxes) - pad
    y0 = min(b[1] for b in boxes) - pad
    x1 = max(b[2] for b in boxes) + pad
    y1 = max(b[3] for b in boxes) + pad
    h, w = frame_sets[0][0].shape[:2]
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def to_cells(frames, box, cell=None):
    """Crop to `box`; if `cell` (w, h) is given, centre horizontally and sit on the bottom."""
    out = []
    for f in frames:
        im = Image.fromarray(f, "RGBA").crop(box)
        if cell:
            cw, ch = cell
            c = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
            c.alpha_composite(im, ((cw - im.width) // 2, ch - im.height))
            im = c
        out.append(im)
    return out


def save_gif(cells, path, fps: int, scale: int = 1, bg=None):
    ims = []
    for c in cells:
        im = c.resize((c.width * scale, c.height * scale), Image.NEAREST)
        if bg:
            b = Image.new("RGBA", im.size, bg)
            b.alpha_composite(im)
            im = b
        ims.append(im)
    ims[0].save(path, format="GIF", save_all=True, append_images=ims[1:], duration=round(1000 / fps), loop=0, disposal=2)


def save_strip(cells, path):
    w, h = cells[0].size
    s = Image.new("RGBA", (w * len(cells), h), (0, 0, 0, 0))
    for i, c in enumerate(cells):
        s.paste(c, (i * w, 0))
    s.save(path)
    return s


def save_sheet(rows: dict, path):
    names = list(rows)
    w, h = rows[names[0]][0].size
    n = max(len(r) for r in rows.values())
    s = Image.new("RGBA", (w * n, h * len(names)), (0, 0, 0, 0))
    for j, name in enumerate(names):
        for i, c in enumerate(rows[name]):
            s.paste(c, (i * w, j * h))
    s.save(path)
    return s


def write_meta(path, sprite: str, rows: dict, fps: int, origin):
    w, h = next(iter(rows.values()))[0].size
    meta = {
        "sprite": sprite,
        "frame_size": [w, h],
        "fps": fps,
        "origin": list(origin),
        "rows": {name: {"row": j, "frames": len(r), "file": f"{sprite}_{name}_strip{len(r)}.png"}
                 for j, (name, r) in enumerate(rows.items())},
        "gamemaker": {
            "note": "Import each strip with 'Import Strip Image', frames = count in the name; set origin to the value above.",
            "image_speed_at_60fps_room": round(fps / 60, 4),
        },
    }
    Path(path).write_text(json.dumps(meta, indent=2))
