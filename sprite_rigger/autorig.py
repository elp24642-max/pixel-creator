"""The model's jobs: group regions into a skeleton, write motions, revise them from notes."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .ai import Provider, extract_json
from .rig import build_view
from .segment import region_info, segment
from .viz import regions_overlay

PRESETS = Path(__file__).parent / "presets"

SYSTEM = ("You are a pixel-art animator rigging small game sprites. You answer with one JSON object "
          "in a ```json block, after at most a few short sentences.")

RIG_PROMPT = """The picture is a {w}x{h} px sprite ({view} view), enlarged, with every region tinted and
numbered; grid lines every 8 px. Region list (bbox is x0,y0,x1,y1 in sprite pixels):
{regions}

Notes from the user: {notes}

Group EVERY region into body parts and build a skeleton. Rules:
- Name parts by role: torso, head, and for limbs arm_l/arm_r, hand_l/hand_r, leg_l/leg_r, foot_l/foot_r
  (l/r = screen left/right) in front/back views; in a side view use arm, hand, leg_near, foot_near.
  Add other parts (tail, cape, hair, weapon...) with plain names if the sprite has them.
- Split long limbs into segments when the drawing has visible joints (thigh/shin) so knees can bend.
- pivot = the joint the part turns around, in sprite pixel coords [x, y] (top of a leg = hip, etc.).
- parent: head/arms/legs hang from torso, hands from arms, feet from legs. Exactly one part has no parent.
- z: draw order, higher is in front.
- In a side view, if a far leg or arm is hidden, add it as a clone:
  "leg_far": {{"clone_of": "leg_near", "offset": [0, 0], "shade": 0, "parent": "torso", "z": 0}}

Answer format:
```json
{{"parts": {{"torso": {{"regions": [..], "pivot": [x, y], "z": 3}},
            "head": {{"regions": [..], "pivot": [x, y], "parent": "torso", "z": 7}}}}}}
```"""

MOTION_PROMPT = """Write a looping "{name}" animation for this rig ({view} view). Requested: {notes}

Parts (name: pivot, parent):
{parts}

Format: each part's curves are lists of evenly spaced keys over ONE cycle; the engine resamples
them to any frame count. Channels: dx, dy (pixels, rounded; +y is down), rot (degrees, positive =
clockwise on screen, so a hanging limb swings backwards for a right-facing side view),
variant (name of an alternative drawing, if the part has any). Children follow parents, so a
planted foot must cancel the torso's drop. Keep pixel moves small (1-3 px) and angles modest.

Example (front walk):
```json
{example}
```

Answer with the motion JSON for this rig, same format, only parts that exist."""

FIX_PROMPT = """Here is the current motion and frames rendered from it (left to right, enlarged).
Feedback: {notes}

Current motion:
```json
{motion}
```
Return the corrected full motion JSON in the same format."""


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def validate_rig(spec: dict, n_regions: int) -> list[str]:
    errs = []
    parts = spec.get("parts")
    if not isinstance(parts, dict) or not parts:
        return ["no parts"]
    used = {}
    roots = []
    for name, p in parts.items():
        if "clone_of" in p:
            if p["clone_of"] not in parts:
                errs.append(f"{name}: clone_of unknown part {p['clone_of']}")
            continue
        for r in p.get("regions", []):
            if not isinstance(r, int) or not 1 <= r <= n_regions:
                errs.append(f"{name}: region {r} does not exist")
            elif r in used:
                errs.append(f"region {r} is in both {used[r]} and {name}")
            else:
                used[r] = name
        if "pivot" not in p:
            errs.append(f"{name}: missing pivot")
        if p.get("parent") is None:
            roots.append(name)
        elif p["parent"] not in parts:
            errs.append(f"{name}: parent {p['parent']} does not exist")
    missing = sorted(set(range(1, n_regions + 1)) - set(used))
    if missing:
        errs.append(f"regions not assigned to any part: {missing}")
    if len(roots) != 1:
        errs.append(f"exactly one part needs no parent, found {roots}")
    return errs


def build_rig(ai: Provider, image: np.ndarray, view: str, notes: str = "", attempts: int = 3) -> dict:
    seg = segment(image)
    info = region_info(seg["masks"])
    pic = png_bytes(regions_overlay(image, seg["masks"], 12))
    h, w = image.shape[:2]
    prompt = RIG_PROMPT.format(w=w, h=h, view=view, notes=notes or "none",
                               regions="\n".join(json.dumps(r) for r in info))
    errs: list[str] = []
    for _ in range(attempts):
        q = prompt if not errs else prompt + "\n\nYour previous answer had problems, fix them:\n- " + "\n- ".join(errs)
        spec = extract_json(ai.ask(q, [pic], SYSTEM))
        errs = validate_rig(spec, len(info))
        if not errs:
            build_view(view, image, spec)  # final check: the engine accepts it
            return spec
    raise RuntimeError("the model could not build a valid rig: " + "; ".join(errs))


def validate_motion(motion: dict, part_names) -> list[str]:
    errs = []
    tracks = motion.get("tracks")
    if not isinstance(tracks, dict) or not tracks:
        return ["no tracks"]
    for p, tr in tracks.items():
        if p not in part_names:
            errs.append(f"track for unknown part {p}")
        for ch, keys in tr.items():
            if ch not in ("dx", "dy", "rot", "variant"):
                errs.append(f"{p}: unknown channel {ch}")
            elif not isinstance(keys, list) or not keys:
                errs.append(f"{p}.{ch}: needs a non-empty list")
    return errs


def write_motion(ai: Provider, view_spec: dict, view: str, name: str, notes: str, attempts: int = 3) -> dict:
    parts = "\n".join(f"{n}: pivot {p.get('pivot')}, parent {p.get('parent')}"
                      + (f", variants {list(p['variants'])}" if p.get("variants") else "")
                      for n, p in view_spec["parts"].items())
    example = (PRESETS / "walk_front.json").read_text()
    prompt = MOTION_PROMPT.format(name=name, view=view, notes=notes or name, parts=parts, example=example)
    errs: list[str] = []
    for _ in range(attempts):
        q = prompt if not errs else prompt + "\n\nFix these problems:\n- " + "\n- ".join(errs)
        m = extract_json(ai.ask(q, [], SYSTEM))
        m.setdefault("name", name)
        m.setdefault("loop", True)
        errs = validate_motion(m, view_spec["parts"])
        if not errs:
            return m
    raise RuntimeError("the model could not write a valid motion: " + "; ".join(errs))


def fix_motion(ai: Provider, motion: dict, frames: list[np.ndarray], notes: str, part_names, attempts: int = 2) -> dict:
    h, w = frames[0].shape[:2]
    strip = Image.new("RGBA", (w * len(frames), h), (52, 104, 86, 255))
    for i, f in enumerate(frames):
        strip.alpha_composite(Image.fromarray(f, "RGBA"), (i * w, 0))
    strip = strip.resize((strip.width * 4, strip.height * 4), Image.NEAREST)
    prompt = FIX_PROMPT.format(notes=notes, motion=json.dumps(motion, indent=1))
    errs: list[str] = []
    for _ in range(attempts):
        q = prompt if not errs else prompt + "\n\nFix these problems:\n- " + "\n- ".join(errs)
        m = extract_json(ai.ask(q, [png_bytes(strip)], SYSTEM))
        errs = validate_motion(m, part_names)
        if not errs:
            return m
    raise RuntimeError("the model could not fix the motion: " + "; ".join(errs))
