"""Run with: python -m pytest tests  (or python tests/test_engine.py)"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sprite_rigger.ai import extract_json  # noqa: E402
from sprite_rigger.autorig import validate_motion, validate_rig  # noqa: E402
from sprite_rigger.cleanup import cleanup  # noqa: E402
from sprite_rigger.motion import animate, load_motion, sample  # noqa: E402
from sprite_rigger.rig import load_rig, render  # noqa: E402
from sprite_rigger.rotsprite import rotate  # noqa: E402
from sprite_rigger.segment import segment  # noqa: E402

RIG = ROOT / "examples" / "robot" / "rig.json"
PRESETS = ROOT / "sprite_rigger" / "presets"


def palette_of(img):
    return {tuple(int(v) for v in p[:3]) for p in img.reshape(-1, 4) if p[3]}


def test_segment_covers_every_pixel():
    img = np.array(Image.open(ROOT / "examples/robot/front.png").convert("RGBA"))
    masks = segment(img)["masks"]
    cover = np.zeros(img.shape[:2], bool)
    for m in masks.values():
        cover |= m
    assert (cover == (img[..., 3] > 0)).all()


def test_rest_pose_renders_unchanged():
    for name, v in load_rig(RIG).items():
        f = render(v, {}, margin=0)
        assert (f == v.image).all() or name == "side", name  # side has clones hidden behind


def test_frames_stay_in_palette_and_alpha_is_binary():
    views = load_rig(RIG)
    for view, motion in (("front", "walk_front"), ("side", "walk_side")):
        frames = animate(views[view], load_motion(PRESETS / f"{motion}.json"), 6)
        pal = palette_of(views[view].image)
        for f in frames:
            assert palette_of(f) <= pal
            assert set(np.unique(f[..., 3])) <= {0, 255}


def test_any_frame_count():
    v = load_rig(RIG)["front"]
    m = load_motion(PRESETS / "walk_front.json")
    for n in (1, 4, 5, 12):
        assert len(animate(v, m, n)) == n


def test_sample_wraps():
    assert sample([0, 4], 4) == [0, 2, 4, 2]


def test_rotate_keeps_colours():
    img = np.array(Image.open(ROOT / "examples/robot/side.png").convert("RGBA"))
    r, _ = rotate(img, 17, (24, 30))
    assert palette_of(r) <= palette_of(img)


def test_cleanup_only_inside_area():
    img = np.array(Image.open(ROOT / "examples/robot/front.png").convert("RGBA"))
    area = np.zeros(img.shape[:2], bool)
    assert (cleanup(img, area) == img).all()


def test_validators_and_json_extraction():
    spec = json.loads(RIG.read_text())["views"]["front"]
    assert validate_rig({"parts": spec["parts"]}, 41) == []
    assert validate_rig({"parts": {"a": {"regions": [1], "pivot": [0, 0]}}}, 2)
    assert validate_motion({"tracks": {"torso": {"dy": [0, 1]}}}, spec["parts"]) == []
    assert validate_motion({"tracks": {"tail": {"dy": [0]}}}, spec["parts"])
    assert extract_json('ok\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('text {"b": {"c": 2}} end') == {"b": {"c": 2}}


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
