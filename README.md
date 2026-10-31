# sprite-rigger

Animate pixel-art sprites from a few still poses. An AI model builds the skeleton and writes the motion; a small local engine draws every frame pixel-exact, in your palette, at any frame count. Built for weak laptops: the app is Python + numpy + Pillow and a page in your browser.

![walk preview](docs/robot_walk.gif)

## What it does

1. **Cuts the sprite into regions.** Pixel art draws parts as fills inside an outline, so the outline tells where the parts are. No AI needed.
2. **Builds a skeleton.** A vision model looks at the numbered regions and groups them into parts (torso, head, arms, legs…) with joints, parents and draw order. You can edit the result as JSON.
3. **Writes the motion.** From a description ("heavy walk, head bobs late") the model writes short curves per part. The curves are resampled to any number of frames, so the same motion works at 4, 6, 8 or 12 frames.
4. **Draws the frames.** Parts move by whole pixels or turn with a pixel-art rotation (RotSprite); gaps at stretched joints are filled; colours never leave your palette.
5. **Touches up.** A rule-based cleanup pass closes outlines and removes stray pixels. An optional small local **refiner model** redraws rough spots. What still needs a human eye you paint in a built-in editor with onion skin; part **variants** (a bent knee, a fist) are drawn once and reused.
6. **Exports** strips, a sheet, GIF previews and metadata for GameMaker (works for any engine that reads strips/sheets).

## Run it

You need Python 3.10+.

```
git clone https://github.com/elp24642-max/pixel-creator
cd pixel-creator
run.bat           (Windows)      ./run.sh        (macOS / Linux)
```

The first run makes a virtual environment and installs numpy and Pillow. The browser opens at `http://127.0.0.1:8765`. Pass a folder to work in: `run.bat my_robot`.

Try the example: `run.bat examples/robot`.

## Drawing sprites with an AI agent

Any coding agent (Claude Code, Codex, Antigravity, Gemini CLI) or a model over the API can draw sprites here: it writes a JSON **recipe** (palette keys, text grids, shapes, mirror, outline), renders it, looks at the enlarged preview and fixes it. The instructions are in [.claude/skills/pixel-sprite/SKILL.md](.claude/skills/pixel-sprite/SKILL.md); `AGENTS.md` / `GEMINI.md` / `CLAUDE.md` point agents to them. Recipes are data, nothing in them is executed, so they are safe to take from any model.

```
python -m sprite_rigger draw art/hero.recipe.json        # PNGs + previews + front/back/side poses
python -m sprite_rigger ascii art/hero.recipe.json hero_front
```

Idea inspired by [claude-pxl](https://github.com/crsolver/claude-pxl) (MIT), rewritten as a data-only format for this project.

## Model connections

Pick one in the left panel:

| Connection | What you need |
|---|---|
| **Claude Code** | Claude Code installed and signed in on this computer. The app runs `claude -p` in the background. |
| **Anthropic API** | An API key. |
| **OpenAI-compatible** | Any server that speaks the OpenAI chat API: OpenAI, OpenRouter, Gemini's OpenAI endpoint, DeepSeek, and local ones (Ollama, LM Studio, llama.cpp). Set the base URL, key and model. |

The model must accept images for skeleton building. Writing and fixing motions works with any text model. The key is saved only in the project's `settings.json`.

There is no request limit: providers show your usage, and local models have none. The one safeguard is for **auto-review**, where the model checks its own frames and revises: it stops after N rounds in a row (settings, default 3, 0 turns it off). The header shows requests and tokens used in this session.

## Settings

Frames, FPS, motion strength, margin around the sprite, fixed cell size (or fit), preview scale, cleanup on/off, refiner on/off, auto-review limit. Mirror rows on export (a left walk from the right one).

## The refiner model

A small U-Net (~0.5 M parameters) that runs in numpy (~50 ms per frame on a laptop CPU). It only changes pixels the engine turned, stretched or swapped, and can only pick colours from your palette.

**Two layers:**

- **Base** (`sprite_rigger/models/refiner_base.npz`) learns from hand-drawn walk cycles: the engine fakes a rough version of each real frame from the standing frame, and the model learns to turn it into what the artist drew. The bundled base was trained on a free Colab T4 (20 000 samples, 30 000 steps, ~0.9 M parameters): 90% of redrawn pixels right on held-out walk frames, against 71% for the rough frame (an earlier short CPU run reached 83%).
- **Tuned to your character**, with no drawing: press *Tune to this character* (3-5 min on a laptop). The app spoils the character's own poses the way the engine does - a part turned through a lossy path - while the answer is the same part turned without loss (a quarter turn, a flip, or back home). So the model learns clean pixels in *this* character's style. The tuned copy is saved in the project folder; the base is never changed, so one character never makes it worse for another. Tuning needs JAX once: run `install_training.bat` / `install_training.sh`.

Honest limits: it cleans and fills; it does not invent a pose that is not there (a turn, a fall). On the example robot's side view (thin legs drawn over each other) the tuned model is still no better than the rule-based cleanup alone, so the refiner is off by default; it helps most where parts turn without overlapping. Draw missing poses, or draw a part variant and let the engine reuse it.

**Train a better base** in a free Colab GPU: open `training/train_refiner_colab.ipynb`, or locally:

```
pip install -r requirements-train.txt
python training/make_dataset.py <LPC spritesheets folder> data.npz 20000 2
python training/train.py data.npz sprite_rigger/models/refiner.npz --steps 30000
```

Training data: [Universal LPC Spritesheet Character Generator](https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator) art (CC-BY-SA 3.0 / GPL 3.0 and others; authors in its `CREDITS.csv`). The art is not included in this repository.

## Files

```
sprite_rigger/
  segment.py    regions from outlines
  rig.py        skeleton, transforms, frame renderer
  rotsprite.py  pixel-art rotation
  motion.py     curves -> per-frame poses
  cleanup.py    rule-based tidy-up
  refiner.py    the small model, numpy inference
  autorig.py    prompts + validation for skeleton / motion / fixes
  ai.py         Claude Code, Anthropic, OpenAI-compatible connections
  server.py     local web app;  web/index.html  the page
  presets/      walk_front, walk_side, idle
training/       dataset builder, trainer (JAX), Colab notebook
examples/robot  a sample rig
```

Command line without the browser:

```
python -m sprite_rigger regions pose.png out/
python -m sprite_rigger animate examples/robot/rig.json --out out/ --anim down:front:walk_front --anim right:side:walk_side --mirror right:left --frames 8
```

## Rig format

```json
{"views": {"front": {"image": "front.png", "parts": {
  "torso": {"regions": [6, 7, 9], "pivot": [24, 29], "z": 3},
  "leg_l": {"regions": [20, 24], "pivot": [18, 30], "parent": "torso", "z": 1,
            "variants": {"bent": {"image": "variants/front_leg_l_bent.png"}}},
  "leg_far": {"clone_of": "leg_near", "offset": [0, 0], "shade": -1, "z": 0}
}}}}
```

Motion: `{"loop": true, "tracks": {"leg_l": {"dy": [0, -2, -3, -1, 0, 0, 0, 0], "rot": [...], "variant": ["", "bent", "", ""]}}}`. `dx`/`dy` in pixels, `rot` in degrees (clockwise), keys evenly spaced over one cycle.

## License

MIT. The example robot sprite belongs to its author and is included only as a demo.
