# Instructions for AI coding agents (Codex, Antigravity, Gemini CLI, Claude Code, ...)

This repository is **sprite-rigger**: a local tool that draws and animates pixel-art sprites.

- To **draw a sprite or character** (or poses for animation), follow
  [.claude/skills/pixel-sprite/SKILL.md](.claude/skills/pixel-sprite/SKILL.md): write a JSON recipe, render it with
  `python -m sprite_rigger draw <recipe.json>`, look at the preview, fix, repeat.
- To **animate** drawn poses, run the app (`run.bat <folder>` / `./run.sh <folder>`) or the CLI
  `python -m sprite_rigger animate` (see README).
- Code style: Python 3.10+, numpy + Pillow only at runtime; tests with `python tests/test_engine.py`.
