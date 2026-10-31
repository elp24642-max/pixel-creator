---
name: pixel-sprite
description: Draw pixel-art game sprites (characters, enemies, items, poses for animation) as JSON recipes that sprite-rigger renders to PNG, then look at the result and fix it. Use when asked to draw, design, redraw or iterate on a sprite or character, or to make front/back/side poses for animation.
---

# Pixel sprites as recipes

You do not paint pixels blindly. You write a small JSON **recipe**: a palette of one-letter keys,
text grids for shapes that need care, and shape ops for the rest. The tool renders it, you **look**
at the result, and you patch what is wrong. Mirroring and outlines are one line each, so you never
draw them by hand.

`SR` below means `python -m sprite_rigger` run from the repository root.

## Workflow

1. Settle the brief before drawing: size (whole figure height in px), view(s), palette, style, what
   must read at a glance (the silhouette). If the project already has sprites, match their size,
   palette and outline.
2. Write `art/<name>.recipe.json` (template below). Start with the **silhouette**: big shapes only.
3. Build: `SR draw art/<name>.recipe.json` (add `--only hero_front` while iterating).
4. **Look at it.** Open `art/out/_preview/<sprite>.png` (enlarged, checkerboard = transparent,
   grid lines every pixel, darker every 8). If you cannot see images, or for exact coordinates, run
   `SR ascii art/<name>.recipe.json <sprite>`: the sprite as palette keys with x/y rulers.
5. Fix with small edits (one item, one grid row) and rebuild. Repeat 3-5 until the checklist passes.
6. Poses for animation: list them in `"poses"` and build; the tool writes `front.png`, `back.png`,
   `side.png` (same size) into the output folder, ready for the rig (`run.bat <folder>`).

Errors name the sprite, the item index and the problem ("unknown colour 'G' (did you mean 'g'?)").
One broken sprite does not stop the others.

## Template

```json
{
  "palette": {"k": "#081820", "d": "#346856", "m": "#88c070", "l": "#e0f8d0"},
  "ramps": ["kdml"],
  "sprites": {
    "_eye": {"grid": ["lk"]},
    "hero_front": {"w": 32, "h": 40, "mirror": "x", "outline": "k", "items": [
      {"ellipse": [15.5, 11, 9, 8, "m"]},
      {"grid": [
        "......kkkk",
        "....kkmmmm",
        "...kmmmlll"], "x": 6, "y": 2},
      {"rect": [10, 19, 12, 10, "d"]},
      {"stamp": ["_eye", 11, 11]},
      {"shade": [[0, 24, 32, 16]]}
    ]},
    "hero_front_red": {"base": "hero_front", "swap": {"m": "r"}}
  },
  "poses": {"front": "hero_front", "back": "hero_back", "side": "hero_side"}
}
```

## Format

Colours: a palette key, `"#rrggbb"`, `"."` (leave as is), `"_"` (erase). Keys are one character
(not `. _ #` or space). `ramps` list keys dark to light; `shade`/`light` step along them.

Sprite fields: `w`, `h` (or inferred from `grid`), `items` (run in order), `grid` (drawn first),
`base` (start from another sprite), then post-processing in this order: `mirror` (`"x"` copies the
left half to the right; `"xc"` shares the centre column for odd widths; `"y"`), `swap`
(`{"m": "r"}` recolour), `flip` (`"x"`, `"y"`), `outline` (key: 1 px around every opaque pixel;
`outline8: true` fills diagonals too), `crop: true`. The canvas never grows: **leave a 1 px free
border** for the outline. Names starting with `_` are parts: usable with `stamp`, not exported.

Items (each object has exactly one op):

| op | value |
|---|---|
| `grid` | list of row strings, one char per pixel; `"x"`, `"y"` offset; `.` and space skip |
| `px` | `[x, y, c]` |
| `rect` / `box` | `[x, y, w, h, c]` filled / 1 px border |
| `line` | `[x0, y0, x1, y1, c, width?]` (width > 1 = round brush: limbs, poles) |
| `circ` / `ring` | `[cx, cy, r, c]` |
| `ellipse` | `[cx, cy, rx, ry, c, ring?]` (.5 centres for even widths) |
| `poly` | `[[x0, y0, x1, y1, ...], c]` filled, vertices on pixel corners |
| `fill` | `[x, y, c]` flood fill |
| `dither` | `[x, y, w, h, c1, c2, density]` |
| `shade` / `light` | `[region, steps?]` along the ramps; region `[x,y,w,h]`, `[cx,cy,r]` or polygon |
| `recolor` | `[{"a": "b"}, region?]` |
| `stamp` | `["_part", x, y, {"flip": "x"}?]` |

## Small-sprite craft (24-64 px figures)

- **Silhouette first.** Fill the whole body in one mid colour and check it reads as the character
  at 1x. If it does not, no shading will save it. Exaggerate: big head, readable hands and feet.
- **Proportions for chibi RPG characters** (Undertale / OneShot / FF6 look): head 1/3 to 2/5 of the
  height, torso about as tall as the head, short legs. Eyes 1-2 px, set low on the face.
- **Few colours, clear ramps.** 3-4 tones per material. One light direction for everything (top
  left by default): light on top/left edges, shadow under the head, under arms, on the right side.
- **Outlines**: `outline` on the sprite, darkest key. Inside the body, separate parts with a line
  of the darkest or second-darkest colour only where they overlap (arm over torso, legs).
- **Clean lines.** Curves step evenly (1-1-2-2-3, never 1-3-1). No "doubles" (L-shaped corners
  of 3 pixels on a 1 px line), no lone pixels of a colour, no pillow shading (dark rim all round).
- **No noise.** Every mark needs a reason: a seam, a fold, a highlight. Flat colour is fine.
- **Symmetry**: draw front and back views with `mirror: "x"` and add asymmetric details after
  (a strap, a scar) by giving them their own `items` in a variant via `base`.

## Poses that animate well (sprite-rigger)

The rig cuts the sprite where the outline separates parts, so draw for it:

- **Same canvas size and feet line** for every view; the figure centred.
- **Close every part with the outline**: head, torso, each arm, hand, leg, foot. A 1 px line
  between touching parts is what makes them separate parts.
- **Side view: keep limbs apart.** The arm hangs slightly in front of or behind the body, not over
  the leg; draw the far leg too (one shade darker) instead of hiding it behind the near one.
- Front and back: arms a pixel or two away from the torso, legs with a gap between them.

## Checklist before you call it done

1. Silhouette reads at 1x; size matches the brief (measure in the ascii view).
2. Only palette colours; 3-4 tones per material; one light direction.
3. Closed outline, no gaps, no doubles, no lone pixels.
4. Proportions consistent across front/back/side; feet on the same row.
5. For poses: parts separated as above.
Then show the preview to the user and ask what to change.
