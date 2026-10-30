"""Local web app: python -m sprite_rigger.server [project_dir] [--port 8765]

Standard library HTTP server + one HTML page. Nothing leaves your computer except the
requests to the model connection you choose.

Project folder layout:
  rig.json            views: image + parts (the skeleton)
  <view>.png          the drawn poses
  motions/<name>.json motions
  variants/...        alternative part drawings
  settings.json       render + connection settings (API key stays in this file only)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import traceback
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from . import autorig
from .ai import Provider
from .export import crop_box, save_gif, to_cells
from .motion import animate
from .rig import build_view
from .segment import region_info, segment
from .viz import regions_overlay

HERE = Path(__file__).parent
PRESETS = HERE / "presets"
DEFAULT_SETTINGS = {
    "frames": 8, "fps": 8, "amplitude": 1.0, "margin": 4, "cell": None,
    "clean": True, "refine": False, "preview_scale": 4,
    "auto_fix_limit": 3,
    "ai": {"kind": "claude-code", "model": "", "api_key": "", "base_url": ""},
}


def b64png(img) -> str:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img, "RGBA")
    b = io.BytesIO()
    img.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


def unb64(s: str) -> np.ndarray:
    if "," in s[:64]:
        s = s.split(",", 1)[1]
    return np.array(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGBA"))


class Project:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "motions").mkdir(exist_ok=True)
        (self.root / "variants").mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self._refiner = None
        self._usage = None

    # -- files
    def rig(self) -> dict:
        p = self.root / "rig.json"
        return json.loads(p.read_text()) if p.exists() else {"name": self.root.name, "views": {}}

    def save_rig(self, rig):
        (self.root / "rig.json").write_text(json.dumps(rig, indent=1))

    def settings(self) -> dict:
        p = self.root / "settings.json"
        s = json.loads(json.dumps(DEFAULT_SETTINGS))
        if p.exists():
            got = json.loads(p.read_text())
            s.update({k: v for k, v in got.items() if k != "ai"})
            s["ai"].update(got.get("ai", {}))
        return s

    def save_settings(self, s):
        (self.root / "settings.json").write_text(json.dumps(s, indent=1))

    def motions(self) -> dict:
        out = {}
        for p in sorted(PRESETS.glob("*.json")):
            out["preset:" + p.stem] = json.loads(p.read_text())
        for p in sorted((self.root / "motions").glob("*.json")):
            out[p.stem] = json.loads(p.read_text())
        return out

    def image(self, view) -> np.ndarray:
        return np.array(Image.open(self.root / self.rig()["views"][view]["image"]).convert("RGBA"))

    def view(self, name):
        spec = self.rig()["views"][name]
        return build_view(name, self.image(name), {**spec, "_dir": str(self.root)})

    def provider(self) -> Provider:
        a = self.settings()["ai"]
        p = Provider(a["kind"], a.get("model", ""), a.get("api_key", ""), a.get("base_url", ""))
        if self._usage is not None:
            p.usage = self._usage
        self._usage = p.usage
        return p

    def refiner(self):
        if self._refiner is None:
            for p in (self.root / "refiner.npz", HERE / "models" / "refiner.npz"):
                if p.exists():
                    from .refiner import Refiner
                    self._refiner = Refiner(p)
                    break
        return self._refiner

    def frames(self, view, motion, s, with_touched=False):
        ref = self.refiner() if s.get("refine") else None
        fr, touched = animate(self.view(view), motion, int(s["frames"]), float(s["amplitude"]), int(s["margin"]),
                              clean=bool(s["clean"]), refiner=ref, with_touched=True)
        patches = motion.get("patches", {}).get(str(len(fr)), {})
        for i, plist in patches.items():
            f = fr[int(i)]
            for x, y, r, g, b, a in plist:
                if 0 <= y < f.shape[0] and 0 <= x < f.shape[1]:
                    f[y, x] = (r, g, b, a)
        return (fr, touched) if with_touched else fr


class Handler(BaseHTTPRequestHandler):
    project: Project = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (HERE / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self._send(200, self.state())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        fn = getattr(self, "api_" + self.path.removeprefix("/api/").replace("/", "_"), None)
        if fn is None:
            return self._send(404, {"error": f"unknown endpoint {self.path}"})
        try:
            with self.project.lock:
                self._send(200, fn(req) or {"ok": True})
        except Exception as e:  # report every failure to the page instead of a dead request
            traceback.print_exc()
            self._send(400, {"error": str(e)})

    # -- endpoints
    def state(self):
        P = self.project
        rig = P.rig()
        views = {}
        for v, spec in rig["views"].items():
            try:
                views[v] = {"image": b64png(P.image(v)), "parts": spec.get("parts", {})}
            except Exception as e:
                views[v] = {"error": str(e)}
        s = P.settings()
        s_public = {**s, "ai": {**s["ai"], "api_key": "•••" if s["ai"].get("api_key") else ""}}
        u = P._usage
        return {"project": str(P.root), "views": views, "motions": P.motions(), "settings": s_public,
                "usage": u.__dict__ if u else {"requests": 0, "input_tokens": 0, "output_tokens": 0},
                "refiner": P.refiner() is not None}

    def api_settings(self, req):
        s = self.project.settings()
        ai = req.pop("ai", None)
        s.update(req)
        if ai:
            if ai.get("api_key") == "•••":
                ai.pop("api_key")
            s["ai"].update(ai)
        self.project.save_settings(s)

    def api_pose(self, req):
        view = req["view"]
        img = unb64(req["png"])
        name = f"{view}.png"
        Image.fromarray(img, "RGBA").save(self.project.root / name)
        rig = self.project.rig()
        rig["views"].setdefault(view, {"image": name, "parts": {}})["image"] = name
        self.project.save_rig(rig)

    def api_regions(self, req):
        img = self.project.image(req["view"])
        seg = segment(img)
        return {"png": b64png(regions_overlay(img, seg["masks"], 12)), "regions": region_info(seg["masks"])}

    def api_rig(self, req):
        rig = self.project.rig()
        rig["views"][req["view"]]["parts"] = req["parts"]
        build_view(req["view"], self.project.image(req["view"]), {**rig["views"][req["view"]], "_dir": str(self.project.root)})
        self.project.save_rig(rig)

    def api_ai_rig(self, req):
        P = self.project
        spec = autorig.build_rig(P.provider(), P.image(req["view"]), req["view"], req.get("notes", ""))
        rig = P.rig()
        rig["views"][req["view"]]["parts"] = spec["parts"]
        P.save_rig(rig)
        return {"parts": spec["parts"]}

    def api_motion(self, req):
        name = req["name"].removeprefix("preset:")
        (self.project.root / "motions" / f"{name}.json").write_text(json.dumps(req["motion"], indent=1))
        return {"name": name}

    def api_ai_motion(self, req):
        P = self.project
        spec = P.rig()["views"][req["view"]]
        m = autorig.write_motion(P.provider(), spec, req["view"], req["name"], req.get("notes", ""))
        (P.root / "motions" / f"{req['name']}.json").write_text(json.dumps(m, indent=1))
        return {"name": req["name"], "motion": m}

    def api_render(self, req):
        P = self.project
        s = {**P.settings(), **req.get("settings", {})}
        fr, touched = P.frames(req["view"], req["motion"], s, with_touched=True)
        return {"frames": [b64png(f) for f in fr], "touched": [b64png(np.dstack([t * 255] * 3 + [t * 160]).astype(np.uint8)) for t in touched],
                "rest": b64png(np.pad(P.image(req["view"]), ((s["margin"],) * 2, (s["margin"],) * 2, (0, 0))))}

    def api_ai_fix(self, req):
        P = self.project
        s = {**P.settings(), **req.get("settings", {})}
        fr = P.frames(req["view"], req["motion"], s)
        m = autorig.fix_motion(P.provider(), req["motion"], fr, req["notes"], P.rig()["views"][req["view"]]["parts"])
        return {"motion": m}

    def api_ai_autofix(self, req):
        """Let the model review its own frames and revise, at most `auto_fix_limit` times."""
        P = self.project
        s = {**P.settings(), **req.get("settings", {})}
        limit = int(s.get("auto_fix_limit", 3))
        if limit <= 0:
            raise RuntimeError("auto-fix is off (limit 0)")
        m = req["motion"]
        parts = P.rig()["views"][req["view"]]["parts"]
        notes = ("Review these frames as a pixel-art animator: feet sliding on the ground, parts detaching, "
                 "jitter, unnatural timing. " + req.get("notes", "") +
                 " If nothing needs changing, return the motion exactly as it is.")
        rounds = 0
        for rounds in range(1, limit + 1):
            fr = P.frames(req["view"], m, s)
            new = autorig.fix_motion(P.provider(), m, fr, notes, parts)
            if json.dumps(new.get("tracks"), sort_keys=True) == json.dumps(m.get("tracks"), sort_keys=True):
                break
            m = {**m, **new}
        return {"motion": m, "rounds": rounds}

    def api_variant(self, req):
        """Save an alternative drawing of a part (full view-size canvas, only that part painted)."""
        P = self.project
        img = unb64(req["png"])
        rel = f"variants/{req['view']}_{req['part']}_{req['name']}.png"
        Image.fromarray(img, "RGBA").save(P.root / rel)
        rig = P.rig()
        part = rig["views"][req["view"]]["parts"][req["part"]]
        part.setdefault("variants", {})[req["name"]] = {"image": rel}
        P.save_rig(rig)

    def api_export(self, req):
        P = self.project
        s = {**P.settings(), **req.get("settings", {})}
        sets = {}
        for row in req["rows"]:
            sets[row["row"]] = P.frames(row["view"], row["motion"], s)
        for src, dst in req.get("mirror", []):
            if src not in sets:
                raise ValueError(f"mirror: there is no row named {src!r}")
            sets[dst] = [np.ascontiguousarray(f[:, ::-1]) for f in sets[src]]
        box = crop_box(list(sets.values()))
        cell = tuple(s["cell"]) if s.get("cell") else None
        rows = {k: to_cells(v, box, cell) for k, v in sets.items()}
        name = req.get("name") or P.rig().get("name", "sprite")
        w, h = next(iter(rows.values()))[0].size
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            sheet = Image.new("RGBA", (w * max(len(r) for r in rows.values()), h * len(rows)), (0, 0, 0, 0))
            meta = {"sprite": name, "frame_size": [w, h], "fps": s["fps"], "origin": [w // 2, h - 2], "rows": {}}
            for j, (k, cells) in enumerate(rows.items()):
                strip = Image.new("RGBA", (w * len(cells), h), (0, 0, 0, 0))
                for i, c in enumerate(cells):
                    strip.paste(c, (i * w, 0))
                    sheet.paste(c, (i * w, j * h))
                fname = f"{name}_{k}_strip{len(cells)}.png"
                b = io.BytesIO(); strip.save(b, "PNG"); z.writestr(fname, b.getvalue())
                g = io.BytesIO(); save_gif(cells, g, s["fps"], s["preview_scale"], (52, 104, 86, 255))
                z.writestr(f"preview/{name}_{k}.gif", g.getvalue())
                meta["rows"][k] = {"row": j, "frames": len(cells), "file": fname}
            b = io.BytesIO(); sheet.save(b, "PNG"); z.writestr(f"{name}_sheet.png", b.getvalue())
            meta["gamemaker"] = {"import": "Sprite editor > Import > strip image; frames = number in the file name",
                                 "origin": [w // 2, h - 2], "image_speed_at_60fps": round(s["fps"] / 60, 4)}
            z.writestr(f"{name}.json", json.dumps(meta, indent=1))
        out = P.root / "export" / f"{name}.zip"
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(buf.getvalue())
        return {"zip": base64.b64encode(buf.getvalue()).decode(), "path": str(out), "frame": [w, h]}


def main():
    ap = argparse.ArgumentParser(prog="sprite_rigger.server")
    ap.add_argument("project", nargs="?", default="my_sprite")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    Handler.project = Project(Path(a.project))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"sprite-rigger: {url}  (project: {Path(a.project).resolve()})  Ctrl+C to stop")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
