"""Model access: the rig builder and motion writer talk to any of these.

  claude-code   your installed `claude` CLI (uses your own sign-in), no key needed
  anthropic     Anthropic API key
  openai        any OpenAI-compatible server: OpenAI, OpenRouter, Gemini's OpenAI
                endpoint, DeepSeek, and local servers (Ollama, LM Studio, llama.cpp)

Only the standard library is used for HTTP, so nothing extra has to be installed.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Provider:
    kind: str                   # claude-code | anthropic | openai
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout: int = 300
    usage: Usage = field(default_factory=Usage)

    def ask(self, prompt: str, images: list[bytes] | None = None, system: str = "") -> str:
        images = images or []
        fn = {"claude-code": self._claude_code, "anthropic": self._anthropic, "openai": self._openai}[self.kind]
        self.usage.requests += 1
        return fn(prompt, images, system)

    # -- claude code CLI ------------------------------------------------------
    def _claude_code(self, prompt, images, system):
        exe = shutil.which("claude")
        if not exe:
            raise RuntimeError("The `claude` command was not found. Install Claude Code and sign in, "
                               "or pick another connection.")
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, b in enumerate(images):
                p = Path(tmp) / f"image_{i}.png"
                p.write_bytes(b)
                paths.append(p)
            text = prompt
            if paths:
                text = ("Look at these image files first (use your Read tool on each):\n"
                        + "\n".join(str(p) for p in paths) + "\n\n" + prompt)
            cmd = [exe, "-p", text, "--output-format", "json", "--allowedTools", "Read",
                   "--add-dir", tmp]
            if system:
                cmd += ["--append-system-prompt", system]
            if self.model:
                cmd += ["--model", self.model]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=tmp)
        if r.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {r.stderr.strip() or r.stdout.strip()}")
        try:
            data = json.loads(r.stdout)
            u = data.get("usage", {})
            self.usage.input_tokens += int(u.get("input_tokens", 0))
            self.usage.output_tokens += int(u.get("output_tokens", 0))
            return data.get("result", "")
        except json.JSONDecodeError:
            return r.stdout

    # -- Anthropic API ---------------------------------------------------------
    def _anthropic(self, prompt, images, system):
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("No Anthropic API key.")
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.b64encode(b).decode()}} for b in images]
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model or "claude-sonnet-5-5", "max_tokens": 8000,
                "messages": [{"role": "user", "content": content}]}
        if system:
            body["system"] = system
        url = (self.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        data = _post(url, body, {"x-api-key": key, "anthropic-version": "2023-06-01"}, self.timeout)
        u = data.get("usage", {})
        self.usage.input_tokens += int(u.get("input_tokens", 0))
        self.usage.output_tokens += int(u.get("output_tokens", 0))
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    # -- OpenAI-compatible -----------------------------------------------------
    def _openai(self, prompt, images, system):
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}
                   for b in images]
        content.append({"type": "text", "text": prompt})
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        data = _post(url, {"model": self.model, "messages": msgs}, headers, self.timeout)
        u = data.get("usage", {}) or {}
        self.usage.input_tokens += int(u.get("prompt_tokens", 0))
        self.usage.output_tokens += int(u.get("completion_tokens", 0))
        return data["choices"][0]["message"]["content"] or ""


def _post(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None


def extract_json(text: str):
    """The last JSON object in a reply (models like to wrap it in prose or ``` fences)."""
    for m in reversed(list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S))):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    depth, start = 0, None
    found = []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                found.append(text[start:i + 1])
    for s in reversed(found):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object in the model's reply")
