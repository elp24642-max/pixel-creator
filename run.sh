#!/bin/sh
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv && .venv/bin/python -m pip install -q -r requirements.txt
fi
exec .venv/bin/python -m sprite_rigger.server "$@"
