#!/bin/sh
# Adds JAX (about 100 MB) so the app can tune the refiner to your characters.
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-train.txt
