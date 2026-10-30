@echo off
rem Adds JAX (about 100 MB) so the app can tune the refiner to your characters.
cd /d "%~dp0"
if not exist .venv python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-train.txt
