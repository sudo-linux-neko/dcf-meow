#!/usr/bin/env bash
# run.sh — one-command build + bench + GUI for DCF-Meow
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple): activates ml_env (mandatory), builds AVX2 lib with all 12 cores, proves speed, opens GUI.
set -e
# 1) Must use ml_env only (your rule: download libs only here, never break system).
if [ -f "$HOME/ml_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/ml_env/bin/activate"
else
  echo "ERROR: ~/ml_env not found. Create it first." >&2
  exit 1
fi
cd "$(dirname "$0")"
echo "[1/3] Building AVX2 core (-O3 -march=native -mavx2 -mfma) with $(nproc) threads..."
make -j"$(nproc)"
echo "[2/3] Benchmarking on YOUR CPU (i5-1335U)..."
python tools/bench.py || echo "(bench warning, continuing)"
echo "[3/3] Launching GUI (PySide6 + pyqtgraph)..."
exec python gui/app.py "$@"
