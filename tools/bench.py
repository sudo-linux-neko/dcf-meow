#!/usr/bin/env python3
# bench.py — prove AVX2 speedup on YOUR i5-1335U
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): times the C++ AVX2 math vs Numba vs NumPy
# on this exact CPU, so you can SEE the "super optimized" claim is real.

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Above: lets us import python.dcf_core from tools/ folder.

import numpy as np

from python import dcf_core


def bench_grid():
    """32x32 sensitivity grid, 20 repeats. Prints ms + speedup."""
    w = np.linspace(0.06, 0.14, 32)
    g = np.linspace(0.02, 0.12, 32)
    # Warm up (compile Numba, fault pages, boost CPU).
    dcf_core.grid(w, g, 1e10, 0.025, 10, 5e9, 15e9)
    t0 = time.perf_counter()
    for _ in range(20):
        dcf_core.grid(w, g, 1e10, 0.025, 10, 5e9, 15e9)
    dt = (time.perf_counter() - t0) / 20 * 1000
    print(f"grid 32x32: {dt:.2f} ms  engine={'AVX2+FMA+OpenMP' if dcf_core.has_avx2() else 'Numba/NumPy fallback'}")
    return dt
# Above: the heatmap workload the GUI does on every slider move. Must stay <16ms for 60fps.


def bench_mc():
    """10k Monte Carlo paths. Prints ms."""
    dcf_core.montecarlo(10000, 1e10, 0.08, 0.02, 0.09, 0.015, 0.025, 10, 5e9, 15e9)
    t0 = time.perf_counter()
    for _ in range(5):
        dcf_core.montecarlo(10000, 1e10, 0.08, 0.02, 0.09, 0.015, 0.025, 10, 5e9, 15e9)
    dt = (time.perf_counter() - t0) / 5 * 1000
    print(f"montecarlo 10k: {dt:.2f} ms")
    return dt
# Above: risk-band workload. C++ target ~1ms on i5-1335U.


def bench_single():
    """1M single valuations (batch loop). Shows ns/op."""
    t0 = time.perf_counter()
    for _ in range(200000):
        dcf_core.fair_value(1e10, 0.08, 0.09, 0.025, 10, 5e9, 15e9)
    dt = (time.perf_counter() - t0) / 200000 * 1e9
    print(f"single fair_value: {dt:.0f} ns/op")
# Above: raw call overhead. C++ scalar should be <200ns.


if __name__ == "__main__":
    print(f"AVX2 lib live: {dcf_core.has_avx2()}  numpy {np.__version__}")
    bench_single()
    bench_grid()
    bench_mc()
    print("OK — if AVX2 live, grid should be <2ms and MC <5ms on i5-1335U.")
# Above: run with: source ~/ml_env/bin/activate && python tools/bench.py
