# live.py — LIVE-ONLY switch: only maths is fixed, everything else is fetched fresh
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): your rule — "only maths formulas are fixed,
# everything else must be live at that moment, fetch new data every single time,
# never use old data". So LIVE_ONLY=True bypasses ALL disk cache. Each fetch
# stamps its moment (UTC) so the GUI can prove freshness.

from __future__ import annotations
import datetime

# Master switch. True = ignore cache_db completely (fresh network every call).
# Only the DCF/MC MATH (cpp/dcf_avx2) stays fixed — that is the formula, not data.
LIVE_ONLY = True


def now_utc() -> str:
    """Current moment stamp, e.g. 2026-10-01 16:40:22 UTC. Shown in GUI as LIVE proof."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
# Above: one clock for all tabs. If two fetches show different stamps, data is fresh.
