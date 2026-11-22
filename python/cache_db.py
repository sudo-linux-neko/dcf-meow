# cache_db.py — SQLite WAL cache (fast offline + quota guard)
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): saves downloaded prices/facts to a tiny
# database on disk, so repeat views are instant and we don't hammer
# SEC (10 req/s limit) or Stooq (daily quota). WAL mode = reads never block writes.

from __future__ import annotations
import sqlite3
import time
from pathlib import Path

# Where cache lives: ~/.cache/dcf-meow/dcf_cache.db (XDG-friendly, no root needed).
CACHE_DIR = Path.home() / ".cache" / "dcf-meow"
DB_PATH = CACHE_DIR / "dcf_cache.db"
# How long entries stay fresh: prices 24h, SEC facts 7 days (filings don't change intraday much).
TTL_PRICE = 24 * 3600
TTL_SEC = 7 * 24 * 3600


def _conn() -> sqlite3.Connection:
    """Open DB, create table, enable WAL for speed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL;")  # WAL: readers + writer in parallel, crash-safe
    c.execute("PRAGMA synchronous=NORMAL;")  # faster commits, still safe with WAL
    c.execute("PRAGMA cache_size=-64000;")  # 64MB page cache (fits in 7.6GB RAM easily)
    c.execute("""CREATE TABLE IF NOT EXISTS kv(
        k TEXT PRIMARY KEY, v BLOB, ts REAL)""")
    return c
# Above: opens/creates the cache file. WAL + big cache = max speed on this laptop.


def get(key: str, ttl: float) -> bytes | None:
    """Return cached bytes if fresh, else None. LIVE_ONLY=True -> always None (fresh fetch)."""
    try:
        from .live import LIVE_ONLY
        if LIVE_ONLY:
            return None  # your rule: never use old data, fetch new every single time
    except Exception:
        pass
    try:
        c = _conn()
        row = c.execute("SELECT v, ts FROM kv WHERE k=?", (key,)).fetchone()
        c.close()
        if not row:
            return None
        v, ts = row
        if time.time() - float(ts) > ttl:
            return None  # stale: caller will refetch
        return bytes(v)
    except Exception:
        return None
# Above: lookup. Returns None on miss/stale/error so caller always falls back to network.


def put(key: str, value: bytes) -> None:
    """Store bytes under key (upsert). LIVE_ONLY=True -> no-op (no old data kept)."""
    try:
        from .live import LIVE_ONLY
        if LIVE_ONLY:
            return
    except Exception:
        pass
    try:
        c = _conn()
        c.execute("INSERT OR REPLACE INTO kv(k,v,ts) VALUES(?,?,?)",
                  (key, bytes(value), time.time()))
        c.commit()
        c.close()
    except Exception:
        pass
# Above: save. Silently ignores errors (cache must never crash valuation).
