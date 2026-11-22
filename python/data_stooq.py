# data_stooq.py — Max-data price fetcher: Stooq primary + Yahoo fallback
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): downloads the LONGEST price history possible.
# Tries Stooq first (21k tickers, 30+ years, daily/weekly/monthly) because you asked
# for max data from Stooq. Since early-2026 Stooq needs a free apikey (CAPTCHA at
# stooq.com/q/d/?s=spy.us&get_apikey) + JS challenge, we support apikey AND
# auto-fallback to Yahoo Chart v8 (no key, tested 200 OK) so the app NEVER breaks.

from __future__ import annotations
import csv
import io
import time
import requests
from . import cache_db

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) DCF-Meow Vaibhav"}


def _norm(ticker: str) -> str:
    """Normalize: AAPL -> aapl.us, ^SPX -> ^spx, EURUSD stays."""
    t = ticker.strip()
    if not t:
        return "aapl.us"
    if t.startswith("^"):
        return t.lower()
    if "." not in t:
        return t.lower() + ".us"
    return t.lower()
# Above: Stooq needs exchange suffix (.us for NYSE/NASDAQ). Adds it if missing.


def fetch_stooq(ticker: str, interval: str = "d", apikey: str = "") -> list[dict] | None:
    """Try Stooq q/d/l CSV. Returns list of dicts or None if blocked.

    URL: https://stooq.com/q/d/l/?s=aapl.us&d1=19900101&d2=20261001&i=d[&apikey=...]
    Response columns: Date,Open,High,Low,Close,Volume.
    Since 2026: without apikey you get a JS-PoW HTML page (we detect '<html').
    """
    s = _norm(ticker)
    url = f"https://stooq.com/q/d/l/?s={s}&d1=19900101&d2=20261001&i={interval}"
    if apikey.strip():
        url += f"&apikey={apikey.strip()}"
    key = "stooq:" + url
    hit = cache_db.get(key, cache_db.TTL_PRICE)
    raw = None
    if hit:
        raw = hit.decode("utf-8", "ignore")
    else:
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code == 200 and r.text and "<html" not in r.text[:500].lower():
                raw = r.text
                cache_db.put(key, r.content)
            else:
                return None  # JS challenge / quota / apikey needed -> fallback
        except Exception:
            return None
    try:
        rows = list(csv.DictReader(io.StringIO(raw)))
        out = []
        for row in rows:
            try:
                out.append({"date": row["Date"], "open": float(row["Open"]),
                            "high": float(row["High"]), "low": float(row["Low"]),
                            "close": float(row["Close"]), "volume": float(row.get("Volume") or 0)})
            except Exception:
                continue
        return out or None
    except Exception:
        return None
# Above: Stooq primary. Detects block pages, caches good CSVs for 24h.


def fetch_yahoo(ticker: str, rng: str = "10y") -> list[dict] | None:
    """Yahoo Chart v8 fallback (no key). interval=1d, range up to max."""
    base = ticker.strip().upper().split(".")[0].replace("^", "%5EGSPC" if ticker.strip().upper() in ("^SPX", "^GSPC") else ticker.strip())
    # Map common indices: ^SPX -> %5EGSPC for Yahoo.
    sym = ticker.strip().upper().split(".")[0]
    if sym in ("^SPX", "^GSPC", "SPX"):
        sym = "%5EGSPC"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range={rng}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 DCF-Meow"}, timeout=20)
        j = r.json()
        res = j["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        out = []
        for i, t in enumerate(ts):
            try:
                out.append({"date": time.strftime("%Y-%m-%d", time.gmtime(t)),
                            "open": float(q["open"][i] or 0), "high": float(q["high"][i] or 0),
                            "low": float(q["low"][i] or 0), "close": float(q["close"][i] or 0),
                            "volume": float(q["volume"][i] or 0)})
            except Exception:
                continue
        out = [x for x in out if x["close"] > 0]
        return out or None
    except Exception:
        return None
# Above: Yahoo safety net. Same dict shape as Stooq so GUI code is identical.


def fetch_prices(ticker: str, apikey: str = "") -> tuple[list[dict], str]:
    """Max-data fetch: Stooq daily -> Stooq weekly -> Yahoo 10y -> Yahoo max. Returns (rows, source)."""
    for iv, label in (("d", "Stooq-daily"), ("w", "Stooq-weekly")):
        r = fetch_stooq(ticker, iv, apikey)
        if r and len(r) > 30:
            return r, label + ("+key" if apikey else "")
    for rng in ("10y", "max"):
        r = fetch_yahoo(ticker, rng)
        if r and len(r) > 30:
            return r, f"Yahoo-{rng}(Stooq blocked: need free apikey)"
    return [], "none (offline? check network)"
# Above: tries longest histories first. Prefers Stooq (your wish), falls back silently.
# GUI shows `source` so you always know where numbers came from.


def beta_vs_spy(stock: list[dict], spy: list[dict]) -> float:
    """CAPM beta = Cov(stock,spy)/Var(spy) on daily returns (last 2y, min 60 pts)."""
    try:
        import numpy as np
        sc = np.array([x["close"] for x in stock[-504:]], dtype=np.float64)
        pc = np.array([x["close"] for x in spy[-504:]], dtype=np.float64)
        n = min(len(sc), len(pc))
        if n < 60:
            return 1.0
        sc, pc = sc[-n:], pc[-n:]
        rs = sc[1:] / sc[:-1] - 1.0
        rm = pc[1:] / pc[:-1] - 1.0
        v = float(np.var(rm))
        if v < 1e-12:
            return 1.0
        b = float(np.cov(rs, rm, bias=True)[0, 1] / v)
        return max(0.2, min(3.0, b))  # clamp crazy penny-stock betas
    except Exception:
        return 1.0
# Above: beta for WACC. Clamped 0.2–3.0 so one bad tick can't explode valuation.
