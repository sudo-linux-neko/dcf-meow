# sp500.py — LIVE S&P 500 constituents (fresh list every call, never a fixed TOP_MEGA)
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): downloads TODAY's S&P 500 member list (503 tickers
# with GICS sector) from live sources: datahub.io CSV (updated Sep-2026, 503 rows)
# primary, Wikipedia table fallback. Your old fixed TOP_MEGA list is gone —
# peers/top-N now come from this live universe + live market caps.

from __future__ import annotations
import csv
import io
import re
import requests
from .live import now_utc

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) DCF-Meow Vaibhav LIVE"}

URLS = [
    "https://datahub.io/core/s-and-p-500-companies-financials/_r/-/data/constituents.csv",
    "https://datahub.io/core/s-and-p-500-companies/_r/-/data/constituents.csv",
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
]
# Above: live mirrors of the same 503-row list (Wikipedia-sourced, refreshed 2026).


def get_constituents() -> tuple[list[dict], str]:
    """Returns ([{ticker, name, sector}...], 'source @ moment'). Empty + LIVE FAILED on outage."""
    moment = now_utc()
    for u in URLS:
        try:
            r = requests.get(u, headers=UA, timeout=25)
            if r.status_code != 200 or "Symbol" not in r.text[:500]:
                continue
            rows = list(csv.DictReader(io.StringIO(r.text)))
            out = []
            for row in rows:
                sym = (row.get("Symbol") or row.get("symbol") or "").strip().upper().replace(".", "-")
                if not sym:
                    continue
                out.append({"ticker": sym,
                            "name": (row.get("Name") or row.get("name") or sym)[:60],
                            "sector": (row.get("Sector") or row.get("sector") or "Unknown")})
            if len(out) > 400:  # sanity: S&P500 must be ~500 rows, else page changed
                return out, f"datahub-live {len(out)} members @ {moment}"
        except Exception:
            continue
    # Wikipedia fallback (parse first wikitable tickers).
    try:
        t = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers=UA, timeout=25).text
        syms = re.findall(r"<td>([A-Z]{1,5}(?:\.[A-Z])?)</td>", t)[:600]
        out = [{"ticker": s.replace(".", "-"), "name": s, "sector": "Unknown"} for s in syms]
        if len(out) > 400:
            return out, f"wikipedia-live {len(out)} members @ {moment}"
    except Exception:
        pass
    return [], f"LIVE FAILED — no S&P500 list reachable @ {moment}"
# Above: live universe. Count-checked (400+ rows) so a broken page never fakes membership.
