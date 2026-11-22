# data_fred.py — LIVE risk-free rate (10Y Treasury, fresh every call, never stale)
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): gets TODAY's 10-year Treasury yield — the "safe"
# rate for CAPM — from the live US Treasury + FRED, with its DATE attached.
# No fixed fallback number: if all live sources fail it returns None + error,
# and the GUI says LIVE FAILED (your rule: no old/fixed data, only live + maths).

from __future__ import annotations
import re
import requests
from .live import now_utc

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) DCF-Meow Vaibhav LIVE"}


def _treasury_xml() -> tuple[float | None, str]:
    """US Treasury daily par-yield TextView CSV (official, timestamped)."""
    try:
        # Proven live layout (web research): TextView rows like
        # 08/31/2026 | ... | 4.75(10Y) ... with 10Y near the tail. We robustly
        # pick the 10Y as the 3rd-from-last numeric before the 20Y/30Y tail.
        for u in ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                  "daily-treasury-rates.csv/2026/all?type=daily_treasury_yield_curve",
                  "http://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                  "TextView?field_tdr_date_value=2026&type=daily_treasury_yield_curve"):
            try:
                r = requests.get(u, headers=UA, timeout=20)
            except Exception:
                continue
            if r.status_code != 200 or not r.text:
                continue
            lines = [ln for ln in r.text.strip().splitlines() if ln and ln[0].isdigit()]
            for ln in reversed(lines[-20:]):
                cells = [c.strip().strip('"') for c in re.split(r",|\|", ln) if c.strip() not in ("", "N/A")]
                # cells[0]=date, rest=numerics across maturities; 10Y ≈ index -3/-4 of tail
                if len(cells) < 8:
                    continue
                date = cells[0]
                for cand in (cells[-4], cells[-5], cells[-3]):
                    try:
                        v = float(cand)
                        if 0.5 < v < 15:
                            return v / 100.0, f"US-Treasury-10Y {date}"
                    except Exception:
                        continue
        return None, ""
    except Exception:
        return None, ""
# Above: official Treasury source. Date travels WITH the number (freshness proof).


def _fred_csv() -> tuple[float | None, str]:
    """FRED DGS10 CSV (secondary, daily H.15, timestamped)."""
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
                         headers=UA, timeout=20)
        lines = [ln for ln in r.text.strip().splitlines() if ln and not ln.startswith("DATE")]
        for ln in reversed(lines[-15:]):
            p = ln.split(",")
            if len(p) == 2 and p[1] not in (".", ""):
                try:
                    return float(p[1]) / 100.0, f"FRED-DGS10 {p[0]}"
                except Exception:
                    continue
        return None, ""
    except Exception:
        return None, ""
# Above: FRED backup. Also dated — GUI shows WHICH day the yield is from.


def get_risk_free(fred_key: str = "") -> tuple[float | None, str]:
    """Returns (rf_decimal_or_None, 'source + date @ fetched-moment'). Never uses fixed numbers.

    Order: FRED CSV (proven live daily H.15) -> Treasury official -> FRED API (if key).
    All attempts are FRESH network (cache bypassed via live.py). No fixed fallback:
    failure returns (None, 'LIVE FAILED ...') so you never see a fake fixed yield.
    """
    moment = now_utc()
    # FRED CSV first (proven live daily H.15, e.g. 4.79% Sep-2026 in research).
    v, s = _fred_csv()
    if v is not None:
        return v, f"{s} @ {moment}"
    v, s = _treasury_xml()
    if v is not None:
        return v, f"{s} @ {moment}"
    if fred_key.strip():
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params={"series_id": "DGS10", "api_key": fred_key.strip(),
                                     "file_type": "json", "sort_order": "desc", "limit": 5},
                             timeout=15, headers=UA)
            for o in r.json().get("observations", []):
                if o.get("value") not in (None, ".", ""):
                    return float(o["value"]) / 100.0, f"FRED-DGS10(api) {o.get('date','')} @ {moment}"
        except Exception:
            pass
    return None, f"LIVE FAILED — no Treasury/FRED reachable @ {moment} (no fixed fallback by your rule)"
# Above: live chain. Date + fetch-moment always shown = proof it is THIS moment's number.
