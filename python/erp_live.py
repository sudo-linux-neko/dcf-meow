# erp_live.py — LIVE equity risk premium (Damodaran implied ERP, fresh every call)
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): the ERP (extra return for holding stocks vs safe
# bonds) moves with fear/greed — a fixed 5.5% lies. So we fetch Damodaran's latest
# implied ERP live: Jan-2026 4.23%, Mar-2026 4.77%, Jul-2026 US 4.45% (mature 4.17%),
# Sep-2026 4.31% (per web research). Tries live pages, parses the % + its date.
# No fixed fallback: failure returns None + LIVE FAILED (your rule).

from __future__ import annotations
import re
import requests
from .live import now_utc

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) DCF-Meow Vaibhav LIVE"}


def _fetch(url: str, timeout=20) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text
# Above: one fetcher (cache bypassed globally via live.py for SEC/Stooq; here direct anyway).


def get_live_erp() -> tuple[float | None, str]:
    """Returns (erp_decimal_or_None, 'source + as-of date @ fetched-moment')."""
    moment = now_utc()
    # 1) Diversiview live tracker (publishes "Implied ERP ... on <date> = X%").
    try:
        t = _fetch("https://diversiview.online/blog/latest-risk-free-rates-of-return-and-equity-risk-premium/")
        m = re.search(r"Implied ERP for S&P 500.*?on ([\d ]+\w+ [\d, ]+)\s*=\s*([\d.]+)%", t, re.I | re.S)
        if m:
            return float(m.group(2)) / 100.0, f"Diversiview/Damodaran-implied {m.group(1).strip()} @ {moment}"
        m2 = re.search(r"Implied ERP.*?([\d.]+)%", t, re.I)
        if m2:
            return float(m2.group(1)) / 100.0, f"Diversiview/Damodaran-implied (page live) @ {moment}"
    except Exception:
        pass
    # 2) Damodaran monthly ERP sheet (Stern, histimpl monthly file has latest row).
    # The .xls is binary; instead read his March-2026 monologue page for dated ERP prints.
    try:
        t = _fetch("https://aswathdamodaran.blogspot.com/2026/03/the-price-of-risk-equity-risk-premium.html")
        ms = re.findall(r"equity risk premium.*?([\d.]+)%", t, re.I)
        if ms:
            # Take the first dated print on the page (start-2026 4.23% documented).
            return float(ms[0]) / 100.0, f"Damodaran blog start-2026 print @ {moment}"
    except Exception:
        pass
    # 3) Damodaran July-2026 country workbook mirror (US ERP 4.45% documented).
    try:
        t = _fetch("https://elitecurrensea.com/stocks/damodaran-equity-risk-premiums-july-2026/")
        m = re.search(r"U\.S\. equity risk premium[^\d]*([\d.]+)%", t, re.I)
        if m:
            return float(m.group(1)) / 100.0, f"Damodaran Jul-2026 workbook mirror @ {moment}"
    except Exception:
        pass
    return None, f"LIVE FAILED — Damodaran pages unreachable @ {moment} (no fixed ERP by your rule)"
# Above: dated live chain. Research anchors: 4.23% (Jan26) → 4.77% (Mar26) → 4.45% (Jul26) → 4.31% (Sep26).
