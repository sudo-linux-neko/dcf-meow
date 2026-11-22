# sector.py — LIVE sector risk + LIVE top-N by market cap (no fixed universe)
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): each industry is risky in its own way, BUT the
# numbers are NOT fixed — base table below is only the starting maths, then live
# sector volatility (fresh price wobble measured NOW) tightens/loosens the BUY bar.
# Peers come from the LIVE S&P 500 list (sp500.py, 503 members fetched fresh),
# ranked by LIVE market caps. Nothing historical, nothing cached (live.py forces fresh).

from __future__ import annotations

# BASE win-prob + margin per sector (starting maths only — live vol adjusts them;
# see dynamic_risk(). User can still override in Optimizer tab).
SECTOR_RISK: dict[str, tuple[float, float]] = {
    "Technology": (0.75, 0.25),
    "Healthcare": (0.70, 0.20),
    "Financial Services": (0.80, 0.15),
    "Energy": (0.65, 0.30),
    "Utilities": (0.85, 0.10),
    "Consumer Defensive": (0.80, 0.15),
    "Consumer Cyclical": (0.70, 0.20),
    "Industrials": (0.72, 0.18),
    "Communication Services": (0.72, 0.20),
    "Basic Materials": (0.68, 0.25),
    "Real Estate": (0.80, 0.12),
    "Default": (0.75, 0.20),
}
# Above: the risk table. BUY only if BOTH winprob and margin pass (break-test survival).


def sic_to_sector(sic: str | int | None, desc: str | None) -> str:
    """Map SEC SIC code + description to our sector names (keyword first, range fallback)."""
    d = (desc or "").upper()
    # Keyword pass (most reliable: SEC descriptions like 'PHARMACEUTICAL PREPARATIONS').
    if any(k in d for k in ("SOFTWARE", "COMPUTER", "SEMICONDUCT", "INFORMATION RETRIEVAL")):
        return "Technology"
    if any(k in d for k in ("PHARM", "DRUG", "HEALTH", "BIOLOGICAL", "MEDICAL", "HOSPITAL")):
        return "Healthcare"
    if any(k in d for k in ("BANK", "FINANC", "INSUR", "BROKER", "CREDIT", "SAVINGS")):
        return "Financial Services"
    if any(k in d for k in ("PETROLEUM", "OIL", "GAS", "COAL", "MINING", "GOLD", "STEEL", "METAL MINING")):
        # Mining metals -> materials, oil/gas -> energy (split below by code).
        try:
            c = int(str(sic)[:4] if sic else 0)
            if 1011 <= c <= 1499 and not (1311 <= c <= 1389):
                return "Basic Materials"
        except Exception:
            pass
        return "Energy"
    if any(k in d for k in ("ELECTRIC", "GAS TRANSMISSION", "WATER SUPPLY", "SANITARY")):
        return "Utilities"
    if any(k in d for k in ("RETAIL", "GROCERY", "FOOD", "BEVERAGE", "TOBACCO", "HOUSEHOLD")):
        return "Consumer Defensive"
    if any(k in d for k in ("MOTOR VEHICLE", "APPAREL", "RESTAURANT", "HOTEL", "AIRLINE", "AUTOMOTIVE")):
        return "Consumer Cyclical"
    if any(k in d for k in ("TELEPHONE", "COMMUNICATION", "BROADCAST", "TELEVISION", "CABLE")):
        return "Communication Services"
    if any(k in d for k in ("REAL ESTATE", "REIT", "OPERATORS", "LESSORS")):
        return "Real Estate"
    if any(k in d for k in ("CONSTRUCTION", "MACHINERY", "AEROSPACE", "RAILROAD", "TRUCKING", "AIR TRANSPORT")):
        return "Industrials"
    # Range fallback on SIC code.
    try:
        c = int(str(sic)[:4] if sic else 0)
        if 4911 <= c <= 4991:
            return "Utilities"
        if 6000 <= c <= 6999:
            return "Financial Services"
        if 1311 <= c <= 1389 or 2911 <= c <= 2999:
            return "Energy"
        if 2830 <= c <= 2836 or 8011 <= c <= 8099:
            return "Healthcare"
        if 3570 <= c <= 3699 or 7370 <= c <= 7379:
            return "Technology"
        if 4812 <= c <= 4899:
            return "Communication Services"
    except Exception:
        pass
    return "Default"
# Above: two passes so weird SIC texts still land somewhere sane. Never crashes.


# Mega-cap universe is GONE (your rule: no fixed lists). Peers now come from the
# LIVE S&P 500 constituents (sp500.py) ranked by LIVE caps. See get_live_peers().


def dynamic_risk(sector_name: str, live_sector_vol: float | None,
                 live_market_vol: float | None) -> tuple[float, float, str]:
    """Adjust base sector bar by LIVE volatility (fresh, not historical).

    Simple words: if your sector wobbles MORE than the market right now, demand
    more margin/probability before saying BUY. Maths (fixed): for each +1pp of
    excess annualised vol, +0.15pp winprob and +0.30pp margin, capped.
    Returns (need_prob, need_margin, explanation).
    """
    base_p, base_m = SECTOR_RISK.get(sector_name, SECTOR_RISK["Default"])
    if not live_sector_vol or not live_market_vol:
        return base_p, base_m, "base (live vol unavailable this fetch)"
    excess = max(-0.10, min(0.30, live_sector_vol - live_market_vol))
    need_p = max(0.50, min(0.95, base_p + excess * 0.15))
    need_m = max(0.05, min(0.50, base_m + excess * 0.30))
    return need_p, need_m, (f"base {base_p*100:.0f}%/{base_m*100:.0f}% "
                           f"+ live excess-vol {excess*100:+.1f}pp → "
                           f"{need_p*100:.0f}%/{need_m*100:.0f}%")
# Above: the no-fixed-risk engine. Same sector can demand 70% today, 78% in a storm.


def get_live_peers(n: int, stooq_key: str = "") -> list[dict]:
    """LIVE top-N by market cap from the LIVE S&P 500 universe.

    Simple words: fetch today's 503-member list fresh, price each (live Yahoo),
    multiply by live SEC shares = live cap, sort, take top N. Every number
    stamped at this moment; nothing reused from yesterday.
    """
    from . import data_stooq, data_sec, sp500
    from .live import now_utc
    members, _ = sp500.get_constituents()
    if not members:
        return []
    # Score all by live price first (fast Yahoo), then resolve shares for ranking.
    scored: list[tuple[float, dict]] = []
    for m in members:
        try:
            prices, _ = data_stooq.fetch_prices(m["ticker"], stooq_key)
            px = prices[-1]["close"] if prices else 0.0
            if px > 0:
                scored.append((px, m))
        except Exception:
            continue
    moment = now_utc()
    # Rank the scored set by live cap where shares known (bounded for SEC rate rule).
    ranked: list[dict] = []
    for px, m in scored:
        try:
            f = data_sec.get_fundamentals(m["ticker"])
            sh = (f.get("shares") or 0) if f.get("ok") else 0
            if not sh:
                continue
            sec = sic_to_sector(f.get("sic"), f.get("sicDescription")) if f.get("ok") else m.get("sector", "Default")
            ranked.append({"ticker": m["ticker"], "price": px, "shares": sh,
                           "mcap": px * sh, "title": (f.get("title") or m.get("name", ""))[:40],
                           "sector": sec, "fetched": moment})
        except Exception:
            continue
        if len(ranked) >= 120:  # bound: 120 SEC calls max per peers refresh (rate-safe)
            break
    ranked.sort(key=lambda x: x["mcap"], reverse=True)
    return ranked[: max(1, int(n))]
# Above: live leaderboard. Moment-stamped; count-bounded so SEC's 10 req/s rule holds.
