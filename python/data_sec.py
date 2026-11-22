# data_sec.py — SEC EDGAR fundamentals (free, no key, authoritative)
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): turns a ticker like AAPL into real cash-flow
# numbers from official 10-K/10-Q filings. Uses SEC's free JSON APIs:
# company_tickers.json -> CIK, companyfacts/CIK.json -> all XBRL facts,
# submissions/CIK.json -> filing list. Respects SEC rules (User-Agent + 10 req/s).

from __future__ import annotations
import json
import time
import requests
from . import cache_db

# SEC requires a real User-Agent with contact (else 403). Keep it honest.
UA = {"User-Agent": "DCF-Meow Vaibhav research contact@example.com",
      "Accept": "application/json", "Host": "data.sec.gov"}
UA_WWW = {"User-Agent": "DCF-Meow Vaibhav research contact@example.com"}
_last = [0.0]


def _throttle():
    """Max 10 req/s to data.sec.gov (SEC fair-use). Sleeps if too fast."""
    dt = time.time() - _last[0]
    if dt < 0.11:
        time.sleep(0.11 - dt)
    _last[0] = time.time()
# Above: politeness guard. SEC blocks abusers; 0.11s gap = ~9 req/s, safe.


def _get_json(url: str, ttl: float, host_data: bool = True):
    """GET JSON with cache + throttle + retry. Returns dict or {}."""
    key = "sec:" + url
    hit = cache_db.get(key, ttl)
    if hit:
        try:
            return json.loads(hit.decode("utf-8"))
        except Exception:
            pass
    h = UA if host_data else UA_WWW
    for attempt in range(3):
        try:
            _throttle()
            r = requests.get(url, headers=h, timeout=20)
            if r.status_code == 200 and r.content:
                cache_db.put(key, r.content)
                return r.json()
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return {}
# Above: cached fetch with 3 tries. Cache first = instant repeat + quota-safe.


_ticker_map: dict[str, dict] | None = None


def ticker_to_cik(ticker: str) -> tuple[int | None, str]:
    """Map 'AAPL'/'aapl.us' -> (CIK int, company title). Uses SEC company_tickers.json."""
    global _ticker_map
    t = ticker.strip().upper().split(".")[0].replace("^", "")
    if _ticker_map is None:
        j = _get_json("https://www.sec.gov/files/company_tickers.json",
                      cache_db.TTL_SEC, host_data=False)
        _ticker_map = {}
        try:
            for _, v in (j.items() if isinstance(j, dict) else []):
                _ticker_map[str(v.get("ticker", "")).upper()] = v
        except Exception:
            pass
    v = (_ticker_map or {}).get(t)
    if not v:
        return None, t
    return int(v["cik_str"]), str(v.get("title", t))
# Above: ticker -> CIK number SEC uses as ID. Handles 'AAPL.US' and '^SPX' gracefully.


def _annual_series(facts: dict, taxonomy: str, tag: str, unit="USD") -> dict[int, float]:
    """Extract {fiscal_year: value} for 10-K annual facts only (form 10-K/20-F/40-F)."""
    out: dict[int, float] = {}
    try:
        arr = facts["facts"][taxonomy][tag]["units"][unit]
    except Exception:
        return out
    for f in arr:
        try:
            if f.get("form") not in ("10-K", "10-K/A", "20-F", "40-F"):
                continue
            fy = int(f.get("fy", 0))
            if fy and f.get("val") is not None:
                out[fy] = float(f["val"])  # last filing wins (amendments overwrite)
        except Exception:
            continue
    return out
# Above: picks only yearly audited numbers (not quarterly noise) for stable DCF inputs.


def _ttm_or_latest(facts: dict, taxonomy: str, tags: list[str]) -> float | None:
    """Best-effort latest annual value trying tag aliases (GAAP names vary by filer)."""
    for tag in tags:
        for unit in ("USD", "shares", "pure"):
            s = _annual_series(facts, taxonomy, tag, unit)
            if s:
                return s[max(s.keys())]
    return None
# Above: GAAP tags differ (e.g. Cash vs CashAndCashEquivalents). Try aliases, take newest year.


def get_fundamentals(ticker: str) -> dict:
    """Fetch + normalize fundamentals for DCF. Returns dict with TTM FCF etc.

    FCF = OperatingCashFlow - CapEx(PaymentsToAcquirePPE).
    NetDebt = TotalDebt - Cash. Shares = latest shares outstanding.
    Also returns 5y history for CAGR + filing list for GUI table.
    """
    cik, title = ticker_to_cik(ticker)
    if not cik:
        return {"ok": False, "error": f"Ticker {ticker} not in SEC (non-US? try price tab).", "title": ticker}
    cik10 = f"{cik:010d}"
    facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json", cache_db.TTL_SEC)
    subs = _get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json", cache_db.TTL_SEC)
    if not facts:
        return {"ok": False, "error": "SEC facts empty (rate-limit? retry in 5s).", "title": title, "cik": cik}

    # Revenue / profit / cash-flow lines (aliases cover most filers incl. AAPL/MSFT).
    rev = _annual_series(facts, "us-gaap", "Revenues")
    if not rev:
        rev = _annual_series(facts, "us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax")
    if not rev:
        rev = _annual_series(facts, "us-gaap", "SalesRevenueNet")
    ocf = _annual_series(facts, "us-gaap", "NetCashProvidedByUsedInOperatingActivities")
    capex = _annual_series(facts, "us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment")
    if not capex:
        capex = _annual_series(facts, "us-gaap", "PaymentsToAcquireProductiveAssets")
    ni = _annual_series(facts, "us-gaap", "NetIncomeLoss")

    # FCF history = OCF - CapEx per year (CapEx stored positive outflow, so subtract).
    fcf_hist: dict[int, float] = {}
    for y in sorted(set(ocf) & set(capex)):
        fcf_hist[y] = ocf[y] - abs(capex[y])
    if not fcf_hist and ocf:  # some filers lack CapEx tag: fall back to OCF*0.85 heuristic
        for y, v in ocf.items():
            fcf_hist[y] = v * 0.85

    yrs = sorted(fcf_hist)
    fcf0 = fcf_hist[yrs[-1]] if yrs else None
    # 3y CAGR of FCF for default growth guess (clamped later in GUI).
    cagr = None
    if len(yrs) >= 4 and fcf_hist[yrs[-4]] > 0 and fcf_hist[yrs[-1]] > 0:
        cagr = (fcf_hist[yrs[-1]] / fcf_hist[yrs[-4]]) ** (1 / 3) - 1

    cash = _ttm_or_latest(facts, "us-gaap", ["CashAndCashEquivalentsAtCarryingValue", "Cash"])
    debt_long = _ttm_or_latest(facts, "us-gaap", ["LongTermDebtNoncurrent", "LongTermDebt"])
    debt_short = _ttm_or_latest(facts, "us-gaap",
                                ["ShortTermBorrowings", "LongTermDebtCurrent", "DebtCurrent"])
    debt = (debt_long or 0) + (debt_short or 0)
    net_debt = debt - (cash or 0)
    shares = _ttm_or_latest(facts, "dei", ["EntityCommonStockSharesOutstanding"])
    equity = _ttm_or_latest(facts, "us-gaap", ["StockholdersEquity"])

    # Recent filings for GUI table (form + date + accession).
    filings: list[tuple[str, str, str]] = []
    try:
        rec = subs.get("filings", {}).get("recent", {})
        forms, dates, accns = rec.get("form", []), rec.get("filingDate", []), rec.get("accessionNumber", [])
        for i in range(min(12, len(forms))):
            filings.append((forms[i], dates[i], accns[i]))
    except Exception:
        pass
    # SIC sector for sector-aware risk (submissions JSON carries it live).
    try:
        sic = str(subs.get("sic") or "")
        sicDesc = str(subs.get("sicDescription") or "")
    except Exception:
        sic, sicDesc = "", ""

    return {"ok": True, "title": title, "cik": cik, "cik10": cik10,
            "rev": rev, "ocf": ocf, "capex": capex, "ni": ni, "fcf_hist": fcf_hist,
            "fcf0": fcf0, "cagr": cagr, "cash": cash, "debt": debt, "net_debt": net_debt,
            "shares": shares, "equity": equity, "filings": filings,
            "sic": sic, "sicDescription": sicDesc}
# Above: the heart of SEC parsing. Returns everything the DCF tab needs, with
# sensible fallbacks when GAAP tags vary. All numbers in USD (or shares).
