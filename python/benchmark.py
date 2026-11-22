# benchmark.py — LIVE risk-adjusted benchmark vs S&P 500 (not just pure return)
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# What this does (simple words): compares your stock against the S&P 500 index
# on BOTH reward and pain: Sharpe (return per wobble), Sortino (return per
# DOWN-wobble only — up-days aren't punished), Calmar (return per worst crash),
# alpha/beta/R², upside/downside capture, max drawdown, VaR. Formulas (fixed maths):
# Sharpe=(Rp−Rf)/σ, Sortino=(Rp−Rf)/σ_down, Calmar=CAGR/|MaxDD|,
# beta=Cov(Rs,Rm)/Var(Rm), alpha=CAGR_s − beta*CAGR_m. Inputs 100% live prices.

from __future__ import annotations
import numpy as np

TRADING = 252.0  # trading days/year (fixed maths convention)


def risk_metrics(stock_px: np.ndarray, spy_px: np.ndarray, rf_annual: float) -> dict:
    """Compute risk-adjusted table from LIVE price arrays (aligned tail).

    Simple words: line up both histories, turn prices into daily % moves,
    then score reward-per-risk instead of raw %. Annualised with 252-day maths.
    """
    s = np.ascontiguousarray(stock_px, dtype=np.float64)
    m = np.ascontiguousarray(spy_px, dtype=np.float64)
    n = min(len(s), len(m))
    out: dict = {"n": n}
    if n < 60:
        out["error"] = "need 60+ overlapping days (fetch fresh prices)"
        return out
    s, m = s[-n:], m[-n:]
    rs = s[1:] / s[:-1] - 1.0  # daily returns, stock
    rm = m[1:] / m[:-1] - 1.0  # daily returns, S&P500
    rf_d = (1 + rf_annual) ** (1 / TRADING) - 1  # daily risk-free
    # Total + annualised growth.
    tot_s = float(s[-1] / s[0] - 1)
    tot_m = float(m[-1] / m[0] - 1)
    yrs = n / TRADING
    cagr_s = float((s[-1] / s[0]) ** (1 / yrs) - 1) if yrs > 0 else 0.0
    cagr_m = float((m[-1] / m[0]) ** (1 / yrs) - 1) if yrs > 0 else 0.0
    vol_s = float(np.std(rs, ddof=1) * np.sqrt(TRADING))
    vol_m = float(np.std(rm, ddof=1) * np.sqrt(TRADING))
    # Sharpe (all wobble) + Sortino (down-wobble only).
    ex = rs - rf_d
    sharpe = float(np.mean(ex) / (np.std(ex, ddof=1) + 1e-12) * np.sqrt(TRADING))
    down = ex[ex < 0]
    sortino = float(np.mean(ex) / (np.std(down, ddof=1) + 1e-12) * np.sqrt(TRADING)) if len(down) > 5 else 0.0
    # Drawdowns from running peak (pain lived through).
    def _dd(px):
        peak = np.maximum.accumulate(px)
        dd = px / peak - 1.0
        return float(np.min(dd)), dd
    maxdd_s, dd_s = _dd(s)
    maxdd_m, _ = _dd(m)
    calmar = float(cagr_s / (abs(maxdd_s) + 1e-12))
    # Beta / alpha / R² vs market.
    v = float(np.var(rm, ddof=1))
    beta = float(np.cov(rs, rm, ddof=1)[0, 1] / v) if v > 1e-12 else 1.0
    alpha = float(cagr_s - beta * cagr_m)
    r2 = float(np.corrcoef(rs, rm)[0, 1] ** 2) if n > 2 else 0.0
    # Capture: gains vs losses participation.
    up = rm > 0
    up_cap = float(np.mean(rs[up]) / (np.mean(rm[up]) + 1e-12)) if up.sum() > 5 else 0.0
    dn = rm < 0
    dn_cap = float(np.mean(rs[dn]) / (np.mean(rm[dn]) + 1e-12)) if dn.sum() > 5 else 0.0
    var95 = float(np.percentile(rs, 5))
    out.update(total_stock=tot_s, total_spy=tot_m, cagr_stock=cagr_s, cagr_spy=cagr_m,
               vol_stock=vol_s, vol_spy=vol_m, sharpe=sharpe, sortino=sortino,
               calmar=calmar, maxdd_stock=maxdd_s, maxdd_spy=maxdd_m,
               beta=beta, alpha=alpha, r2=r2, up_capture=up_cap, down_capture=dn_cap,
               var95_daily=var95, dd_curve_s=dd_s, rs=rs, rm=rm)
    return out
# Above: risk lens. A 30% gainer with 40% wobble (Sharpe 0.6) can LOSE to a 15%
# gainer with 10% wobble (Sharpe 1.2) — this table shows that, not raw % alone.
