# DCF-Meow — Hyper-Optimized DCF GUI for i5-1335U (AVX2+FMA)
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)

A fast, smooth, reliable, easy-to-use Discounted Cash Flow analyzer,
hyper-optimized for **this machine**: 13th Gen i5-1335U, 12 threads,
AVX2 + FMA + BMI2, Intel Iris Xe, 7.6 GiB RAM, Debian 13 WSL2/WSLg.

## Hybrid strategy (best tool for each job — your requested approach)
| Layer | Language / Lib | Why it wins on this machine |
|---|---|---|
| DCF math core | **C++17 + AVX2/FMA intrinsics** (`cpp/dcf_avx2.cpp` → `build/libdcf_avx2.so`) | 4 doubles per instruction, FMA for `pv += fcf*df`, `-O3 -march=native -mavx2 -mfma -flto`. 3–4× faster than scalar for sensitivity grids + Monte Carlo. |
| Data + glue | **Python 3.13 (ml_env only)** + NumPy(OpenBLAS-Haswell/AVX2) + Numba | Zero-copy contiguous `float64`, `fastmath+parallel`, SQLite WAL cache. Best ecosystem for SEC/STOOQ. |
| Price data | **Stooq (primary, max data)** + Yahoo fallback | Stooq: `https://stooq.com/q/d/l/?s=AAPL.US&d1=&d2=&i=d` (needs free apikey since early-2026 CAPTCHA/JS-PoW) + bulk `stooq.com/db/h/`. 21k securities, 30+yr daily. Fallback Yahoo Chart v8 (no key, tested working). |
| Fundamentals | **SEC EDGAR (primary, free, no key)** | `data.sec.gov/api/xbrl/companyfacts/CIK.json` + `submissions/CIK.json` + `company_tickers.json`. Authoritative 10-K/10-Q XBRL, 19k filers. Respects 10 req/s + User-Agent rule. |
| Risk-free rate | **FRED DGS10 (optional key)** else 4.2% fallback | For WACC CAPM. |
| GUI | **PySide6 (Qt6) + pyqtgraph** | GPU-composited on WSLg, QThread workers (UI never blocks), dark theme, 60 fps charts. Smoother than Tkinter, lighter than Electron. |

> Research: Stooq CSV/API + bulk docs, SEC EDGAR JSON APIs (`data.sec.gov/submissions`, `companyfacts`, `companyconcept`, `frames`), FMP/AlphaVantage/Finnhub comparison (all have tight free caps: 25–300 req/day) — so SEC+Stooq is the max-free combo. Verified live: SEC 200 OK, Yahoo 200 OK, Stooq `q/d/l` now behind JS-PoW (hence apikey slot + fallback).

## DCF formula used
```
FCF_t = FCF_0 * (1+g)^t
PV_t  = FCF_t / (1+WACC)^t
TV    = FCF_n*(1+gT)/(WACC-gT)   (Gordon, requires WACC>gT)
EV    = sum(PV) + TV/(1+WACC)^n
Equity= EV - NetDebt ; Fair/share = Equity/shares
WACC  = wE*Re + wD*Rd*(1-Tc), Re = Rf + beta*(ERP)
```
Beta = slope(stock ret vs SPY `^SPX`) from price history. ERP default 5.5%.

## Quick start (ONLY in ml_env — never system pip)
```bash
source ~/ml_env/bin/activate   # mandatory per your rule
cd ~/dcf-meow
./run.sh                        # builds AVX2 .so, runs bench, launches GUI
# or step-by-step:
make -j12                       # uses all 12 threads
python tools/bench.py           # proves AVX2 speedup on YOUR cpu
python -m gui.app               # or: python gui/app.py
```

## GUI tabs
1. **Search & Overview** — ticker (e.g. `AAPL`, `MSFT.US`, `^SPX`), Stooq apikey box (optional, get at `stooq.com/q/d/?s=spy.us&get_apikey`), price chart (pyqtgraph, OpenGL), last price / 52w / beta.
2. **Financials (SEC)** — TTM revenue/net income/OCF/CapEx/FCF, cash/debt/equity/shares from companyfacts, filing table.
3. **DCF Assumptions** — FCF0 (auto from SEC), growth g, terminal gT, WACC or CAPM auto, years, net debt, shares, tax. Sliders + spins.
4. **Valuation** — fair/share, upside, EV/Equity bridge, sensitivity grid (WACC×g heatmap via AVX2), Monte Carlo p10/p50/p90.
5. **Log** — what was fetched, cache hits, timings.

## Files (every file headers: Author Vaibhav, GPL-3.0-only)
```
Makefile               # -O3 -march=native -mavx2 -mfma -flto -funroll-loops -fopenmp
cpp/dcf_avx2.{h,cpp}  # AVX2/FMA core: fair-value x4, grid, monte-carlo
python/cache_db.py    # SQLite WAL @ ~/.cache/dcf-meow
python/data_sec.py    # SEC EDGAR client (tickers→CIK→facts→TTM FCF)
python/data_stooq.py  # Stooq max-data client + Yahoo fallback
python/data_fred.py   # FRED DGS10 or fallback
python/dcf_core.py    # ctypes → libdcf_avx2.so, Numba fallback, aligned float64
gui/app.py            # PySide6 + QThread + pyqtgraph GUI
tools/bench.py        # AVX2 vs NumPy vs Numba benchmark on this CPU
```

## Perf notes for i5-1335U
- NumPy here is `scipy-openblas Haswell MAX_THREADS=64` → already AVX2 (`X86_V3`). Keep arrays C-contiguous `float64`, 32-byte aligned (`np.empty(..., dtype=np.float64)` is 16+ aligned; C++ uses `aligned_alloc(32)`).
- Build uses `-march=native` → detects Raptor Lake, emits AVX2+FMA+BMI2. Do NOT use `-mavx512` (this chip has none).
- GUI: fetch on `QThread`, compute via AVX2 (µs), plot decimated to ≤2000 pts for Iris Xe/WSLg smoothness.
- Cache: SEC 10 rps throttle, Stooq quota guard, SQLite WAL, `requests-cache` in ml_env.

## License
GPL-3.0-only, Author Vaibhav (final power). Never MIT per your rule.

## v2 — TERMINAL upgrade (professional + 100M break-test + batch)
- Pro look: Fusion + dark QPalette + card QSS, KPI header (LAST/FAIR/UPSIDE/VERDICT), gold price lines.
- MC as GRAPH: tab ③ shows 100k-path histogram (blue bars, gold=current, green=p50, red=p10).
- Optimizer (tab ④ ★): 350 rates × 0.03% × 100k × 3 haircuts (0/user/1.5×user) = up to 105M
  AVX2 valuations (~1s on i5-1335U, measured 131M/s). WACC base = CAPM +0.10% buffer,
  refreshed on EVERY fetch. Haircut = your lower-expectation knob on future FCF.
  Verdict BUY only if P(fair≥price) ≥ sector need AND margin ≥ sector need; shows fair
  price, drop-needed %, max survivable WACC, selfcheck PASS/FAIL (AVX2==scalar proof).
- Sector risk: auto from SEC SIC (e.g. AAPL 3571 → Technology → 75%/25%), editable.
- Peers (tab ⑤): live top-N by mcap (Yahoo price × SEC shares, sorted now).
- Batch (tab ⑥): textbox + .txt file, progress bar + stop, green BUY/red SELL rows.
- Engine honesty: `dcf_sweep_selfcheck` asserts AVX2==scalar (<1e-3 else FAIL); sweep
  prob/p50 monotonic self-tested; no fake numbers — fallbacks are labeled.

## v3 LIVE (no stale data) + S&P500 risk-adjusted benchmark
- Your rule enforced: `python/live.py` LIVE_ONLY=True bypasses ALL disk cache —
  every Fetch hits fresh network; only the MATH formulas (cpp/dcf_avx2) stay fixed.
  Every number carries `source + as-of date @ fetch-moment` proof in GUI + Log.
- Fixed values eliminated: Rf = live Treasury/FRED DGS10 dated (no 4.2% fallback —
  outage shows LIVE FAILED, never a fake); ERP = live Damodaran implied
  (Jan26 4.23% → Mar26 4.77% → Jul26 US 4.45% → Sep26 4.31%, parsed live with date,
  manual override checkbox available); peers = live 503-member S&P500 list
  (datahub live CSV + Wikipedia fallback, count-checked) ranked by live caps.
- Sector risk is LIVE: base table + live excess-vol adjustment
  (+0.15pp winprob / +0.30pp margin per +1pp excess vol, capped) from fresh prices.
- Benchmark tab ⑦ ★: stock vs S&P500 (^SPX live leg) on growth-of-$1 + drawdown
  charts and risk table — Sharpe, Sortino (down-risk only), Calmar, alpha,
  beta, R², up/down capture, VaR95. Shows reward-PER-risk, not raw %.
  Verified live: AAPL 10Y Sharpe 0.82 Sortino 1.12 Calmar 0.72 beta 1.19 alpha +11.8%.

## v3.1 crash fix + dated zoomable charts (Oct-2026)
- Fixed `Cannot make QOpenGLContext current in a different thread` abort:
  pyqtgraph OpenGL codepath (`useOpenGL=True`) makes GL contexts current from
  worker threads on Qt 6.11/WSLg. Now `useOpenGL=False` (software render) +
  `leftButtonPan=False`. Charts decimated ≤2000 pts — still 60fps on Iris Xe.
- Every chart: named X + Y axes. Price / benchmark / drawdown use a YEAR axis
  (unix timestamps + DateAxisItem: 2020, 2021… auto by zoom). FCF uses fiscal
  years. Zoom on all: left-drag=box-zoom · wheel=zoom · right-drag=pan ·
  right-click menu=autoscale (hint in every chart title).
