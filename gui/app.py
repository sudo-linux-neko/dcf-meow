#!/usr/bin/env python3
# app.py — DCF-Meow TERMINAL v2: professional, MC graphs, 350-rate optimizer, batch
# SPDX-License-Identifier: GPL-3.0-only
# Author: Vaibhav (holds final power over the project)
# Copyright (C) 2026 Vaibhav — GPL v3 only (never MIT).
#
# What this does (simple words): a Bloomberg-style dark terminal.
# Fetch -> prices (Stooq max / Yahoo backup) + SEC fundamentals + live sector.
# DCF tab -> Monte Carlo HISTOGRAM graph (100k futures as bars).
# Optimizer tab -> 350 interest rates x 0.03% x 100k futures x 3 haircuts = 105M
#   valuations in C++ AVX2+FMA+OpenMP (~5s on i5-1335U), finds the max rate your
#   thesis survives, fair price, drop-needed %, BUY/WAIT verdict vs sector risk.
# Peers tab -> live top-N by market cap. Batch tab -> 1000s tickers with progress.

from __future__ import annotations
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Above: find python/ package when launched as `python gui/app.py`.

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QTabWidget, QTableWidget,
                               QTableWidgetItem, QDoubleSpinBox, QSpinBox, QTextEdit,
                               QGroupBox, QFormLayout, QHeaderView, QProgressBar,
                               QFileDialog, QSplitter, QComboBox, QCheckBox)
import pyqtgraph as pg
# Above: Qt widgets + pyqtgraph charts (OpenGL, 60fps on Iris Xe).

from python import dcf_core, data_sec, data_stooq, data_fred, sector

# --- Professional terminal palette (Bloomberg-like density, soft on eyes in WSLg) ---
BG, CARD, BORDER = "#0d1117", "#161b26", "#2a3042"
TXT, DIM = "#e6e9f2", "#8b93a7"
GREEN, RED, ACCENT, GOLD = "#00d68f", "#ff4d6d", "#5aa2ff", "#ffc857"

QSS = f"""
QMainWindow, QWidget {{ background:{BG}; color:{TXT}; font-size:13px; }}
QTabWidget::pane {{ border:1px solid {BORDER}; border-radius:8px; background:{CARD}; }}
QTabBar::tab {{ background:{BG}; color:{DIM}; padding:8px 16px; border-top-left-radius:8px;
  border-top-right-radius:8px; margin-right:4px; font-weight:bold; }}
QTabBar::tab:selected {{ background:{CARD}; color:{TXT}; border:1px solid {BORDER}; border-bottom:none; }}
QGroupBox {{ border:1px solid {BORDER}; border-radius:8px; margin-top:14px; background:{CARD};
  font-weight:bold; padding-top:10px; }}
QGroupBox::title {{ subcontrol-origin:margin; left:12px; color:{ACCENT}; }}
QLineEdit, QTextEdit, QTableWidget {{ background:#0a0e14; border:1px solid {BORDER};
  border-radius:6px; color:{TXT}; selection-background-color:{ACCENT}; }}
QPushButton {{ background:#1f2637; border:1px solid {BORDER}; border-radius:6px; padding:7px 14px;
  font-weight:bold; }}
QPushButton:hover {{ border-color:{ACCENT}; }}
QDoubleSpinBox, QSpinBox, QComboBox {{ background:#0a0e14; border:1px solid {BORDER};
  border-radius:6px; padding:4px; }}
QProgressBar {{ border:1px solid {BORDER}; border-radius:6px; background:#0a0e14; text-align:center; }}
QProgressBar::chunk {{ background:{ACCENT}; border-radius:5px; }}
QHeaderView::section {{ background:#1f2637; color:{TXT}; border:none; padding:6px; font-weight:bold; }}
"""
# Above: one stylesheet for the whole terminal. Cards + borders = professional density.


def _card_value(v: float, fmt="${:,.2f}") -> str:
    try:
        return fmt.format(v)
    except Exception:
        return "—"


ZOOM_HINT = " (left-drag=box-zoom · wheel=zoom · right-drag=pan · right-click=autoscale)"


def _mkplot(title: str, xlabel: str, ylabel: str, dates: bool = False) -> "pg.PlotWidget":
    """Build one zoomable chart: year/price labels, grid, mouse zoom, date axis if asked.

    Simple words: every chart gets x-name + y-name, dotted grid, mouse
    zoom/pan (drag moves, wheel zooms, right-click menu resets). Price charts
    get a YEAR axis (unix timestamps → 2020, 2021… auto by zoom level).
    Software render (no OpenGL) — OpenGL contexts crash across threads on WSLg.
    """
    import pyqtgraph as _pg
    if dates:
        axis = _pg.DateAxisItem(orientation="bottom")
        w = _pg.PlotWidget(title=title + ZOOM_HINT, axisItems={"bottom": axis})
    else:
        w = _pg.PlotWidget(title=title + ZOOM_HINT)
    w.setBackground(BG)
    w.setLabel("bottom", xlabel)
    w.setLabel("left", ylabel)
    w.showGrid(x=True, y=True, alpha=0.25)
    vb = w.getViewBox()
    vb.setMouseEnabled(x=True, y=True)  # drag=pan, wheel=zoom, box-zoom via right-drag axis
    vb.menuEnabled = True  # right-click menu: autoscale / mouse mode
    return w


def _dates_to_ts(dates: list[str]) -> np.ndarray:
    """'YYYY-MM-DD' strings -> unix timestamps (float) for the YEAR axis. Bad rows -> NaN."""
    import datetime as _dt
    import numpy as _np
    ts = []
    for dstr in dates:
        try:
            dt = _dt.datetime.strptime(dstr[:10], "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
            ts.append(dt.timestamp())
        except Exception:
            ts.append(float("nan"))
    return _np.array(ts, dtype=float)
# Above: date helper. NaN rows are skipped by connect='finite' so one bad date never kills a chart.


class FetchWorker(QObject):
    """Background: LIVE prices + LIVE SEC + LIVE Rf + LIVE ERP + sector (never blocks UI).

    Your rule: only maths is fixed — every number is fetched fresh at this moment
    (cache bypassed in live.py), each with its own timestamp as proof.
    """
    done = Signal(dict)
    log = Signal(str)

    def __init__(self, ticker: str, stooq_key: str, fred_key: str, erp_override: float | None = None):
        super().__init__()
        self.ticker, self.stooq_key = ticker, stooq_key
        self.fred_key, self.erp_override = fred_key, erp_override

    def run(self):
        from python import erp_live
        from python.live import now_utc
        t0 = time.perf_counter()
        try:
            moment = now_utc()
            self.log.emit(f"LIVE fetch {self.ticker} @ {moment} (no cache, fresh network)…")
            prices, psrc = data_stooq.fetch_prices(self.ticker, self.stooq_key)
            spy, spsrc = data_stooq.fetch_prices("^SPX", self.stooq_key)
            beta = data_stooq.beta_vs_spy(prices, spy) if prices and spy else 1.0
            fund = data_sec.get_fundamentals(self.ticker)
            rf, rfsrc = data_fred.get_risk_free(self.fred_key)  # live Treasury/FRED w/ date, None on outage
            erp, erpsrc = erp_live.get_live_erp()  # live Damodaran implied ERP w/ date
            if self.erp_override is not None:
                erp, erpsrc = float(self.erp_override), f"manual override {self.erp_override*100:.2f}% @ {moment}"
            sec_name = "Default"
            if fund.get("ok"):
                sec_name = sector.sic_to_sector(fund.get("sic"), fund.get("sicDescription"))
            if rf is None or erp is None:
                self.log.emit(f"LIVE FAILED: rf={rfsrc} erp={erpsrc} — showing data but WACC needs both live")
                self.done.emit({"prices": prices, "psrc": f"{psrc} @ {moment}",
                                "spy": spy, "spsrc": spsrc, "beta": beta, "fund": fund,
                                "rf": rf, "rfsrc": rfsrc, "erp": erp, "erpsrc": erpsrc,
                                "sector": sec_name, "wacc_base": None, "last": prices[-1]["close"] if prices else 0.0,
                                "secs": time.perf_counter() - t0, "moment": moment})
                return
            # Auto-WACC base = LIVE CAPM + 0.10% buffer, refreshed on EVERY fetch (your rule).
            wacc_base = float(np.clip(rf + beta * erp + 0.001, 0.03, 0.30))
            last = prices[-1]["close"] if prices else 0.0
            self.log.emit(f"LIVE OK @ {moment}: {len(prices)} prices ({psrc}) beta {beta:.2f} | "
                          f"{fund.get('title', '?')} [{sec_name}] FCF ${(fund.get('fcf0') or 0)/1e9:.1f}B | "
                          f"Rf {rf*100:.2f}% ({rfsrc}) ERP {erp*100:.2f}% ({erpsrc}) → WACC base {wacc_base*100:.2f}%")
            self.done.emit({"prices": prices, "psrc": f"{psrc} @ {moment}",
                            "spy": spy, "spsrc": spsrc, "beta": beta, "fund": fund,
                            "rf": rf, "rfsrc": rfsrc, "erp": erp, "erpsrc": erpsrc,
                            "sector": sec_name, "wacc_base": wacc_base, "last": last,
                            "secs": time.perf_counter() - t0, "moment": moment})
        except Exception:
            self.log.emit(traceback.format_exc())
            self.done.emit({"prices": [], "psrc": "error", "spy": [], "spsrc": "", "beta": 1.0,
                            "fund": {"ok": False, "error": "crash"}, "rf": None,
                            "rfsrc": "crash", "erp": None, "erpsrc": "crash", "sector": "Default",
                            "wacc_base": None, "last": 0.0, "secs": 0, "moment": now_utc()})
# Above: live fetch thread. No fixed Rf/ERP anywhere — both arrive dated or the run is flagged FAILED.


class SweepWorker(QObject):
    """105M break-test in chunks (7x50 rates per haircut) with live progress. C++ AVX2 inside."""
    prog = Signal(int, str)
    done = Signal(dict)
    log = Signal(str)

    def __init__(self, p: dict):
        super().__init__()
        self.p = p  # base_fcf,g,wacc_base,tgrowth,years,debt,shares,last,haircuts,mc,seed

    def run(self):
        try:
            import time as _t
            t0 = _t.perf_counter()
            haircuts = self.p["haircuts"]
            n_rates, step = self.p["n_rates"], self.p["wacc_step"]
            CH = 50  # rates per C++ call -> smooth progress (5M vals ~0.25s each)
            total_calls = len(haircuts) * ((n_rates + CH - 1) // CH)
            call = 0
            results = {}
            total_vals = 0
            for h in haircuts:
                fcf_h = self.p["base_fcf"] * (1.0 - h)
                P50 = np.empty(n_rates)
                P10 = np.empty(n_rates)
                PR = np.empty(n_rates)
                for s in range(0, n_rates, CH):
                    e = min(n_rates, s + CH)
                    c = e - s
                    a, b, c_ = dcf_core.sweep_rates(
                        c, self.p["wacc_base"] + s * step, step,
                        fcf_h, self.p["g"], self.p["g_sd"], self.p["w_jit"],
                        self.p["tgrowth"], self.p["years"], self.p["debt"],
                        self.p["shares"], self.p["last"],
                        mc_per_rate=self.p["mc"], seed=self.p["seed"] + s + int(h * 1e4))
                    P50[s:e], P10[s:e], PR[s:e] = a, b, c_
                    total_vals += c * self.p["mc"]
                    call += 1
                    self.prog.emit(int(100 * call / total_calls),
                                   f"haircut {h*100:.0f}% rates {s}-{e}/{n_rates}")
                results[h] = (P50, P10, PR)
            dev = dcf_core.selfcheck(self.p["base_fcf"], self.p["g"], self.p["wacc_base"],
                                     self.p["tgrowth"], self.p["years"],
                                     self.p["debt"], self.p["shares"])
            dt = _t.perf_counter() - t0
            self.log.emit(f"Sweep {total_vals/1e6:.0f}M valuations in {dt:.1f}s "
                          f"({total_vals/dt/1e6:.1f}M/s AVX2) selfcheck dev {dev:.2e} "
                          f"{'PASS' if dev < 1e-3 else 'FAIL'}")
            self.done.emit({"results": results, "selfcheck": dev, "secs": dt,
                            "total_vals": total_vals})
        except Exception:
            self.log.emit(traceback.format_exc())
            self.done.emit({"results": {}, "selfcheck": 1.0, "secs": 0, "total_vals": 0})
# Above: chunked sweep = progress bar moves + UI alive. 105M never allocated at once.


class BatchWorker(QObject):
    """Batch 1000s tickers: LIVE Rf + LIVE ERP once per run, then quick DCF each."""
    prog = Signal(int, str)
    done = Signal(list)
    log = Signal(str)

    def __init__(self, tickers: list[str], stooq_key: str):
        super().__init__()
        self.tickers, self.stooq_key = tickers, stooq_key
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from python import erp_live
        from python.live import now_utc
        rows = []
        try:
            moment = now_utc()
            rf, rfsrc = data_fred.get_risk_free("")
            erp, erpsrc = erp_live.get_live_erp()
            self.log.emit(f"Batch LIVE inputs @ {moment}: Rf {rfsrc} ERP {erpsrc}")
            if rf is None or erp is None:
                self.log.emit("Batch LIVE FAILED (Rf/ERP) — rows flagged, no fixed numbers used")
            for i, t in enumerate(self.tickers):
                if self._stop:
                    break
                try:
                    prices, _ = data_stooq.fetch_prices(t, self.stooq_key)
                    last = prices[-1]["close"] if prices else 0.0
                    f = data_sec.get_fundamentals(t)
                    if rf is None or erp is None or not f.get("ok") or not f.get("fcf0") or not last:
                        rows.append((t, "NO-LIVE-DATA", 0, 0, 0, "SKIP"))
                    else:
                        g = float(np.clip((f.get("cagr") or 0.06), -0.05, 0.25))
                        beta = 1.0
                        w = float(np.clip(rf + beta * erp + 0.001, 0.04, 0.25))
                        fv = dcf_core.fair_value(f["fcf0"], g, w, 0.025, 10,
                                                 f.get("net_debt") or 0, f.get("shares") or 1)
                        up = (fv / last - 1) * 100 if last else 0
                        rows.append((t.upper(), f"{last:.2f}", f"{fv:.2f}", f"{up:+.1f}%",
                                     f"${(f['fcf0'])/1e9:.1f}B",
                                     "BUY" if up > 20 else ("HOLD" if up > 0 else "SELL")))
                except Exception as e:
                    rows.append((t.upper(), "ERR", 0, 0, 0, str(e)[:30]))
                self.prog.emit(int(100 * (i + 1) / max(1, len(self.tickers))), f"{i+1}/{len(self.tickers)} {t}")
            self.done.emit(rows)
        except Exception:
            self.log.emit(traceback.format_exc())
            self.done.emit(rows)
# Above: batch loop. LIVE inputs once (same moment for all rows = fair compare); per-row progress.


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DCF-Meow TERMINAL v3 LIVE — Author Vaibhav (GPL-v3) — AVX2 — no stale data")
        self.resize(1400, 900)
        self.prices: list[dict] = []
        self.spy: list[dict] = []
        self.fund: dict = {"ok": False}
        self.beta, self.rf, self.erp = 1.0, None, None
        self.rfsrc, self.erpsrc, self.moment = "—", "—", "—"
        self.sector = "Default"
        self.wacc_base, self.last = 0.09, 0.0
        self.sweep_cache: dict = {}
        self.thread = None
        self._build()
        self._recalc_quick()
    # Above: state. rf/erp start None (no fixed numbers) until first LIVE fetch stamps them.

    # ---------- layout ----------
    def _build(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)
        # Header bar: brand + inputs + fetch.
        hdr = QHBoxLayout()
        brand = QLabel("◈ DCF-MEOW <span style='color:#5aa2ff'>TERMINAL</span>")
        brand.setStyleSheet("font-size:17px;font-weight:900;")
        hdr.addWidget(brand)
        hdr.addWidget(QLabel("Ticker:"))
        self.ticker = QLineEdit("AAPL")
        self.ticker.setFixedWidth(110)
        hdr.addWidget(self.ticker)
        hdr.addWidget(QLabel("Stooq key:"))
        self.skey = QLineEdit("")
        self.skey.setPlaceholderText("optional apikey")
        self.skey.setFixedWidth(150)
        hdr.addWidget(self.skey)
        hdr.addWidget(QLabel("ERP:"))
        self.erp_live = QCheckBox("LIVE")
        self.erp_live.setChecked(True)
        self.erp_live.setToolTip("Checked = live Damodaran implied ERP every fetch. Uncheck = use manual %.")
        hdr.addWidget(self.erp_live)
        self.erp = QDoubleSpinBox(value=4.3, suffix=" %", minimum=2, maximum=8)
        self.erp.setFixedWidth(90)
        self.erp.setEnabled(False)
        self.erp_live.toggled.connect(lambda on: self.erp.setEnabled(not on))
        hdr.addWidget(self.erp)
        self.live_badge = QLabel("● LIVE-ONLY (no cache)")
        self.live_badge.setStyleSheet(f"color:{GREEN};font-weight:900;")
        self.live_badge.setToolTip("Every number is fetched fresh at its stamped moment. Only maths formulas are fixed.")
        hdr.addWidget(self.live_badge)
        self.btn = QPushButton("⟳ FETCH & VALUE")
        self.btn.setStyleSheet(f"background:{ACCENT};color:#060a12;font-weight:900;padding:8px 18px;")
        self.btn.clicked.connect(self.on_fetch)
        hdr.addWidget(self.btn)
        eng = "⚡ AVX2+FMA+OpenMP" if dcf_core.has_avx2() else "fallback"
        self.badge = QLabel(f"{eng}")
        self.badge.setStyleSheet(f"color:{GREEN};font-weight:900;")
        hdr.addWidget(self.badge)
        self.sector_badge = QLabel("Sector: —")
        self.sector_badge.setStyleSheet(f"color:{GOLD};font-weight:bold;")
        hdr.addWidget(self.sector_badge)
        hdr.addStretch(1)
        root.addLayout(hdr)
        # KPI cards row.
        kpi = QHBoxLayout()
        self.kpi_last = self._kpi("LAST", "—")
        self.kpi_fair = self._kpi("FAIR p50", "—")
        self.kpi_up = self._kpi("UPSIDE", "—")
        self.kpi_verd = self._kpi("VERDICT", "—")
        for w in (self.kpi_last, self.kpi_fair, self.kpi_up, self.kpi_verd):
            kpi.addWidget(w)
        root.addLayout(kpi)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._tab_overview()
        self._tab_fin()
        self._tab_dcf_mc()
        self._tab_optimizer()
        self._tab_peers()
        self._tab_batch()
        self._tab_benchmark()
        self.log = QTextEdit(readOnly=True)
        self.tabs.addTab(self.log, "⑧ Log")
    # Above: header + KPI cards + 8 tabs. Benchmark = risk-adjusted S&P500 compare.

    def _kpi(self, title: str, val: str) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{CARD};border:1px solid {BORDER};border-radius:8px;")
        l = QVBoxLayout(w)
        t = QLabel(title)
        t.setStyleSheet(f"color:{DIM};font-size:11px;font-weight:bold;")
        v = QLabel(val)
        v.setStyleSheet("font-size:19px;font-weight:900;")
        l.addWidget(t)
        l.addWidget(v)
        w._v = v  # type: ignore
        return w
    # Above: KPI card helper. _v holds the big number label for updates.

    def _tab_overview(self):
        w = QWidget()
        l = QHBoxLayout(w)
        self.px = _mkplot("Price — decimated ≤2000 pts (60fps)", "Year", "Price ($)", dates=True)
        l.addWidget(self.px, 3)
        self.ov = QTextEdit(readOnly=True)
        l.addWidget(self.ov, 1)
        self.tabs.addTab(w, "① Overview")
    # Above: price chart (YEAR x-axis, PRICE $ y-axis) + stats.

    def _tab_fin(self):
        w = QWidget()
        l = QHBoxLayout(w)
        self.ft = QTableWidget(0, 2)
        self.ft.setHorizontalHeaderLabels(["Item", "Value"])
        self.ft.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        l.addWidget(self.ft, 1)
        self.fcf = _mkplot("FCF history — SEC 10-K", "Year", "FCF ($B)")
        l.addWidget(self.fcf, 2)
        self.tabs.addTab(w, "② Financials")
    # Above: SEC table + cash-flow bars (YEAR x, $B y).

    def _tab_dcf_mc(self):
        w = QWidget()
        l = QHBoxLayout(w)
        box = QGroupBox("Assumptions (instant AVX2)")
        f = QFormLayout(box)
        self.i_fcf = QDoubleSpinBox(decimals=2, minimum=0, maximum=1e15, value=98e9)
        self.i_g = QDoubleSpinBox(minimum=-20, maximum=50, value=6.0, suffix=" %")
        self.i_tg = QDoubleSpinBox(minimum=0, maximum=6, value=2.5, suffix=" %")
        self.i_w = QDoubleSpinBox(minimum=2, maximum=35, value=9.0, suffix=" %")
        self.i_y = QSpinBox(minimum=1, maximum=20, value=10)
        self.i_d = QDoubleSpinBox(decimals=2, minimum=-1e15, maximum=1e15, value=54e9)
        self.i_s = QDoubleSpinBox(decimals=0, minimum=1, maximum=1e14, value=14.7e9)
        self.i_gsd = QDoubleSpinBox(minimum=0, maximum=10, value=2.0, suffix=" %")
        self.i_wj = QDoubleSpinBox(minimum=0, maximum=5, value=1.5, suffix=" %")
        for spin, name in ((self.i_fcf, "FCF₀"), (self.i_g, "Growth g"), (self.i_tg, "Terminal gT"),
                           (self.i_w, "WACC"), (self.i_y, "Years"), (self.i_d, "Net debt"),
                           (self.i_s, "Shares"), (self.i_gsd, "Growth σ"), (self.i_wj, "WACC jitter σ")):
            f.addRow(name, spin)
            spin.valueChanged.connect(lambda _=None: self._recalc_quick())
        l.addWidget(box, 1)
        right = QVBoxLayout()
        self.mc_hist = _mkplot("Monte Carlo distribution — 100k fair values", "Fair value ($/share)", "Frequency")
        right.addWidget(self.mc_hist, 3)
        self.mc_txt = QTextEdit(readOnly=True)
        self.mc_txt.setFixedHeight(130)
        right.addWidget(self.mc_txt, 1)
        rw = QWidget()
        rw.setLayout(right)
        l.addWidget(rw, 2)
        self.tabs.addTab(w, "③ DCF + Monte Carlo")
    # Above: left = inputs, right = MC HISTOGRAM graph (your request) + p10/p50/p90.

    def _tab_optimizer(self):
        w = QWidget()
        l = QHBoxLayout(w)
        box = QGroupBox("Break-test setup — 350 rates × 0.03% × 100k × haircuts = 100M+")
        f = QFormLayout(box)
        self.o_hair = QDoubleSpinBox(minimum=0, maximum=60, value=20.0, suffix=" % haircut")
        self.o_mc = QSpinBox(minimum=10000, maximum=200000, value=100000)
        self.o_n = QSpinBox(minimum=50, maximum=350, value=350)
        self.o_step = QDoubleSpinBox(minimum=0.01, maximum=0.2, value=0.03, suffix=" % step")
        self.o_reqp = QDoubleSpinBox(minimum=50, maximum=99, value=75.0, suffix=" % win")
        self.o_reqm = QDoubleSpinBox(minimum=0, maximum=60, value=20.0, suffix=" % margin")
        self.o_topn = QSpinBox(minimum=3, maximum=18, value=8)
        self.o_autorisk = QCheckBox("Auto sector risk (override below)")
        self.o_autorisk.setChecked(True)
        f.addRow("Earnings haircut", self.o_hair)
        f.addRow("MC per rate", self.o_mc)
        f.addRow("Rates", self.o_n)
        f.addRow("Step", self.o_step)
        f.addRow("Need win-prob", self.o_reqp)
        f.addRow("Need margin", self.o_reqm)
        f.addRow("Peer top-N", self.o_topn)
        f.addRow(self.o_autorisk)
        self.btn_sweep = QPushButton("▶ RUN 100M BREAK-TEST")
        self.btn_sweep.setStyleSheet(f"background:{GREEN};color:#04120c;font-weight:900;padding:10px;")
        self.btn_sweep.clicked.connect(self.on_sweep)
        f.addRow(self.btn_sweep)
        self.sweep_bar = QProgressBar()
        f.addRow(self.sweep_bar)
        l.addWidget(box, 1)
        right = QVBoxLayout()
        self.verdict = QLabel("Verdict: —")
        self.verdict.setStyleSheet("font-size:20px;font-weight:900;padding:10px;")
        self.verdict.setAlignment(Qt.AlignCenter)
        right.addWidget(self.verdict)
        self.sweep_plot = _mkplot("Survival vs rate: p50 (blue) p10 (red) vs current price (gold)",
                                    "WACC (%)", "Fair value ($/share)")
        right.addWidget(self.sweep_plot, 3)
        self.sweep_txt = QTextEdit(readOnly=True)
        self.sweep_txt.setFixedHeight(170)
        right.addWidget(self.sweep_txt, 1)
        rw = QWidget()
        rw.setLayout(right)
        l.addWidget(rw, 2)
        self.tabs.addTab(w, "④ ★ Optimizer")
    # Above: optimizer = your core ask. Haircut + 350 rates + sector risk + verdict.

    def _tab_peers(self):
        w = QWidget()
        l = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Live top-N by market cap (price × SEC shares):"))
        self.btn_peers = QPushButton("↻ Load peers")
        self.btn_peers.clicked.connect(self.on_peers)
        row.addWidget(self.btn_peers)
        row.addStretch(1)
        l.addLayout(row)
        self.pt = QTableWidget(0, 5)
        self.pt.setHorizontalHeaderLabels(["Ticker", "Company", "Price", "MktCap", "Sector"])
        self.pt.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        l.addWidget(self.pt)
        self.tabs.addTab(w, "⑤ Peers")
    # Above: live leaderboard. Proves "seen live" top-N.

    def _tab_batch(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Tickers (comma or newline): e.g. AAPL, MSFT, NVDA — or Load .txt"))
        self.bt = QTextEdit("AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, JPM")
        self.bt.setFixedHeight(80)
        l.addWidget(self.bt)
        row = QHBoxLayout()
        self.btn_file = QPushButton("📂 Load .txt")
        self.btn_file.clicked.connect(self.on_file)
        self.btn_batch = QPushButton("▶ RUN BATCH")
        self.btn_batch.setStyleSheet(f"background:{ACCENT};color:#060a12;font-weight:900;padding:8px 20px;")
        self.btn_batch.clicked.connect(self.on_batch)
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        row.addWidget(self.btn_file)
        row.addWidget(self.btn_batch)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        l.addLayout(row)
        self.bbar = QProgressBar()
        self.blab = QLabel("idle")
        l.addWidget(self.bbar)
        l.addWidget(self.blab)
        self.br = QTableWidget(0, 6)
        self.br.setHorizontalHeaderLabels(["Ticker", "Last", "Fair", "Upside", "FCF", "Signal"])
        self.br.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        l.addWidget(self.br)
        self.tabs.addTab(w, "⑥ Batch")
    # Above: batch = 1000s tickers with progress bar (your request).

    def _tab_benchmark(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("S&P 500 benchmark — LIVE prices both legs, RISK-ADJUSTED (not raw %). "
                           "Sharpe=(Rp−Rf)/σ · Sortino uses down-σ only · Calmar=CAGR/|MaxDD| · "
                           "beta=Cov/Var · alpha=CAGR−beta·CAGR_m."))
        mid = QHBoxLayout()
        self.bm_eq = _mkplot("Growth of $1 — stock (blue) vs S&P500 (gold), LIVE", "Year", "Growth ($)", dates=True)
        mid.addWidget(self.bm_eq, 3)
        self.bm_dd = _mkplot("Drawdown — pain lived through (stock red, SPY gray)", "Year", "Drawdown (%)", dates=True)
        mid.addWidget(self.bm_dd, 2)
        l.addLayout(mid, 3)
        self.bm_table = QTableWidget(0, 3)
        self.bm_table.setHorizontalHeaderLabels(["Metric (risk lens)", "Stock", "S&P 500"])
        self.bm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        l.addWidget(self.bm_table, 2)
        self.tabs.addTab(w, "⑦ ★ Benchmark vs S&P500")
    # Above: benchmark tab. Equity curves + drawdown + risk table prove reward-PER-risk.

    # ---------- helpers ----------
    @Slot(str)
    def say(self, m: str):
        self.log.append(m)
    # Above: log helper. @Slot = Qt knows its thread affinity (runs in GUI thread).

    def inputs(self):
        """Current DCF inputs from tab ③."""
        return (self.i_fcf.value(), self.i_g.value() / 100, self.i_tg.value() / 100,
                self.i_w.value() / 100, self.i_y.value(), self.i_d.value(),
                self.i_s.value(), self.i_gsd.value() / 100, self.i_wj.value() / 100)
    # Above: one place reading all spins (no drift between tabs).

    # ---------- fetch ----------
    def on_fetch(self):
        if self.thread and self.thread.isRunning():
            return
        self.btn.setEnabled(False)
        self.say(f"—— LIVE fetch {self.ticker.text()} ——")
        override = None if self.erp_live.isChecked() else self.erp.value() / 100
        self.thread = QThread()
        self.wk = FetchWorker(self.ticker.text().strip() or "AAPL", self.skey.text(),
                              "", override)
        self.wk.moveToThread(self.thread)
        self.thread.started.connect(self.wk.run)
        # QueuedConnection = slot ALWAYS runs in the GUI thread, never the worker.
        # (Plain lambdas have no thread affinity and would run IN the worker thread,
        # touching widgets off-thread -> "Cannot create children..." + segfault.)
        self.wk.log.connect(self.say, Qt.QueuedConnection)
        self.wk.done.connect(self.on_data, Qt.QueuedConnection)
        self.wk.done.connect(self.thread.quit)
        self.thread.finished.connect(self._forget_thread)
        self.thread.start()
    # Above: background fetch (UI alive). ERP live unless you uncheck + type manual %.

    @Slot()
    def _forget_thread(self):
        """Step the finished-thread refs aside (emitted in GUI thread — safe)."""
        try:
            if self.thread is not None and not self.thread.isRunning():
                self.thread = None
        except Exception:
            self.thread = None
    # Above: lets old finished QThreads die instead of piling up. Not a worker slot.

    def closeEvent(self, e):
        """Never tear down a live worker: stop it first or Qt aborts."""
        try:
            try:
                self.bw.stop()
            except Exception:
                pass
            if self.thread is not None and self.thread.isRunning():
                self.thread.quit()
                self.thread.wait(5000)
        except Exception:
            pass
        e.accept()
    # Above: closing mid-fetch/sweep/batch exits cleanly instead of segfaulting.

    @Slot(dict)
    def on_data(self, d: dict):
        from python import benchmark
        self.btn.setEnabled(True)
        self.prices, self.spy = d.get("prices", []), d.get("spy", [])
        self.beta, self.fund = d["beta"], d["fund"]
        self.rf, self.rfsrc = d.get("rf"), d.get("rfsrc", "—")
        self.erp, self.erpsrc = d.get("erp"), d.get("erpsrc", "—")
        self.sector, self.moment = d["sector"], d.get("moment", "—")
        if d.get("wacc_base") is not None:
            self.wacc_base = d["wacc_base"]
        self.last = d["last"]
        self.live_badge.setText(f"● LIVE @ {self.moment}")
        self.sector_badge.setText(f"Sector: {self.sector}")
        # LIVE dynamic sector risk: base + live excess-vol adjustment (fresh, not fixed).
        if self.o_autorisk.isChecked():
            try:
                s_px = np.array([p["close"] for p in self.prices]) if self.prices else np.array([])
                m_px = np.array([p["close"] for p in self.spy]) if self.spy else np.array([])
                sv = float(np.std(s_px[1:] / s_px[:-1] - 1) * np.sqrt(252)) if len(s_px) > 60 else None
                mv = float(np.std(m_px[1:] / m_px[:-1] - 1) * np.sqrt(252)) if len(m_px) > 60 else None
                rp, rm, why = sector.dynamic_risk(self.sector, sv, mv)
            except Exception:
                rp, rm, why = (*sector.SECTOR_RISK.get(self.sector, sector.SECTOR_RISK["Default"]), "base")
            self.o_reqp.setValue(rp * 100)
            self.o_reqm.setValue(rm * 100)
            self.say(f"Sector risk LIVE: {self.sector} → need {rp*100:.0f}%/{rm*100:.0f}% ({why})")
        # Refresh WACC + inputs from LIVE data (None-safe: keep old WACC if live failed).
        if d.get("wacc_base") is not None:
            self.i_w.setValue(self.wacc_base * 100)
        try:
            self.px.clear()
            if self.prices:
                cl = np.array([p["close"] for p in self.prices])
                ts = _dates_to_ts([p["date"] for p in self.prices])
                st = max(1, len(cl) // 2000)
                self.px.plot(ts[::st], cl[::st], pen=pg.mkPen(ACCENT, width=2), connect="finite")
                rf_t = f"{self.rf*100:.2f}%" if self.rf is not None else "LIVE FAILED"
                erp_t = f"{self.erp*100:.2f}%" if self.erp is not None else "LIVE FAILED"
                wb_t = f"{self.wacc_base*100:.2f}%" if d.get("wacc_base") is not None else "— (live failed)"
                self.ov.setText(f"{self.ticker.text().upper()}  last ${cl[-1]:.2f}  LIVE @ {self.moment}\n"
                                f"rows {len(cl)} ({d['psrc']})  beta {self.beta:.2f}\n"
                                f"Rf {rf_t} ({self.rfsrc})\nERP {erp_t} ({self.erpsrc})\nWACC base {wb_t}")
        except Exception:
            self.say(traceback.format_exc())
        try:
            f = self.fund
            self.ft.setRowCount(0)
            if f.get("ok"):
                if f.get("fcf0"):
                    self.i_fcf.setValue(float(f["fcf0"]))
                if f.get("net_debt") is not None:
                    self.i_d.setValue(float(f["net_debt"]))
                if f.get("shares"):
                    self.i_s.setValue(float(f["shares"]))
                if f.get("cagr") is not None:
                    self.i_g.setValue(float(np.clip(f["cagr"] * 100, -10, 30)))
                rows = [("Company", f["title"]), ("SIC", f"{f.get('sic','')} {f.get('sicDescription','')}"),
                        ("Fetched", self.moment),
                        ("TTM FCF", f"${f['fcf0']/1e9:.2f}B" if f.get("fcf0") else "—"),
                        ("Cash", f"${(f.get('cash') or 0)/1e9:.2f}B"),
                        ("Debt", f"${(f.get('debt') or 0)/1e9:.2f}B"),
                        ("Net debt", f"${(f.get('net_debt') or 0)/1e9:.2f}B"),
                        ("Shares", f"{(f.get('shares') or 0)/1e9:.2f}B")]
                for k, v in rows:
                    r = self.ft.rowCount()
                    self.ft.insertRow(r)
                    self.ft.setItem(r, 0, QTableWidgetItem(k))
                    self.ft.setItem(r, 1, QTableWidgetItem(v))
                h = f.get("fcf_hist", {})
                self.fcf.clear()
                if h:
                    ys = sorted(h)
                    vals = np.array([h[y] / 1e9 for y in ys])
                    self.fcf.addItem(pg.BarGraphItem(x=list(range(len(ys))), height=vals,
                                                     width=0.6, brush=GREEN))
                    self.fcf.getAxis("bottom").setTicks([[(i, str(y)) for i, y in enumerate(ys)]])
        except Exception:
            self.say(traceback.format_exc())
        # Benchmark vs S&P 500 — risk-adjusted, LIVE both legs.
        try:
            if self.prices and self.spy and self.rf is not None:
                s_px = np.array([p["close"] for p in self.prices])
                m_px = np.array([p["close"] for p in self.spy])
                m = benchmark.risk_metrics(s_px, m_px, self.rf)
                if "error" not in m:
                    n = m["n"]
                    eq_s, eq_m = s_px[-n:] / s_px[-n], m_px[-n:] / m_px[-n]
                    ts_m = _dates_to_ts([p["date"] for p in self.spy])[-n:]
                    dd_pct = m["dd_curve_s"] * 100.0
                    self.bm_eq.clear()
                    self.bm_eq.plot(ts_m, eq_s, pen=pg.mkPen(ACCENT, width=2), name="stock", connect="finite")
                    self.bm_eq.plot(ts_m, eq_m, pen=pg.mkPen(GOLD, width=2), name="SP500", connect="finite")
                    self.bm_dd.clear()
                    self.bm_dd.plot(ts_m, dd_pct, pen=pg.mkPen(RED, width=1.5), fillLevel=0,
                                    fillBrush=(255, 77, 109, 60), connect="finite")
                    rows_b = [("CAGR (ann.)", f"{m['cagr_stock']*100:.1f}%", f"{m['cagr_spy']*100:.1f}%"),
                              ("Total", f"{m['total_stock']*100:.1f}%", f"{m['total_spy']*100:.1f}%"),
                              ("Volatility σ", f"{m['vol_stock']*100:.1f}%", f"{m['vol_spy']*100:.1f}%"),
                              ("Sharpe (per all risk)", f"{m['sharpe']:.2f}", "—"),
                              ("Sortino (per DOWN risk)", f"{m['sortino']:.2f}", "—"),
                              ("Calmar (per worst crash)", f"{m['calmar']:.2f}", "—"),
                              ("Max drawdown", f"{m['maxdd_stock']*100:.1f}%", f"{m['maxdd_spy']*100:.1f}%"),
                              ("Beta vs SPY", f"{m['beta']:.2f}", "1.00"),
                              ("Alpha (ann.)", f"{m['alpha']*100:.1f}%", "—"),
                              ("R²", f"{m['r2']:.2f}", "1.00"),
                              ("Up capture", f"{m['up_capture']*100:.0f}%", "100%"),
                              ("Down capture", f"{m['down_capture']*100:.0f}%", "100%"),
                              ("VaR95 daily", f"{m['var95_daily']*100:.2f}%", "—")]
                    self.bm_table.setRowCount(0)
                    for k, a, b in rows_b:
                        r = self.bm_table.rowCount()
                        self.bm_table.insertRow(r)
                        self.bm_table.setItem(r, 0, QTableWidgetItem(k))
                        self.bm_table.setItem(r, 1, QTableWidgetItem(a))
                        self.bm_table.setItem(r, 2, QTableWidgetItem(b))
                    better = "BETTER per risk" if m["sharpe"] > 1.0 and m["sortino"] > 1.0 else "mixed — see Sharpe/Sortino"
                    self.say(f"Benchmark LIVE ({n}d): Sharpe {m['sharpe']:.2f} Sortino {m['sortino']:.2f} "
                             f"Calmar {m['calmar']:.2f} beta {m['beta']:.2f} alpha {m['alpha']*100:.1f}% "
                             f"maxDD {m['maxdd_stock']*100:.1f}% → {better}")
        except Exception:
            self.say(traceback.format_exc())
        self._recalc_quick()
    # Above: LIVE data arrival. Stamps, dynamic sector risk, risk-adjusted benchmark. Nothing stale.

    # ---------- quick recalc (single + 100k MC histogram graph) ----------
    def _recalc_quick(self):
        try:
            fcf0, g, tg, w, yrs, debt, sh, gsd, wj = self.inputs()
            last = self.prices[-1]["close"] if self.prices else self.last
            fv = dcf_core.fair_value(fcf0, g, w, tg, yrs, debt, sh)
            up = (fv / last - 1) * 100 if last > 0 else float("nan")
            self.kpi_last._v.setText(_card_value(last))
            self.kpi_fair._v.setText(_card_value(fv))
            self.kpi_up._v.setText(f"{up:+.1f}%" if last > 0 else "—")
            # MC histogram graph (exact path, 100k).
            mc = dcf_core.montecarlo(100000, fcf0, g, gsd, w, wj, tg, yrs, debt, sh)
            p10, p50, p90 = (float(np.percentile(mc, q)) for q in (10, 50, 90))
            hist, edges = np.histogram(mc, bins=60)
            self.mc_hist.clear()
            xs = (edges[:-1] + edges[1:]) / 2
            self.mc_hist.addItem(pg.BarGraphItem(x=xs, height=hist, width=(edges[1] - edges[0]) * 0.9,
                                                 brush="#5aa2ff"))
            for x, col in ((last, GOLD), (p50, GREEN), (p10, RED)):
                if x and np.isfinite(x):
                    self.mc_hist.addLine(x=x, pen=pg.mkPen(col, width=2,
                                         style=Qt.DashLine if col != GOLD else Qt.SolidLine))
            verdict = "WAIT"
            color = GOLD
            if last > 0 and np.isfinite(p50):
                if p50 > last * 1.2 and p10 > last:
                    verdict, color = "BUY", GREEN
                elif p50 < last:
                    verdict, color = "SELL / OVERVALUED", RED
            self.kpi_verd._v.setText(verdict)
            self.kpi_verd._v.setStyleSheet(f"font-size:19px;font-weight:900;color:{color};")
            self.mc_txt.setText(f"MC 100k  p10 ${p10:,.2f}  p50 ${p50:,.2f}  p90 ${p90:,.2f}\n"
                                 f"gold=solid current price  green=p50  red=p10  "
                                 f"{'AVX2' if dcf_core.has_avx2() else 'fallback'}")
        except Exception:
            self.say(traceback.format_exc())
    # Above: instant single + MC GRAPH (bars + price lines). Sliders feel live (AVX2 µs-ms).

    # ---------- sweep ----------
    def on_sweep(self):
        if self.thread and self.thread.isRunning():
            self.say("busy — wait for fetch/sweep")
            return
        if self.rf is None or self.erp is None:
            self.sweep_txt.setText("LIVE Rf/ERP missing — hit FETCH first (no fixed rates used, by your rule).")
            self.say("Sweep blocked: live Rf/ERP not yet fetched (refusing fixed numbers).")
            return
        fcf0, g, tg, _, yrs, debt, sh, gsd, wj = self.inputs()
        last = self.prices[-1]["close"] if self.prices else self.last
        if last <= 0 or fcf0 <= 0:
            self.sweep_txt.setText("Fetch a real ticker first (need price + SEC FCF).")
            return
        h = self.o_hair.value() / 100
        haircuts = sorted({0.0, h, min(0.5, h * 1.5 or 0.15)})
        params = {"base_fcf": fcf0, "g": g, "g_sd": gsd, "wacc_base": self.wacc_base,
                  "wacc_step": self.o_step.value() / 100, "n_rates": self.o_n.value(),
                  "tgrowth": tg, "years": yrs, "debt": debt, "shares": sh, "last": last,
                  "haircuts": haircuts, "mc": self.o_mc.value(), "seed": 7,
                  "w_jit": wj}
        self.btn_sweep.setEnabled(False)
        self.sweep_bar.setValue(0)
        self.thread = QThread()
        self.sw = SweepWorker(params)
        self.sw.moveToThread(self.thread)
        self.thread.started.connect(self.sw.run)
        self.sw.prog.connect(self.on_sweep_prog, Qt.QueuedConnection)
        self.sw.log.connect(self.say, Qt.QueuedConnection)
        self.sw.done.connect(self.on_sweep_done, Qt.QueuedConnection)
        self.sw.done.connect(self.thread.quit)
        self.thread.finished.connect(self._forget_thread)
        self.thread.start()
        self.say(f"Sweep {self.o_n.value()} rates × {self.o_mc.value()/1000:.0f}k × {len(haircuts)} haircuts "
                 f"= {self.o_n.value()*self.o_mc.value()*len(haircuts)/1e6:.0f}M vals…")
    # Above: launches 100M break-test in background with progress bar.

    @Slot(int, str)
    def on_sweep_prog(self, v: int, t: str):
        """Progress bar step — bound Slot + QueuedConnection = always GUI thread."""
        self.sweep_bar.setValue(v)
        self.sweep_txt.setText(f"{v}% {t}…")
    # Above: was a lambda (ran IN the worker thread -> segfault). Never again.

    @Slot(dict)
    def on_sweep_done(self, d: dict):
        self.btn_sweep.setEnabled(True)
        self.sweep_bar.setValue(100)
        self.sweep_cache = d
        try:
            res = d["results"]
            if not res:
                return
            reqp, reqm = self.o_reqp.value() / 100, self.o_reqm.value() / 100
            n = self.o_n.value()
            step = self.o_step.value() / 100
            ws = np.array([self.wacc_base + i * step for i in range(n)]) * 100
            self.sweep_plot.clear()
            self.sweep_plot.addLegend()
            fair0, drop_txt, verdict, color = 0, "", "SELL", RED
            for h, (P50, P10, PR) in sorted(res.items()):
                self.sweep_plot.plot(ws, P50, pen=pg.mkPen(ACCENT if h == 0 else (GREEN if h == max(res) else DIM),
                                                           width=3 if h == 0 else 2),
                                     name=f"p50 hair {h*100:.0f}%")
                if h == 0:
                    self.sweep_plot.plot(ws, P10, pen=pg.mkPen(RED, width=1, style=Qt.DashLine), name="p10")
                    fair0 = float(P50[0])
                    # Max survivable rate: last rate where prob >= required.
                    surv = np.where(PR >= reqp)[0]
                    maxr = f"{ws[surv[-1]]:.2f}%" if len(surv) else "NONE (fails at base)"
                    prob0, p10_0 = float(PR[0]), float(P10[0])
                    marg = (fair0 / self.last - 1) if self.last else -1
                    drop = (1 - fair0 / self.last) * 100 if self.last and fair0 < self.last else 0
                    if prob0 >= reqp and marg >= reqm:
                        verdict, color = "BUY", GREEN
                        drop_txt = "already fair — margin ok"
                    elif fair0 >= self.last:
                        verdict, color = "WAIT (thin margin)", GOLD
                        drop_txt = f"hold; needs -{max(0, reqm*100 - marg*100):.1f}pp more margin"
                    else:
                        verdict, color = "SELL / WAIT FOR DIP", RED
                        drop_txt = f"needs -{drop:.1f}% to ${fair0:,.2f}"
                    self.sweep_txt.setText(
                        f"Fair p50 (base) ${fair0:,.2f} vs last ${self.last:,.2f} → {marg*100:+.1f}%\n"
                        f"P(fair≥price) {prob0*100:.1f}% (need {reqp*100:.0f}%) p10 ${p10_0:,.2f}\n"
                        f"Max survivable WACC {maxr} | {drop_txt}\n"
                        f"selfcheck dev {d['selfcheck']:.1e} {'PASS' if d['selfcheck'] < 1e-3 else 'FAIL'} | "
                        f"{d['total_vals']/1e6:.0f}M vals in {d['secs']:.1f}s | sector {self.sector}")
            self.sweep_plot.addLine(y=self.last, pen=pg.mkPen(GOLD, width=2))
            self.verdict.setText(f"{verdict} — fair ${fair0:,.2f} | {drop_txt}")
            self.verdict.setStyleSheet(f"font-size:20px;font-weight:900;padding:10px;color:{color};"
                                       f"border:2px solid {color};border-radius:10px;")
            self.kpi_fair._v.setText(_card_value(fair0))
            up = (fair0 / self.last - 1) * 100 if self.last else 0
            self.kpi_up._v.setText(f"{up:+.1f}%")
            self.kpi_verd._v.setText(verdict.split(" ")[0])
            self.kpi_verd._v.setStyleSheet(f"font-size:19px;font-weight:900;color:{color};")
        except Exception:
            self.say(traceback.format_exc())
    # Above: verdict engine. BUY only if break-test survival beats SECTOR risk table.
    # Shows fair price, drop-needed %, max survivable rate, selfcheck PASS/FAIL.

    # ---------- peers ----------
    def on_peers(self):
        try:
            self.say(f"Loading live top-{self.o_topn.value()} by mcap…")
            peers = sector.get_live_peers(self.o_topn.value(), self.skey.text())
            self.pt.setRowCount(0)
            for p in peers:
                r = self.pt.rowCount()
                self.pt.insertRow(r)
                mc = f"${p['mcap']/1e12:.2f}T" if p["mcap"] >= 1e12 else f"${p['mcap']/1e9:.0f}B"
                for c, v in ((0, p["ticker"]), (1, p["title"]), (2, f"${p['price']:.2f}"),
                             (3, mc), (4, p["sector"])):
                    self.pt.setItem(r, c, QTableWidgetItem(v))
            self.say(f"Peers live: {', '.join(p['ticker'] for p in peers)}")
        except Exception:
            self.say(traceback.format_exc())
    # Above: live leaderboard proves top-N by CURRENT cap.

    # ---------- batch ----------
    def on_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Tickers .txt", str(Path.home()), "*.txt")
        if p:
            try:
                self.bt.setText(Path(p).read_text()[:20000])
            except Exception as e:
                self.say(str(e))
    # Above: load 1000s tickers from file.

    def on_batch(self):
        if self.thread and self.thread.isRunning():
            return
        raw = self.bt.toPlainText().replace("\n", ",")
        ticks = sorted({t.strip().upper().split(".")[0] for t in raw.split(",") if t.strip()})[:5000]
        if not ticks:
            return
        self.thread = QThread()
        self.bw = BatchWorker(ticks, self.skey.text())
        self.bw.moveToThread(self.thread)
        self.thread.started.connect(self.bw.run)
        self.bw.prog.connect(self.on_batch_prog, Qt.QueuedConnection)
        self.bw.log.connect(self.say, Qt.QueuedConnection)
        self.bw.done.connect(self.on_batch_done, Qt.QueuedConnection)
        self.bw.done.connect(self.thread.quit)
        self.thread.finished.connect(self._forget_thread)
        self.thread.start()
    # Above: batch with progress bar (your request). Up to 5000 tickers.

    @Slot(int, str)
    def on_batch_prog(self, v: int, t: str):
        """Progress step — bound Slot + QueuedConnection = always GUI thread."""
        self.bbar.setValue(v)
        self.blab.setText(t)
    # Above: was a lambda (ran IN the worker thread -> segfault). Never again.

    def on_stop(self):
        try:
            self.bw.stop()
        except Exception:
            pass
    # Above: stop button.

    @Slot(list)
    def on_batch_done(self, rows: list):
        self.br.setRowCount(0)
        for t, last, fair, up, fcf, sig in rows:
            r = self.br.rowCount()
            self.br.insertRow(r)
            for c, v in enumerate((t, last, fair, up, fcf, sig)):
                it = QTableWidgetItem(str(v))
                if sig == "BUY":
                    it.setBackground(QColor("#0d3b2e"))
                elif sig == "SELL":
                    it.setBackground(QColor("#3b1119"))
                self.br.setItem(r, c, it)
        self.blab.setText(f"done {len(rows)} tickers")
        self.say(f"Batch done: {len(rows)} tickers")
    # Above: results table, green BUY / red SELL rows.


def main():
    import faulthandler
    faulthandler.enable()  # segfault? next time you get a Python traceback instead of silence
    # Software render on purpose: pyqtgraph OpenGL codepath makes a QOpenGLContext
    # current from worker threads on Qt 6.10+/WSLg and aborts with
    # "Cannot make QOpenGLContext current in a different thread".
    # Our threads never touch GL (signals only), but the GL viewport still trips it,
    # so OpenGL stays OFF. Charts are decimated ≤2000 pts — still 60fps on Iris Xe.
    pg.setConfigOptions(antialias=True, useOpenGL=False, leftButtonPan=False)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # mandatory for dark on all platforms
    pal = QPalette()  # professional dark palette (titlebar + widgets)
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TXT))
    pal.setColor(QPalette.Base, QColor("#0a0e14"))
    pal.setColor(QPalette.AlternateBase, QColor(CARD))
    pal.setColor(QPalette.Text, QColor(TXT))
    pal.setColor(QPalette.Button, QColor("#1f2637"))
    pal.setColor(QPalette.ButtonText, QColor(TXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    w = Main()
    w.show()
    sys.exit(app.exec())
# Above: Fusion + palette + QSS = professional terminal look on Debian/WSLg.


if __name__ == "__main__":
    main()
# Above: run: source ~/ml_env/bin/activate && python gui/app.py
