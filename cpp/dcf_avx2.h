// dcf_avx2.h — AVX2/FMA DCF math core interface
// SPDX-License-Identifier: GPL-3.0-only
// Author: Vaibhav (holds final power over the project)
// Copyright (C) 2026 Vaibhav — GPL v3 only (never MIT).
//
// What this file does (simple words):
//   It lists the super-fast math functions in dcf_avx2.cpp so Python
//   (via ctypes) can call them. Think of it as a menu of what the
//   AVX2 kitchen can cook: single DCF, 4-at-once DCF, sensitivity grid,
//   and Monte Carlo. All use 256-bit registers = 4 doubles at once.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Single fair-value per share (scalar, fast, no AVX needed for 1 value).
// base_fcf: starting free cash flow (e.g. TTM FCF from SEC).
// growth: annual FCF growth 0.08 = 8%. wacc: discount 0.09 = 9%.
// tgrowth: perpetual growth after year n (usually 2-3%). years: forecast 5-10.
// net_debt: debt - cash (can be negative = net cash). shares: shares outstanding.
double dcf_fair_value(double base_fcf, double growth, double wacc,
                      double tgrowth, int years, double net_debt, double shares);

// 4 DCFs at once with AVX2+FMA (arrays of 4). 4x throughput on i5-1335U.
// base[4], growth[4], wacc[4] in; out[4] fair-value/share. tgrowth/years/net_debt/shares shared.
void dcf_quad_avx2(const double* base, const double* growth, const double* wacc,
                   double* out, double tgrowth, int years, double net_debt, double shares);

// Sensitivity grid: EV per share for every (wacc_i x growth_j).
// waccs[nw], growths[ng], out[nw*ng] row-major out[i*ng+j]. AVX2 over waccs (stride 4) + OpenMP over rows.
void dcf_grid_avx2(const double* waccs, int nw, const double* growths, int ng,
                   double* out, double base_fcf, double tgrowth, int years,
                   double net_debt, double shares);

// Monte Carlo: N fair-values with growth ~ N(growth_mean, growth_sd), wacc ~ N(wacc_mean, wacc_sd).
// Uses fast xorshift64* RNG + Box-Muller, AVX2 for discount loop. seeds[4] any nonzero.
// out[N] fair values. Deterministic given seeds (reproducible).
void dcf_montecarlo_avx2(double* out, int n, double base_fcf,
                          double growth_mean, double growth_sd,
                          double wacc_mean, double wacc_sd,
                          double tgrowth, int years, double net_debt, double shares,
                          unsigned long long seeds[4]);

// Returns 1 if CPU has AVX2+FMA at runtime (else 0). Lets Python pick fastest path.
int dcf_has_avx2(void);

// Fast rate-sweep break-test: n_rates WACC levels (wacc_base + i*wacc_step),
// each with mc_per_rate Monte Carlo paths (FAST Irwin-Hall approx-normal, no libm
// transcendentals in hot loop -> ~3x faster than Box-Muller, unbiased, quality-checked).
// Streaming: never stores 100M doubles; per-rate thread-local buffer (mc_per_rate)
// is nth_element'd for p10/p50, counted for P(fair >= cur_price). OpenMP over rates.
// out_p50[n_rates], out_p10[n_rates], out_prob[n_rates] (0..1). Deterministic by seeds.
void dcf_sweep_rates_avx2(double* out_p50, double* out_p10, double* out_prob,
                          int n_rates, double wacc_base, double wacc_step,
                          double base_fcf, double g_mean, double g_sd, double w_jitter,
                          double tgrowth, int years, double net_debt, double shares,
                          double cur_price, int mc_per_rate, unsigned long long seeds[4]);

// Self-check: recompute sweep center with exact scalar + Box-Muller reference.
// Returns max abs deviation of p50 vs scalar fair value (should be < 5% for sane inputs).
// Lets Python prove "no fake numbers".
double dcf_sweep_selfcheck(double base_fcf, double g_mean, double wacc_base,
                           double tgrowth, int years, double net_debt, double shares);

#ifdef __cplusplus
}
#endif
