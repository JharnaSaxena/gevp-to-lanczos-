#!/usr/bin/env python3
"""
# figures.py
# Run:  python figures.py
# Writes: output/fig2_full.txt, output/higher_rank.txt, output/ordering.txt,
#         fig2_data.csv, fig2_reproduction.png, fig333.png,
#         fig6_reproduction.png, fig7_reproduction.png
# Fig 2 pipeline
# print_history(method, history, E_true)
#   in  : method      str label ("block" / "scalar_C00" / ...)
#         history     list[dict] from extract_all
#         E_true      list[mpf] true energies
#   out : (prints a Ritz-value table per m)
#
# plot_fig2(all_data, png="fig2_reproduction.png")
#   in  : all_data    dict{block, scalar_C00, scalar_C11 -> history}
#         png         output file name
#   out : (saves png; prints data-integrity hashes)
#
# run_figure()
#   in  : (none)
#   out : (all_data, df)       history dict + pandas DataFrame
#   does: full fig 2 pipeline — runs block Lanczos (m=8) + both scalar
#         Lanczos (m=16), writes fig2_data.csv, plots fig 2, checks caption
#   eq  : (15), (24), (36)-(37), (60)
#
# Post-hoc checks (read fig2_data.csv)
# run_higher_rank_check(csv_path="fig2_data.csv")
#   in  : csv_path    str
#   out : (none — prints; captured to output/higher_rank.txt)
#   does: checks saturation at m=8/16, per-rank convergence, non-degeneracy
#   eq  : cross-checks paper claims about (37)
#
# run_ordering_diagnostic(csv_path="fig2_data.csv")
#   in  : csv_path    str
#   out : (none — prints; captured to output/ordering.txt)
#   does: per-rank trajectory of nearest true level vs m; lock-on detection
#   eq  : ordered-Ritz diagnostic on (37)
#
# Fig 3 — overlap factors |Z_ka^(m)|
# run_fig3(C, E, Z_true)
#   in  : C       list[mp.matrix]  correlator
#         E       list[mpf]        true energies
#         Z_true  list[(mpf, mpf)] true overlaps
#   out : (block_hist, scalar_hist)     both lists[dict] with Z arrays
#   does: plots |Z_ka^(m)| for block + scalar Lanczos; saves fig333.png
#   eq  : (36)-(37) + overlap convention
#
# Fig 6 — relative error + KPS bound
# tan2_theta(n, r, Zlist, C0)
#   in  : n         int, level index
#         r       int, block size
#         Zlist   list[(complex, complex)]  overlaps
#         C0      mp.matrix 2×2   C(0)
#   out : float, tan²θ_n
#   eq  : (86)
#
# kps_bound(n, r, m, tan2, E, K_n)
#   in  : n, r, m           int
#         tan2            float (from tan2_theta)
#         E               list[mpf] true energies
#         K_n             float, K_n^(m) factor
#   out : float, KPS upper bound
#   eq  : (85)
#
# run_fig6(C, E, Z, block_hist, scalar_hist)
#   in  : C, E, Z, block_hist, scalar_hist
#   out : (saves fig6_reproduction.png)
#   does: relative error curves + KPS bounds for n=0,1
#   eq  : (85), (86)
#
# Fig 7 — residual bounds on E_k^(m)
# run_fig7(C, E)
#   in  : C    list[mp.matrix]
#         E    list[mpf]
#   out : (saves fig7_reproduction.png)
#   does: error bars from residual B_k, requires block_lanczos(…, m+1)
#   eq  : (60)-type residual bound via γ_{m+1}
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from mpmath import mp

from lanczos_core import (
    build_mock, block_lanczos, assemble_T, ritz_energies, ritz_sorted,
    extract_all, _read_csv_lossless, _nearest_true_index, NSTR_DIGITS,
)

OUT = Path.cwd()
# Fig. 2 — reproduction + data export
def print_history(method, history, E_true):
    print(f"[{method}] all Ritz values per iteration")
    for entry in history:
        m, t = entry["m"], entry["t"]
        E_vals = entry["energies"]
        print(f"  m={m:2d}  t=2m-1={t:2d}   ({len(E_vals)} eigenvalues)")
        for k, E_k in enumerate(E_vals):
            true_val = E_true[k] if k < len(E_true) else None
            err = (f"|err|={mp.nstr(abs(mp.re(E_k) - true_val), 3)}"
                   if true_val is not None else "—")
            tv = mp.nstr(true_val, 4) if true_val is not None else "—"
            print(f"    rank {k:2d}: λ = {mp.nstr(mp.re(entry['lambdas'][k]), 8):<14}"
                  f"E^(m) = {mp.nstr(mp.re(E_k), 18):<24} "
                  f"true = {tv:<6} {err}")
    print()


def plot_fig2(all_data, png="fig2_reproduction.png"):
    def _snapshot(data):
        return {
            m: [float(mp.re(Ek)) for e in data[m] for Ek in e["energies"]]
            for m in ("block", "scalar_C00", "scalar_C11")
        }

    snap_before = _snapshot(all_data)
    hash_before = {
        m: hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]
        for m, v in snap_before.items()
    }

    csv_path = "fig2_data.csv"
    csv_hash_before = None
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            csv_hash_before = hashlib.sha256(f.read()).hexdigest()[:16]

    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for k in range(16):
        ax.axhline(0.1 * (k + 1), color="black", lw=0.6, zorder=1)

    styles = {
        "block": ("tab:blue", r"$C$"),
        "scalar_C00": ("tab:orange", r"$C_{00}$"),
        "scalar_C11": ("tab:red", r"$C_{11}$"),
    }

    for method, (col, _lab) in styles.items():
        hist = all_data[method]
        xs0, ys0, xs1, ys1, xsh, ysh = [], [], [], [], [], []
        for e in hist:
            t = e["t"]
            E_vals = e["energies"]
            if len(E_vals) >= 1:
                xs0.append(t); ys0.append(float(mp.re(E_vals[0])))
            if len(E_vals) >= 2:
                xs1.append(t); ys1.append(float(mp.re(E_vals[1])))
            for k in range(2, len(E_vals)):
                xsh.append(t); ysh.append(float(mp.re(E_vals[k])))

        if xsh:
            ax.plot(xsh, ysh, linestyle="None", marker="o", markersize=5.0,
                    color=col, markeredgecolor="white", markeredgewidth=1.0,
                    alpha=1.0, zorder=2)
        if xs0:
            ax.plot(xs0, ys0, "-", color=col, lw=1.0, marker="o", markersize=5.0,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        if xs1:
            ax.plot(xs1, ys1, "--", color=col, lw=1.0, marker="o", markersize=5.0,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)

    handles = [Line2D([0], [0], color=col, lw=1.2, label=lab)
               for (col, lab) in styles.values()]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10,
              handlelength=1.2, borderpad=0.3, labelspacing=0.3)

    ax.set_xlim(-0.5, 32)
    ax.set_ylim(0.05, 1.65)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.set_xlabel(r"$t = 2m-1$", fontsize=11)
    ax.set_ylabel(r"$E_k^{(m)} = -\log\,\lambda_k^{(m)}$", fontsize=11)
    ax.grid(False)
    plt.tight_layout(pad=0.3)
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    snap_after = _snapshot(all_data)
    hash_after = {
        m: hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]
        for m, v in snap_after.items()
    }

    csv_hash_after = None
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            csv_hash_after = hashlib.sha256(f.read()).hexdigest()[:16]

    print("fig. 2 — point counts and data integrity")
    print(f"  {'method':<12}{'rank0':>8}{'rank1':>8}{'rank>=2':>10}{'total':>8}")
    for method in ("block", "scalar_C00", "scalar_C11"):
        hist = all_data[method]
        n0 = sum(1 for e in hist if len(e["energies"]) >= 1)
        n1 = sum(1 for e in hist if len(e["energies"]) >= 2)
        nh = sum(max(0, len(e["energies"]) - 2) for e in hist)
        nt = sum(len(e["energies"]) for e in hist)
        print(f"  {method:<12}{n0:>8d}{n1:>8d}{nh:>10d}{nt:>8d}")

    print(f"\n  all_data hash (before) : {hash_before}")
    print(f"  all_data hash (after)  : {hash_after}")
    print(f"  all_data unchanged     : {hash_before == hash_after}")

    if csv_hash_before is not None:
        print(f"\n  csv hash (before)      : {csv_hash_before}")
        print(f"  csv hash (after)       : {csv_hash_after}")
        print(f"  csv unchanged          : {csv_hash_before == csv_hash_after}")
    else:
        print(f"\n  csv file not found at '{csv_path}' — nothing to compare")

    print(f"\n  saved {png}")


def run_figure():
    C, E, _Zt, _Z = build_mock(32)
    print("fig. 2 reproduction")
    print(f"  mp.dps  = {mp.dps}   nstr_digits = {NSTR_DIGITS}")
    print(f"  nt      = {len(C)}   states = {len(E)}   x-axis = t = 2m-1")

    block_hist = extract_all(C, 8, 2)
    print_history("block lanczos (r=2) on full 2x2 c(t)", block_hist, E)
    print()

    c00_hist = extract_all([mp.matrix([[C[t][0, 0]]]) for t in range(32)], 16, 1)
    print_history("scalar lanczos on c_00(t)", c00_hist, E)
    print()

    c11_hist = extract_all([mp.matrix([[C[t][1, 1]]]) for t in range(32)], 16, 1)
    print_history("scalar lanczos on c_11(t)", c11_hist, E)

    all_data = {"block": block_hist, "scalar_C00": c00_hist, "scalar_C11": c11_hist}

    print("building fig2_data.csv "
          "(lossless: mp.nstr with {} sig figs)".format(NSTR_DIGITS))
    rows = []
    for method, hist in all_data.items():
        for entry in hist:
            for rank, E_k in enumerate(entry["energies"]):
                rows.append(dict(
                    method=method, m=entry["m"], t=entry["t"],
                    energy_rank=rank,
                    lambda_real=mp.nstr(mp.re(entry["lambdas"][rank]), NSTR_DIGITS),
                    extracted_energy=mp.nstr(mp.re(E_k), NSTR_DIGITS),
                    true_energy=(mp.nstr(E[rank], NSTR_DIGITS)
                                 if rank < len(E) else "nan"),
                ))

    df = pd.DataFrame(rows)
    df.to_csv("fig2_data.csv", index=False)
    print(f"  {len(df)} rows written (string columns, round-trip lossless)")
    for method in ["block", "scalar_C00", "scalar_C11"]:
        s = df[df.method == method]
        print(f"    {method:<12} : {len(s)} rows, "
              f"m={s.m.min()}..{s.m.max()}, max rank={s.energy_rank.max()}")

    plot_fig2(all_data)

    print()
    print("paper caption check")
    checks = [
        ("block m=8   e_0", block_hist[7]["energies"][0], mp.mpf("0.1")),
        ("block m=8   e_1", block_hist[7]["energies"][1], mp.mpf("0.2")),
        ("c_00  m=16  e_0", c00_hist[15]["energies"][0], mp.mpf("0.1")),
        ("c_00  m=16  e_1", c00_hist[15]["energies"][1], mp.mpf("0.2")),
        ("c_11  m=16  e_0", c11_hist[15]["energies"][0], mp.mpf("0.1")),
        ("c_11  m=16  e_1", c11_hist[15]["energies"][1], mp.mpf("0.2")),
    ]
    for name, val_mp, target in checks:
        diff = abs(mp.re(val_mp) - target)
        ok = diff < mp.mpf("1e-50")
        print(f"  {name} = {mp.nstr(mp.re(val_mp), 20)}   "
              f"|δ| = {mp.nstr(diff, 3)}   {'pass' if ok else 'fail'}")

    return all_data, df
# Post-hoc checks on fig2_data.csv
def run_higher_rank_check(csv_path="fig2_data.csv"):
    TOL_SAT = mp.mpf("1e-50")
    TOL_DEGEN = mp.mpf("1e-3")
    TRUE_E = [mp.mpf("0.1") * (k + 1) for k in range(16)]

    df = _read_csv_lossless(csv_path)

    print("higher-rank eigenvalue check vs. paper claims")
    print(f"  loaded         : {len(df)} rows from {csv_path} (lossless read)")
    print(f"  true spectrum  : e_k = 0.1(k+1), k = 0..15")
    print(f"  tol_sat        : {mp.nstr(TOL_SAT, 3)}")
    print(f"  tol_degen      : {mp.nstr(TOL_DEGEN, 3)}")

    print("part 1 — saturation: full spectrum at the paper's claimed step")
    sat = {"block": 8, "scalar_C00": 16, "scalar_C11": 16}
    paper_claim = {
        "block": "solves for the true eigenvalues exactly after 8 steps",
        "scalar_C00": "requires 16 steps",
        "scalar_C11": "requires 16 steps",
    }

    for method in ["block", "scalar_C00", "scalar_C11"]:
        m_sat = sat[method]
        s = df[(df.method == method) & (df.m == m_sat)].sort_values("energy_rank")
        E = s.extracted_energy.values
        ranks = s.energy_rank.values
        t = int(s.t.values[0])

        print(f"\n  [{method}]  m={m_sat}, x=t={t}")
        print(f"  paper claim: '{paper_claim[method]}'")
        print(f"  {'rank':>4}  {'E_k^(m)':>26}  {'E_k^true':>8}  "
              f"{'|δ|':>11}  {'nearest':>7}  match")

        all_match = True
        bijection = True
        for k in range(len(E)):
            r = int(ranks[k])
            ni = _nearest_true_index(E[k], TRUE_E)
            d_true = abs(E[k] - TRUE_E[r])
            ok = (d_true < TOL_SAT) and (ni == r)
            all_match &= ok
            bijection &= (ni == r)
            print(f"  {r:>4}  {mp.nstr(E[k], 25):>26}  "
                  f"{mp.nstr(TRUE_E[r], 4):>8}  "
                  f"{mp.nstr(d_true, 3):>11}  "
                  f"{mp.nstr(TRUE_E[ni], 4):>7}  "
                  f"{'ok' if ok else 'fail'}")

        print(f"  all 16 ranks within {mp.nstr(TOL_SAT, 3)} : {all_match}")
        print(f"  rank k nearest to true level k for all k   : {bijection}")

    print("part 2 — per-rank convergence (all 16 ranks, every method)")
    print("  table entry = |e_k^(m) − e_k^true|   ('.' = converged below 1e-10)")
    for method in ["block", "scalar_C00", "scalar_C11"]:
        sub = df[df.method == method]
        mmax = int(sub.m.max())
        rmax = int(sub.energy_rank.max())
        print(f"\n  [{method}]   m = 1 .. {mmax}")
        header = "  rank | " + " ".join(f"m={m:<2}" for m in range(1, mmax + 1))
        print(header)

        violations = []
        for r in range(rmax + 1):
            line = f"  {r:>4} | "
            prev_delta = None
            for m in range(1, mmax + 1):
                row = sub[(sub.m == m) & (sub.energy_rank == r)]
                if row.empty:
                    line += " .   "
                    prev_delta = None
                    continue
                E = row.extracted_energy.values[0]
                delta = abs(E - TRUE_E[r])
                if delta < mp.mpf("1e-10"):
                    line += "  .  "
                elif delta >= 1000:
                    line += f"{mp.nstr(delta, 2):<5}"
                elif delta >= 1:
                    line += f"{mp.nstr(delta, 3):<5}"
                else:
                    line += f"{mp.nstr(delta, 2):<5}"
                if prev_delta is not None and delta > prev_delta + mp.mpf("1e-8"):
                    violations.append((r, m - 1, m, prev_delta, delta))
                prev_delta = delta
            print(line)

        if violations:
            print(f"  non-monotone in {len(violations)} steps (expected for oblique):")
            for r, m1, m2, d1, d2 in violations[:5]:
                print(f"    rank {r}: m={m1} δ={mp.nstr(d1, 3)} -> "
                      f"m={m2} δ={mp.nstr(d2, 3)}")
        else:
            print("  convergence from above (monotone) holds for all ranks")

    print("part 3 — within-m non-degeneracy")
    print(f"  distinct eigenvalues at same m should be > {mp.nstr(TOL_DEGEN, 3)} apart.")
    any_degen = False
    for method in ["block", "scalar_C00", "scalar_C11"]:
        sub = df[df.method == method]
        worst_gap, worst_m = mp.inf, None
        for m in sorted(sub.m.unique()):
            E = sub[sub.m == m].sort_values("energy_rank").extracted_energy.values
            if len(E) < 2:
                continue
            gaps = [abs(E[i + 1] - E[i]) for i in range(len(E) - 1)]
            g = min(gaps)
            if g < worst_gap:
                worst_gap, worst_m = g, m
            if g < TOL_DEGEN:
                any_degen = True
                print(f"  [{method}] m={m}: min gap = {mp.nstr(g, 3)}  < "
                      f"{mp.nstr(TOL_DEGEN, 3)}   ← degenerate")
        print(f"  [{method}]  worst gap over all m: {mp.nstr(worst_gap, 3)} "
              f"at m={worst_m}   "
              f"{'ok' if worst_gap >= TOL_DEGEN else 'degenerate'}")

    print(f"\n  any near-degeneracy anywhere : {any_degen}")

    print("verdict")
    def sat_ok(method):
        m_sat = sat[method]
        s = df[(df.method == method) & (df.m == m_sat)].sort_values("energy_rank")
        E = s.extracted_energy.values
        return all(abs(E[k] - TRUE_E[k]) < TOL_SAT for k in range(len(E)))

    print(f"  block m=8, all 16 ranks within {mp.nstr(TOL_SAT, 3)}  : "
          f"{'pass' if sat_ok('block') else 'fail'}")
    print(f"  scalar c_00 m=16, all 16 ranks within {mp.nstr(TOL_SAT, 3)} : "
          f"{'pass' if sat_ok('scalar_C00') else 'fail'}")
    print(f"  scalar c_11 m=16, all 16 ranks within {mp.nstr(TOL_SAT, 3)} : "
          f"{'pass' if sat_ok('scalar_C11') else 'fail'}")
    print(f"  non-degenerate at every m (gap > {mp.nstr(TOL_DEGEN, 3)})   : "
          f"{'pass' if not any_degen else 'fail'}")
    print()


def run_ordering_diagnostic(csv_path="fig2_data.csv"):
    TOL_CONV = mp.mpf("1e-6")
    TOL_GAP = mp.mpf("1e-10")
    df = _read_csv_lossless(csv_path)
    Etrue = [mp.mpf("0.1") * (k + 1) for k in range(16)]

    print("ordering diagnostic — every eigenvalue, every rank, every iteration")
    print(f"loaded   : {len(df)} rows from {csv_path} (lossless)")
    print(f"true e_n : 0.1(n+1), n = 0..15")
    print(f"tol_conv = {mp.nstr(TOL_CONV, 3)}   tol_gap = {mp.nstr(TOL_GAP, 3)}")
    print()
    print("columns:")
    print("  rank     : 0 = lowest extracted re e, 1 = 2nd lowest, ...")
    print("  near     : index n of nearest true level e_n")
    print("  |δ_near| : |e − e_near|")
    print("  true_r   : e_rank  (rank-matched true level)")
    print("  |δ_rank| : |e − e_rank|")
    print("  *        : nearest true level ≠ rank  (still converging)")
    print("  sorted   : extracted energies strictly increasing in rank")

    summary = {}

    for method in ["block", "scalar_C00", "scalar_C11"]:
        sub = df[df.method == method]
        mmax = int(sub.m.max())
        print(f"# method: {method}   (m = 1..{mmax})")
        print("\n[part a] per-iteration table")

        hdr = (f"{'m':>3}{'t':>5}  {'rank':>4}  {'E_k^(m)':>22}  "
               f"{'near':>5}  {'|δ_near|':>11}  {'true_r':>7}  "
               f"{'|δ_rank|':>11}  sorted")
        print("-" * len(hdr)); print(hdr); print("-" * len(hdr))

        lock_on = {}
        any_unsorted = False

        for m in sorted(sub.m.unique()):
            s = sub[sub.m == m].sort_values("energy_rank")
            E = s.extracted_energy.values
            rk = s.energy_rank.values
            t = int(s.t.values[0])
            is_sorted = all(E[i + 1] > E[i] for i in range(len(E) - 1))
            if not is_sorted:
                any_unsorted = True

            for k, E_k in enumerate(E):
                r = int(rk[k])
                ni = _nearest_true_index(E_k, Etrue)
                dn = abs(Etrue[ni] - E_k)
                Er = Etrue[r] if r < len(Etrue) else mp.nan
                dr = abs(E_k - Er)
                if dr < TOL_CONV and r not in lock_on:
                    lock_on[r] = m
                mark = "" if ni == r else "*"
                print(f"{m:>3}{t:>5}  {r:>4}  {mp.nstr(E_k, 20):>22}  "
                      f"{ni:>4}{mark}  {mp.nstr(dn, 3):>11}  "
                      f"{mp.nstr(Er, 3):>7}  {mp.nstr(dr, 3):>11}  {is_sorted}")

            gaps = [abs(E[i + 1] - E[i]) for i in range(len(E) - 1)]
            if len(gaps) > 0 and min(gaps) < TOL_GAP:
                print(f"    !!! m={m}: min gap = {mp.nstr(min(gaps), 3)} "
                      f"< {mp.nstr(TOL_GAP, 3)}")
            print("-" * len(hdr))

        print("\n[part b] per-rank trajectory: nearest true index vs m")
        print("  (no mark -> nearest == rank, * -> nearest ≠ rank, . -> rank absent)")
        rank_max = int(sub.energy_rank.max())
        print(f"\n  {'rank':>4}  " +
              "".join(f"m={m:<3}" for m in range(1, mmax + 1)))

        for r in range(rank_max + 1):
            line = f"  {r:>4}  "
            for m in range(1, mmax + 1):
                row = sub[(sub.m == m) & (sub.energy_rank == r)]
                if row.empty:
                    line += ".    "
                else:
                    E_k = row.extracted_energy.values[0]
                    ni = _nearest_true_index(E_k, Etrue)
                    tag = f"{ni}" if ni == r else f"{ni}*"
                    line += f"{tag:<5}"
            print(line)

        print(f"\n[part c] first m with |e_rank − e_rank^true| < {mp.nstr(TOL_CONV, 3)}")
        for r in range(rank_max + 1):
            cm = lock_on.get(r, None)
            status = f"m = {cm}" if cm is not None else "never within tol"
            print(f"  rank {r:>2}: {status}")

        print(f"\n[part d] at m = {mmax} (saturation): rank r maps to true level r?")
        s_sat = sub[sub.m == mmax].sort_values("energy_rank")
        E_sat = s_sat.extracted_energy.values
        all_ok = True
        for k in range(len(E_sat)):
            ni = _nearest_true_index(E_sat[k], Etrue)
            if ni != k:
                print(f"  rank {k:>2}: nearest = {ni}   mismatch  "
                      f"(e = {mp.nstr(E_sat[k], 12)}, "
                      f"nearest true = {mp.nstr(Etrue[ni], 3)})")
                all_ok = False
        if all_ok:
            print(f"  all {len(E_sat)} ranks map correctly to true levels  -> ok")

        summary[method] = dict(mmax=mmax, n_at_sat=len(E_sat), bijection=all_ok,
                               lock_on=lock_on, unsorted=any_unsorted)

    print("global summary")
    print(f"  {'method':<14}{'n at sat':>10}{'bijection':>12}"
          f"{'ranks converged':>18}{'sorted throughout':>20}")
    for method in ["block", "scalar_C00", "scalar_C11"]:
        s = summary[method]
        nconv = len(s["lock_on"])
        print(f"  {method:<14}{s['n_at_sat']:>10}"
              f"{str(s['bijection']):>12}"
              f"{nconv:>10} / 16"
              f"{str(not s['unsorted']):>20}")

# Fig. 3 — overlap factors |Z_ka^(m)|
def run_fig3(C, E, Z_true):
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "savefig.dpi": 300,
    })
    r, M = 2, 8
    out = block_lanczos(C, M, corrected=True)
    T_full = assemble_T(out["alpha"], out["beta"], out["gamma"], M, r)
    b1 = out["beta"][1]

    block_hist = []
    for m in range(1, M + 1):
        T_m = mp.matrix(m * r, m * r)
        for i in range(m * r):
            for j in range(m * r):
                T_m[i, j] = T_full[i, j]
        lam, eng, Z, _ = ritz_sorted(T_m, r, b1)
        block_hist.append({"m": m, "t": 2 * m - 1,
                           "lambdas": lam, "energies": eng, "Z": Z})

    scalar_hist = {}
    for d in (0, 1):
        Cd = [mp.matrix([[C[t][d, d]]]) for t in range(32)]
        od = block_lanczos(Cd, 16, corrected=True)
        T_d = assemble_T(od["alpha"], od["beta"], od["gamma"], 16, 1)
        b1d = od["beta"][1]
        hist = []
        for m in range(1, 17):
            T_m = mp.matrix(m, m)
            for i in range(m):
                for j in range(m):
                    T_m[i, j] = T_d[i, j]
            lam, eng, Z, _ = ritz_sorted(T_m, 1, b1d)
            hist.append({"m": m, "t": 2 * m - 1,
                         "lambdas": lam, "energies": eng, "Z": Z})
        scalar_hist[d] = hist

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True,
                             gridspec_kw={"wspace": 0.045})
    YMIN, YMAX = 0.04, 4.0
    BLUE, ORANGE, RED, LINE = "#287DB3", "#FF851B", "#E34245", "0.42"
    MS_BLOCK_TOP, MS_BLOCK = 5.8, 4.5
    MS_SCALAR_TOP, MS_SCALAR = 3.4, 2.7

    for a_idx, ax in enumerate(axes):
        # true overlap levels
        for k in range(16):
            zval = float(abs(Z_true[k][a_idx]))
            if not (YMIN <= zval <= YMAX):
                continue
            is_top = k < 2
            if k == 0 and a_idx == 1:
                linestyle = "--"
            elif k % 2 == 0:
                linestyle = "-"
            else:
                linestyle = ":"
            ax.axhline(zval,
                       color="black" if is_top else LINE,
                       linewidth=0.75 if is_top else 0.45,
                       linestyle=linestyle, zorder=0)

        # connectors for the two leading block estimates
        for k in (0, 1):
            for h0, h1 in zip(block_hist[:-1], block_hist[1:]):
                z0 = abs(float(h0["Z"][k, a_idx]))
                z1 = abs(float(h1["Z"][k, a_idx]))
                if YMIN <= z0 <= YMAX and YMIN <= z1 <= YMAX:
                    ax.plot([h0["t"], h1["t"]], [z0, z1], color=BLUE,
                            linestyle=":", linewidth=0.8, alpha=0.8, zorder=1)

        scalar_color = ORANGE if a_idx == 0 else RED
        for k in (0, 1):
            hist = scalar_hist[a_idx]
            for h0, h1 in zip(hist[:-1], hist[1:]):
                if k >= h0["Z"].shape[0] or k >= h1["Z"].shape[0]:
                    continue
                z0 = abs(float(h0["Z"][k, 0]))
                z1 = abs(float(h1["Z"][k, 0]))
                if YMIN <= z0 <= YMAX and YMIN <= z1 <= YMAX:
                    ax.plot([h0["t"], h1["t"]], [z0, z1], color=scalar_color,
                            linestyle=":", linewidth=0.65, alpha=0.75, zorder=1)

        # block estimates
        for h in block_hist:
            for k in range(h["m"] * r):
                zv = h["Z"][k, a_idx]
                az = float(abs(zv))
                if az < 1e-14:
                    continue
                is_top = k < 2
                ms = MS_BLOCK_TOP if is_top else MS_BLOCK
                face = BLUE if zv > 0 else "white"
                if az < YMIN:
                    if a_idx == 1:
                        ax.plot(h["t"], YMIN, marker="v", markersize=ms,
                                markerfacecolor=face, markeredgecolor=BLUE,
                                markeredgewidth=0.85, linestyle="none",
                                clip_on=False, zorder=4)
                    continue
                ax.plot(h["t"], az, marker="o", markersize=ms,
                        markerfacecolor=face, markeredgecolor=BLUE,
                        markeredgewidth=0.65, linestyle="none", zorder=3)

        # scalar estimates
        for h in scalar_hist[a_idx]:
            for k in range(h["m"]):
                zv = h["Z"][k, 0]
                az = float(abs(zv))
                if az < YMIN or az > YMAX or az < 1e-14:
                    continue
                is_top = k < 2
                ms = MS_SCALAR_TOP if is_top else MS_SCALAR
                face = scalar_color if zv > 0 else "white"
                ax.plot(h["t"], az, marker="o", markersize=ms,
                        markerfacecolor=face, markeredgecolor=scalar_color,
                        markeredgewidth=0.55, linestyle="none",
                        alpha=0.9, zorder=2)

        ax.set_yscale("log")
        ax.set_xlim(0, 32); ax.set_ylim(YMIN, YMAX)
        ax.set_xticks([0, 8, 16, 24, 32])
        ax.set_xlabel(r"$t = 2m-1$", fontsize=12)
        ax.set_title(rf"$a={a_idx}$", fontsize=12, pad=7)
        ax.tick_params(axis="both", which="both", direction="in",
                       top=True, right=True, labelsize=9)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    axes[0].set_ylabel(r"$|Z_{ka}^{(m)}|$", fontsize=12, labelpad=5)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17,
                        top=0.89, wspace=0.045)
    path = OUT / "fig333.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")

    return block_hist, scalar_hist

# Fig. 6 — relative error + KPS bound
def tan2_theta(n, r, Zlist, C0):
    Z = mp.matrix(r, r)
    for i in range(r):
        for j in range(r):
            if n + i < len(Zlist) and j < len(Zlist[n + i]):
                Z[i, j] = Zlist[n + i][j]
            else:
                Z[i, j] = 0
    Cinv = mp.inverse(C0)
    X = mp.matrix(r, r)
    for i in range(r):
        for j in range(r):
            s = mp.mpf(0)
            for u in range(r):
                for v in range(r):
                    s += Z[i, u] * Cinv[u, v] * Z[j, v]
            X[i, j] = s
    return float(mp.re(mp.inverse(X)[0, 0])) - 1


def kps_bound(n, r, m, tan2, E, K_n):
    delta = float(mp.re(E[n + r])) - float(mp.re(E[n]))
    Gamma = 2.0 * np.exp(delta) - 1.0
    k = m - n - 1
    if k < 0:
        return np.inf
    mu = np.arccosh(Gamma)
    return K_n ** 2 * tan2 / np.cosh(k * mu) ** 2


def run_fig6(C, E, Z, block_hist, scalar_hist):
    E_np = np.array([float(mp.re(e)) for e in E])
    lam_true = np.exp(-E_np)

    Zlist = [(complex(z[0]), complex(z[1])) for z in Z]
    tan2_block = [tan2_theta(n, 2, Zlist, C[0]) for n in (0, 1)]

    Zc00 = [(complex(z[0]),) for z in Z]
    Zc11 = [(complex(z[1]),) for z in Z]
    C00_0 = mp.matrix([[C[0][0, 0]]])
    C11_0 = mp.matrix([[C[0][1, 1]]])
    tan2_sc = {
        0: [tan2_theta(n, 1, Zc00, C00_0) for n in (0, 1)],
        1: [tan2_theta(n, 1, Zc11, C11_0) for n in (0, 1)],
    }

    def K_of_n(n, engs):
        """K_n^(m) = Π_{l=0}^{n-1} λ_l^(m) / (λ_l^(m) - λ_n)"""
        K = 1.0
        for l in range(n):
            if l >= len(engs) or not np.isfinite(engs[l]):
                return np.nan
            ll = np.exp(-engs[l])
            ln = np.exp(-E_np[n])
            if abs(ll - ln) < 1e-12:
                return np.nan
            K *= ll / (ll - ln)
        return abs(K)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for n, ax in enumerate(axes):
        cases = [
            ('block', block_hist, 'tab:blue', 5.5),
            ('scalar C00', scalar_hist[0], 'tab:orange', 4.0),
            ('scalar C11', scalar_hist[1], 'tab:red', 4.0),
        ]
        for label, hist, col, ms in cases:
            xs, ys = [], []
            for h in hist:
                if (len(h['lambdas']) > n
                        and np.isfinite(h['lambdas'][n])
                        and h['lambdas'][n] > 0):
                    err = abs(h['lambdas'][n] - lam_true[n]) / lam_true[n]
                    if 1e-15 < err < np.inf:
                        xs.append(h['t']); ys.append(err)
            if xs:
                ax.plot(xs, ys, 'o--', color=col, ms=ms, lw=0.9,
                        label=label, zorder=3)

        xs_m = [2 * m - 1 for m in range(1, 17)]

        K_at_8 = K_of_n(n, block_hist[7]['energies']) if n > 0 else 1.0
        yb = []
        for m in range(1, 17):
            if m - n - 1 < 0:
                yb.append(np.nan); continue
            if m <= 8:
                Kn = K_of_n(n, block_hist[m - 1]['energies']) if n > 0 else 1.0
            else:
                Kn = K_at_8
            yb.append(kps_bound(n, 2, m, tan2_block[n], E, Kn))
        ax.plot(xs_m, yb, '-', color='tab:blue', lw=1.4,
                label='block KPS', zorder=2)

        for d, col, name in [(0, 'tab:orange', 'C00'),
                             (1, 'tab:red', 'C11')]:
            ys = []
            for m in range(1, 17):
                if m - n - 1 < 0:
                    ys.append(np.nan); continue
                hist = scalar_hist[d]
                if m - 1 >= len(hist):
                    ys.append(np.nan); continue
                Kn = K_of_n(n, hist[m - 1]['energies']) if n > 0 else 1.0
                ys.append(kps_bound(n, 1, m, tan2_sc[d][n], E, Kn))
            ax.plot(xs_m, ys, '-', color=col, lw=1.0, alpha=0.7,
                    label=f'scalar {name} KPS', zorder=2)

        ax.set_yscale('log')
        ax.set_xlim(0, 32); ax.set_ylim(1e-12, 1e6)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_yticks([10.0 ** p for p in range(-12, 7)])
        ax.yaxis.set_minor_locator(
            matplotlib.ticker.LogLocator(base=10, subs=[], numticks=100))
        ax.set_xlabel(r'$t = 2m-1$', fontsize=11)
        ax.set_ylabel(r'relative error', fontsize=11)
        ax.set_title(f'n={n}', fontsize=11)
        ax.legend(fontsize=8, loc='lower left', frameon=False)
        ax.grid(True, which='major', ls=':', alpha=0.3, zorder=1)

    fig.tight_layout()
    path = OUT / "fig6_reproduction.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# Fig. 7 — residual bounds on E_k^(m)
def run_fig7(C, E):
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.top": True, "ytick.right": True,
    })
    r, M = 2, 8
    out = block_lanczos(C, M + 1, corrected=True)
    T_full = assemble_T(out['alpha'], out['beta'], out['gamma'], M + 1, r)
    gam = out['gamma']; b1 = out['beta'][1]
    E_true = [float(mp.re(E[k])) for k in range(16)]
    YBOT, YTOP = 0.0, 1.75

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for Ek in E_true:
        ax.axhline(Ek, color='black', lw=0.55, zorder=0)

    for m in range(1, M + 1):
        T_m = mp.matrix(m * r, m * r)
        for i in range(m * r):
            for j in range(m * r):
                T_m[i, j] = T_full[i, j]
        lam, eng, Z, Vs = ritz_sorted(T_m, r, b1)
        gn = gam[m + 1]
        for k in range(m * r):
            if lam[k] <= 0 or lam[k] > 1:
                continue
            B = mp.mpf(0)
            for u in range(r):
                for v in range(r):
                    guv = mp.mpf(0)
                    for w in range(r):
                        guv += gn[w, u] * gn[w, v]
                    B += Vs[(m - 1) * r + u, k] * guv * Vs[(m - 1) * r + v, k]
            B = abs(float(mp.re(B)))
            if B <= 0:
                continue
            lam_val = lam[k]
            sqrt_B = np.sqrt(B)
            lam_lo, lam_hi = lam_val - sqrt_B, lam_val + sqrt_B
            E_c = -np.log(lam_val)
            E_lo = -np.log(lam_hi) if lam_hi > 0 else YBOT - 1.0
            E_hi = -np.log(lam_lo) if lam_lo > 0 else YTOP + 1.0
            ax.errorbar(2 * m - 1, E_c,
                        yerr=[[max(E_c - E_lo, 0.0)],
                              [max(E_hi - E_c, 0.0)]],
                        fmt='o', color='tab:blue', ms=3.5, capsize=2.0,
                        alpha=0.85, elinewidth=0.9, capthick=0.9, zorder=2)

    ax.set_xlim(0, 16); ax.set_ylim(YBOT, YTOP)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.set_xlabel(r'$t = 2m-1$', fontsize=11)
    ax.set_ylabel(r'$E_k^{(m)}$', fontsize=11)
    ax.set_title('Fig. 7 — residual bounds', fontsize=11, pad=8)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    path = OUT / "fig7_reproduction.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    from lanczos_core import capture_and_show_summary
    C, E, Zt, Z = build_mock(32)
    print("Fig. 2 reproduction")
    capture_and_show_summary(
        run_figure,
        "output/fig2_full.txt",
        "paper caption check",
    )
    print()
    print("Higher-rank check")
    capture_and_show_summary(
        run_higher_rank_check,
        "output/higher_rank.txt",
        "verdict",
    )
    print()
    print("Ordering diagnostic")
    capture_and_show_summary(
        run_ordering_diagnostic,
        "output/ordering.txt",
        "global summary",
    )
    print()
    print("Fig. 3")
    block_hist, scalar_hist = run_fig3(C, E, Z)
    print("Fig. 6")
    run_fig6(C, E, Z, block_hist, scalar_hist)
    print("Fig. 7")
    run_fig7(C, E)