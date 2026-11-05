#!/usr/bin/env python3
"""
noisy_run_all.py — Section V pipeline orchestrator.

filter_one_bootstrap(C_boot_np, m_max=8, F_ZCW=10)
  in  : bootstrapped C(t) array (Nt, r, r), m_max, F_ZCW
  out : dict{m_H, m_H_strict, m_H_contig, eps_ZCW, survivors, diagnostics, error}
  note: calls analyze_hermitian (Eqs. 107,108) + compute_dzcw (Eq. 119), then find_m_H_* (Eq. 116) + compute_eps_ZCW (Eq. 117) + apply_cut (Eq. 113)
print_m_H_comparison(all_results, m_max)
  in  : bootstrap results, m_max
  out : printed table comparing strict vs contiguous m_H
run_noisy_pipeline(n_cfg=80, n_boot=200, m_max=8, seed=42)
  does: full pipeline + diagnostics
  out : all_results list (or [] on early exit)
main
  does: noiseless check -> factorization test -> eq56 vs 54 -> noisy pipeline
"""
from mpmath import mp
import numpy as np
from lanczos_core import build_mock, block_lanczos, assemble_T, eig_values_vectors
from krylov_verification import verify_block_lanczos, test_factorization_options, verify_eq56_vs_54
from noise_generator import generate_noisy_data, to_mpmath_correlator, symmetrize
from hermitian_filter import analyze_hermitian
from zcw_filter import compute_dzcw, find_m_H_strict, find_m_H_contiguous, compute_eps_ZCW, apply_cut
# Filter one bootstrap
def filter_one_bootstrap(C_boot_np, m_max=8, F_ZCW=10):
    r = 2 # become 2 by 2
    try:
        C_mp = symmetrize(to_mpmath_correlator(C_boot_np))
        out = block_lanczos(C_mp, m_max + 1, method="symmetric")
        beta1, gamma1 = out['beta'][1], out['gamma'][1]
        per_m_diag = {}
        for m in range(1, m_max + 1):
            T_m = assemble_T(out['alpha'], out['beta'], out['gamma'], m, r)
            vals, V = eig_values_vectors(T_m)
            try:
                Vinv = mp.inverse(V)
            except Exception:
                Vinv = mp.inverse(V + mp.eye(V.rows) * mp.mpf("1e-40"))
            diag = analyze_hermitian(V, Vinv, vals, beta1, gamma1, r)
            dzcw = compute_dzcw(V, Vinv, r)
            for k, d in enumerate(diag):
                d['dZCW_k'] = dzcw[k]
            per_m_diag[m] = diag
        m_H_strict = find_m_H_strict(per_m_diag, m_max)
        m_H_contig = find_m_H_contiguous(per_m_diag, m_max)
        m_H = m_H_strict
        eps_ZCW = compute_eps_ZCW(per_m_diag, m_H, F_ZCW)
        survivors = {}
        for m in range(1, m_max + 1):
            survivors[m] = apply_cut(per_m_diag.get(m, []), eps_ZCW)
        return dict(m_H=m_H, m_H_strict=m_H_strict, m_H_contig=m_H_contig, eps_ZCW=eps_ZCW, survivors=survivors, diagnostics=per_m_diag, error=None)
    except Exception as exc:
        return dict(error=f"{exc}", m_H=0, m_H_strict=0, m_H_contig=0, eps_ZCW=1e-2, survivors={}, diagnostics={})
# m_H comparison table
def print_m_H_comparison(all_results, m_max):
    print(f"\n[m_H comparison] strict (literal Eq. 116) vs contiguous")
    strict_vals = [r['m_H_strict'] for r in all_results if not r.get('error')]
    contig_vals = [r['m_H_contig'] for r in all_results if not r.get('error')]
    if not strict_vals:
        print("(no successful bootstraps hence nothing to compare)")
        return
    n_boot = len(strict_vals)
    n_diff = sum(1 for s, c in zip(strict_vals, contig_vals) if s != c)
    print(f"  {'interpretation':<16} | {'median':>6} | {'min':>3} | "
          f"{'max':>3} | {'m=1':>5} | {'m=2':>5} | {'m=3':>5}")
    for name, vals in [("strict (Eq. 116)", strict_vals), ("contiguous", contig_vals)]:
        c1 = sum(1 for v in vals if v == 1)
        c2 = sum(1 for v in vals if v == 2)
        c3 = sum(1 for v in vals if v == 3)
        print(f"  {name:<16} | {int(np.median(vals)):>6} | "
              f"{min(vals):>3} | {max(vals):>3} | "
              f"{c1:>5} | {c2:>5} | {c3:>5}")
    print(f"\n  bootstraps where the two interpretations differ: "
          f"{n_diff} / {n_boot}")
    if n_diff > 0:
        print(f"  (some iteration past the first failure transiently re-cleaned)")
    else:
        print(f"  (interpretations agree on every bootstrap — "
              f"no transient re-cleaning)")
# Full noisy pipeline
def run_noisy_pipeline(n_cfg=80, n_boot=200, m_max=8, seed=42):
    print("HERMITIAN AND BLOCK ZCW FILTERING")
    C_mean_mp, _, _ = build_mock(32)
    _, _, boot_means = generate_noisy_data(C_mean_mp, n_cfg=n_cfg, n_boot=n_boot, seed=seed, rel0=1e-3)
    first = filter_one_bootstrap(boot_means[0], m_max=m_max)
    if first.get('error'):
        print(f"ERROR: {first['error']}")
        return []
    print(f"\n[debug] m_H = {first['m_H']},  ε_ZCW = {first['eps_ZCW']:.4e}")
    print(f"\n  {'m':>3} | {'k':>3} | {'herm':>5} | {'λ':>10} | "
          f"{'Δ^ZCW':>10} | status")
    print(f"  {'-'*3}-+-{'-'*3}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+--------")
    for m in range(1, m_max + 1):
        for d in first['diagnostics'].get(m, []):
            herm = 'Y' if d['is_herm'] else 'n'
            lam_v = mp.nstr(mp.re(d['lam']), 8)
            dzcw = mp.nstr(mp.re(d['dZCW_k']), 5)
            keep = 'keep' if (d['is_real'] and d['is_herm']
                              and float(mp.re(d['dZCW_k'])) >= first['eps_ZCW']) \
                          else 'cut'
            print(f"  {m:>3} | {d['k']:>3} | {herm:>5} | {lam_v:>10} | "
                  f"{dzcw:>10} | {keep}")
    print(f"\n survivors per m: "
          f"{ {m: len(v) for m, v in first['survivors'].items()} }")
    print(f"\n[run] {n_boot} bootstraps...")
    all_results = []
    for b in range(n_boot):
        all_results.append(filter_one_bootstrap(boot_means[b], m_max=m_max))
        if (b + 1) % 50 == 0:
            print(f"  {b+1}/{n_boot}")
    counts = {m: [] for m in range(1, m_max + 1)}
    for res in all_results:
        for m, phys in res.get('survivors', {}).items():
            counts[m].append(len(phys))
    print(f"\n[results] state counts:")
    print(f"  {'m':>3} | {'mean':>6} | {'med':>4} | {'min':>4} | {'max':>4}")
    for m in range(1, m_max + 1):
        c = np.array(counts[m], dtype=float)
        if c.size == 0:
            print(f"  {m:>3} | {'—':>6} | {'—':>4} | {'—':>4} | {'—':>4}")
        else:
            print(f"  {m:>3} | {c.mean():>6.1f} | {int(np.median(c)):>4} | "
                  f"{int(c.min()):>4} | {int(c.max()):>4}")

    ok = [r for r in all_results if not r.get('error')]
    n_err = len(all_results) - len(ok)
    if ok:
        m_Hs = [r['m_H'] for r in ok]
        eps = [r['eps_ZCW'] for r in ok]
        print(f"\n[diag]  m_H: {min(m_Hs)}..{max(m_Hs)}  "
              f"ε_ZCW: {min(eps):.3e}..{max(eps):.3e}  "
              f"errors: {n_err}/{n_boot}")
    else:
        print(f"\n[diag]  no successful bootstraps; errors: {n_err}/{n_boot}")

    return all_results
# Main
if __name__ == "__main__":
    ok = verify_block_lanczos()
    if not ok:
        print("\n!!!!!!!!!!FIX block_lanczos BEFORE proceeding!!!!!!!!!")
        import sys
        sys.exit(1)
    test_factorization_options()
    verify_eq56_vs_54()
    all_results = run_noisy_pipeline(n_cfg=80, n_boot=200, m_max=8, seed=42)
    print_m_H_comparison(all_results, m_max=8)