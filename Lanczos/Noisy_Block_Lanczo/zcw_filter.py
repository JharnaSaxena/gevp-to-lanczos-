#!/usr/bin/env python3
"""
zcw_filter.py — Sec. V.C block ZCW filter (Eqs. 113, 116, 117, 119)
compute_dzcw(omega, omega_inv, r)
  in  : eigenvectors omega, omega_inv, block size r
  out : list of |Delta_k^ZCW| per state k
  eq  : (119)  Delta_k = | sum_a omega_{1ak} (omega^-1)_{k1a} |
find_m_H_strict(per_m_diag, m_max)
  eq : (116) literal max over all m
find_m_H_contiguous(per_m_diag, m_max)
  eq : (116) last m in the initial clean range
compute_eps_ZCW(per_m_diag, m_H, F_ZCW=10)
  in  : {m -> list of dicts}, m_H, F_ZCW hyperparameter
  out : eps_ZCW cutoff
  eq  : (117)  eps_ZCW = (1/F_ZCW) * min_k Delta_k^ZCW(m_H)
apply_cut(diag_at_m, eps_ZCW)
  in  : list of state dicts at one m, cutoff
  out : sorted list of surviving |lambda_k|
  eq  : (113)  keep only Delta_k^ZCW >= eps_ZCW, real & herm 
"""
from mpmath import mp
def compute_dzcw(omega, omega_inv, r):
    """Eq 119: block ZCW quantity per state k"""
    n_states = omega.rows
    return [abs(sum(omega[a, k] * omega_inv[k, a] for a in range(r))) for k in range(n_states)]

def find_m_H_strict(per_m_diag, m_max):
    """Eq 116 max over ALL m satisfying the condition"""
    m_H = 0
    for m in range(1, m_max + 1):
        diag = per_m_diag.get(m, [])
        all_phys = bool(diag) and all(
            d['is_real'] and d['is_herm']
            and 0 < float(mp.re(d['lam'])) < 1
            for d in diag)
        if all_phys:
            m_H = m
    return m_H


def find_m_H_contiguous(per_m_diag, m_max):
    """Eq 116 contiguous reading last m in the clean range from m=1"""
    m_H = 0
    found_first_bad = False
    for m in range(1, m_max + 1):
        diag = per_m_diag.get(m, [])
        all_phys = bool(diag) and all(
            d['is_real'] and d['is_herm']
            and 0 < float(mp.re(d['lam'])) < 1
            for d in diag)
        if all_phys and not found_first_bad:
            m_H = m
        elif not all_phys:
            found_first_bad = True
    return m_H

def compute_eps_ZCW(per_m_diag, m_H, F_ZCW=10):
    """Eq 117 with m_H=1 fallback per paper Sec. V(C3)"""
    if m_H <= 1:
        return 1e-2
    dZCW_mH = [float(mp.re(d['dZCW_k'])) for d in per_m_diag[m_H]
               if d['is_real'] and d['is_herm']]
    if not dZCW_mH:
        return 1e-2
    return max(min(dZCW_mH) / F_ZCW, 1e-15)
def apply_cut(diag_at_m, eps_ZCW):
    phys = []
    for d in diag_at_m:
        if not (d['is_real'] and d['is_herm']):
            continue
        if float(mp.re(d['dZCW_k'])) < eps_ZCW:
            continue
        phys.append(float(mp.re(d['lam'])))
    phys.sort(reverse=True)
    return phys
if __name__ == "__main__":
    print("zcw_filter.py — library module")
    print("Import from noisy_run_all.py.")
    print("Eq. 119: Delta_k^ZCW = |sum_a omega_{1ak} (omega^-1)_{k1a}|")
    print("Eq. 116: m_H")
    print("Eq. 117: eps_ZCW = (1/F_ZCW) * min Delta_k^ZCW(m_H)")
    print("Eq. 113: cut Delta_k^ZCW < eps_ZCW")