#!/usr/bin/env python3
"""
hermitian_filter.py — Sec V(B) Hermitian-subspace filter (Eqs 107, 108)
analyze_hermitian(omega, omega_inv, lam, beta1, gamma1, r, eps_real=1e-8, eps_imag=1e-8, eps_indep=1e-6)
  in  : eigenvectors (omega, omega_inv), eigenvalues lam,
        initial blocks beta1, gamma1, block size r
  out : list of dicts, one per state k:
          {k, lam, is_real, is_herm, Nk_vals}
        (dZCW_k is NOT computed here — see zcw_filter.py)
  eq  : (107) reality of lam |Im lam / lam| < eps_real
        (108) block norm trick |N_k|^2 = real, positive, a-independent
"""
from mpmath import mp
def analyze_hermitian(omega, omega_inv, lam, beta1, gamma1, r, eps_real=1e-8, eps_imag=1e-8, eps_indep=1e-6):
    n = len(lam)
    results = []
    for k in range(n):
        lam_k = lam[k]
        lam_abs = abs(lam_k)
        is_real = (lam_abs > 0
                   and float(abs(mp.im(lam_k)) / lam_abs) < eps_real)
        if not is_real:
            results.append(dict(k=k, lam=lam_k, is_real=False, is_herm=False, Nk_vals=[]))
            continue
        Nk_vals = []
        for a in range(r):
            num = sum(omega_inv[k, b] * gamma1[b, a] for b in range(r))
            den = sum(beta1[a, c] * omega[c, k] for c in range(r))
            if abs(den) < mp.mpf("1e-40"):
                Nk_vals.append(mp.mpf(0))
                continue
            Nk_vals.append(num / mp.conj(den))
        is_herm = True
        for Nk in Nk_vals:
            if abs(mp.im(Nk)) > eps_imag * max(abs(Nk), mp.mpf("1e-40")):
                is_herm = False
                break
            if mp.re(Nk) <= 0:
                is_herm = False
                break
        if is_herm and len(Nk_vals) > 1:
            ref = mp.re(Nk_vals[0])
            for Nk in Nk_vals[1:]:
                rel = abs(mp.re(Nk) - ref) / max(abs(ref), mp.mpf("1e-40"))
                if float(rel) > eps_indep:
                    is_herm = False
                    break
        results.append(dict(k=k, lam=lam_k, is_real=True, is_herm=is_herm, Nk_vals=Nk_vals))
    return results
if __name__ == "__main__":
    print("hermitian_filter.py — library module")
    print("Import from noisy_run_all.py.")
    print("Eq 107: |Im lam / lam| < eps_float")
    print("Eq 108: |N_k|^2 real, positive, a-independent")