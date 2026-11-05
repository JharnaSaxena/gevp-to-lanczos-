#!/usr/bin/env python3
"""
krylov_verification.py
Functions:
  compute_krylov_K(m, r, alphas, betas, gammas)
    eq : (44), (45), (46)
  compute_overlaps_eq56_vs_54(C, m, r, omega, omega_inv, beta_1, gamma_1, KR, KL, N_k)
    in : C, eigenvectors, beta1, gamma1, Krylov coefficients, N_k
    out: ZR56, ZL56, ZR54, ZL54, diff_R, diff_L
    eq : (48), (54), (56)
  verify_block_lanczos()
    eq : (30), (37)  — on noiseless mock, m=8
  test_factorization_options()
    eq : (8), (10)-(11)  — symmetric vs cholesky vs LU
  verify_eq56_vs_54()
    eq : (44)-(46), (48), (54), (56)
"""
from mpmath import mp
from lanczos_core import (build_mock, block_lanczos, assemble_T, eig_values_vectors,)
def compute_krylov_K(m, r, alphas, betas, gammas):
    """Eqs 44-46: Krylov coefficients KR_{t,j}, KL_{t,j}"""
    KR, KL = {}, {}
    for j in range(1, m + 1):
        for t in range(j):
            KR[(t, j)] = mp.zeros(r, r)
            KL[(t, j)] = mp.zeros(r, r)
    for a in range(r):
        for b in range(r):
            if a == b:
                KR[(0, 1)][a, b] = 1
                KL[(0, 1)][a, b] = 1
    if m >= 2:
        g2_inv = mp.inverse(gammas[2])
        b2_inv = mp.inverse(betas[2])
        for a in range(r):
            for b in range(r):
                KR[(1, 2)][a, b] = g2_inv[a, b]
                KL[(1, 2)][a, b] = b2_inv[a, b]
                KR[(0, 2)][a, b] = -sum(alphas[1][a, c] * g2_inv[c, b] for c in range(r))
                KL[(0, 2)][a, b] = -sum(b2_inv[a, c] * alphas[1][c, b] for c in range(r))
    for j in range(2, m):
        gj1_inv = mp.inverse(gammas[j + 1])
        bj1_inv = mp.inverse(betas[j + 1])
        for t in range(j + 1):
            for a in range(r):
                for b in range(r):
                    term = mp.zeros(r, r)
                    if t - 1 >= 0 and (t - 1, j) in KR:
                        term += KR[(t - 1, j)]
                    if (t, j) in KR:
                        term -= KR[(t, j)] * alphas[j]
                    if (t, j - 1) in KR:
                        term -= KR[(t, j - 1)] * betas[j]
                    KR[(t, j + 1)][a, b] = sum(term[a, c] * gj1_inv[c, b]
                                              for c in range(r))
                    term_L = mp.zeros(r, r)
                    if t - 1 >= 0 and (t - 1, j) in KL:
                        term_L += KL[(t - 1, j)]
                    if (t, j) in KL:
                        term_L -= alphas[j] * KL[(t, j)]
                    if (t, j - 1) in KL:
                        term_L -= gammas[j] * KL[(t, j - 1)]
                    KL[(t, j + 1)][a, b] = sum(bj1_inv[a, c] * term_L[c, b] for c in range(r))
    return KR, KL
def compute_overlaps_eq56_vs_54(C, m, r, omega, omega_inv, beta_1, gamma_1, KR, KL, N_k):
    """Two-route overlap factors: Eq 54 (cheap) vs Eq 56 (via P_kta)."""
    num_states = m*r
    ZR56 = mp.zeros(num_states, r); ZL56 = mp.zeros(num_states, r)
    ZR54 = mp.zeros(num_states, r); ZL54 = mp.zeros(num_states, r)
    beta_1_inv = mp.inverse(beta_1)
    gamma_1_inv = mp.inverse(gamma_1)
    for k in range(num_states):
        for a in range(r):
            ZR54[k, a] = N_k[k] * sum(beta_1[a, b] * omega[b, k] for b in range(r))
            ZL54[k, a] = (1.0 / N_k[k]) * sum(omega_inv[k, b] * gamma_1[b, a] for b in range(r))
        for a in range(r):
            sum_r = mp.mpf(0); sum_l = mp.mpf(0)
            for t in range(m):
                P_R = [mp.mpf(0)] * r
                P_L = [mp.mpf(0)] * r
                for i in range(t + 1, m + 1):
                    if (t, i) in KR:
                        for b in range(r):
                            for d in range(r):
                                P_R[b] += KR[(t, i)][b, d] * omega[(i - 1) * r + d, k]
                    if (t, i) in KL:
                        for b in range(r):
                            for d in range(r):
                                P_L[b] += omega_inv[k, (i - 1) * r + d] * KL[(t, i)][d, b]
                for b in range(r):
                    P_R[b] *= N_k[k]
                    P_L[b] *= 1.0 / N_k[k]
                for b in range(r):
                    for c in range(r):
                        sum_r += C[t][a, c] * gamma_1_inv[c, b] * P_R[b]
                        sum_l += P_L[b] * beta_1_inv[b, c] * C[t][c, a]
            ZR56[k, a] = sum_r
            ZL56[k, a] = sum_l
    diff_R = max(abs(ZR56[k, a] - ZR54[k, a])
                 for k in range(num_states) for a in range(r))
    diff_L = max(abs(ZL56[k, a] - ZL54[k, a])
                 for k in range(num_states) for a in range(r))
    return ZR56, ZL56, ZR54, ZL54, diff_R, diff_L

def verify_block_lanczos():
    print("VERIFICATION")
    C, E, _ = build_mock(32)
    out = block_lanczos(C, 8)
    r = 2
    print("\n[Check 1] Eq. 30: G_j(1) = gamma_j, B_j(1) = beta_j")
    worst = 0
    for j in range(2, 9):
        eg = max(abs(out["G"][j][1][u, v] - out["gamma"][j][u, v])
                 for u in range(r) for v in range(r))
        eb = max(abs(out["B"][j][1][u, v] - out["beta"][j][u, v])
                 for u in range(r) for v in range(r))
        worst = max(worst, float(eg), float(eb))
    print(f"  worst = {worst:.2e}  "
          f"{'pass' if worst < 1e-20 else 'fail'}")
    print("\n[Check 2] Eq. 37: Ritz values at m=8 vs true e^(-E_k)")
    T8 = assemble_T(out["alpha"], out["beta"], out["gamma"], 8, r)
    vals, _ = eig_values_vectors(T8)
    vals_sorted = sorted(vals, key=lambda v: -mp.re(v))
    max_err = 0
    for k in range(16):
        lam_true = mp.e**(-E[k])
        err = float(abs(vals_sorted[k] - lam_true))
        max_err = max(max_err, err)
    print(f"  max |err| = {max_err:.2e}  "
          f"{'pass' if max_err < 1e-20 else 'fail'}")
    return worst < 1e-20 and max_err < 1e-20
def test_factorization_options():
    """symmetric vs Cholesky vs LU on noiseless mock at m=8"""
    print("Factorization Options (Symmetric, Cholesky, LU)")
    C, _, _ = build_mock(32)
    methods = ["symmetric", "cholesky", "lu"]
    results = {}
    for method in methods:
        out = block_lanczos(C, 8, method=method)
        T8 = assemble_T(out['alpha'], out['beta'], out['gamma'], 8, 2)
        vals, _ = eig_values_vectors(T8)
        vals_sorted = sorted(vals, key=lambda v: -mp.re(v))
        results[method] = vals_sorted
        top3 = [mp.nstr(v, 6) for v in vals_sorted[:3]]
        print(f"  Method '{method:<10}': Top 3 lambda = {top3}")

    print("\n  [Cross-Check] Max |Delta lambda| between methods:")
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            diffs = [abs(results[m1][k] - results[m2][k]) for k in range(16)]
            max_diff = max(float(d) for d in diffs)
            if max_diff < 1e-20:
                status = "pass"
            elif "lu" in (m1, m2):
                status = "fail (expected for LU breaks beta=gamma^T symmetry)"
            else:
                status = "fail"
            print(f"    {m1:<10} vs {m2:<10}: {max_diff:.2e}  {status}")

def verify_eq56_vs_54():
    print("Eq. 56 (via P_kta) vs Eq. 54 (via omega)")
    C, _, _ = build_mock(32)
    r = 2
    m_max = 3
    out = block_lanczos(C, m_max, method="symmetric")
    beta1, gamma1 = out['beta'][1], out['gamma'][1]
    KR, KL = compute_krylov_K(m_max, r, out['alpha'], out['beta'], out['gamma'])
    all_pass = True
    for m in range(1, m_max + 1):
        T_m = assemble_T(out['alpha'], out['beta'], out['gamma'], m, r)
        vals, V = eig_values_vectors(T_m)
        try:
            Vinv = mp.inverse(V)
        except Exception:
            Vinv = mp.inverse(V + mp.eye(V.rows) * mp.mpf("1e-40"))
        N_k = [mp.mpf(1)] * (m * r)
        ZR56, ZL56, ZR54, ZL54, diff_R, diff_L = compute_overlaps_eq56_vs_54(C, m, r, V, Vinv, beta1, gamma1, KR, KL, N_k)
        ok = (float(diff_R) < 1e-20 and float(diff_L) < 1e-20)
        all_pass = all_pass and ok
        print(f"\n  m={m} ({m*r} states):")
        print(f"    max |Z^R_56 − Z^R_54| = {mp.nstr(diff_R, 3)}")
        print(f"    max |Z^L_56 − Z^L_54| = {mp.nstr(diff_L, 3)}")
        print(f"    {'pass' if ok else 'fail'}")
    print(f"\n  Overall: {'pass' if all_pass else 'fail'}\n")
    return all_pass
if __name__ == "__main__":
    print("krylov_verification.py — library module")
    print("Import from noisy_run_all.py.")