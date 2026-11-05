#!/usr/bin/env python3
"""
lanczos_core.py
# MOCK DATA 
# build_mock(Nt=32)
#   in  : Nt                    int, number of time slices
#   out : C   list[mp.matrix 2×2]   correlator C(t)
#         E   list[mpf]            energies E_k = 0.1(k+1)
#         Zt  list[(mpf, mpf)]     raw overlaps Z~_ka
#         Z   list[(mpf, mpf)]     normalized Z_ka / sqrt(2 E_k)
#   eq  : (71)-(72)
# print_mock(C, E, Zt, Z)
#   in  : mock tuple
#   out : (prints a table)
#   eq  : (71)-(72)
# EIGEN / FACTORIZATION HELPERS
# eig_values_vectors(A)
#   in  : A                     mp.matrix
#   out : vals [mpf], ER mp.matrix
#   note: wrapper around mp.eig that tolerates both return shapes
#
# factor_delta(Delta)
#   in  : Delta                 mp.matrix (r×r)
#   out : (beta, gamma)         mp.matrix, with Delta = beta * gamma
#   eq  : (15)  [also reused for (21), (25)]
# BLOCK LANCZOS RECURSION
# block_lanczos(C, m, corrected=True)
#   in  : C                     list[mp.matrix] correlator
#         m                     int, steps
#         corrected             bool  eq. 24 toggle
#                                  True then leading term A1(t+2) (correct)
#                                  False leading term A1(t+1) (printed, fails)
#   out : dict{alpha, beta, gamma, A, G, B}
#   eq  : (15), (17), (20), (21), (22)-(23), (24), (25), (26)-(27), (28), (29)
# SPECTRAL / TRIDIAGONAL ASSEMBLY
# assemble_T(alphas, betas, gammas, m, r)
#   in  : block lists, m steps, r block size
#   out : T^(m)                 mp.matrix (m*r × m*r)
#   eq  : (32)
#
# submatrix(T, n)
#   in  : T, n                  matrix, size
#   out : T[:n, :n]             top-left n×n block
#
# ritz_energies(T)
#   in  : T                     mp.matrix
#   out : list[mpf]             E_k = -log(λ_k), sorted ascending by Re
#   eq  : (37)
#
# ritz_sorted(T_m, r, beta1)
#   in  : T_m, r, beta1         submatrix, block size, β₁
#   out : (lam, E, Z, Vs)       λ descending, E ascending,
#                                Z = overlaps (n×r), Vs = eigenvector matrix
#   eq  : (36)-(37)  [+ overlap convention]
#   used: figures.py (figs 3, 6, 7)
#
# reconstruct_C(T, b1, g1, Nt, r)
#   in  : T, beta1, gamma1, Nt, r
#   out : list[C_rec(t)]        reconstruction of C(t)
#   eq  : (60)  C(t) = β₁ · [T^t]_(1,1) · γ₁
# DIAGNOSTIC / DEBUGGING HELPERS
# maxdiff(A, B, r)          in: two r×r matrices   out: mpf, max |A-B|
# matrix_max_abs(A, r)      in: r×r matrix         out: mpf, max |a_ij|
# HISTORY EXTRACTION USED BY FIGURES
# extract_all(C, mmax, r, corrected=True)
#   in  : C, mmax, r, bool
#   out : list[dict{m, t, lambdas, energies}] used by fig 2
#
# extract_hist(C, mmax, r, corrected=True)
#   in  : C, mmax, r, bool
#   out : list[dict{m, t, energies}] used by eq24 plot_case
# CSV HELPERS (LOSLESS WAS IMPORTANT OR DATA WOULD'VE PLOTTED 0)
# _read_csv_lossless(csv_path)   in: path out: DataFrame (mpf columns)
# _nearest_true_index(E, TRUE_E) in: value, list out: int
# capture_and_show_summary(func, filepath, delimiter, summary=None)
#   in  : func (callable), filepath (str), delimiter (str), summary (dict|None)
#   out : whatever `func` returns
#   does: redirects func's stdout filepath, prints the tail after `delimiter`
#         (or the `summary` dict if supplied). Terminal stays clean.
"""
import hashlib
import io
import json
import os
from contextlib import redirect_stdout
import pandas as pd
from mpmath import mp

mp.dps = 100
NSTR_DIGITS = 80
OUTPUT_DIR = "output"
# Mock correlator (eqs. 71-72)
def build_mock(Nt=32):
    r, Ns = 2, 16
    E, Zt, Z = [], [], []
    for k in range(Ns):
        Ek = mp.mpf("0.1") * (k + 1)
        if k % 2 == 0:
            z0 = mp.mpf("1.0")
            z1 = mp.mpf("-0.1") if k == 0 else mp.mpf("0.1")
        else:
            z0 = mp.mpf("0.1")
            z1 = mp.mpf("1.0")
        Zt.append((z0, z1))
        Z.append((z0 / mp.sqrt(2 * Ek), z1 / mp.sqrt(2 * Ek)))
        E.append(Ek)
    C = []
    for t in range(Nt):
        Ct = mp.matrix(r, r)
        for k in range(Ns):
            ek = mp.exp(-E[k] * t)
            for a in range(r):
                for b in range(r):
                    Ct[a, b] += Z[k][a] * Z[k][b] * ek
        C.append(Ct)
    return C, E, Zt, Z
def print_mock(C, E, Zt, Z):
    print("[Eq 71-72] MOCK CORRELATOR CONSTRUCTION")
    print(f"  Nt       = {len(C)}")
    print(f"  N_states = {len(E)}")
    print()
    print("  k    E_k      Z~_{k,0}    Z~_{k,1}     Z_{k,0}         Z_{k,1}")
    for k in range(len(E)):
        print(
            f"  {k:2d}   {mp.nstr(E[k], 4):<8} "
            f"{mp.nstr(Zt[k][0], 4):<10} {mp.nstr(Zt[k][1], 4):<10} "
            f"{mp.nstr(Z[k][0], 10):<14} {mp.nstr(Z[k][1], 10)}"
        )
    print()
    print("  C(t) sample values:")
    sample_times = [t for t in [0, 1, 2, 5, 10, 15, 20, 31] if t < len(C)]
    for t in sample_times:
        print(f"  t={t:2d}:  C = [[{mp.nstr(C[t][0,0], 10)}, "
              f"{mp.nstr(C[t][0,1], 10)}],")
        print(f"               [{mp.nstr(C[t][1,0], 10)}, "
              f"{mp.nstr(C[t][1,1], 10)}]]")
# Eigen / factorization helpers
def eig_values_vectors(A):
    E, ER = mp.eig(A)
    try:
        vals = [E[k, 0] for k in range(E.rows)]
    except (TypeError, AttributeError):
        vals = list(E)
    if not hasattr(ER, "rows"):
        ER = mp.matrix(ER)
    return vals, ER


def factor_delta(Delta):
    """Factor Delta = beta * gamma (branch-unchecked, same as reference)."""
    r = Delta.rows
    if r == 1:
        s = mp.sqrt(Delta[0, 0])
        return mp.matrix([[s]]), mp.matrix([[s]])
    vals, V = eig_values_vectors(Delta)
    Vinv = mp.inverse(V)
    S = mp.diag([mp.sqrt(vals[k]) for k in range(r)])
    return V * S, S * Vinv
# Block Lanczos recursion
def block_lanczos(C, m, corrected=True):
    """
    Run m steps of block Lanczos on C
    corrected=True eq. 24 leading term A_1(t+2)  (correct)
    corrected=False eq. 24 leading term A_1(t+1)  (print fails)
    """
    Nt = len(C)
    r = C[0].rows
    if Nt < 2 * m:
        raise ValueError(f"block_lanczos requires Nt >= 2m; got Nt={Nt}, m={m}")
    alphas = [None] * (m + 1)
    betas = [None] * (m + 1)
    gammas = [None] * (m + 1)
    As = [None] * (m + 1)
    Gs = [None] * (m + 1)
    Bs = [None] * (m + 1)
    # eq. 15
    b1, g1 = factor_delta(C[0])
    betas[1], gammas[1] = b1, g1
    ib1, ig1 = mp.inverse(b1), mp.inverse(g1)
    # eq. 20
    A1 = [ib1 * C[t] * ig1 for t in range(Nt)]
    As[1] = A1
    alphas[1] = A1[1]
    if m == 1:
        return dict(alpha=alphas, beta=betas, gamma=gammas,A=As, G=Gs, B=Bs)
    a1 = alphas[1]
    # eq. 21
    Delta2 = A1[2] - a1 * a1
    b2, g2 = factor_delta(Delta2)
    betas[2], gammas[2] = b2, g2
    ib2, ig2 = mp.inverse(b2), mp.inverse(g2)
    # eqs. 22-23
    Gs[2] = [ib2 * (A1[t + 1] - a1 * A1[t]) for t in range(Nt - 1)]
    Bs[2] = [(A1[t + 1] - A1[t] * a1) * ig2 for t in range(Nt - 1)]
    # eq. 24 (toggle)
    if corrected:
        As[2] = [ib2 * (A1[t + 2] + a1 * A1[t] * a1- (a1 * A1[t + 1] + A1[t + 1] * a1)) * ig2 for t in range(Nt - 2)]
    else:
        As[2] = [ib2 * (A1[t + 1] + a1 * A1[t] * a1- (a1 * A1[t + 1] + A1[t + 1] * a1)) * ig2 for t in range(Nt - 1)]
    alphas[2] = As[2][1]
    for j in range(2, m):
        Aj, Gj, Bj, Ajm1 = As[j], Gs[j], Bs[j], As[j - 1]
        aj, bj, gj = alphas[j], betas[j], gammas[j]
        # eq. 25
        Delta = Aj[2] - aj * aj - gj * bj
        bn, gn = factor_delta(Delta)
        betas[j + 1], gammas[j + 1] = bn, gn
        ibn, ign = mp.inverse(bn), mp.inverse(gn)
        # eqs. 26-27
        Gs[j + 1] = [ibn * (Aj[t + 1] - aj * Aj[t] - gj * Bj[t]) for t in range(min(len(Aj) - 1, len(Bj)))]
        Bs[j + 1] = [(Aj[t + 1] - Aj[t] * aj - Gj[t] * bj) * ign for t in range(min(len(Aj) - 1, len(Gj)))]
        # eq. 28
        LA = min(len(Aj) - 2, len(Ajm1), len(Bj) - 1, len(Gj) - 1)
        As[j + 1] = []
        for t in range(LA):
            term = (Aj[t + 2]- (aj * Aj[t + 1] + Aj[t + 1] * aj)+ aj * Aj[t] * aj+ gj * Ajm1[t] * bj- (gj * Bj[t + 1] + Gj[t + 1] * bj)+ gj * Bj[t] * aj+ aj * Gj[t] * bj)
            As[j + 1].append(ibn * term * ign)
        alphas[j + 1] = As[j + 1][1]
    return dict(alpha=alphas, beta=betas, gamma=gammas,A=As, G=Gs, B=Bs)

# Tridiagonal assembly + spectral helpers
def assemble_T(alphas, betas, gammas, m, r):
    T = mp.matrix(m * r, m * r)
    for i in range(m):
        for a in range(r):
            for b in range(r):
                T[i * r + a, i * r + b] = alphas[i + 1][a, b]
    for i in range(m - 1):
        for a in range(r):
            for b in range(r):
                T[i * r + a, (i + 1) * r + b] = betas[i + 2][a, b]
                T[(i + 1) * r + a, i * r + b] = gammas[i + 2][a, b]
    return T

def submatrix(T, n):
    return mp.matrix([[T[i, j] for j in range(n)] for i in range(n)])
def ritz_energies(T):
    vals, _ = eig_values_vectors(T)
    out = [-mp.log(v) for v in vals]
    out.sort(key=lambda x: mp.re(x))
    return out
def ritz_sorted(T_m, r, beta1):
    """
    Eigendecompose T_m; return (lambdas desc, E asc, Z (n,r), Vs) Z uses a convention for overlap factors first column sign-fixed
    """
    import numpy as np
    n = T_m.rows
    vals, V = eig_values_vectors(T_m)
    vals = [mp.re(v) for v in vals]
    idx = sorted(range(n), key=lambda k: -vals[k])
    lam = np.array([float(vals[k]) for k in idx])
    E = np.array([-np.log(l) if l > 0 else np.nan for l in lam])
    Vs = np.zeros((n, n))
    for j, k in enumerate(idx):
        for i in range(n):
            Vs[i, j] = float(mp.re(V[i, k]))
    Z = np.zeros((n, r))
    for k in range(n):
        for u in range(r):
            acc = mp.mpf(0)
            for v in range(r):
                acc += beta1[u, v] * Vs[v, k]
            Z[k, u] = float(acc)
        if Z[k, 0] < 0:
            Z[k, :] *= -1
    return lam, E, Z, Vs
def reconstruct_C(T, b1, g1, Nt, r):
    out, P = [], mp.eye(T.rows)
    for _ in range(Nt):
        block = mp.matrix([[P[a, b] for b in range(r)] for a in range(r)])
        out.append(b1 * block * g1)
        P = P * T
    return out
def maxdiff(A, B, r):
    return max(abs(A[a, b] - B[a, b]) for a in range(r) for b in range(r))
def matrix_max_abs(A, r):
    return max(abs(A[a, b]) for a in range(r) for b in range(r))
# History extraction
def extract_all(C, mmax, r, corrected=True):
    """Ritz values for every m = 1..mmax (used by fig 2)."""
    out = block_lanczos(C, mmax, corrected=corrected)
    T_full = assemble_T(out["alpha"], out["beta"], out["gamma"], mmax, r)
    history = []
    for m in range(1, mmax + 1):
        T_m = submatrix(T_full, m * r)
        E_vals = ritz_energies(T_m)
        lambdas = [mp.e ** (-E) for E in E_vals]
        history.append(dict(m=m, t=2 * m - 1, lambdas=lambdas, energies=E_vals))
    return history
def extract_hist(C, mmax, r, corrected=True):
    """Energies only (used by eq24 plot_case)."""
    out = block_lanczos(C, mmax, corrected=corrected)
    T_full = assemble_T(out["alpha"], out["beta"], out["gamma"], mmax, r)
    hist = []
    for m in range(1, mmax + 1):
        T_m = submatrix(T_full, m * r)
        hist.append(dict(m=m, t=2 * m - 1, energies=ritz_energies(T_m)))
    return hist
# CSV helpers
def _read_csv_lossless(csv_path):
    df = pd.read_csv(
        csv_path,
        dtype={"extracted_energy": str, "true_energy": str,
               "lambda_real": str},
    ).sort_values(["method", "m", "energy_rank"]).reset_index(drop=True)
    for col in ["m", "t", "energy_rank"]:
        df[col] = df[col].astype(int)
    df["extracted_energy"] = df["extracted_energy"].map(lambda x: mp.mpf(x))
    df["true_energy"] = df["true_energy"].map(
        lambda x: mp.mpf(x) if x != "nan" else mp.nan)
    return df
def _nearest_true_index(E_value, TRUE_E):
    return min(range(len(TRUE_E)), key=lambda j: abs(TRUE_E[j] - E_value))
# stdout capture helper
def capture_and_show_summary(func, filepath, delimiter, summary=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = func()
    text = buf.getvalue()
    with open(filepath, "w") as f:
        f.write(text)
    print(f"\n[{os.path.basename(filepath)}]  {text.count(chr(10))} lines ->  {filepath}")
    if summary is not None:
        print(f"\n{delimiter}\n  " + "\n  ".join(f"{k}: {v}" for k, v in summary.items()))
    elif delimiter and delimiter in text:
        print(f"\n{delimiter}{text.split(delimiter, 1)[1]}")
    return result
if __name__ == "__main__":
    print("lanczos_core.py — library module")
    print("Import from eq24_comparison.py or figures.py.")
    print(f"mp.dps = {mp.dps}, nstr_digits = {NSTR_DIGITS}")