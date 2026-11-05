#!/usr/bin/env python3
"""
lanczos_core.py — core library for Section V noisy-data pipeline.
Functions:
  build_mock(Nt=32, Ns=16)
    in  : Nt (int), Ns (int)
    out : C list[mp.matrix 2x2], E list[mpf], Z list[(mpf, mpf)]
    eq  : (71)-(72)
  eig_values_vectors(A)
    in  : A (mp.matrix)
    out : vals [mpf], ER (mp.matrix)
    note: wrapper around mp.eig, tolerant to return shapes
  factor_delta(Delta, method="symmetric")
    in  : Delta (mp.matrix), method in {"symmetric","cholesky","lu"}
    out : (beta, gamma) with Delta = beta * gamma
    eq  : (8), (10)-(11)
  block_lanczos(C, m, method="symmetric")
    in  : C (list[mp.matrix]), m (int), method
    out : dict{alpha, beta, gamma, A, G, B}
    eq  : (15), (17), (20), (21), (22)-(23), (24) corrected, (25), (26)-(27), (28) full, (29)
  assemble_T(alphas, betas, gammas, m, r)
    in  : block lists, m, r
    out : T^(m)  (mp.matrix m*r x m*r)
    eq  : (31)-(32)
"""
from mpmath import mp
mp.dps = 40
def build_mock(Nt=32, Ns=16):
    """Eqs 71-72 demo 2x2 mock correlator from 16 hidden states"""
    E, Z = [], []
    for k in range(Ns):
        Ek = mp.mpf("0.1") * (k + 1)
        if k % 2 == 0:
            z0 = mp.mpf("1.0")
            z1 = mp.mpf("-0.1") if k == 0 else mp.mpf("0.1")
        else:
            z0 = mp.mpf("0.1")
            z1 = mp.mpf("1.0")
        Z.append((z0 / mp.sqrt(2 * Ek), z1 / mp.sqrt(2 * Ek)))
        E.append(Ek)
    C = []
    for t in range(Nt):
        Ct = mp.matrix(2, 2)
        for k in range(Ns):
            ek = mp.exp(-E[k] * t)
            for a in range(2):
                for b in range(2):
                    Ct[a, b] += Z[k][a] * Z[k][b] * ek
        C.append(Ct)
    return C, E, Z
def eig_values_vectors(A):
    E, ER = mp.eig(A)
    try:
        vals = [E[k, 0] for k in range(E.rows)]
    except (TypeError, AttributeError):
        vals = list(E)
    if not hasattr(ER, "rows"):
        ER = mp.matrix(ER)
    return vals, ER
def factor_delta(Delta, method="symmetric"):
    """
    Eq 8: Delta = beta * gamma.
    method="symmetric" Eq. 11 (paper's choice for numerical results)
    method="cholesky" valid only for real positive-definite Delta
    method="lu" valid only for complex non-Hermitian Delta (breaks beta=gamma^T, T^(m) non-Hermitian)
    """
    r = Delta.rows
    if r == 1:
        s = mp.sqrt(abs(mp.re(Delta[0, 0])))
        return mp.matrix([[s]]), mp.matrix([[s]])
    Ds = mp.matrix(r, r)
    for i in range(r):
        for j in range(r):
            Ds[i, j] = mp.re(Delta[i, j] + Delta[j, i]) / 2
    if method == "cholesky":
        try:
            L = mp.cholesky(Ds)
            return L, L.T
        except Exception:
            method = "symmetric"
    if method == "lu":
        try:
            P, L, U = mp.lu(Ds)
            Pinv = mp.inverse(P)
            return Pinv * L, U
        except Exception:
            method = "symmetric"
    vals, V = eig_values_vectors(Ds)
    S = mp.diag([mp.sqrt(abs(mp.re(v))) if abs(mp.re(v)) > mp.mpf("1e-40") else mp.mpf("1e-20") for v in vals])
    Vinv = mp.inverse(V)
    return V * S, S * Vinv
def block_lanczos(C, m, method="symmetric"):
    """Eqs 15-29 with Eq 24 corrected (A1(t+2)) and full Eq 28 B/G terms"""
    Nt, r = len(C), C[0].rows
    if Nt < 2 * m:
        raise ValueError(f"need Nt >= 2m; got Nt={Nt}, m={m}")
    a = [None] * (m + 1); b = [None] * (m + 1); g = [None] * (m + 1)
    A = [None] * (m + 1); G = [None] * (m + 1); B = [None] * (m + 1)
    b1, g1 = factor_delta(C[0], method=method)
    b[1], g[1] = b1, g1
    ib1, ig1 = mp.inverse(b1), mp.inverse(g1)
    A1 = [ib1 * C[t] * ig1 for t in range(Nt)]
    A[1] = A1
    a[1] = A1[1]
    if m == 1:
        return dict(alpha=a, beta=b, gamma=g, A=A, G=G, B=B)
    a1 = a[1]
    b2, g2 = factor_delta(A1[2] - a1 * a1, method=method)
    b[2], g[2] = b2, g2
    ib2, ig2 = mp.inverse(b2), mp.inverse(g2)
    G[2] = [ib2 * (A1[t + 1] - a1 * A1[t]) for t in range(Nt - 1)]
    B[2] = [(A1[t + 1] - A1[t] * a1) * ig2 for t in range(Nt - 1)]
    A2 = [ib2 * (A1[t + 2] + a1 * A1[t] * a1 - (a1 * A1[t + 1] + A1[t + 1] * a1)) * ig2 for t in range(Nt - 2)]
    A[2] = A2
    a[2] = A2[1]
    for j in range(2, m):
        Aj, Gj, Bj, Ajm1 = A[j], G[j], B[j], A[j - 1]
        aj, bj, gj = a[j], b[j], g[j]
        bn, gn = factor_delta(Aj[2] - aj * aj - gj * bj, method=method)
        b[j + 1], g[j + 1] = bn, gn
        ibn, ign = mp.inverse(bn), mp.inverse(gn)
        G[j + 1] = [ibn * (Aj[t + 1] - aj * Aj[t] - gj * Bj[t]) for t in range(len(Aj) - 1)]
        B[j + 1] = [(Aj[t + 1] - Aj[t] * aj - Gj[t] * bj) * ign for t in range(len(Aj) - 1)]
        LA = min(len(Aj) - 2, len(Ajm1), len(Bj) - 1, len(Gj) - 1)
        A[j + 1] = []
        for t in range(LA):
            term = (Aj[t + 2] - (aj * Aj[t + 1] + Aj[t + 1] * aj) + aj * Aj[t] * aj + gj * Ajm1[t] * bj - (gj * Bj[t + 1] + Gj[t + 1] * bj)+ gj * Bj[t] * aj + aj * Gj[t] * bj)
            A[j + 1].append(ibn * term * ign)
        a[j + 1] = A[j + 1][1]
    return dict(alpha=a, beta=b, gamma=g, A=A, G=G, B=B)
def assemble_T(a, b, g, m, r):
    """Eq 32: block-tridiagonal T^(m) from alpha, beta, gamma"""
    T = mp.matrix(m * r, m * r)
    for i in range(m):
        for u in range(r):
            for v in range(r):
                T[i * r + u, i * r + v] = a[i + 1][u, v]
    for i in range(m - 1):
        for u in range(r):
            for v in range(r):
                T[i * r + u, (i + 1) * r + v] = b[i + 2][u, v]
                T[(i + 1) * r + u, i * r + v] = g[i + 2][u, v]
    return T

if __name__ == "__main__":
    print("lanczos_core.py — library module")
    print("Import from krylov_verification.py, hermitian_filter.py, "
          "zcw_filter.py, or noisy_run_all.py.")
    print(f"mp.dps = {mp.dps}")