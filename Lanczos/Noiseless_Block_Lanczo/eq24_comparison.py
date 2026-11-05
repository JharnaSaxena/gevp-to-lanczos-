#!/usr/bin/env python3
"""
# eq24_comparison.py
# Run:  python eq24_comparison.py
# Writes: output/diagnostic.txt, output/eq24_side_by_side.txt, fig2_eq24_printed.png, fig2_eq24_corrected.png
# run_recurrence_audit()— full audit, corrected recurrence
#   in  : (none — builds its own mock)
#   out : (none — prints; captured by __main__ to output/diagnostic.txt)
#   does: walks through the entire block-Lanczos recurrence and prints a row
#         for every identity, ending with a PASS/FAIL "SUMMARY" block
#   eq  : (15), (17), (20), (21), (22-23), (24 candidate), (25-29),(30), (32), (36-37), (60)
#
# run_diagnostics(corrected)
#   in  : corrected             bool
#                                  True then use A1(t+2) (correct)
#                                  False then use A1(t+1) (printed, fails)
#   out : dict{corrected, a2_err, worst, rec_err_t15, fail_count}
#         a2_err        ‖A₂(0) − I‖, should be ~0 for corrected
#         worst         worst |G(1) − γ| or |B(1) − β| over j=2..8
#         rec_err_t15   eq. 60 reconstruction error at t=15
#         fail_count    # of 16 Ritz levels with |err| > 0.01
#   eq  : (24) toggle + (30) + (60) + Ritz check
# plot_case(corrected, png)
#   in  : corrected bool have str, output file name
#   out : (saves png)
#   does: fig.-2-style plot of block + scalar-C00 + scalar-C11 runs
#   eq  : uses eq. (15), (24), (60) implicitly via block_lanczos
# run()
#   in  : (none)
#   out : (none — prints side-by-side table)
#   does: runs run_diagnostics(False) -> run_diagnostics(True) and prints the printed-vs-corrected summary table + verdict
#   eq  : (24) as the central comparison
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpmath import mp

from lanczos_core import (
    build_mock, print_mock, factor_delta, block_lanczos,
    assemble_T, submatrix, ritz_energies, reconstruct_C,
    maxdiff, matrix_max_abs, extract_hist,
)
# Part 1 — full recurrence audit (corrected candidate only)
def run_recurrence_audit():
    C, E, Zt, Z = build_mock(32)
    r = 2
    print_mock(C, E, Zt, Z)

    # eq. 15
    print()
    print("[Eq. 15] INITIAL FACTORIZATION  C(0) = β₁ γ₁")
    b1, g1 = factor_delta(C[0])
    init_err = maxdiff(b1 * g1, C[0], r)
    print(f"  β₁ = [[{mp.nstr(b1[0,0], 10)}, {mp.nstr(b1[0,1], 10)}],")
    print(f"        [{mp.nstr(b1[1,0], 10)}, {mp.nstr(b1[1,1], 10)}]]")
    print(f"  γ₁ = [[{mp.nstr(g1[0,0], 10)}, {mp.nstr(g1[0,1], 10)}],")
    print(f"        [{mp.nstr(g1[1,0], 10)}, {mp.nstr(g1[1,1], 10)}]]")
    print(f"  ‖β₁ γ₁ − C(0)‖_max = {mp.nstr(init_err, 3)}")

    m = 8
    out = block_lanczos(C, m, corrected=True)
    alphas, betas, gammas = out["alpha"], out["beta"], out["gamma"]
    As, Gs, Bs = out["A"], out["G"], out["B"]

    # eq. 17
    print()
    print("[Eq. 17] α₁ = β₁⁻¹ C(1) γ₁⁻¹")
    a1 = alphas[1]
    print(f"  α₁ = [[{mp.nstr(a1[0,0], 10)}, {mp.nstr(a1[0,1], 10)}],")
    print(f"        [{mp.nstr(a1[1,0], 10)}, {mp.nstr(a1[1,1], 10)}]]")

    # eq. 20
    print()
    print("[Eq. 20] A₁(t) = β₁⁻¹ C(t) γ₁⁻¹")
    for t in [0, 1, 2, 5, 10, 15]:
        A1t = As[1][t]
        print(f"  t={t:2d}: A₁ = [[{mp.nstr(A1t[0,0], 8)}, {mp.nstr(A1t[0,1], 8)}],")
        print(f"              [{mp.nstr(A1t[1,0], 8)}, {mp.nstr(A1t[1,1], 8)}]]")

    print(f"  ‖A₁(0) − I‖_max  = {mp.nstr(maxdiff(As[1][0], mp.eye(r), r), 3)}")
    print(f"  ‖A₁(1) − α₁‖_max = {mp.nstr(maxdiff(As[1][1], a1, r), 3)}")

    # eq. 21
    print()
    print("[Eq. 21] Δ₂ = A₁(2) − α₁ α₁ = β₂ γ₂")
    Delta2 = As[1][2] - a1 * a1
    print(f"  Δ₂ = [[{mp.nstr(Delta2[0,0], 10)}, {mp.nstr(Delta2[0,1], 10)}],")
    print(f"        [{mp.nstr(Delta2[1,0], 10)}, {mp.nstr(Delta2[1,1], 10)}]]")
    print(f"  ‖β₂ γ₂ − Δ₂‖_max = {mp.nstr(maxdiff(betas[2] * gammas[2], Delta2, r), 3)}")

    # eqs. 22-23
    print()
    print("[Eq. 22-23] G₂(t), B₂(t)")
    for t in [1, 2, 5, 10]:
        G2t, B2t = Gs[2][t], Bs[2][t]
        print(f"  t={t:2d}: G₂ = [[{mp.nstr(G2t[0,0], 8)}, {mp.nstr(G2t[0,1], 8)}],"
              f" [{mp.nstr(G2t[1,0], 8)}, {mp.nstr(G2t[1,1], 8)}]]")
        print(f"        B₂ = [[{mp.nstr(B2t[0,0], 8)}, {mp.nstr(B2t[0,1], 8)}],"
              f" [{mp.nstr(B2t[1,0], 8)}, {mp.nstr(B2t[1,1], 8)}]]")

    # eq. 24 candidate
    print()
    print("=" * 72)
    print("[Eq. 24 CANDIDATE] A₂(t) with A₁(t+2) leading term")
    print("=" * 72)
    for t in [0, 1, 2, 5]:
        A2t = As[2][t]
        print(f"  t={t:2d}: A₂ = [[{mp.nstr(A2t[0,0], 8)}, {mp.nstr(A2t[0,1], 8)}],"
              f" [{mp.nstr(A2t[1,0], 8)}, {mp.nstr(A2t[1,1], 8)}]]")

    a2_err = maxdiff(As[2][0], mp.eye(r), r)
    print(f"  ‖A₂(0) − I‖_max = {mp.nstr(a2_err, 3)}")
    print("  This identity check tests one consequence of the candidate recurrence;")
    print("  it does not prove the full recurrence.")

    # Falsification of the printed form
    print()
    print("[FALSIFICATION] PRINTED Eq. 24 (A₁(t+1) leading term) must fail")
    ib2, ig2 = mp.inverse(betas[2]), mp.inverse(gammas[2])
    A2_printed = (
        ib2
        * (As[1][1] + a1 * As[1][0] * a1
           - (a1 * As[1][1] + As[1][1] * a1))
        * ig2
    )
    err_p = maxdiff(A2_printed, mp.eye(r), r)
    print(f"  Printed form  : ‖A₂(0) − I‖_max = {mp.nstr(err_p, 5)}")
    print(f"  Candidate form: ‖A₂(0) − I‖_max = {mp.nstr(a2_err, 3)}")

    # eqs. 25-29 loop
    print()
    print("[Eq. 25-29] LOOP j = 3..8")
    print(f"  j    ‖β_j γ_j − Δ_j‖       ‖α_j‖       ‖β_j‖       ‖γ_j‖")
    for j in range(2, m + 1):
        if j > 2:
            Delta_j = (As[j - 1][2]
                       - alphas[j - 1] * alphas[j - 1]
                       - gammas[j - 1] * betas[j - 1])
            consistency = mp.nstr(maxdiff(betas[j] * gammas[j], Delta_j, r), 3)
        else:
            consistency = "—"
        aj, bj, gj = alphas[j], betas[j], gammas[j]
        nA = mp.nstr(matrix_max_abs(aj, r), 4)
        nB = mp.nstr(matrix_max_abs(bj, r), 4)
        nG = mp.nstr(matrix_max_abs(gj, r), 4)
        print(f"  {j:2d}   {consistency:<18} {nA:<10} {nB:<10} {nG}")

    # eq. 30
    print()
    print("[Eq. 30] CONSISTENCY  G_j(1) = γ_j,  B_j(1) = β_j")
    print(f"  j    ‖G_j(1) − γ_j‖       ‖B_j(1) − β_j‖")
    worst = mp.mpf(0)
    for j in range(2, m + 1):
        eg = maxdiff(Gs[j][1], gammas[j], r)
        eb = maxdiff(Bs[j][1], betas[j], r)
        worst = max(worst, eg, eb)
        print(f"  {j:2d}   {mp.nstr(eg, 3):<18} {mp.nstr(eb, 3)}")
    print(f"  WORST = {mp.nstr(worst, 3)}")

    # eq. 32
    T8 = assemble_T(alphas, betas, gammas, m, r)
    print()
    print("[Eq. 32] T^(8) — 16×16 block tridiagonal")
    print("  Block structure (j-row × i-col): α diag, β upper, γ lower")
    for i in range(m):
        line = "  "
        for j in range(m):
            if i == j:
                line += " α "
            elif i == j - 1:
                line += " γ "
            elif i == j + 1:
                line += " β "
            else:
                line += " . "
        print(line)
    print("  Diagonal blocks α_j:")
    for j in range(1, m + 1):
        aj = alphas[j]
        print(f"    α_{j} = [[{mp.nstr(aj[0,0], 8)}, {mp.nstr(aj[0,1], 8)}],"
              f" [{mp.nstr(aj[1,0], 8)}, {mp.nstr(aj[1,1], 8)}]]")

    # eqs. 36-37
    print()
    print("[Eq. 36-37] RITZ DECOMPOSITION of T^(8) -> λ_k -> E_k = -ln λ_k")
    Er = ritz_energies(T8)
    print(f"  k    λ_k                E_k^(8)                   E_k true   |err|")
    all_ok = True
    for k in range(16):
        lam = mp.e ** (-Er[k])
        err = abs(mp.re(Er[k]) - E[k])
        all_ok &= (err < mp.mpf("1e-50"))
        print(f"  {k:2d}   {mp.nstr(mp.re(lam), 12):<18}"
              f"{mp.nstr(mp.re(Er[k]), 20):<24}"
              f"{mp.nstr(E[k], 4):<10}{mp.nstr(err, 3)}")
    print(f"  All |E_k − E_k^true| < 1e-50 : {all_ok}")

    # eq. 60
    print()
    print("[Eq. 60] RECONSTRUCTION  C(t) = β₁ [T^(8)^t]_{(1,1)} γ₁")
    Crec = reconstruct_C(T8, betas[1], gammas[1], 32, r)
    print(f"  t    ‖C_recon(t) − C(t)‖_max")
    for t in range(16):
        print(f"  {t:2d}   {mp.nstr(maxdiff(Crec[t], C[t], r), 3)}")
    rec_err = max(maxdiff(Crec[t], C[t], r) for t in range(16))

    # scalar lanczos on diagonals
    print()
    print("[Scalar m=16] LANCZOS ON DIAGONAL C_00(t) AND C_11(t)")
    for d in (0, 1):
        Cd = [mp.matrix([[C[t][d, d]]]) for t in range(32)]
        o = block_lanczos(Cd, 16, corrected=True)
        Td = assemble_T(o["alpha"], o["beta"], o["gamma"], 16, 1)
        Ed = ritz_energies(Td)
        print(f"  C_{d}{d}(t):")
        print(f"    E_0^(16) = {mp.nstr(mp.re(Ed[0]), 18)}  (true 0.1)  "
              f"|err| = {mp.nstr(abs(mp.re(Ed[0]) - E[0]), 3)}")
        print(f"    E_1^(16) = {mp.nstr(mp.re(Ed[1]), 18)}  (true 0.2)  "
              f"|err| = {mp.nstr(abs(mp.re(Ed[1]) - E[1]), 3)}")
        Cr = reconstruct_C(Td, o["beta"][1], o["gamma"][1], 32, 1)
        rerr = max(abs(Cr[t][0, 0] - Cd[t][0, 0]) for t in range(32))
        print(f"    Eq. 60 reconstruction (t≤31): max err = {mp.nstr(rerr, 3)}")

    # summary
    print()
    print("SUMMARY")
    eq30_ok = float(worst) < 1e-80
    a2_ok = float(a2_err) < 1e-80
    printed_fails = float(err_p) > 1e-3
    init_ok = float(init_err) < 1e-80
    rec_ok = float(rec_err) < 1e-50

    print(f"  Eq. 15  β₁ γ₁ = C(0)              : "
          f"{'PASS' if init_ok else 'FAIL'}  ({mp.nstr(init_err, 3)})")
    print(f"  Eq. 30  G_j(1)=γ_j, B_j(1)=β_j    : "
          f"{'PASS' if eq30_ok else 'FAIL'}  (worst {mp.nstr(worst, 3)})")
    print(f"  Eq. 24  A₂(0) = I (candidate)     : "
          f"{'PASS' if a2_ok else 'FAIL'}  ({mp.nstr(a2_err, 3)})")
    print(f"  Eq. 24  Printed form fails A₂(0)=I: "
          f"{'PASS' if printed_fails else 'FAIL'}  (printed gives {mp.nstr(err_p, 3)})")
    print(f"  Eq. 37  Block spectrum m=8        : "
          f"{'PASS' if all_ok else 'FAIL'}")
    print(f"  Eq. 60  Reconstruction            : "
          f"{'PASS' if rec_ok else 'FAIL'}  ({mp.nstr(rec_err, 3)})")

# Part 2 — eq. 24 side-by-side (printed vs corrected)
def run_diagnostics(corrected):
    tag = "corrected (A_1(t+2))" if corrected else "as printed (A_1(t+1))"
    C, E, _Zt, _Z = build_mock(32)
    r = 2
    out8 = block_lanczos(C, 8, corrected=corrected)

    print()
    print(f"eq. (24) {tag}")

    # d1: biorthonormality at t=0
    a2_err = maxdiff(out8["A"][2][0], mp.eye(r), r)
    print()
    print("[d1] ||A_2(0) - I||  (should be zero by biorthonormality)")
    print(f"     ||A_2(0) - I||_max = {mp.nstr(a2_err, 5)}")
    print(f"     >>> {'pass: working precision' if float(a2_err) < 1e-3 else 'fail: order-one error'}")

    # d2: eq. 30 identity
    print()
    print("[d2] eq. (30): G_j(1) = gamma_j,  B_j(1) = beta_j")
    worst = mp.mpf(0)
    for j in range(2, 9):
        eg = maxdiff(out8["G"][j][1], out8["gamma"][j], r)
        eb = maxdiff(out8["B"][j][1], out8["beta"][j], r)
        worst = max(worst, eg, eb)
        print(f"     j={j}:  |G-gamma|={mp.nstr(eg, 3):<10}  |B-beta|={mp.nstr(eb, 3):<10}")
    print(f"     worst = {mp.nstr(worst, 5)}")

    # d3: eq. 60 reconstruction
    print()
    print("[d3] eq. (60): C(t) = beta_1 [T^t]_{11} gamma_1,  t <= 15")
    T8 = assemble_T(out8["alpha"], out8["beta"], out8["gamma"], 8, r)
    Crec = reconstruct_C(T8, out8["beta"][1], out8["gamma"][1], 32, r)
    print("     t   ||C_recon(t) - C(t)||_max")
    rec_errs = []
    for t in range(16):
        d = maxdiff(Crec[t], C[t], r)
        rec_errs.append(d)
        flag = "" if float(d) < 1e-20 else "   <<< fail"
        print(f"     {t:2d}   {mp.nstr(d, 5)}{flag}")

    # d4: block spectrum at m=8
    print()
    print("[d4] block lanczos spectrum at m=8  (target 0.1, 0.2, ..., 1.6)")
    Er = ritz_energies(T8)
    fail_count = 0
    for k in range(min(16, len(Er))):
        E_re = mp.re(Er[k])
        err = abs(E_re - E[k])
        if float(err) > 0.01:
            fail_count += 1
            print(f"     {k:2d}   E = {mp.nstr(E_re, 20):<24}  "
                  f"true = {mp.nstr(E[k], 4):<6}  "
                  f"|err| = {mp.nstr(err, 3):<10}   <<< fail")
        else:
            print(f"     {k:2d}   E = {mp.nstr(E_re, 20):<24}  "
                  f"true = {mp.nstr(E[k], 4):<6}  "
                  f"|err| = {mp.nstr(err, 3)}")
    if fail_count:
        print(f"     >>> {fail_count} of 16 ranks off by > 0.01")
    else:
        print("     >>> all 16 ranks within 0.01")

    return dict(corrected=corrected, a2_err=a2_err, worst=worst,
                rec_err_t15=rec_errs[15], fail_count=fail_count)


def plot_case(corrected, png):
    tag = "corrected  (A$_1$(t+2))" if corrected else "as printed  (A$_1$(t+1))"
    subtitle = ("reproduces fig. 2: block m=8, scalar m=16 "
                "converge exactly to black reference lines"
                if corrected else
                "fails: curves do not converge to black reference lines")

    C, _E, _Zt, _Z = build_mock(32)
    block_hist = extract_hist(C, 8, 2, corrected=corrected)
    c00_hist = extract_hist(
        [mp.matrix([[C[t][0, 0]]]) for t in range(32)], 16, 1, corrected=corrected)
    c11_hist = extract_hist(
        [mp.matrix([[C[t][1, 1]]]) for t in range(32)], 16, 1, corrected=corrected)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(16):
        ax.axhline(0.1 * (k + 1), color="k", lw=0.5, alpha=0.3)

    for hist, color, label in [
        (block_hist, "tab:blue", "block"),
        (c00_hist, "tab:orange", r"scalar $C_{00}$"),
        (c11_hist, "tab:red", r"scalar $C_{11}$"),
    ]:
        xs0, ys0, xs1, ys1 = [], [], [], []
        for e in hist:
            t = e["t"]
            Evals = sorted([mp.re(v) for v in e["energies"]])
            if len(Evals) >= 1 and -2 < Evals[0] < 5:
                xs0.append(t); ys0.append(float(Evals[0]))
            if len(Evals) >= 2 and -2 < Evals[1] < 5:
                xs1.append(t); ys1.append(float(Evals[1]))
        if xs0:
            ax.plot(xs0, ys0, "o-", color=color, lw=1.5, ms=4,
                    label=label + " rank 0")
        if xs1:
            ax.plot(xs1, ys1, "s--", color=color, lw=1.2, ms=3,
                    label=label + " rank 1")

    ax.set_xlabel(r"$t = 2m - 1$", fontsize=12)
    ax.set_ylabel(r"$E_k^{(m)} = -\log \lambda_k^{(m)}$", fontsize=12)
    ax.set_title(f"eq. (24) {tag}\n{subtitle}", fontsize=11)
    ax.set_xlim(0, 32); ax.set_ylim(-2, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close(fig)
    print(f"     saved {png}")


def run():
    print("eq. (24) -- as printed vs corrected")
    s_printed = run_diagnostics(corrected=False)
    print()
    print("[plot] fig. 2 style, eq. (24) as printed")
    plot_case(corrected=False, png="fig2_eq24_printed.png")

    s_corrected = run_diagnostics(corrected=True)
    print()
    print("[plot] fig. 2 style, eq. (24) corrected")
    plot_case(corrected=True, png="fig2_eq24_corrected.png")

    print()
    print("side-by-side summary")
    print(f"  {'quantity':<28} {'printed':>22} {'corrected':>22}")
    print(f"  {'||A_2(0) - I||':<28} "
          f"{mp.nstr(s_printed['a2_err'], 5):>22} "
          f"{mp.nstr(s_corrected['a2_err'], 5):>22}")
    print(f"  {'eq. 30 worst':<28} "
          f"{mp.nstr(s_printed['worst'], 5):>22} "
          f"{mp.nstr(s_corrected['worst'], 5):>22}")
    print(f"  {'eq. 60 err at t=15':<28} "
          f"{mp.nstr(s_printed['rec_err_t15'], 5):>22} "
          f"{mp.nstr(s_corrected['rec_err_t15'], 5):>22}")
    print(f"  {'spectrum failures (of 16)':<28} "
          f"{s_printed['fail_count']:>22} "
          f"{s_corrected['fail_count']:>22}")
    print()
    vp = "fail" if (s_printed["fail_count"] > 0
                     or float(s_printed["a2_err"]) > 1e-3) else "pass"
    vc = "fail" if (s_corrected["fail_count"] > 0
                    or float(s_corrected["a2_err"]) > 1e-3) else "pass"
    print(f"  verdict  printed   : {vp}")
    print(f"  verdict  corrected : {vc}")


if __name__ == "__main__":
    from lanczos_core import capture_and_show_summary
    print("Eq. 24 / recurrence audit")
    capture_and_show_summary(
        run_recurrence_audit,
        "output/diagnostic.txt",
        "SUMMARY",
    )

    print()
    print("Eq. 24 side-by-side")
    capture_and_show_summary(
        run,
        "output/eq24_side_by_side.txt",
        "side-by-side summary",
    )