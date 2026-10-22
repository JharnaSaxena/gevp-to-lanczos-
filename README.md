# GEVP to Lanczos

Implementation of §II.B of Hackett & Wagman, [arXiv:2412.04444](https://arxiv.org/abs/2412.04444) — oblique block Lanczos for lattice QCD correlator matrices — with verification against the paper's noiseless mock data (§III).

## Overview

Block Lanczos generalizes the standard Lanczos algorithm to correlator matrices built from multiple interpolating operators. It extracts Ritz values and vectors from the Krylov subspace spanned by all operators simultaneously, with faster convergence and better signal-to-noise than scalar Lanczos, and formally contains GEVP as its one-step limit.

This repository implements the oblique block Lanczos recursion and validates it on the paper's noiseless mock data, including a check of Eq. (24) whose leading term appears to be misprinted in the paper.

## Repository structure
GEVP-to-Lanczos/
├── lanczos.ipynb # main notebook — II.B implementation, verification, Fig. 2
├── requirements.txt
├── README.md
├── fig2_reproduction.png # Fig. 2 — corrected Eq. (24)
├── fig2_eq24_printed.png # Fig. 2 style — Eq. (24) as printed
├── fig2_eq24_corrected.png # Fig. 2 style — Eq. (24) with A₁(t+2)
├── fig2_data.csv # Ritz table, 80 sig figs, lossless round-trip
└── output/
├── diagnostic.txt # per-equation trace (Eqs. 15–60)
├── fig2_full.txt # all Ritz values per iteration
├── higher_rank.txt # saturation, per-rank convergence, non-degeneracy
├── ordering.txt # per-iteration ordering and bijection diagnostics
└── ritz_values.json # structured Ritz data (100-digit strings)

 

## Notebook

Two workflow cells:

**Cell 3 — II.B verification and Fig. 2.** Implements the recursion (Eqs. 15–30), assembles the block-tridiagonal `T^(m)` (Eq. 32), diagonalizes it for Ritz values `λ_k` and energies `E_k = −ln λ_k` (Eqs. 36–37), and verifies the reconstruction `C(t) = β₁ [T^t]_{11} γ₁` (Eq. 60) for `t ≤ 2m−1`. Reproduces Fig. 2 and checks the full 16-rank spectrum at saturation. Verbose output goes to `output/`; console shows only the summary.

**Cell 4 — Eq. (24) comparison.** Runs the recursion on identical mock data with the paper's Eq. (24) as printed (leading term `A₁(t+1)`) and with the corrected form (`A₁(t+2)`), and reports the side-by-side diagnostic.

All computations at `mp.dps = 100`, matching the paper's noiseless prescription. Deterministic — same 100-digit output on every run.

## Results

**Verification with corrected Eq. (24):**

| Check | Result |
|---|---|
| Eq. 15 — `‖β₁γ₁ − C(0)‖` | `5.7e-101` |
| Eq. 30 — `‖G_j(1) − γ_j‖`, `‖B_j(1) − β_j‖` | `1.4e-101` |
| Eq. 60 — reconstruction for `t ≤ 15` | `2.0e-100` |
| Block Lanczos at `m = 8` — all 16 Ritz values | match `0.1, …, 1.6` to `< 1e-88` |
| Scalar `C₀₀`, `C₁₁` at `m = 16` | all 16 ranks to `< 1e-80` |
| Rank ↔ true level at saturation | bijection, all three methods |
| Non-degeneracy at every `m` | worst gap `0.021`, above `1e-3` threshold |

**Fig. 2 reproduced.** Block Lanczos (`r = 2`) saturates the 16-state system at `m = 8` (Krylov dimension `2m = 16`); scalar Lanczos on `C₀₀` or `C₁₁` requires `m = 16`.

**Eq. (24) — printed vs corrected.** The paper's Eq. (24) uses `A₁(t+1)` in the leading term. This violates `A₂(0) = I`, which follows from biorthonormality `⟨v^L_{ja}|v^R_{jb}⟩ = δ_ab` and is not a matter of convention, by an order-one amount:

| Quantity | Printed `A₁(t+1)` | Corrected `A₁(t+2)` |
|---|---|---|
| `‖A₂(0) − I‖` | `4.4684` | `1.1e-100` |
| Eq. 60 error at `t = 15` | `30.17`, growing with `t` | `2.0e-100`, flat |
| Ranks matching true spectrum | `0 / 16` | `16 / 16` |
| Negative `E` (i.e. `λ > 1`) | `5` | `0` |
| Exact eigenvalue degeneracies | `4` | `0` |

The leading term of Eq. (28) reduced to `j = 1` is also `A₁(t+2)`, so the paper's printed Eq. (24) and Eq. (28) are internally inconsistent with each other. The `A₁(t+2)` form is used throughout the notebook. This is treated as a likely typographical error in the paper; confirmation from the authors is pending.

## How to run

**Google Colab:**
1. Open `lanczos.ipynb` in Colab
2. Runtime → Run all (~60 s)
3. Console shows summary tables; full traces are written to `output/`

**Local:**
pip install -r requirements.txt
jupyter notebook lanczos.ipynb
requirements.txt:
mpmath>=1.3.0
matplotlib>=3.7
pandas>=2.0
numpy>=1.24
Scope
Implements §II.B (oblique block Lanczos recursion) and §III (noiseless demonstration) of the paper. Not yet implemented:

§II.C–F — Ritz vectors, Krylov coefficients, overlap factors

§II.G — Operator matrix elements

§IV — Two-sided bounds

§V — Noisy data, ZCW filtering, bootstrap

§VI — GEVP comparison
---
## What to check before committing

1. **`requirements.txt` doesn't exist yet in your repo** — create it with the four lines above, or delete that reference from the README. If you're not going to add it, remove the "Local" section entirely and keep only the Colab instructions.

2. **`numpy` is imported in the notebook indirectly** (via pandas and matplotlib). We don't use `numpy` directly, so strictly it isn't a dependency. But listing it is standard since pandas pulls it in. Either include it or drop it — no functional difference.

3. **`fig2_data.csv` is written to CWD, not `output/`.** That matches the code as written. If you later move it into `output/`, update the README structure tree.

4. **One-sentence AI disclosure.** The README doesn't mention it. That's fine for a research repo — most don't. If you want to be extra thorough, add after Scope:

Note on Research Process

This work was developed by the author with mathematical guidance from a faculty mentor and supplementary assistance from internet based tools during debugging and technical review. The author carried out the implementation, numerical experiments, and verification of the discrepancy between Eqs. (24) and (28).
```markdown
