#!/usr/bin/env python3
"""
noise_generator.py mock-noise generation and preprocessing
Functions:
  generate_noisy_data(C_mean_mp, n_cfg=80, n_boot=200, seed=42, rel0=1e-3, E_noise=0.03, rho_t=0.5)
    in  : noiseless C(t), config count, bootstrap count, seed, noise params
    out : samples (n_cfg, Nt, r, r), mean_noisy, boot_means (n_boot, Nt, r, r)
    note: AR(1) time-correlated noise (our choice, paper uses real data)
  to_mpmath_correlator(C_np)
    in  : numpy (Nt, r, r)
    out : list[mp.matrix]
  symmetrize(C_mp)
    in  : list[mp.matrix]
    out : list[mp.matrix]
    eq  : (99) real-symmetric projection
"""
import numpy as np
from mpmath import mp
def generate_noisy_data(C_mean_mp, n_cfg=80, n_boot=200, seed=42, rel0=1e-3, E_noise=0.03, rho_t=0.5):
    rng = np.random.default_rng(seed)
    Nt = len(C_mean_mp)
    r = C_mean_mp[0].rows
    C_mean = np.zeros((Nt, r, r))
    for t in range(Nt):
        for a in range(r):
            for b in range(r):
                C_mean[t, a, b] = float(mp.re(C_mean_mp[t][a, b]))
    g = rng.normal(size=(n_cfg, Nt, r, r))
    g = np.triu(g)
    g = g + np.triu(g, k=1).swapaxes(-1, -2)
    z = np.empty_like(g)
    z[:, 0] = g[:, 0]
    for t in range(1, Nt):
        z[:, t] = rho_t * z[:, t - 1] + np.sqrt(1 - rho_t ** 2) * g[:, t]
    diag0 = np.diag(C_mean[0])
    scale0 = np.sqrt(np.outer(diag0, diag0))
    t_arr = np.arange(Nt)
    sigma = rel0 * np.exp(-E_noise * t_arr)[:, None, None] * scale0[None, :, :]
    samples = C_mean[None] + z * sigma[None]
    mean_noisy = samples.mean(axis=0)
    boot_idx = rng.integers(0, n_cfg, size=(n_boot, n_cfg))
    boot_means = samples[boot_idx].mean(axis=1)
    return samples, mean_noisy, boot_means

def to_mpmath_correlator(C_np):
    Nt, r, _ = C_np.shape
    return [mp.matrix([[mp.mpf(str(C_np[t, a, b])) for b in range(r)]
                       for a in range(r)]) for t in range(Nt)]

def symmetrize(C_mp):
    """Eq 99: C'_ab(t) = (1/2) Re[C_ab(t) + C_ba(t)]"""
    out = []
    for Ct in C_mp:
        r = Ct.rows
        Cs = mp.matrix(r, r)
        for a in range(r):
            for b in range(r):
                Cs[a, b] = (mp.re(Ct[a, b]) + mp.re(Ct[b, a])) / 2
        out.append(Cs)
    return out

if __name__ == "__main__":
    print("noise_generator.py — library module")
    print("Import from noisy_run_all.py.")