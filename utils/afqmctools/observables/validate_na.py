#!/usr/bin/env python3

# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from afqmctools.analysis.rdm import get_afqmc_rdm_samples
from afqmctools.observables.rhonk import momentum_distribution, nk_discrete_distribution


def _load_latest_rdm(rdm_stat_file):
    dm_dict = get_afqmc_rdm_samples(rdm_stat_file)
    avg_keys = sorted(k for k in dm_dict.keys() if k.startswith("a"))
    if not avg_keys:
        raise ValueError("No AFQMC RDM averages found in statistics file")
    key = avg_keys[-1]
    return dm_dict[key]["dm_mean"], dm_dict[key]["dm_error"], key


def _bin_by_magnitude(kvecs, nkm, nbins):
    kmag = np.linalg.norm(kvecs, axis=1)
    edges = np.linspace(0.0, kmag.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(nbins, np.nan)

    inds = np.digitize(kmag, edges) - 1
    for i in range(nbins):
        mask = inds == i
        if np.any(mask):
            binned[i] = np.mean(nkm[mask])
    return centers, binned


def main():
    parser = argparse.ArgumentParser(description="Validate AFQMC momentum workflow for bcc Na")
    parser.add_argument("--orbital-source", type=Path, required=True, help="Path to QE folder or Coqui orbs.h5")
    parser.add_argument("--rdm-stat", type=Path, required=True, help="Path to qmc.s000.stat.h5")
    parser.add_argument("--kcut", type=float, default=1.5, help="Momentum cutoff in 1/Bohr")
    parser.add_argument("--nsample", type=int, default=64, help="Bootstrap samples for error propagation")
    parser.add_argument("--expected-supercell-electrons", type=float, default=64.0)
    parser.add_argument("--expected-per-cell-electrons", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for electron-count checks")
    parser.add_argument("--nbins", type=int, default=80, help="Bins for n(|q|) plot")
    parser.add_argument("--plot-file", type=Path, default=Path("na_nofq_binned.png"))
    args = parser.parse_args()

    rdm, error_rdm, avg_key = _load_latest_rdm(args.rdm_stat)
    print(f"[validate_na] Loaded RDM average key: {avg_key}")
    print(f"[validate_na] rdm shape: {rdm.shape}, error shape: {error_rdm.shape}")

    kvecs, nkm, nke = momentum_distribution(
        rdm=rdm,
        error_rdm=error_rdm,
        orbital_source=args.orbital_source,
        kcut=args.kcut,
        nsample=args.nsample,
        verbose=True,
    )

    kpts_cart, nk_mean, nk_error = nk_discrete_distribution(
        rdm=rdm,
        error_rdm=error_rdm,
        orbital_source=args.orbital_source,
        nsample=args.nsample,
        verbose=True,
        band_resolved=False,
    )

    nq_sum = np.sum(nkm)
    nkpts = kpts_cart.shape[0]
    nk_avg_per_cell = np.sum(nk_mean) / nkpts

    print(f"[validate_na] sum_q n(q): {nq_sum:.8f}")
    print(f"[validate_na] sum_k n(k)/Nk: {nk_avg_per_cell:.8f}")

    if not np.isclose(nq_sum, args.expected_supercell_electrons, rtol=args.rtol):
        raise AssertionError(
            "Unfolded sum rule failed: "
            f"sum_q n(q)={nq_sum:.8f}, expected {args.expected_supercell_electrons:.8f}, "
            f"rtol={args.rtol}"
        )

    if not np.isclose(nk_avg_per_cell, args.expected_per_cell_electrons, rtol=args.rtol):
        raise AssertionError(
            "Discrete-mesh sum rule failed: "
            f"sum_k n(k)/Nk={nk_avg_per_cell:.8f}, expected {args.expected_per_cell_electrons:.8f}, "
            f"rtol={args.rtol}"
        )

    eps = max(5.0 * float(np.nanmean(nke)), 1e-6)
    if np.any(nkm < -eps) or np.any(nkm > 2.0 + eps):
        nmin = float(np.min(nkm))
        nmax = float(np.max(nkm))
        raise AssertionError(
            "Physical bounds failed for n(q): "
            f"min={nmin:.6f}, max={nmax:.6f}, allowed=[{-eps:.6f}, {2.0 + eps:.6f}]"
        )

    imag_ratio = float(np.max(np.abs(np.imag(nkm))) / max(np.max(np.abs(np.real(nkm))), 1.0))
    if imag_ratio > 1e-2:
        raise AssertionError(
            "Imaginary-part check failed: "
            f"max|Im|/max|Re|={imag_ratio:.3e}, threshold=1e-2"
        )

    kmag_centers, nkm_binned = _bin_by_magnitude(kvecs, np.real(nkm), args.nbins)
    plt.figure(figsize=(7, 4.5))
    plt.plot(kmag_centers, nkm_binned, marker="o", ms=3, lw=1)
    plt.axvline(0.26, color="red", ls="--", lw=1, label="k_F ~ 0.26 1/Bohr")
    plt.xlabel("|q| [1/Bohr]")
    plt.ylabel("n(|q|)")
    plt.title("bcc Na AFQMC Momentum Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_file, dpi=160)
    print(f"[validate_na] Saved plot to: {args.plot_file}")

    print("[validate_na] All validation checks passed")


if __name__ == "__main__":
    main()
