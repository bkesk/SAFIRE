# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
AFQMCTOOLs

analysis of the 1-rdm

author: Kyle Eskridge
date: 1/29/2024
"""
from pathlib import Path
from warnings import warn

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

from afqmctools.analysis.average import estimate_error_eig
from stats import stat_h5

from .common import get_metadata, get_bp_taus, _Neq_from_Teq

def get_afqmc_rdm_samples(rdm_fname,kappa=None,nequil=0,group='BackPropagated'):
    """  
    Read 1-rdm from AFQMC format
    """
    rdm_fname = Path(rdm_fname)
    if group == "BackPropagated":
        taus = get_bp_taus(rdm_fname)
        dm_map = {'taus': taus}
    else:
        taus = [get_metadata(rdm_fname)['Timestep']]
        dm_map = { 'taus' : taus }

    with h5.File(rdm_fname,'r') as f:
        for iav, tau in enumerate(taus):
            name = 'a%d' % iav
            dm, de = stat_h5.afobs(
                f,
                'FullOneRDM',
                nequil,
                numer='one_rdm',
                iav=iav,
                kappa=kappa,
                group=group
            )
            dm_map[name] = {'dm_mean': dm, 'dm_error': de}
    return dm_map

def average_afqmc_rdm(
        rdm_file="qmc.s000.stat.h5",
        dm_map=None,
        kappa=None,
        Neq=None,
        Teq=None,
        group='BackPropagated'
        ):
    """
    Average the AFQMC rdms

    Parameters
    ----------
    rdm_file : str 
        path to the AFQMC 1-rdm file
    dm_map : dict, optional
        dictionary containing the 1-rdm samples. If not given, the samples
        are read from rdm_file.
    kappa : float, optional
        autocorrelation length in units of blocks. If not given, the autocorrelation
        length is automatically computed from the samples.
    Neq : int, optional
        number of blocks in equilibration. If not given, it is computed from Teq.
    Teq : float, optional
        equilibration time. If only Teq is given, then Neq is computed
           as Neq = Teq / (dT/block). If neither Teq nor Neq given, Neq is set to 0. 
    
    Returns
    -------
    dm_means:np.array with shape (Navgs,Nspins,Nbasis,Nbasis)
    dm_errors:np.array with shape (Navgs,Nspins,Nbasis,Nbasis)

    Notes
    -----
    For solids, the basis includes k-point dependence implicitly. (i.e. Nbasis = Nbands * Nkpts) 
    """

    if Neq is None and Teq is not None:
        Neq = _Neq_from_Teq(Teq,dm_map['taus'])
    elif Neq is None and Teq is None:
        Neq = 0
    elif Neq is not None and Teq is not None:
        warn("average_afqmc_rdm: both Neq and Teq are given. Using Neq.")

    if dm_map is None:
        dm_map = get_afqmc_rdm_samples(
            rdm_fname=rdm_file,
            kappa=kappa,
            nequil=Neq,
            group=group
        )

    dm_means = np.array([
        dm_map[key]['dm_mean'] for key in 
        [ key for key in dm_map.keys() if key.startswith("a")]
    ])

    dm_errors = np.array([
        dm_map[key]['dm_error'] for key in 
        [ key for key in dm_map.keys() if key.startswith("a")]
    ])

    return dm_means,dm_errors

def resample(marr, earr, nsample): #TODO: move to the rdm module
    """
    Resample the mean reduced density matrix (RDM).

    Parameters
    ----------
    marr : numpy.ndarray
        The mean reduced density matrix to be resampled. It should be a square matrix.
    earr : numpy.ndarray
        The error array corresponding to the mean RDM. It should have the same shape as `marr`.
    nsample : int
        The number of samples to generate.
    
    Returns
    -------
    numpy.ndarray
        A new array of shape `(nsample, *marr.shape)` containing the resampled RDMs.

    Notes
    -----
    to set the random seed, use `numpy.random.seed(seed)` immediately before calling this function.
    """
    shape = marr.shape
    assert np.allclose(shape, earr.shape)
    noise = earr * np.random.randn(nsample, *shape)
    new_marr = marr[np.newaxis] + noise
    return new_marr

def check_1rdm_convergence(dm_means,dm_errors,sigma=2):
    """
    Check the convergence of the 1-rdm.

    The 1-rdms are considered converged if the difference between the
    last two samples is within the maximum (over spin and orbitals) 
    joint error of the two samples.

    Parameters
    ----------
    dm_means:np.array with shape (Navgs,Nspins,Nbasis,Nbasis)
    dm_errors:np.array with shape (Navgs,Nspins,Nbasis,Nbasis)

    Returns
    -------
    converged:bool
        True if the 1-rdm is converged
    """
    dm_diff = np.diff(dm_means,axis=0)
    dm_diff_errors = np.zeros_like(dm_diff)
    for a in range(dm_diff_errors.shape[0]):
        dm_diff_errors[a] = np.sqrt(np.sum(np.power(dm_errors[a:a+1],2),axis=0))

    _sigma_actual = np.max(np.abs(dm_diff)/dm_diff_errors,axis=(1,2,3)).real

    converged_avg = np.where(_sigma_actual < sigma)[0]

    if converged_avg.size > 0:
        print("1-RDM is converged for averages: ",converged_avg + 1)
        for a in converged_avg:
            print(f"Average {a+1}: same to within {_sigma_actual[a]} sigma")
        return True
    else:
        print("1-RDM is not converged.")
        for a in range(len(_sigma_actual)):
            print(f"Average {a+1}: same to within {_sigma_actual[a]} sigma")
        print("sigma threshold = ",sigma)
        return False

def plot_rdm_diagonal(dm_means,dm_errors,ax,spin=None,**kwargs):
    """
    plot the diagonal of the 1-RDM vs basis set index
    with errorbars

    Parameters
    ----------
    rdm : np.array with shape (Nspins,Nbasis,Nbasis)
        1-RDM matrix
    delta_rdm : np.array with shape (Nspins,Nbasis,Nbasis)
        error in the 1-RDM matrix
    ax : matplotlib.pyplot.axis
        axis to plot the diagonal
    spin : int, optional
        spin index to plot. If None, sum over spins.
    **kwargs : dict
        keyword arguments to pass to plt.errorbar

    Notes
    -----
    This function is useful for checking convergence
    of the 1-RDM.

    It may be useful to pass a label to the errorbar
    for the legend.
    """

    nbasis = dm_means.shape[-1]

    x = np.arange(nbasis)

    # check for real or complex
    if np.all(not np.iscomplex(dm_means)):
        dm_means = dm_means.real
        dm_errors = dm_errors.real
    else:
        max_imag = np.max(np.abs(dm_means.imag))
        warn(f"rdm has non-zero imaginary component; discarding imaginary part. Max imaginary part: {max_imag}")
        dm_means = dm_means.real
        dm_errors = dm_errors.real
        
    # trace over spin
    if spin is None:
        dm_means = np.sum(dm_means,axis=0)
        dm_errors = np.sqrt(np.sum(dm_errors**2,axis=0))
    else:
        dm_means = dm_means[spin]
        dm_errors = dm_errors[spin]
    
    ax.errorbar(
        x,
        dm_means[x,x],
        yerr=dm_errors[x,x],
        **kwargs
    )

def get_natural_orbs(dm_mean,dm_error,bootstrap=True,Nbootstraps=100,force_hermitian=True,rtol=0.001):
    """
    Get the natural orbitals from the 1-rdm.

    Uses bootstrapping to estimate the error in the natural occupancies.
    The natural orbitals are the eigenvectors of the 1-rdm with no error 
    propagation.

    Parameters
    ----------
    dm_mean:np.array with shape (Nbasis,Nbasis)
    dm_error:np.array with shape (Nbasis,Nbasis)
    bootstrap:bool, optional
        whether to use bootstrapping to estimate the error
    Nbootstraps:int, optional
        number of bootstraps to use
    force_hermitian:bool, optional
        whether to force the 1-rdm to be hermitian
    rtol:float, optional
        relative tolerance for checking hermiticity

    Returns
    -------
    natural_orbs:np.array with shape (Nbasis,Nbasis)
    natural_occs:np.array with shape (Nbasis)
    natural_occs_errs:np.array with shape (Nbasis)
    """
    # check if the 1-rdm is hermitian
    if not np.allclose(
            dm_mean,dm_mean.T.conj(),
            atol=np.average(dm_error),
            rtol=rtol
        ):
        warn("1-rdm is not hermitian.")
        print("Max deviation from hermitian: ",np.max(np.abs(dm_mean - dm_mean.T.conj())))

    if force_hermitian:
        print("Forcing 1-rdm to be hermitian.")
        dm_mean = 0.5*(dm_mean + dm_mean.T.conj())
        natural_occs, natural_orbs = np.linalg.eigh(dm_mean)
    else:
        natural_occs, natural_orbs = np.linalg.eig(dm_mean)
    if bootstrap:
        natural_occs_errs = estimate_error_eig(dm_mean,dm_error,nsamp=Nbootstraps)
    else:
        natural_occs_errs = np.zeros_like(natural_occs)
    return natural_orbs,natural_occs,natural_occs_errs

def plot_1rdm_convergence(dm_means,dm_errors,spin_trace=False,force_hermitian=False):
    """
    Plot the convergence of the 1-rdm.

    Parameters
    ----------
    dm_means:np.array with shape (Navgs,Nspins,Nbasis,Nbasis)
    dm_errors:np.array with shape (Navgs,Nspins,Nbasis,Nbasis)
    """

    _pretty_spin = {0:r"$\alpha$ sector",1:r"$\beta$ sector",None:"spin-traced"}

    nspin,nbasis,_ = dm_means.shape[1:]
    
    if spin_trace and nspin > 1:
        dm_means = np.sum(dm_means,axis=1,keepdims=True)
        dm_errors = np.sqrt(np.sum(dm_errors**2,axis=1,keepdims=True))
        nspin = 1
        # TODO: double check the shape!
    
    fig, axs = plt.subplots(1,nspin,figsize=(6*nspin,4))

    for spin in range(nspin):
        ax = axs[spin] if nspin > 1 else axs
        for a, rho in enumerate(dm_means):
            # diagonalize the 1-rdm
            _, natural_occs, delta_natural_occs = get_natural_orbs(
                rho[spin],dm_errors[a,spin],
                force_hermitian=force_hermitian
            )
            ax.errorbar(
                np.arange(nbasis),
                natural_occs.real,
                yerr=delta_natural_occs[a],
                label=f"Average {a}",
                fmt="o-",
                capsize=5  # Add caps to the error bars
            )

        ax.set_title(f"Natural Occupancies for the {_pretty_spin[(spin if nspin > 1 else None)]} 1-rdm")
        ax.set_xlabel("Natural orbital index")
        ax.set_ylabel("Natural Occupancy")
        ax.legend()

    plt.show()

    return fig, axs

