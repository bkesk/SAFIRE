# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import logging
logger = logging.getLogger(__name__)
from warnings import warn

import numpy as np
import h5py
import scipy

from afqmctools.utils.io import (
    to_complex, 
    add_group, 
    add_dataset
)

logger.dev(
    "[Warning] (for Devs) afqmctools.utils.slater_types is indirectly imported "
    "from afqmctools.wavefunction.common; Refactor to direct imports and remove " 
    "the import here."
)
from afqmctools.utils.slater_types import (
    _SlaterType,
    _slater2dims,
    _slater_enum_map,
    _get_slater_type
)

def write_wfn(
    filename,
    wfn,
    walker_type,
    nelec,
    norb,
    init=None,
    orbmat=None,
    verbose=False
    ):
    """
    Write wavefunction to HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 file to write wavefunction to.
    wfn : tuple
        Wavefunction description. Either a list containing an array coefficients and occupation numbers (PHMSD)
        or a tuple containing an array of coefficients and an array of determinants (NOMSD).
    walker_type : str
        Type of walker to write. Options are 'closed', 'collinear', 'noncollinear', 'fullypolarized'.
    nelec : tuple
        Number of alpha and beta electrons.
    norb : int
        Number of orbitals.
    init : tuple
        Initial Slater determinants for AFQMC.
    orbmat : tuple
        Orbital matrices for PHMSD wavefunctions.
    verbose : bool
        Print additional information.
    """
    # User defined wavefunction.
    # PHMSD is a list of tuple of (ci, occa, occb).
    # NOMSD is a tuple of (list, np.ndarray).
    
    walker_type = _slater_enum_map(walker_type)

    if len(wfn) == 3:
        coeffs, occa, occb = wfn
        wfn_type = 'PHMSD'
    elif len(wfn) == 2:
        coeffs, wfn = wfn
        wfn_type = 'NOMSD'
    else:
        raise ValueError("Unknown wavefunction type passed.")

    with h5py.File(filename, 'a') as fh5:
        nalpha, nbeta = nelec

        # Apply some user input corrections
        if walker_type == _SlaterType.COLLINEAR and nelec[1] == 0:
            print(
                "collinear/uhf/rohf walker type requested with no beta electrons: "
                "using fully spin-polarized walkers"
            )
            walker_type = _SlaterType.FULLYPOLARIZED
        
        if wfn_type == 'PHMSD':
            print("PHMSD trial wavfunction -> using uhf-like walkers")
            if nelec[1] > 0:
                walker_type = _SlaterType.COLLINEAR
            else:
                walker_type = _SlaterType.FULLYPOLARIZED

            wfn_group = add_group(fh5, 'Wavefunction/PHMSD')
            write_phmsd(wfn_group, occa, occb, nelec, norb,
                        init=init, orbmat=orbmat)
        
        elif wfn_type == 'NOMSD':
            wfn_group = add_group(fh5, 'Wavefunction/NOMSD')
            
            if (
                walker_type == _SlaterType.CLOSED or 
                walker_type == _SlaterType.COLLINEAR
            ):
                write_nomsd(
                    fh5=wfn_group, 
                    wfn=wfn,
                    uhf=walker_type==_SlaterType.COLLINEAR,
                    nelec=nelec,
                    init=init
                )
            elif (
                walker_type == _SlaterType.NONCOLLINEAR or 
                walker_type == _SlaterType.FULLYPOLARIZED
            ):
                if walker_type == _SlaterType.NONCOLLINEAR:
                    nelec = (nalpha+nbeta,0)
                write_nomsd_ghf(
                    fh5=wfn_group,
                    wfn=wfn,
                    nelec=nelec,
                    init=init
                )
        
        if coeffs.dtype == float:
            if verbose:
                print(" # Found real MSD coefficients. Converting to complex.")
            coeffs = np.array(coeffs, dtype=np.complex128)
        wfn_group['ci_coeffs'] = to_complex(coeffs)
        
        if walker_type == _SlaterType.NONCOLLINEAR:
            dims = [norb, nalpha+nbeta, 0, _slater2dims(walker_type), len(coeffs)]
        else:
            dims = [norb, nalpha, nbeta, _slater2dims(walker_type), len(coeffs)]


        wfn_group['dims'] = np.array(dims, dtype=np.int32)


def write_nomsd_old(fh5, wfn, uhf, nelec, thresh=1e-8, init=None):
    """Write NOMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    wfn : :class:`np.ndarray`
        NOMSD trial wavefunctions.
    uhf : bool
        UHF style wavefunction.
    nelec : tuple
        Number of alpha and beta electrons.
    thresh : float
        Threshold for writing wavefunction elements.
    """
    nalpha, nbeta = nelec

    wfn[abs(wfn) < thresh] = 0.0
    if init is not None:
        add_dataset(fh5, 'Psi0_alpha', to_complex(init[0]))
        add_dataset(fh5, 'Psi0_beta', to_complex(init[1]))
    else:
        # TODO: add leading dimension to wfn!! (outside of this!)
        add_dataset(fh5, 'Psi0_alpha',
                    to_complex(wfn[0,:,:nalpha].copy()))  
        if uhf:
            warn(
                "Using UHF Slater determinant for initial Walkers with a Collinear Trial Wavefunction. " 
                "This can lead to very slow equilibration in AFQMC calculations. " 
                "using ROHF Slater determinants for intiail Walkers is recommended."
            )
            add_dataset(
                fh5, 'Psi0_beta',
                to_complex(
                    wfn[0,:,nalpha:nalpha+nbeta].copy()
                )
            )
    
    for idet, w in enumerate(wfn):
        ix = 2*idet if uhf else idet
        psia = scipy.sparse.csr_array(w[:,:nalpha].conj().T)
        write_nomsd_single(fh5, psia, ix)
        if uhf and nbeta > 0:
            ix = 2*idet + 1
            psib = scipy.sparse.csr_array(w[:,nalpha:nalpha+nbeta].conj().T)
            write_nomsd_single(fh5, psib, ix)


def write_nomsd_ghf(fh5, wfn, nelec, thresh=1e-8, init=None):
    """Write 'GHF'/'G. S. DET' NOMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    wfn : :class:`np.ndarray`
        NOMSD trial wavefunctions.
    nelec : tuple
        Number of alpha and beta electrons.
    thresh : float
        Threshold for writing wavefunction elements.
    """
    nalpha, nbeta = nelec

    if nbeta != 0:
        raise ValueError(
            "nbeta is non-zero while attempting to write NOMSD for either"
            " NONCOLLINER or FULLYPOLARIZED Slater determinants."
        )

    wfn[abs(wfn) < thresh] = 0.0
    
    wfn = np.array(wfn,dtype=np.complex128) 

    if init is not None:
        add_dataset(fh5, 'Psi0_alpha', to_complex(init[0]))
    else:
        add_dataset(fh5, 'Psi0_alpha',to_complex(wfn[0,:,:nalpha]))
    
    for idet, w in enumerate(wfn):
        psia = scipy.sparse.csr_array(w[:,:nalpha].conj().T)
        write_nomsd_single(fh5, psia, idet)

write_nomsd = write_nomsd_old

def write_nomsd_single(fh5, psi, idet):
    """Write single component of NOMSD to hdf.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    psi : :class:`scipy.sparse.csr_array`
        Sparse representation of trial wavefunction.
    idet : int
        Determinant number.
    """
    base = 'PsiT_{:d}/'.format(idet)
    dims = [psi.shape[0], psi.shape[1], psi.nnz]
    fh5[base+'dims'] = np.array(dims, dtype=np.int32)
    fh5[base+'data_'] = to_complex(psi.data)
    fh5[base+'jdata_'] = psi.indices
    fh5[base+'pointers_begin_'] = psi.indptr[:-1]
    fh5[base+'pointers_end_'] = psi.indptr[1:]

def write_phmsd(fh5, occa, occb, nelec, norb, init=None, orbmat=None):
    """Write NOMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    nelec : tuple
        Number of alpha and beta electrons.
    """
    # TODO: Update if we ever wanted "mixed" phmsd type wavefunctions.
    na, nb = nelec
    if init  is None:
        init = [
            np.eye(norb,dtype=np.complex128),
            np.eye(norb,dtype=np.complex128)
            ]

    add_dataset(fh5, 'Psi0_alpha', to_complex(init[0][:,occa[0]].copy()))
    if nb > 0:
        add_dataset(fh5, 'Psi0_beta', to_complex(init[1][:,occb[0]].copy()))
    if orbmat is not None:
        fh5['type'] = 1
        # Expects conjugate transpose.
        oa = scipy.sparse.csr_array(orbmat[0].conj().T)
        write_nomsd_single(fh5, oa, 0)
        if nb > 0:
            ob = scipy.sparse.csr_array(orbmat[1].conj().T)
            write_nomsd_single(fh5, ob, 1)
    else:
        fh5['type'] = 0
    occs = np.zeros((len(occa), na+nb), dtype=np.int32)
    occs[:,:na] = np.array(occa)
    if nb > 0:
        occs[:,na:] = norb+np.array(occb)
    fh5['occs'] = occs.ravel()
