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
Free-Electron Trial Wavefunction builder

Author: Kyle Eskridge
GitHub: bkesk
"""
from warnings import warn
from pathlib import Path

import numpy as np
import h5py as h5
import scipy.sparse.linalg as spsl
import scipy.linalg as spl
import scipy.sparse as sps

from afqmctools.hamiltonian.model.director import HamiltonianDirector
from afqmctools.hamiltonian.model.ham_class import Hamiltonian,SpinSymm,get_spin_symm_enum
from afqmctools.systems.lattice import Lattice
from afqmctools.hamiltonian.converter import read_hamiltonian
from afqmctools.systems.lattice import get_lattice
from afqmctools.utils.io import read_input_params

from autohf import lattice_hf, AutoHFHamiltonian


# TODO: Validate a "good" default small irrational twist angle
# For Free Electron Trial wavefunctions - small *irrational* twist angle
THETA_X = 1/np.sqrt(592560607) # 592560607 is prime
THETA_Y = 1/np.sqrt(47603)     # 47603 is prime


def free_electron(source,nelec,twist=None,spin_symm=None,use_dense=True,lattice=None,return_autohf=False,measure_spin=True):
    """
    Builds a free-electron trial wavefunction based on the 'source' lattice hamiltonian.

    Parameters
    ----------
    source : str | dict | Hamiltonian
        str - the name of an input file with lattice and hamiltonian blocks defined
        dict - a dict containing lattice and hamiltonian blocks
    """
    if twist is None:
        twist = 0.1*np.array((THETA_X,THETA_Y))
    else:
        twist = np.array(twist)

    print(f"Generating free-electron trial wavefunction with twist = {twist}")

    if isinstance(source,str|Path|dict):
        # convert Path to str
        if isinstance(source,Path):
            source = str(source)

        if isinstance(source,str) and h5.is_hdf5(source):
            hamiltonian = read_hamiltonian(source)
        
        elif isinstance(source,str|dict):
            if isinstance(source,str) and source.endswith(".toml"):
                source = read_input_params(source)

            lattice_params = source.get("lattice",dict())
            lattice_params["twist"] = twist
            lattice = get_lattice(params=lattice_params)

            hamiltonian_director = HamiltonianDirector(
                source=source,
                lattice=lattice
                )
            hamiltonian = hamiltonian_director.build()
    elif isinstance(source,Hamiltonian):

        hamiltonian = source

        if np.any(twist != hamiltonian.twist):
            warn(
                "Twist angle of the Hamiltonian does not match the twist angle "
                "provided to the free electron wavefunction builder. "
                "Using the twist angle from the Hamiltonian."
            )

    else:
        raise ValueError("source must be a str|Path, dict, or Hamiltonian instance")

    if spin_symm is None:
        spin_symm = hamiltonian.spin_symm

    wfn,spin_symm_wfn = _free_electron(
        hamiltonian=hamiltonian,
        nelec=nelec,
        spin_symm=spin_symm,
        use_dense=use_dense
    )

    # TODO: put this somewhere else, probably parse from an input file
    settings = dict(
        ansatz = 'SD',
        numSteps = -1, # prints only the reference energy
        verbose = True,
        nelec = nelec,
        numTrials = 1,
        measure_spin = measure_spin
    )

    if spin_symm_wfn == SpinSymm.NONCOLLINEAR:
        settings["noncollinear"] = True

    results = lattice_hf(
        AutoHFHamiltonian(source=hamiltonian),
        settings=settings,
        initial_guess=wfn[1][0], # TODO: make a wfn container class so that we don't need to have to take awkward indices like this
        suppress_logo = True,
    )

    if return_autohf:
        return wfn,spin_symm,results
    else:
        return wfn,spin_symm_wfn
    

def _free_electron(hamiltonian:Hamiltonian,nelec,spin_symm=None,use_dense=True):
    """
    Builds a free-electron trial wavefunction for the given Hamiltonian

    Parameters
    ----------        
    hamiltonian : Hamiltonian
        a Lattice Model Hamiltonian instance to build the Hamiltonian for.
    nelec : tuple(int,int)
        an iterable with the number of electrons per spin to use.

    Returns
    -------
    wfn : list
        a list containing the coefficients and Slater determinant arrays of a Wavefunction
    spin_symm : SpinSymm
        the spin symmetry of the wave
    """

    norb = hamiltonian.nbands * hamiltonian.nsites
    T = hamiltonian.get_one_body()

    if spin_symm is None:
        spin_symm = hamiltonian.spin_symm
    else:
        spin_symm = get_spin_symm_enum(spin_symm)

    if spin_symm == SpinSymm.CLOSED:
        raise NotImplementedError("Free electron wavefunction with closed spin symmmetry is not implemented")
    elif spin_symm == SpinSymm.COLLINEAR:
        wfn = _collinear_free_elec(T,nelec=nelec,Nmo=norb,use_dense=use_dense)
        return wfn, spin_symm
    elif spin_symm == SpinSymm.NONCOLLINEAR:
        wfn = _noncollinear_free_elec(T,nelec=nelec,Nmo=norb,use_dense=use_dense)
        return wfn, spin_symm
    else:
        raise ValueError("Can't make free electron wavefunction. Uknown spin_symm")


def _get_1body_eig(H1body,num_eigenvals,use_dense=True):
    """
    Get the eigenvalues and vectors of the given one-boyd Hamiltonian.

    This function will attempt to use the dense representation of the
      1-body Hamiltonian by default, but will use the sparse representation if requested and if possible. The sparse
      rep. is not recommended. It uses spipy.sparse.linalg.eigh which
      is known to not properly handle complex-valued matrices.
    
    Parameters
    ----------
    H1body : scipy.sparse.csr_array
        a sparse array representing the 1-body Hamiltonian
    num_eigenvals : int
        the number of eigenvalues to find
    use_dense : bool
        whether to use the dense representation of the 1-body Hamiltonian

    Returns
    -------
    sorted_eigenvals : np.ndarray
        the sorted eigenvalues of the 1-body Hamiltonian
    sorted_orbitals : np.ndarray
        the sorted eigenvectors of the 1-body Hamiltonian
    """

    N = H1body.shape[0]
    if num_eigenvals > N:
        print("Requested more eigenvalues than N for NxN array: finding N eigenvalues instead")
        num_eigenvals = N

    if use_dense or num_eigenvals == N:
        print("using dense representation of 1-body Hamiltonian to find eigenvalues and orbitals")
        sorted_eigenvals,sorted_orbitals = spl.eigh(
            a=H1body.toarray()
        )
    else:
        print("using sparse representation of 1-body Hamiltonian to find eigenvalues and orbitals")
        # only valid for num_eigenvales <= N-1 
        warn("Using sparse representation of 1-body Hamiltonian. This may not work for complex-valued matrices.")
        eigenvalues,orbitals = spsl.eigsh(
            A=H1body,
            k=num_eigenvals,
            which='SA' # smallest algebraic eigenvalues
            )
        # sort by eignvalue
        sorted_eigenvals,sorted_orbitals = zip(*sorted(zip(eigenvalues,orbitals.T),key=lambda x: x[0]))
        sorted_orbitals = np.array(sorted_orbitals).T

    return sorted_eigenvals,sorted_orbitals



def _collinear_free_elec(Hfree,nelec,Nmo,use_dense=True):
    """
    Given a 1-body Hamiltonian, generates a 'free-electron' wavefunction in
       a format that can be directly passed to `write_wfn()`.

    Parameters
    ----------
    Hfree : np.ndarray
        a 1-body Hamiltonian
    nelec : tuple(int,int)
        the number of electrons per spin
    Nmo : int
        the number of basis set functions
    use_dense : bool
        whether to use the dense representation of the 1-body Hamiltonian
    
    Returns
    -------
    wfn : list
        a list containing the coefficients and Slater determinant arrays of a Wavefunction
    """
    num_eigenvals = max(nelec)
    Nup,Ndown = nelec
    Hfree_sigma = (Hfree[:Nmo,:],Hfree[Nmo:,:])

    eigenvalues_sigma,orbitals_sigma = [],[]

    for Hfree in Hfree_sigma:
        eigenvals,orbitals = _get_1body_eig(Hfree,num_eigenvals,use_dense=use_dense)
        print("Eigenvalues of the non-interacting Hamiltonian: ", eigenvals)
        eigenvalues_sigma.append(eigenvals)
        orbitals_sigma.append(orbitals)

    wfn = np.zeros((1,Nmo,sum(nelec)), dtype=np.complex128)
    coeffs = np.array([1.0+0j])
    wfn[0,:,:Nup] = orbitals_sigma[0][:,:Nup]
    wfn[0,:,Nup:] = orbitals_sigma[1][:,:Ndown]

    return [coeffs,wfn]

def _noncollinear_free_elec(Hfree,nelec,Nmo,use_dense=True):
    """
    Given a 1-body Hamiltonian, generates a 'free-electron' wavefunction in
         a format that can be directly passed to `write_wfn()`.

    Parameters
    ----------
    Hfree : np.ndarray
        a 1-body Hamiltonian
    nelec : tuple(int,int)
        the number of electrons per spin
    Nmo : int
        the number of basis set functions
    use_dense : bool
        whether to use the dense representation of the 1-body Hamiltonian

    Returns
    -------
    wfn : list
        a list containing the coefficients and Slater determinant arrays of a Wavefunction
    """
    Nup,Ndown = nelec
    num_eigenvals = (Nup+Ndown)
    nelec = (Nup+Ndown,0)
    # assumes that the input is collinear
    if Hfree.shape == (2*Nmo,Nmo):
        Hfree = sps.block_diag([Hfree[:Nmo,:],Hfree[Nmo:,:]],format="csr")
    elif Hfree.shape == (Nmo,Nmo):
        Hfree = sps.block_diag([Hfree,Hfree],format="csr")
    elif Hfree.shape != (2*Nmo,2*Nmo):
        raise ValueError(
            "Hfree has an invalid shape. "
            "shape must be any of (Nmo,Nmo) - Closed H, (2*Nmo,Nmo) - Collinear H "
            " - (2*Nmo,2*Nmo) Noncollinear H"
        )

    eigenvals,orbitals = _get_1body_eig(Hfree,num_eigenvals,use_dense=use_dense)
    
    print("Eigenvalues of the non-interacting Hamiltonian: ", eigenvals)

    npol = 2
    wfn = np.zeros((1,npol*Nmo,sum(nelec)), dtype=np.complex128)
    coeffs = np.array([1.0+0j])
    wfn[0,:,:] = orbitals[:,:Nup+Ndown]

    return [coeffs,wfn]

