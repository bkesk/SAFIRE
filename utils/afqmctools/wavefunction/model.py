# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from afqmctools.wavefunction.mol import write_wfn
from afqmctools.utils.io import read_one_body,get_hamiltonian_spin_symm,read_nmo
from afqmctools.hamiltonian.model.ham_class import SpinSymm,get_spin_symm_enum


# TODO: remove once the new "free_electron" submodule is integrated
def make_free_elec(hamiltonian_fname,nelec,spin_symm=None):
    """
    Make a free-electron wavefunction with `nelec` = (Nup,Ndown) electrons based on the one-body Hamiltonian in `hamiltonian_fame`.

    Parameters
    ----------
    hamiltonian_fname : str
        Path to the Hamiltonian file.
    nelec : tuple
        Number of electrons. (Nup,Ndown)
    spin_symm : str | SpinSymm
        Spin symmetry of the wavefunction. If None, the spin symmetry is inferred from the Hamiltonian file.
    
    Returns
    -------
    wfn : Wavefunction
        Free-electron wavefunction.
    spin_symm : SpinSymm
        Spin symmetry of the wavefunction.
    """    

    from afqmctools.wavefunction.free_electron import _collinear_free_elec,_noncollinear_free_elec 

    Nmo = read_nmo(fname=hamiltonian_fname)
    Hfree = read_one_body(fname=hamiltonian_fname)

    if spin_symm is None:
        spin_symm = get_hamiltonian_spin_symm(
            fname=hamiltonian_fname
        )
    else:
        spin_symm = get_spin_symm_enum(spin_symm)
    
    if spin_symm == SpinSymm.CLOSED:
        raise NotImplementedError("Free electron wavefunction with closed spin symmetry is not implemented")
    elif spin_symm == SpinSymm.COLLINEAR:
        wfn = _collinear_free_elec(Hfree,nelec=nelec,Nmo=Nmo)
        return wfn, spin_symm
    elif spin_symm == SpinSymm.NONCOLLINEAR:
        wfn = _noncollinear_free_elec(Hfree,nelec=(sum(nelec),0),Nmo=Nmo)
        return wfn, spin_symm
    else:
        raise ValueError("Can't make free electron wavefunction. Unknown spin_symm")


def write_free_electron_wfn(hamiltonian_fname,nelec,output_fname=None,spin_symm=None):
    """
    Write a free-electron wavefunction with `nelec` = (Nup,Ndown) electrons based on the one-body Hamiltonian in `hamiltonian_fame`.
    Optionally save the wavefunction to a different file (`output_fname`)

    Parameters
    ----------
    hamiltonian_fname : str
        Path to the Hamiltonian file.
    nelec : tuple
        Number of electrons. (Nup,Ndown)
    output_fname : str
        Path to the output wavefunction file.
    spin_symm : str | SpinSymm
        Spin symmetry of the wavefunction. If None, the spin symmetry is inferred from the Hamiltonian file.
    """
    print("Generating a free-electron trial wavefunction")
    if output_fname is None:
        output_fname = hamiltonian_fname
    wfn,spin_symm = make_free_elec(hamiltonian_fname,nelec=nelec,spin_symm=spin_symm)
    
    write_wfn(
        filename=output_fname,
        wfn=wfn,
        walker_type=spin_symm,
        nelec=nelec,
        norb=read_nmo(fname=hamiltonian_fname)
    )
