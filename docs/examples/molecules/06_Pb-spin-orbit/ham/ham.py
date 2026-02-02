# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import h5py as h5
import numpy as np

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from afqmctools.hamiltonian.mol import write_hamil_mol
from afqmctools.wavefunction.mol import write_wfn


def main():
    # inputs
    basis_chk = '../scf/rohf.chk'
    ghf_chkfile = '../scf/ghf.chk'
    ghf_soc_chkfile = '../scf/ghf_soc.chk'
    chol_tol = 1e-5
    
    npol = 2

    # output
    fout = 'afqmc_sf.h5'
    fout_soc = 'afqmc_soc.h5'

    #####################################
    #                                   #
    #  Write Hamiltonian in ROHF basis  #
    #                                   #
    #####################################
    scf_data = load_from_pyscf_chk_mol(basis_chk, 'scf')
    mol = scf_data['mol']
    nelec = mol.nelec
    na,nb = nelec
    overlap = mol.intor('int1e_ovlp')
    basis = scf_data['mo_coeff']
    nmo = basis.shape[-1]

    transform_marix = basis.conj().T @ overlap
    def to_orbital_basis(orbitals):
        return transform_marix @ orbitals

    print("="*20 + " scf_info " + "="*20)
    for key,value in scf_data.items():
        print(f"{key}  :  {value}")

    write_hamil_mol(
        scf_data,
        fout, 
        chol_tol, 
        dense=True, 
        real_chol=True, 
        verbose=True,
        walker_type='ghf'
    )

    write_hamil_mol(
       scf_data, 
       fout_soc, 
       chol_tol, 
       dense=True, 
       real_chol=True, 
       verbose=True,
       walker_type='ghf',
       with_soc=True
    )

    #####################################
    #                                   #
    # Write Spin-Free GHF Wavefunction  #
    #                                   #
    #####################################

    # TODO: Make a function to do this
    with h5.File(ghf_chkfile,'r') as f:
        mo_coeff = f['/scf/mo_coeff'][...]

    # convert to |ROHF> x |sigma> basis:
    phi = np.zeros(
        shape=(npol*nmo,sum(nelec)),
        dtype=np.complex128
    )
    phi[:nmo,:na+nb] = to_orbital_basis(mo_coeff[:nmo,:na+nb])
    phi[nmo:,:na+nb] = to_orbital_basis(mo_coeff[nmo:,:na+nb])
    
    ci = np.array([1.0], dtype=np.complex128)    
    wfn = (ci,np.array([phi]))

    write_wfn(
       filename=fout,
       wfn=wfn,
       walker_type='ghf',
       nelec=(sum(nelec),0),
       norb=nmo
       )
    
    #####################################
    #                                   #
    #    Write GHF Wavefunction         #
    #                                   #
    #####################################
    
    # TODO: make this into a function in the tooling
    with h5.File(ghf_soc_chkfile,'r') as f:
        mo_coeff = f['/scf/mo_coeff'][...]

    # convert to |ROHF> x |sigma> basis:
    phi = np.zeros(
        shape=(npol*nmo,sum(nelec)),
        dtype=np.complex128
    )

    phi[:nmo,:na+nb] = to_orbital_basis(mo_coeff[:nmo,:na+nb])
    phi[nmo:,:na+nb] = to_orbital_basis(mo_coeff[nmo:,:na+nb])

    # generate fake CI coeff:
    ci = np.array([1.0], dtype=np.complex128)    
    wfn = (ci,np.array([phi]))

    write_wfn(
       filename=fout_soc,
       wfn=wfn,
       walker_type='ghf',
       nelec=(sum(nelec),0),
       norb=nmo
       )


if __name__ == '__main__':
    main()
