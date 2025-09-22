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
from afqmctools.wavefunction.mol import write_wfn,slater_gto2mo


def main():

    # inputs
    chk = '../scf/rohf_single.chk'
    chol_tol = 1e-5

    cas_afqmc = (3,32)

    # output
    fout = 'afqmc.h5'

    #####################################
    #                                   #
    #  Write Hamiltonian in ROHF basis  #
    #                                   #
    #####################################

    scf_data = load_from_pyscf_chk_mol(chk, 'scf')

    mol = scf_data['mol']
    nelec = mol.nelec
    na,nb = nelec
    overlap = mol.intor('int1e_ovlp')
    basis = scf_data['mo_coeff']
    nmo = basis.shape[-1]

    ncore = nelec[0] - cas_afqmc[0]
    assert ncore >= 0

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
        walker_type='fullypolarized',
        cas=cas_afqmc
    )
    
    #####################################
    #                                   #
    #      Write Trial Wavefunction     #
    #                                   #
    #####################################

    nmo_active = cas_afqmc[1]
    nelec_total_active = cas_afqmc[0]
    
    phi_active = np.eye(nmo_active,nelec_total_active)
    nelec_active = (nelec_total_active,0)

    ci = np.array([1.0], dtype=np.complex128)    
    wfn = (ci,np.array([phi_active]))

    write_wfn(
       filename=fout,
       wfn=wfn,
       walker_type='fully_polarized',
       nelec=nelec_active,
       norb=nmo_active
       )

if __name__ == '__main__':
    main()
