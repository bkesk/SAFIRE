# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import pytest
import h5py
import numpy as np

#skip this file if pyscf can't import
pyscf = pytest.importorskip("pyscf")

from afqmctools.inputs.from_pyscf import pyscf_to_afqmc
from afqmctools.inputs.energy import calculate_hf_energy
from afqmctools.utils.linalg import get_ortho_ao_mol


@pytest.mark.pyscf
class TestEnergy:


    def test_from_pyscf(self,tmp_path,neon_rhf):
        
        mf,energy = neon_rhf

        assert np.isclose(energy, -126.60452499805)
        
        pyscf_to_afqmc(
            mf.chkfile,
            tmp_path / 'afqmc.h5',
            1e-8, 
            wfn_file= tmp_path / 'afqmc.h5',
            real_chol=True
            )
        
        etot,_,_ = calculate_hf_energy(
            hamil_file=tmp_path/'afqmc.h5',
            wfn_file=tmp_path/'afqmc.h5'
            )
        assert np.isclose(etot, -126.60452499805)


    @pytest.mark.slow
    def test_from_pyscf_lindep(self,tmp_path,neon_rhf_tz):
        from pyscf.scf.addons import remove_linear_dep_
        
        mf,_ = neon_rhf_tz
        atom = mf.mol
        remove_linear_dep_(mf, 0.1, 0.1)        
        energy = mf.kernel()
       
        X = get_ortho_ao_mol(atom.intor('int1e_ovlp_sph'), 0.1)
        with h5py.File(
            mf.chkfile, 
            #tmp_path/'scf.chk', 
            'r+') as fh5:
            fh5['scf/orthoAORot'] = X

        assert np.isclose(energy,  -128.11089429105502)
        
        pyscf_to_afqmc(
            mf.chkfile, 
            tmp_path/'afqmc.h5', 
            1e-8, 
            wfn_file=tmp_path/'afqmc.h5',
            real_chol=True, 
            ortho_ao=True
            )
        
        etot,_,_ = calculate_hf_energy(
            hamil_file=tmp_path/'afqmc.h5',
            wfn_file=tmp_path/'afqmc.h5'
            )
        assert np.isclose(etot, -128.11089429105502)

