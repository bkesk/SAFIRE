# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import os

import pytest
import numpy as np
import h5py

from afqmctools.wavefunction import mol,pbc

#skip this file if pyscf can't import
pyscf = pytest.importorskip("pyscf")

@pytest.mark.pyscf
class TestMolWavefunction:


    def test_write_wfn_mol(self,neon_atom,neon_rhf,tmp_path):
        
        atom = neon_atom
        mf,_ = neon_rhf
        
        scf_data = {
            'mol': atom,
            'mo_coeff': mf.mo_coeff,
            'mo_occ': mf.mo_occ,
            'nelec': atom.nelec,
            'X': mf.mo_coeff,
            'norb' : mf.mo_coeff.shape[-1],
            'walker_type': 'closed',
            'mo_energy': mf.mo_energy
            }
        
        mol.write_wfn_mol(scf_data, tmp_path/'wfn.h5')

        with h5py.File(tmp_path/'wfn.h5', 'r') as fh5:
            dims = fh5['Wavefunction/NOMSD/dims'][:]

        assert (dims == [5,5,5,1,1]).all()


@pytest.mark.pyscf
class TestPBCWavefunction:


    def test_write_wfn_pbc(self,diamond,diamond_lda_k221,tmp_path):
        cell = diamond
        mf,kpts = diamond_lda_k221

        nk = [2,2,1]
        kpts = cell.make_kpts(nk)

        nmo_pk = np.array([C.shape[-1] for C in mf.mo_coeff])

        hcore = mf.get_hcore()
        fock = hcore + mf.get_veff()
        scf_data = {
            'cell': cell,
            'mo_coeff': mf.mo_coeff,
            'Xocc': mf.mo_occ,
            'X': mf.mo_coeff,
            'fock': fock,
            'walker_type': 'closed',
            'hcore': hcore,
            'nmo_pk': nmo_pk,
            'mo_energy': mf.mo_energy,
            'kpts': kpts
        }
        
        pbc.write_wfn_pbc(scf_data, True, tmp_path/'wfn.h5', rediag=True)
        
        with h5py.File(tmp_path/'wfn.h5', 'r') as fh5:
            dims = fh5['Wavefunction/NOMSD/dims'][:]
        
        assert (dims == [32,16,16,1,1]).all()

