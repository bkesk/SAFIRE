# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

'''
Questions:
1. are tests guarunteed to run in the order of declaration?
2. is it possible to use fixtures within mark.parametrize?

'''

import pytest
import numpy as np

#skip this file if pyscf can't import
pyscf = pytest.importorskip("pyscf")

from afqmctools.inputs.from_pyscf import pyscf_to_afqmc,calculate_hf_energy
from afqmctools.hamiltonian.converter import read_hamiltonian
from afqmctools.wavefunction.converter import read_wavefunction


class TestOxygen:
    """
    Workflow tests based on an Oxygen atom
    """
    @pytest.fixture(scope="session")
    def run_file(self,tmp_path_factory):
        return tmp_path_factory.mktemp("test_write_oxygen") / f"afqmc.h5"


    def test_write(self,run_file,oxygen_uhf):
        mf,_ = oxygen_uhf

        pyscf_to_afqmc(
            chkfile=mf.chkfile,
            hamil_file=run_file,
            threshold=1e-4,
            ortho_ao=True,
            real_chol=True
            )
        assert True

    
    def test_read_hamiltonian(self,run_file):
        hamil = read_hamiltonian(run_file)
        assert hamil is not None


    def test_read_wavefunction(self,run_file):
        wfn = read_wavefunction(run_file)
        assert wfn is not None


    def test_energy(self,run_file,oxygen_uhf):
        _,Euhf = oxygen_uhf

        etot,_,_ = calculate_hf_energy(
            hamil_file=run_file,
            wfn_file=run_file
            )

        assert np.isclose(etot,Euhf)
