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
Workflow test: Neon atom

Test Goals:
1. cover reading/writing PHMSD wavefunctions


"""
import pytest
import numpy as np
import h5py as h5

#skip this file if pyscf can't import
pyscf = pytest.importorskip("pyscf")

from afqmctools.inputs.from_pyscf import pyscf_to_afqmc,calculate_hf_energy
from afqmctools.hamiltonian.converter import read_hamiltonian
from afqmctools.wavefunction.converter import read_wavefunction
from afqmctools.wavefunction.mol import write_cas_wfn


class TestNeon:
    '''
    Workflow tests based on Diamond
    
    Attributes:
    - `test_files` : a dictionary for holding the 
                       the path where temporary 
                       test files are located. The
                       intention is to allow writing
                       and reading to occur in separate 
                       tests.

    Currently only tests that everythings runs without errors,
      we want to varify that we read the same stuff that we tried 
      to write.
    '''


    @pytest.fixture(scope="session")
    def run_dir(self,tmp_path_factory):
        return tmp_path_factory.mktemp(f"test_write_neon")


    @pytest.fixture
    def run_file(self,run_dir):
        return run_dir / f"afqmc.h5"


    def test_write(self,run_file,neon_casscf):
        mf = neon_casscf

        pyscf_to_afqmc(
            chkfile=mf.chkfile,
            hamil_file=run_file,
            threshold=1e-4,
            ortho_ao=True,
            real_chol=True,
            verbose=True
            )
        assert True


    def test_read_hamiltonian(self,run_file):
        hamil = read_hamiltonian(run_file)
        assert hamil is not None


    def test_read_wavefunction(self,run_file):
        wfn = read_wavefunction(run_file)
        assert wfn is not None


    def test_write_phmsd(self,run_dir,neon_casscf):
        mc = neon_casscf

        write_cas_wfn(
            mol=mc.mol,
            cas_chkfile=mc.chkfile,
            tol_trunc=1.0E-4,
            outname=run_dir/'afqmc_cas.h5'
        )


    def test_read_phmsd(self,run_dir):
        wfn,psi0,(na,nb) = read_wavefunction(run_dir/'afqmc_cas.h5')

        print("wfn is: ", wfn)
        print("Psi0 is: ", psi0)
        print("(na,nb) is: ", (na,nb))

        assert wfn is not None
        assert psi0 is not None
        assert na is not None
        assert nb is not None


    @pytest.mark.xfail(reason="calculate_hf_energy() assumes a NOMSD wavefunction")
    def test_read_phmsd(self,run_dir,neon_casscf,run_file):
        with h5.File(neon_casscf.chkfile,'r') as f:
            energy_ref = f['/mcscf/e_tot'][...]

        etot,_,_ = calculate_hf_energy(
            hamil_file=run_file,
            wfn_file=run_dir/'afqmc_cas.h5'
            )

        assert energy_ref == etot

