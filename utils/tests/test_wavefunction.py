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
import scipy.sparse as sps

from afqmctools.wavefunction import mol,pbc
from afqmctools.utils.io import from_complex
from afqmctools.wavefunction.free_electron import (
    _group_eigenvalues_by_shell,
    _fill_shells,
    _collinear_free_elec
)
from afqmctools.utils.slater_types import _SlaterType
from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol

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

    @pytest.mark.parametrize("ortho_ao,cas", [
        (False, None),
        (True, None),
        (False, (8, 4)),
        (True, (8, 4)),
    ])
    def test_make_slater_old_matches_generate_wavefunction(self, neon_hf, ortho_ao, cas):
        mf, _ = neon_hf
        scf_data = load_from_pyscf_chk_mol(mf.chkfile)
        C = scf_data["mo_coeff"]
        norb = scf_data["norb"]
        needs_ortho_ao = C.ndim > 2 or C.shape[0] == 2 * norb

        if ortho_ao:
            scf_data = {**scf_data, "orthAO": True}

        if ortho_ao and cas is not None:
            with pytest.raises(ValueError):
                mol.generate_wavefunction(scf_data, scf_data, ortho_ao=True, cas=cas)
            return

        if needs_ortho_ao and not ortho_ao:
            with pytest.raises(ValueError):
                mol.generate_wavefunction(scf_data, scf_data, ortho_ao=False, cas=cas)
            return

        old = mol.make_slater_old(scf_data, cas=cas)
        new, _, _ = mol.generate_wavefunction(scf_data, scf_data, ortho_ao=ortho_ao, cas=cas)

        assert old.shape == new.shape
        assert np.allclose(old, new, atol=1e-10)


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


class TestShellFilling:
    """Tests for shell-based orbital filling in free electron wavefunctions"""

    def test_shell_grouping_basic(self):
        """Test that eigenvalues are correctly grouped into shells"""
        # Create eigenvalues with clear shell structure
        eigenvalues = np.array([
            -2.0, -2.0,           # Shell 0: doubly degenerate
            -1.0,                 # Shell 1: non-degenerate
            0.0, 0.0, 0.0,        # Shell 2: triply degenerate
            1.0, 1.0              # Shell 3: doubly degenerate
        ])
        
        # Create dummy orbitals (identity matrix)
        norb = len(eigenvalues)
        orbitals = np.eye(norb, dtype=complex)
        
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        assert len(shells) == 4, f"Expected 4 shells, got {len(shells)}"
        assert shells[0]['degeneracy'] == 2, "Shell 0 should have degeneracy 2"
        assert shells[1]['degeneracy'] == 1, "Shell 1 should have degeneracy 1"
        assert shells[2]['degeneracy'] == 3, "Shell 2 should have degeneracy 3"
        assert shells[3]['degeneracy'] == 2, "Shell 3 should have degeneracy 2"
        
        # Check indices are correct
        assert shells[0]['indices'] == [0, 1]
        assert shells[1]['indices'] == [2]
        assert shells[2]['indices'] == [3, 4, 5]
        assert shells[3]['indices'] == [6, 7]

    def test_aufbau_filling(self):
        """Test Aufbau filling strategy"""
        eigenvalues = np.array([-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        orbitals = np.eye(len(eigenvalues), dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        n_electrons = 5
        occupied_orbitals, occupied_indices = _fill_shells(
            shells, n_electrons, strategy='aufbau', verbose=False
        )
        
        expected_indices = [0, 1, 2, 3, 4]  # First 5 orbitals
        assert occupied_indices == expected_indices, \
            f"Expected indices {expected_indices}, got {occupied_indices}"
        assert occupied_orbitals.shape == (8, 5), \
            f"Expected shape (8, 5), got {occupied_orbitals.shape}"

    def test_balanced_filling(self):
        """Test balanced filling strategy for partially filled shell"""
        eigenvalues = np.array([-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        orbitals = np.eye(len(eigenvalues), dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        # Fill to partially occupy the triply degenerate shell
        n_electrons = 5
        occupied_orbitals, occupied_indices = _fill_shells(
            shells, n_electrons, strategy='balanced', verbose=False
        )
        
        # The first 3 electrons fill shells 0 and 1 completely
        # The last 2 electrons should be evenly spaced in shell 2
        assert len(occupied_indices) == 5, f"Expected 5 occupied orbitals"
        assert occupied_orbitals.shape == (8, 5)

    def test_all_degenerate_shell(self):
        """Test with all orbitals in one degenerate shell"""
        eigenvalues = np.array([0.0, 0.0, 0.0, 0.0])
        orbitals = np.eye(4, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        assert len(shells) == 1 and shells[0]['degeneracy'] == 4
        
        # Fill partially with balanced strategy
        occ_orb, occ_idx = _fill_shells(shells, 2, strategy='balanced', verbose=False)
        assert len(occ_idx) == 2
        assert occ_orb.shape == (4, 2)

    def test_no_degeneracy(self):
        """Test with no degenerate orbitals"""
        eigenvalues = np.array([-2.0, -1.0, 0.0, 1.0])
        orbitals = np.eye(4, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        assert len(shells) == 4 and all(s['degeneracy'] == 1 for s in shells)
        
        occ_orb, occ_idx = _fill_shells(shells, 3, strategy='aufbau', verbose=False)
        assert occ_idx == [0, 1, 2]

    def test_near_degeneracy_1e8(self):
        """Test eigenvalue differences ~1e-8 with tol=1e-6"""
        eigenvalues = np.array([
            -1.0,
            0.0, 0.0 + 5e-9, 0.0 + 1.2e-8,  # Shell with ~1e-8 differences
            1.0
        ])
        orbitals = np.eye(len(eigenvalues), dtype=complex)
        
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        assert len(shells) == 3, f"Expected 3 shells, got {len(shells)}"
        assert shells[1]['degeneracy'] == 3, \
            f"Middle shell should have degeneracy 3, got {shells[1]['degeneracy']}"

    def test_near_degeneracy_1e7(self):
        """Test eigenvalue differences ~1e-7 with tol=1e-6"""
        eigenvalues = np.array([0.0, 5e-8, 1e-7, 2e-7, 4e-7, 8e-7])
        orbitals = np.eye(len(eigenvalues), dtype=complex)
        
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        # All should be in one shell since consecutive differences are < tol
        assert len(shells) == 1, f"Expected 1 shell, got {len(shells)}"
        assert shells[0]['degeneracy'] == 6

    def test_mixed_degeneracies(self):
        """Test mixed - some within tolerance, some outside"""
        eigenvalues = np.array([
            -1.0, -1.0 + 1e-8,              # Shell 0: diff ~1e-8
            0.0, 0.0 + 5e-7, 0.0 + 9e-7,    # Shell 1: diffs ~1e-7
            1.0, 1.0 + 2e-6,                # Shell 2 & 3: diff > tol (should split)
            2.0                             # Shell 4: single
        ])
        orbitals = np.eye(len(eigenvalues), dtype=complex)
        
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        assert len(shells) == 5, f"Expected 5 shells, got {len(shells)}"
        assert shells[0]['degeneracy'] == 2, "Shell 0 should have degeneracy 2"
        assert shells[1]['degeneracy'] == 3, "Shell 1 should have degeneracy 3"
        assert shells[2]['degeneracy'] == 1, "Shell 2 should have degeneracy 1"
        assert shells[3]['degeneracy'] == 1, "Shell 3 should have degeneracy 1"
        assert shells[4]['degeneracy'] == 1, "Shell 4 should have degeneracy 1"

    def test_tight_tolerance(self):
        """Test with very tight tolerance (tol=1e-10)"""
        eigenvalues = np.array([0.0, 1e-11, 5e-11, 1e-10, 1e-8])
        orbitals = np.eye(len(eigenvalues), dtype=complex)
        
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-10)
        
        # First 3 within 1e-10 of first value
        # 4th value (1e-10) is NOT within tol of 0.0 (boundary case)
        assert len(shells) == 3, f"Expected 3 shells, got {len(shells)}"
        assert shells[0]['degeneracy'] == 3, "First shell should have degeneracy 3"

    def test_balanced_filling_near_degenerate(self):
        """Test balanced filling with near-degenerate shell"""
        eigenvalues = np.array([
            -1.0,
            0.0, 0.0 + 1e-8, 0.0 + 2e-8, 0.0 + 3e-8, 0.0 + 4e-8,  # 5 nearly degenerate
            1.0
        ])
        orbitals = np.eye(len(eigenvalues), dtype=complex)
        
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        assert shells[1]['degeneracy'] == 5, "Middle shell should have degeneracy 5"
        
        # Fill with balanced strategy
        n_electrons = 4  # 1 from first shell, 3 from middle, 0 from last
        occupied_orbitals, occupied_indices = _fill_shells(
            shells, n_electrons, strategy='balanced', verbose=False
        )
        
        assert len(occupied_indices) == 4, "Should have 4 occupied orbitals"
        assert occupied_indices[0] == 0, "First orbital should be from shell 0"

    def test_collinear_free_elec_integration(self):
        """Integration test with _collinear_free_elec"""
        # Create a simple tight-binding Hamiltonian with degeneracies
        Nmo = 6
        nelec = (3, 2)  # 3 up, 2 down electrons
        
        # Create eigenvalue structure with degeneracies
        H_up = np.diag([-2.0, -2.0, -1.0, 0.0, 0.0, 1.0])
        H_down = np.diag([-2.0, -2.0, -1.0, 0.0, 0.0, 1.0])
        
        # Stack into collinear format
        Hfree = sps.csr_array(np.vstack([H_up, H_down]))
        
        # Test Aufbau strategy
        wfn_aufbau = _collinear_free_elec(
            Hfree, nelec, Nmo, 
            use_dense=True, 
            filling_strategy='aufbau',
            shell_tol=1e-6
        )
        
        coeffs, wfn_matrix = wfn_aufbau
        assert wfn_matrix.shape == (1, Nmo, sum(nelec)), \
            f"Expected shape (1, {Nmo}, {sum(nelec)}), got {wfn_matrix.shape}"
        assert len(coeffs) == 1 and np.abs(coeffs[0] - 1.0) < 1e-10
        
        # Test Balanced strategy
        wfn_balanced = _collinear_free_elec(
            Hfree, nelec, Nmo, 
            use_dense=True, 
            filling_strategy='balanced',
            shell_tol=1e-6
        )
        
        coeffs_bal, wfn_matrix_bal = wfn_balanced
        assert wfn_matrix_bal.shape == (1, Nmo, sum(nelec))

    def test_zero_electrons(self):
        """Test edge case with zero electrons"""
        eigenvalues = np.array([0.0, 1.0, 2.0])
        orbitals = np.eye(3, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        occ_orb, occ_idx = _fill_shells(shells, 0, strategy='aufbau', verbose=False)
        assert occ_orb.shape == (3, 0)
        assert occ_idx == []

    def test_fill_all_orbitals(self):
        """Test filling all available orbitals"""
        eigenvalues = np.array([0.0, 1.0, 2.0])
        orbitals = np.eye(3, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        occ_orb, occ_idx = _fill_shells(shells, 3, strategy='aufbau', verbose=False)
        assert occ_orb.shape == (3, 3)
        assert occ_idx == [0, 1, 2]

    def test_insufficient_orbitals(self):
        """Test that requesting more electrons than orbitals raises an error"""
        eigenvalues = np.array([0.0, 1.0])
        orbitals = np.eye(2, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        with pytest.raises(ValueError, match="Not enough orbitals"):
            _fill_shells(shells, 5, strategy='aufbau', verbose=False)

    def test_invalid_strategy(self):
        """Test that invalid filling strategy raises an error"""
        # Use a degenerate shell that will be partially filled
        # so the strategy actually matters
        eigenvalues = np.array([0.0, 0.0, 0.0])
        orbitals = np.eye(3, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        with pytest.raises(ValueError, match="Unknown filling strategy"):
            _fill_shells(shells, 2, strategy='invalid_strategy', verbose=False)

    def test_alternating_filling_basic(self):
        """Test alternating filling strategy (from edges inward)"""
        # Create a shell with 6 degenerate orbitals
        eigenvalues = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        orbitals = np.eye(6, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        # Fill 4 electrons: should get indices [0, 5, 1, 4]
        occ_orb, occ_idx = _fill_shells(shells, 4, strategy='alternating', verbose=False)
        
        expected_indices = [0, 5, 1, 4]  # 0, -1, 1, -2
        assert occ_idx == expected_indices, f"Expected {expected_indices}, got {occ_idx}"
        assert occ_orb.shape == (6, 4)

    def test_alternating_filling_odd(self):
        """Test alternating filling with odd number of electrons"""
        eigenvalues = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        orbitals = np.eye(5, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        # Fill 3 electrons: should get indices [0, 4, 1]
        occ_orb, occ_idx = _fill_shells(shells, 3, strategy='alternating', verbose=False)
        
        expected_indices = [0, 4, 1]  # 0, -1, 1
        assert occ_idx == expected_indices, f"Expected {expected_indices}, got {occ_idx}"

    def test_alternating_filling_multishell(self):
        """Test alternating filling across multiple shells"""
        eigenvalues = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        orbitals = np.eye(7, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        # Fill 5 electrons: should fill first shell completely (aufbau),
        # then use alternating for partial fill of second shell
        occ_orb, occ_idx = _fill_shells(shells, 5, strategy='alternating', verbose=False)
        
        # First 2 from shell 0: [0, 1]
        # Next 3 from shell 1 with alternating: [2, 5, 3]
        expected_indices = [0, 1, 2, 5, 3]
        assert occ_idx == expected_indices, f"Expected {expected_indices}, got {occ_idx}"

    def test_alternating_vs_balanced(self):
        """Compare alternating and balanced strategies"""
        eigenvalues = np.array([0.0] * 10)
        orbitals = np.eye(10, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        # Alternating with 5 electrons
        occ_orb_alt, occ_idx_alt = _fill_shells(shells, 5, strategy='alternating', verbose=False)
        expected_alt = [0, 9, 1, 8, 2]  # edges inward
        assert occ_idx_alt == expected_alt, f"Alternating: expected {expected_alt}, got {occ_idx_alt}"
        
        # Balanced with 5 electrons
        occ_orb_bal, occ_idx_bal = _fill_shells(shells, 5, strategy='balanced', verbose=False)
        expected_bal = [0, 2, 4, 6, 8]  # evenly spaced
        assert occ_idx_bal == expected_bal, f"Balanced: expected {expected_bal}, got {occ_idx_bal}"
        
        # They should be different
        assert occ_idx_alt != occ_idx_bal, "Alternating and balanced should produce different results"

    def test_alternating_fill_all(self):
        """Test alternating filling when filling entire shell"""
        eigenvalues = np.array([0.0, 0.0, 0.0, 0.0])
        orbitals = np.eye(4, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        # When filling all orbitals, should use aufbau (just fill in order)
        occ_orb, occ_idx = _fill_shells(shells, 4, strategy='alternating', verbose=False)
        
        expected_indices = [0, 1, 2, 3]  # all in order when completely filled
        assert occ_idx == expected_indices

    def test_alternating_single_electron(self):
        """Test alternating filling with single electron in shell"""
        eigenvalues = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        orbitals = np.eye(5, dtype=complex)
        shells = _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6)
        
        # Fill 1 electron: should get index 0 (first from left)
        occ_orb, occ_idx = _fill_shells(shells, 1, strategy='alternating', verbose=False)
        
        assert occ_idx == [0], f"Expected [0], got {occ_idx}"


class TestNoncollinearShellFilling:
    """Tests for noncollinear shell-based orbital filling"""

    def test_noncollinear_basic(self):
        """Test basic noncollinear wavefunction generation"""
        from afqmctools.wavefunction.free_electron import _noncollinear_free_elec
        
        # Create a simple Hamiltonian
        Nmo = 4
        nelec = (2, 1)  # 2 up, 1 down -> 3 total electrons
        
        # Create a block diagonal Hamiltonian (2*Nmo x 2*Nmo)
        H_block = np.diag([-2.0, -1.0, 0.0, 1.0])
        Hfree = sps.block_diag([H_block, H_block], format="csr")
        
        # Test with aufbau strategy
        wfn = _noncollinear_free_elec(
            Hfree, nelec, Nmo,
            use_dense=True,
            filling_strategy='aufbau',
            shell_tol=1e-6
        )
        
        coeffs, wfn_matrix = wfn
        # Shape should be (1, 2*Nmo, Nup+Ndown)
        expected_shape = (1, 2*Nmo, nelec[0] + nelec[1])
        assert wfn_matrix.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {wfn_matrix.shape}"
        assert len(coeffs) == 1 and np.abs(coeffs[0] - 1.0) < 1e-10

    def test_noncollinear_alternating(self):
        """Test alternating filling strategy for noncollinear"""
        from afqmctools.wavefunction.free_electron import _noncollinear_free_elec
        
        Nmo = 5
        nelec = (3, 2)  # 5 total electrons
        
        # Create Hamiltonian with degenerate orbitals
        eigenvals = np.array([0.0] * 10)  # 10 degenerate orbitals (2*Nmo)
        H = np.diag(eigenvals)
        Hfree = sps.csr_array(H)
        
        wfn = _noncollinear_free_elec(
            Hfree, nelec, Nmo,
            use_dense=True,
            filling_strategy='alternating',
            shell_tol=1e-6
        )
        
        coeffs, wfn_matrix = wfn
        expected_shape = (1, 2*Nmo, sum(nelec))
        assert wfn_matrix.shape == expected_shape

    def test_noncollinear_balanced(self):
        """Test balanced filling strategy for noncollinear"""
        from afqmctools.wavefunction.free_electron import _noncollinear_free_elec
        
        Nmo = 6
        nelec = (4, 2)  # 6 total electrons
        
        # Create Hamiltonian
        H = np.diag([-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        Hfree = sps.csr_array(H)
        
        wfn = _noncollinear_free_elec(
            Hfree, nelec, Nmo,
            use_dense=True,
            filling_strategy='balanced',
            shell_tol=1e-6
        )
        
        coeffs, wfn_matrix = wfn
        expected_shape = (1, 2*Nmo, sum(nelec))
        assert wfn_matrix.shape == expected_shape

    def test_noncollinear_near_degenerate(self):
        """Test noncollinear with near-degenerate eigenvalues"""
        from afqmctools.wavefunction.free_electron import _noncollinear_free_elec
        
        Nmo = 4
        nelec = (2, 1)  # 3 total electrons
        
        # Create Hamiltonian with near-degenerate eigenvalues
        eigenvals = np.array([
            -1.0,
            0.0, 0.0 + 1e-8, 0.0 + 2e-8,  # nearly degenerate
            1.0, 1.0 + 1e-8, 1.0 + 2e-8, 1.0 + 3e-8
        ])
        H = np.diag(eigenvals)
        Hfree = sps.csr_array(H)
        
        wfn = _noncollinear_free_elec(
            Hfree, nelec, Nmo,
            use_dense=True,
            filling_strategy='aufbau',
            shell_tol=1e-6
        )
        
        coeffs, wfn_matrix = wfn
        expected_shape = (1, 2*Nmo, sum(nelec))
        assert wfn_matrix.shape == expected_shape

    def test_noncollinear_from_collinear_input(self):
        """Test noncollinear with collinear-shaped input"""
        from afqmctools.wavefunction.free_electron import _noncollinear_free_elec
        
        Nmo = 3
        nelec = (2, 1)
        
        # Create collinear-shaped input (2*Nmo, Nmo)
        H_up = np.diag([-1.0, 0.0, 1.0])
        H_down = np.diag([-1.0, 0.0, 1.0])
        Hfree = sps.csr_array(np.vstack([H_up, H_down]))
        
        wfn = _noncollinear_free_elec(
            Hfree, nelec, Nmo,
            use_dense=True,
            filling_strategy='aufbau',
            shell_tol=1e-6
        )
        
        coeffs, wfn_matrix = wfn
        expected_shape = (1, 2*Nmo, sum(nelec))
        assert wfn_matrix.shape == expected_shape

    def test_noncollinear_from_closed_input(self):
        """Test noncollinear with closed-shell-shaped input"""
        from afqmctools.wavefunction.free_electron import _noncollinear_free_elec
        
        Nmo = 4
        nelec = (2, 2)
        
        # Create closed-shell-shaped input (Nmo, Nmo)
        H = np.diag([-1.0, -0.5, 0.0, 0.5])
        Hfree = sps.csr_array(H)
        
        wfn = _noncollinear_free_elec(
            Hfree, nelec, Nmo,
            use_dense=True,
            filling_strategy='aufbau',
            shell_tol=1e-6
        )
        
        coeffs, wfn_matrix = wfn
        expected_shape = (1, 2*Nmo, sum(nelec))
        assert wfn_matrix.shape == expected_shape

    def test_noncollinear_all_strategies(self):
        """Test all filling strategies work for noncollinear"""
        from afqmctools.wavefunction.free_electron import _noncollinear_free_elec
        
        Nmo = 5
        nelec = (3, 2)
        
        # Create Hamiltonian with shells
        eigenvals = np.array([-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0])
        H = np.diag(eigenvals)
        Hfree = sps.csr_array(H)
        
        strategies = ['aufbau', 'balanced', 'alternating', 'hund']
        
        for strategy in strategies:
            wfn = _noncollinear_free_elec(
                Hfree, nelec, Nmo,
                use_dense=True,
                filling_strategy=strategy,
                shell_tol=1e-6
            )
            
            coeffs, wfn_matrix = wfn
            expected_shape = (1, 2*Nmo, sum(nelec))
            assert wfn_matrix.shape == expected_shape, \
                f"Strategy '{strategy}' produced wrong shape: {wfn_matrix.shape}"


