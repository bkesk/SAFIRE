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
Define functional tests for BH
"""
import enum
from dataclasses import dataclass

from afqmctools.hamiltonian.model.ham_class import SpinSymm

from pathlib import Path
from warnings import warn

import pytest

# TODO: generalize and centralize if useful!
class HamiltonianClass(enum.Enum):
    """
    """
    GENERIC_DENSE = enum.auto()
    GENERIC_SPARSE = enum.auto()

# TODO: generalize and centralize if useful!
class WavefunctionClass(enum.Enum):
    """
    """
    NOMSD = enum.auto()
    PHMSD = enum.auto()

Pb_TEST_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = Pb_TEST_ROOT / "afqmc_inputs"
REF_DATA_DIR = Pb_TEST_ROOT / "afqmc_ref_runs"

@dataclass
class AFQMCHamiltonian:
    path:Path
    spin_symm:SpinSymm
    type:HamiltonianClass

@dataclass
class AFQMCWavefunction:
    path:Path
    spin_symm:SpinSymm
    type:WavefunctionClass

@dataclass
class AFQMCWalker:
    name:str
    spin_symm:SpinSymm

@dataclass
class AFQMCInputSet:
    """
    Holds metadata related to AFQMC inputs for sorting into
        'correct behavior' categories.
    """
    hamiltonian:AFQMCHamiltonian
    wavefunction:AFQMCWavefunction
    walker:AFQMCWalker
    reference:Path


# Some general filtering rules to use in the tests
def wfn_is_implemented(case):
    """
    Filters cases based on whether the wavefunction is implemented
    or not.

    TODO: Filter based on which Hamiltonian / wavefunction combinations are implemented.
    """
    if case.wavefunction.type == WavefunctionClass.PHMSD:
        # currently, only collinear PHMSD wavefunctions are implemented
        if case.wavefunction.spin_symm == SpinSymm.COLLINEAR:
            return True
        else:
            return False
    elif case.wavefunction.type == WavefunctionClass.NOMSD:
        return True
    else:
        raise ValueError(f"Unrecognized wavefunction type: {case.wavefunction.type}")

def compatible_spin_H_wfn(case:AFQMCInputSet):
    # high int value means less spin symmetry!
    if case.wavefunction.spin_symm >= case.hamiltonian.spin_symm:
        return True
    else:
        return False
    
def compatible_spin_H_walker(case:AFQMCInputSet):
    # high int value means less spin symmetry!
    if case.walker.spin_symm >= case.hamiltonian.spin_symm:
        return True
    else:
        return False
    
def compatible_spin_wfn_walker(case:AFQMCInputSet):
    # high int value means less spin symmetry!
    if case.walker.spin_symm >= case.wavefunction.spin_symm:
        return True
    else:
        return False

def get_all_rules():
    return [
        wfn_is_implemented,
        compatible_spin_H_wfn,
        compatible_spin_H_walker,
        compatible_spin_wfn_walker
    ]

def hamil_type_dictionary():
    """Generates and returns a dict of Hamiltonian
    types to test against. the dict is structured using
    a descriptive name for the Hamiltonian as a key, and 
    the value is the path to the HDF5 file (including it's name)
    in SAFIRE format that contains that wavefunction.
    """
    return {
        "dense_rhf_basis_noncollinear_sf" :  AFQMCHamiltonian(
            INPUTS_DIR/"afqmc_H_rhf_basis_noncollinear_sf.h5",
            SpinSymm.NONCOLLINEAR,
            HamiltonianClass.GENERIC_DENSE
        ),
        "dense_rhf_basis_noncollinear_soc" :  AFQMCHamiltonian(
            INPUTS_DIR/"afqmc_H_rhf_basis_noncollinear_soc.h5",
            SpinSymm.NONCOLLINEAR,
            HamiltonianClass.GENERIC_DENSE
        ),
    }

def wavefunction_type_dictionary():
    """Generates and returns a dict of wavefunction
    types to test against. the dict is structured using
    a descriptive name for the wavefunction as a key, and 
    the value is the path to the HDF5 file (including it's name)
    in SAFIRE format that contains that wavefunction.
    """
    return {
        "uhf_nomsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_uhf_nomsd.h5",
            SpinSymm.COLLINEAR,
            WavefunctionClass.NOMSD
        ),
        "ghf_sf_nomsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_ghf_sf_nomsd.h5",
            SpinSymm.NONCOLLINEAR,
            WavefunctionClass.NOMSD
        ),
        "ghf_soc_nomsd": AFQMCWavefunction(
            INPUTS_DIR/"afqmc_ghf_soc_nomsd.h5",
            SpinSymm.NONCOLLINEAR,
            WavefunctionClass.NOMSD
        )
    }

def walker_type_list():
    return [
        AFQMCWalker("CLOSED",SpinSymm.CLOSED),
        AFQMCWalker("COLLINEAR",SpinSymm.COLLINEAR),
        AFQMCWalker("NONCOLLINEAR",SpinSymm.NONCOLLINEAR),
    ]


def generate_param_list(
        hamil_type_dictionary=hamil_type_dictionary(),
        wavefunction_type_dictionary=wavefunction_type_dictionary(),
        walker_type_list=walker_type_list(),
        ref_data_dir=REF_DATA_DIR
    ):
    """
    Simple function to generate the full list of combinations of Hamiltonian types,
      wavefunction types, and walker types for use with the pytest.mark.parameterize decorator.
    """
    params = []
    for h_name in hamil_type_dictionary:
        hamil_path = ref_data_dir/ h_name
        if not hamil_path.exists:
            warn(f"requested a test against non-existant Hamiltonian directory: {hamil_path}; SKIPPING!")
            continue
        for trial_wfn_name in wavefunction_type_dictionary:
            trial_wfn_path = hamil_path / trial_wfn_name
            if not trial_wfn_path.exists:
                warn(f"requested a test against non-existant wavefunction directory: {trial_wfn_path}; SKIPPING!")
                continue
            for walker in walker_type_list:
                walker_path = trial_wfn_path / walker.name.lower()
                if not walker_path.exists:
                    warn(f"requested a test against non-existant walker type directory: {walker_path}; SKIPPING!")
                    continue
                params.append(
                    AFQMCInputSet(
                        hamiltonian=hamil_type_dictionary[h_name],
                        wavefunction=wavefunction_type_dictionary[trial_wfn_name],
                        walker=walker,
                        reference=walker_path/"results.h5"
                    )
                )
    return params


def should_run_successfully(all_cases=generate_param_list()):
    rules = get_all_rules()
    return [ case for case in all_cases if all( (rule(case) for rule in rules) ) ]

def should_exit_werror(all_cases=generate_param_list()):
    rules = get_all_rules()
    return [ case for case in all_cases if any(( not rule(case) for rule in rules )) ]

def should_run_warn(all_cases=generate_param_list()):
    """Generates list of cases where SAFIRE should run with a warning
    """
    rules = []
    return [ case for case in all_cases if any(( not rule(case) for rule in rules )) ]


def _expected_success(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case):
    """
    Implements the test for the expected success cases
    """
    print(case)
    afqmc_helper.run_afqmc(
        run_path=tmp_path,
        fname="afqmc.json",
        hamil_file=case.hamiltonian.path,
        wfn_file=case.wavefunction.path,
        walker_type=case.walker.name,
        afqmc_runmode=afqmc_runmode,
        run_bp=False
    )
    return result_checker.results_are_same(
        fname_test=tmp_path/"results.h5",
        fname_ref=case.reference,
        )

def _expected_fail(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case
    ):
    """
    Implements the test for the expected failure cases
    """
    print(case)
    afqmc_helper.run_afqmc(
        run_path=tmp_path,
        fname="afqmc.json",
        hamil_file=case.hamiltonian.path,
        wfn_file=case.wavefunction.path,
        walker_type=case.walker.name,
        afqmc_runmode=afqmc_runmode,
        run_bp=False
    )
    return result_checker.same_error(
        fname_test=tmp_path/"results.h5",
        fname_ref=case.reference,
        )

# per push tests
@pytest.mark.functional
@pytest.mark.push
@pytest.mark.parametrize("case",[
    AFQMCInputSet(
        hamiltonian=AFQMCHamiltonian(
            INPUTS_DIR/"afqmc_H_rhf_basis_noncollinear_soc.h5",
            SpinSymm.NONCOLLINEAR,
            HamiltonianClass.GENERIC_DENSE
        ),
        wavefunction=AFQMCWavefunction(
            INPUTS_DIR/"afqmc_ghf_soc_nomsd.h5",
            SpinSymm.NONCOLLINEAR,
            WavefunctionClass.NOMSD
        ),
        walker= AFQMCWalker("NONCOLLINEAR",SpinSymm.NONCOLLINEAR),
        reference=REF_DATA_DIR/"dense_rhf_basis_noncollinear_soc/ghf_soc_nomsd/noncollinear/results.h5"
    )
])
def test_success_push(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    case
    ):
    assert _expected_success(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case
    )

@pytest.mark.functional
@pytest.mark.push
@pytest.mark.parametrize("case",[
    AFQMCInputSet(
        hamiltonian=AFQMCHamiltonian(
            INPUTS_DIR/"afqmc_H_rhf_basis_noncollinear_soc.h5",
            SpinSymm.NONCOLLINEAR,
            HamiltonianClass.GENERIC_DENSE
        ),
        wavefunction=AFQMCWavefunction(
            INPUTS_DIR/"afqmc_ghf_soc_nomsd.h5",
            SpinSymm.NONCOLLINEAR,
            WavefunctionClass.NOMSD
        ),
        walker= AFQMCWalker("COLLINEAR",SpinSymm.COLLINEAR),
        reference=REF_DATA_DIR/"dense_rhf_basis_noncollinear_soc/ghf_soc_nomsd/collinear/results.h5"
    )
])
def test_fail_push(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    case
    ):
    assert _expected_fail(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case
    )

# Weekly tests
@pytest.mark.functional
@pytest.mark.weekly
@pytest.mark.parametrize("case",should_run_successfully())
def test_success_weekly(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    case
    ):
    assert _expected_success(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case
    )

@pytest.mark.functional
@pytest.mark.weekly
@pytest.mark.parametrize("case",should_exit_werror())
def test_fail_weekly(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    case
    ):
    assert _expected_fail(
        afqmc_helper,
        result_checker,
        afqmc_runmode,
        tmp_path,
        case
    )

@pytest.mark.skip
@pytest.mark.dev
def test_expected_warn(
    afqmc_helper,
    result_checker,
    afqmc_runmode,
    tmp_path,
    should_run_wwarn
    ):
    raise NotImplementedError
