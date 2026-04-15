# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from afqmctools.hamiltonian.supercell import write_hamil_supercell
from afqmctools.hamiltonian.kpoint import write_hamil_kpoints
from afqmctools.hamiltonian.mol import write_hamil_mol
from afqmctools.inputs.energy import calculate_hf_energy
from afqmctools.utils.pyscf_utils import (
        chk_is_pbc,
        load_from_pyscf_chk,
        load_from_pyscf_chk_mol
        )
from afqmctools.wavefunction.mol import write_wfn_mol
from afqmctools.wavefunction.pbc import write_wfn_pbc

from afqmctools.utils.slater_types import (
    _slater_enum_map,
    _get_slater_type,
)

# TODO: rename this across entire code base
def pyscf_to_afqmc(chkfile, hamil_file, threshold, comm=None,
                  ortho_ao=False, df=False, kpoint=False, verbose=False,
                  cas=None, wfn_file=None,
                  write_hamil=True, ndet_max=None, real_chol=False,
                  phdf=False, low=0.1, high=0.95, dense=False, 
                  walker_type=None, with_sfx2c=False,
                  with_x2c=False):
    """Dispatching routine dependent on options.

    This needs documentation badly!! many options, none are documented ...
    """
    pbc = chk_is_pbc(chkfile=chkfile)

    if wfn_file is None:
        wfn_file = hamil_file

    if pbc:
        if with_sfx2c or with_x2c:
            raise NotImplementedError("x2c and sfx2c not yet implemented with pbc.") 
        if comm.rank == 0 and verbose:
            print(" # Generating Hamiltonian and wavefunction from pyscf cell"
                  " object.")
        scf_data = load_from_pyscf_chk(chkfile, orthoAO=ortho_ao)
        if write_hamil:
            if kpoint:
                write_hamil_kpoints(comm, scf_data, hamil_file, threshold,
                                    verbose=verbose, cas=cas,
                                    ortho_ao=ortho_ao, phdf=phdf)
            else:
                write_hamil_supercell(comm, scf_data, hamil_file, threshold,
                                      verbose=verbose, cas=cas,
                                      ortho_ao=ortho_ao)
        if comm.rank == 0:
            nelec = write_wfn_pbc(scf_data, ortho_ao, wfn_file, verbose=verbose,
                                  ndet_max=ndet_max, low=low, high=high)
    else:
        if verbose:
            print(" # Generating Hamiltonian and wavefunction from pyscf mol"
                  " object.")
        if comm is not None and comm.size > 1:
            raise ValueError(
                " # Error molecular integral generation must be done "
                "in serial."
            )

        soc_type = None
        if with_sfx2c:
            soc_type = 'sfx2c'
        elif with_x2c:
            soc_type = 'x2c'
        
        scf_data = load_from_pyscf_chk_mol(
            chkfile,
            soc_type=soc_type,
        )
        scf_data['orthAO'] = ortho_ao

        if walker_type is None:
            walker_type = _get_slater_type(
                scf_data['mo_coeff'],
                scf_data['nelec'],
                scf_data['norb']
            )
        else: # override the scf_data based on user input!
            walker_type = _slater_enum_map(walker_type)
            scf_data['walker'] = walker_type
        
        if write_hamil:
            write_hamil_mol(
                scf_data,
                hamil_file,
                threshold,
                verbose=verbose,
                cas=cas,
                ortho_ao=ortho_ao,
                real_chol=real_chol,
                dense=dense,
                df=df,
                walker_type=walker_type
            )
        write_wfn_mol(
            scf_data,
            wfn_file
        )
        if verbose > 1:
            print(" # Recomputing single-determinant Hartree--Fock energy.")
            etot, e1b, e2b = calculate_hf_energy(hamil_file, wfn_file)
            print(" # ETotal : {: 13.10f}".format(etot.real))
            print(" # E1Body : {: 13.10f}".format(e1b.real))
            print(" # E2Body : {: 13.10f}".format(e2b.real))
