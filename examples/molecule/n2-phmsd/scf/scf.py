#!/usr/bin/env python3

# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import h5py

from pyscf import gto, scf, mcscf

def n2_dimer(bond_length_in_bohr, basis):
  elem = ['N', 'N']
  pos = [
    [0, 0, 0],
    [0, 0, bond_length_in_bohr],
  ]
  mol = gto.M(
    atom = [(e, p) for e, p in zip(elem, pos)],
    basis = basis,
    unit = 'Bohr',
  )
  return mol

def main():
  bond_length_in_bohr = 3.0
  basis = 'cc-pvdz'
  chkfile = 'chkfile.h5'

  mol = n2_dimer(bond_length_in_bohr, basis)

  rhf = scf.RHF(mol)
  rhf.chkfile = chkfile
  rhf.run()
  nmo = rhf.mo_energy.shape[-1]

  # Replicate the calculations from J. Chem. Phys. 127, 144101 (2007).
  # They find a CASSCF energy of -108.916484 Ha, and a ph-AFQMC energy of
  # -109.1975(6) Ha with a 97 determinant CASSCF trial.
  nactive = 12
  ncore = 4  # freeze 1s 2s
  nelec = [n-ncore for n in mol.nelec]
  mc = mcscf.CASSCF(rhf, nactive, nelec)
  mc.chkfile = chkfile
  mc.run()

  # save determinant expansion coefficients
  with h5py.File(chkfile, 'a') as fp:
    fp['mcscf']['ci'] = mc.ci

if __name__ == '__main__':
  main()  # set no global variable
