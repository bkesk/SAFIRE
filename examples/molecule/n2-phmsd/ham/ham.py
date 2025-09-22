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
import numpy as np
from pyscf.fci.addons import large_ci
from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from afqmctools.hamiltonian.mol import write_hamil_mol
from afqmctools.wavefunction.mol import write_wfn

def ci_wavefunction(ciab, norb, nelec, ncore, tol):
  nalpha, nbeta = nelec
  ci, occa, occb = zip(*large_ci(ciab, norb, (nalpha, nbeta),
                      tol=tol, return_strs=False))
  ## no need to sort by ci magnitute
  #ixs = np.argsort(np.abs(ci))[::-1]
  #ci = np.array(ci)[ixs]
  #occa = np.array(occa)[ixs]
  #occb = np.array(occb)[ixs]
  # Reinsert frozen core to fully specify each determinant
  core = [i for i in range(ncore)]
  occa = [np.array(core + [o + ncore for o in oa]) for oa in occa]
  occb = [np.array(core + [o + ncore for o in ob]) for ob in occb]
  return ci, occa, occb

def read_cas_meta(chkfile, group='mcscf'):
  meta = dict()
  keys = ['ci', 'ncore', 'ncas']
  with h5py.File(chkfile, 'r') as f:
    for key in keys:
      meta[key] = f[group][key][()]
  return meta

def main():
  # inputs
  chkfile = '../scf/chkfile.h5'
  ci_tol = 0.02
  chol_tol = 1e-5
  # output
  fout = 'afqmc.h5'

  # read molecule and orbitals
  scf_data = load_from_pyscf_chk_mol(chkfile, 'mcscf')
  mol = scf_data['mol']
  nmo = len(scf_data['mo_coeff'])
  # read CAS information
  cas_meta = read_cas_meta(chkfile)
  ncas = cas_meta['ncas']
  ncore = cas_meta['ncore']
  nvals = [n-ncore for n in mol.nelec]
  ciab = cas_meta['ci']

  # Extract ci expansion in the form of a tuple: (ci coeff, occ_a, occ_b).
  # Note the ci_tol param which will return wavefunction elements with abs(ci) > tol.
  ci, occa, occb = ci_wavefunction(ciab, ncas, nvals, ncore, ci_tol)
  ndet = len(ci)
  print('number of determinants: %d' % ndet)

  # write Hamiltonian
  write_hamil_mol(scf_data, fout, chol_tol, dense=True, real_chol=True, verbose=True)
  # write Wavefunction
  ci = np.array(ci, dtype=np.complex128)
  uhf = True # UHF always true for CI expansions.
  write_wfn(fout, (ci, occa, occb), uhf, mol.nelec, nmo)

if __name__ == '__main__':
  main()  # set no global variable
