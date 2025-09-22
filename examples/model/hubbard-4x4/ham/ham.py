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
import numpy as np
from afqmctools.hamiltonian.hubbard import write_hubbard
from afqmctools.wavefunction.mol import write_wfn

def read_nelec(fchk):
  from pyscf.lib.chkfile import load_mol
  mol = load_mol(fchk)
  return mol.nelec

def read_hcore(fchk):
  with h5py.File(fchk, 'r') as fp:
    hcore = fp['scf/hcore'][()]
  return hcore

def read_hubbard(fchk):
  with h5py.File(fchk, 'r') as fp:
    U = fp['scf/HubbardU'][()]
  return U

def read_mo_coeff(fchk, scf='scf'):
  from pyscf.lib.chkfile import load
  mo_coeff = load(fchk, '%s/mo_coeff' % scf)
  return mo_coeff

def main():
  chkfile = '../scf/n4,4-U4.00-ne8,8.h5'
  nelec = read_nelec(chkfile)

  # write Hamiltonian
  h1 = read_hcore(chkfile)
  U = read_hubbard(chkfile)
  # uniform on-site U
  norb = len(h1)
  Uij = U*np.diag(np.ones(norb))
  write_hubbard('afqmc.h5', h1, Uij, nelec)

  # write Wavefunction
  mo_coeff = read_mo_coeff(chkfile)

  nup, ndn = nelec
  ci = np.array([1.0+0j])
  wfn = np.zeros((len(ci), norb, nup+ndn), dtype=np.complex128)
  wfn[0,:,:nup] = mo_coeff[0][:,:nup]
  wfn[0,:,nup:] = mo_coeff[1][:,:ndn]

  uhf = True
  write_wfn('afqmc.h5', (ci, wfn), uhf,
                    nelec, norb, verbose=True)

if __name__ == '__main__':
  main()  # set no global variable
