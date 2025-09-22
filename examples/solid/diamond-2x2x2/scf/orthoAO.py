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
from pyscf.pbc import scf
from afqmctools.utils.pyscf_utils import mf_from_chkfile
from afqmctools.utils.linalg import get_ortho_ao

def store_rotation_matrix(chkfile):
  rhf = mf_from_chkfile(chkfile, scf.KRHF, scf)
  kpts = rhf.kpts
  # get rotation matrix X to produce orthonormal basis
  X, nmo_per_kpt = get_ortho_ao(rhf.cell, kpts)
  #
  hcore = rhf.get_hcore()
  fock = hcore+rhf.get_veff()
  # make conversion
  with h5py.File(chkfile, 'a') as fp:
    fp['scf/orthoAORot'] = X
    fp['scf/nmo_per_kpt'] = nmo_per_kpt
    fp['scf/hcore'] = hcore
    fp['scf/fock'] = fock
  return rhf

def main():
  chkfile = '../scf/chkfile.h5'
  store_rotation_matrix(chkfile)

if __name__ == '__main__':
  main()  # set no global variable
