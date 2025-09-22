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
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

def get_cell(alat, pseudo, basis, mesh, verbose=5):
  axes = alat/2*(np.ones(3)-np.eye(3))
  elem = ['C', 'C']
  pos = np.array([
    [0, 0, 0],
    [0.25, 0.25, 0.25],
  ])*alat
  cellfac = pgto.Cell(
    a = axes,
    atom = [(e, p) for e, p in zip(elem, pos)],
    basis = basis,
    pseudo = pseudo,
    verbose=verbose,
  )
  cell = cellfac.build(mesh=mesh)
  return cell

def run_rhf(chkfile, cell, nkx, ndim=3):
  kpts = cell.make_kpts((nkx,)*ndim)
  rhf = scf.KRHF(cell, kpts=kpts)
  rhf.chkfile = chkfile
  rhf.run()
  return rhf

def main():
  mesh = (25,)*3  # defines ecut
  alat = 3.6  # angstrom
  nkx = 2  # kmesh
  basis = 'gth-szv'
  pseudo = 'gth-pade'
  chkfile = 'chkfile.h5'
  cell = get_cell(alat, pseudo, basis, mesh)
  rhf = run_rhf(chkfile, cell, nkx)

if __name__ == '__main__':
  main()
