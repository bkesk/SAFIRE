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
from pyscf import gto, scf

def hubbard_hop(lat, neighbors, order='C'):
  """Examples:

  >>> h1 = hubbard_hop( (3, 4) )
  >>> h1 = hubbard_hop( (2, 1) )
  """
  ndim = len(lat)
  nndim = neighbors.shape[1]
  if nndim != ndim:
    msg = 'neighbor list dimension %d != lattice dimension %d' % (nndim, ndim)
    raise RuntimeError(msg)
  nbas = np.prod(lat)
  h1 = np.zeros([nbas, nbas], dtype=int)
  for i in range(nbas):
    ilat = np.unravel_index(i, lat, order=order)
    for neib in neighbors:
      jlat = ilat + neib
      for l in range(ndim):
        jlat[l] = jlat[l] % lat[l]
      j = np.ravel_multi_index(jlat, lat, order=order)
      tup = (i, j)
      if np.allclose(tup, tup[0]): continue
      h1[tup] += 1
  return h1

def define_hubbard_model(h1, U, nup_ndn):
  mol = gto.M()
  mol.nelec = nup_ndn
  mol.incore_anyway = True

  def get_veff(mol, dm, *args):
    j_a = np.diag(np.einsum('ii->i', dm[0]) * U)
    k_a = np.diag(np.einsum('ii->i', dm[0]) * U)
    j_b = np.diag(np.einsum('ii->i', dm[1]) * U)
    k_b = np.diag(np.einsum('ii->i', dm[1]) * U)
    j = j_a + j_b
    veff_a = j-k_a
    veff_b = j-k_b
    return veff_a, veff_b
  
  mf = scf.UHF(mol)
  mf.get_hcore = lambda *args: h1
  mf.get_ovlp = lambda *args: np.eye(len(h1))
  mf.get_veff = get_veff
  return mf

def checkerboard_sites(lat, order='C'):
  nbas = np.prod(lat)
  sel = np.zeros(nbas, dtype=bool)
  for i in range(nbas):
    ilat = np.unravel_index(i, lat, order=order)
    if (sum(ilat) % 2):
      sel[i] = True
  return sel

def run_uhf(fout, lat, h1, U, nelec, verbose=1):
  # define UHF model
  mf = define_hubbard_model(h1, U, nelec)
  mf.max_cycle = 1000
  mf.verbose = verbose

  # AFM initial guess
  sel = checkerboard_sites(lat)
  nsite = np.prod(lat)
  d0up = np.zeros(nsite)
  d0up[sel] = 1
  d0dn = np.zeros(nsite)
  d0dn[~sel] = 1
  dm0 = np.array([np.diag(d0up), np.diag(d0dn)])

  # run UHF
  mf.chkfile = fout
  mf.kernel(dm0)

def main():
  # define lattice
  nx = ny = 4
  lat = (nx, ny)
  nsite = np.prod(lat)

  # define filling
  nup = ndn = nsite//2
  nelec = (int(nup), int(ndn))

  # define one-body part
  t = 1.0
  neighbors = np.array([
    [ 0, -1],
    [-1,  0],
    [ 1,  0],
    [ 0,  1],
  ])
  h1 = hubbard_hop(lat, neighbors)
  h1 = -t*h1

  # define two-body part
  U = 4.0

  # output file
  chkfile = 'n%d,%d-U%.2f-ne%d,%d.h5' % (nx, ny, U, nup, ndn)

  run_uhf(chkfile, lat, h1, U, nelec, verbose=5)

  # add Hubbard Parameters to chkfile
  fp = h5py.File(chkfile, 'a')
  fp['scf/hcore'] = h1
  fp['scf/HubbardU'] = U
  fp.close()

if __name__ == '__main__':
  main()
# end __main__
