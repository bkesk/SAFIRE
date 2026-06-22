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

import warnings

import h5py
import numpy as np
from scipy.sparse import csr_array
from afqmctools.utils.io import write_csr

warnings.warn("Deprecation Warning: The afqmctools.hamiltonian.hubbard submodule will be removed")

def write_hubbard(
  hamil_file, h1, Uij, nup_ndn,
  spin_type='closed',
  hst_type='continuous_charge',
):
  warnings.warn("Deprecation Warning: The afqmctools.hamiltonian.hubbard.write_hubbard function will be removed")
  nbas = len(h1)
  nup, ndn = nup_ndn
  dims = [0, 0, 0, nbas, nup, ndn, 0, 0]
  # write hamiltnoian file
  fp = h5py.File(hamil_file, 'w')
  ham = fp.create_group('Hamiltonian')
  ham['dims'] = np.array(dims, dtype=int)
  ham['Energies'] = np.array([0, 0])  # !!!! what are these?
  mham = ham.create_group('ModelHamiltonian')
  mham['number_of_components'] = 2  # !!!! why always 2?
  write_one_body(mham, h1, spin_type)
  write_two_body(mham, Uij, hst_type)
  fp.close()

def write_one_body(grp, h1, spin_type):
  warnings.warn("Deprecation Warning: The afqmctools.hamiltonian.hubbard.write_one_body function will be removed")
  g = grp.create_group('ModelComponent_0')
  g['model_type'] = 'one_body'
  g['spin_type'] = spin_type
  slab = g.create_group('tij')
  mat = csr_array(h1.astype(np.complex128))
  write_csr(slab, mat)

def write_two_body(grp, Uij, hst_type):
  warnings.warn("Deprecation Warning: The afqmctools.hamiltonian.hubbard.write_two_body function will be removed")
  g = grp.create_group('ModelComponent_1')
  g['model_type'] = 'hubbard_u'
  g['hst_type'] = hst_type
  mat = csr_array(Uij.astype(np.complex128))
  slab = g.create_group('Uij')
  write_csrm(slab, mat)
