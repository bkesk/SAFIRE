# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import pytest
import numpy as np

from afqmctools.observables.spin import spin_squared,spin_z
from afqmctools.observables.greens import greens_1body

@pytest.mark.dev
@pytest.mark.weekly
@pytest.mark.parametrize("M,nelec,Sz,S2",
    [
        (4,(2,0),1,2),
        (4,(3,0),1.5,3.75),
        (4,(3,1),1,2),
        (4,(1,3),-1,2),
        (4,(1,0),0.5,0.75),
    ]
)
def test_spin_squared(M,nelec,Sz,S2):
    """Reality test for <S^2> on some simple cases
    """
    M = 4

    rho_uu = np.zeros((M,M))
    for i in range(nelec[0]):
        rho_uu[i,i] = 1.0

    rho_dd = np.zeros_like(rho_uu)
    for i in range(nelec[1]):
        rho_dd[i,i] = 1.0

    rho_du = np.zeros_like(rho_uu)
    rho_ud = rho_du

    rdm = np.block([
        [rho_uu,rho_ud],
        [rho_du,rho_dd]]
    )

    actual_Sz = spin_z(rdm,spin_symm='noncollinear')
    # up only terms
    actual_S2 = spin_squared(rdm,spin_symm='noncollinear')

    assert actual_Sz == Sz and actual_S2 == S2

@pytest.mark.dev
@pytest.mark.weekly
@pytest.mark.parametrize("H",
    [
        np.arange(64).reshape(8,8),
        np.arange(64).reshape(8,8)\
            +np.triu(np.linspace(0,1,64).reshape(8,8),k=1)*1j
    ]
)
def test_greens(H):
  H = H+np.swapaxes(H,-1,-2).conj()
  assert np.allclose(H,H.conj().T)
  Es,vecs = np.linalg.eigh(H)
  orbitals = vecs[:,:4]
  energy = np.sum(Es[:4])
  rdm = greens_1body(orbitals)
  energy_greens = np.sum(H*rdm)

  assert np.allclose(energy,energy_greens)

