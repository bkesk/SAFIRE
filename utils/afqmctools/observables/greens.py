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
Defines shared mathematical operations for
    computing observables.

Contents:
-

#TODO: if performance is an issue, use numba!
"""
import numpy as np

def greens_1body(psi_left:np.array,psi_right:np.array=None,X:np.array=None):
    r"""
    Compute the 1-body Green's function as:

    .. math::
        G_{ij} = < \Psi_L | \sum_{i,j} \hat{c}^\dagger_i \hat{c}_j | \Psi_R > / <\Psi_L|\Psi_R>
           = [ \Psi_R ( \Psi_L^\dagger \Psi_R )^{-1} \Psi_L^\dagger ]_{ji}
           = [ \Psi_L^* ( \Psi_R^T \Psi_L^* )^{-1} \Psi_R^T ]_{ij}

    Parameters
    ----------
    psi_left:np.array
        the Slater matrix of the "left" wavefunction :math:`|\Psi_L>` with shape (Nmo,Nelec)
    psi_right:np.array, opitonal
        the Slater matrix of the "right" wavefunction :math:`|\Psi_R>` with shape (Nmo,Nelec).
        If None, then psi_right = psi_left.
    - X:np.array, optional
        a transformation matrix X with shape (Nmo,Nmo).
        If given, it is applied to the basis of psi_left as `psi_left = X @ psi_left.`

    Examples
    --------
    >>> import numpy as np
    >>> from afqmctools.observables.greens import greens_1body
    >>> Nmo,Nelec = 5,2
    >>> psi_left = np.eye(Nmo,Nelec)
    >>> G = greens_1body(psi_left)
    """

    if X is not None:
        psi_left = X @ psi_left

    if psi_right is None:
        psi_right = psi_left

    # this keeps things vectorized
    psi_right_T = np.swapaxes(psi_right,-1,-2)
    psi_left_c = psi_left.conj()

    return psi_left_c @ np.linalg.solve( psi_right_T @ psi_left_c, psi_right_T)


def sd_overlap(psi_left:np.array,psi_right:np.array,X:np.array=None):
    r"""
    Compute the overlap between two sets of orbitals. Does not take into account spin.

    .. math::
        O = < \Psi_L | \Psi_R >

    Parameters
    ----------
    psi_left:np.array
        the Slater matrix of the "left" wavefunction :math:`|\Psi_L>` with shape (Nmo,Nelec)
    psi_right:np.array, opitonal
        the Slater matrix of the "right" wavefunction :math:`|\Psi_R>` with shape (Nmo,Nelec).
    - X:np.array, optional
        a transformation matrix X with shape (Nmo,Nmo).
        If given, it is applied to the basis of psi_left as `psi_left = X @ psi_left.`

    Examples
    --------
    >>> import numpy as np
    >>> from afqmctools.observables.greens import sd_overlap
    >>> Nmo,Nelec = 5,2
    >>> psi_l= np.eye(Nmo,Nelec)
    >>> psi_r= np.eye(Nmo,Nelec)
    >>> O = sd_overlap(psi_l,psi_r)
    >>> psi_l = np.stack([psi_l,psi_l]) # UHF like
    >>> psi_r = np.stack([psi_r,psi_r]) # UHF like
    >>> # wrap in prod to take into account spin
    >>> O = np.prod(sd_overlap(psi_l,psi_r))
    """

    if X is not None:
        psi_left = X @ psi_left

    if psi_right is None:
        psi_right = psi_left

    # this keeps things vectorized
    psi_right_T = np.swapaxes(psi_right,-1,-2)
    psi_left_c = psi_left.conj()

    return np.linalg.det(psi_right_T @ psi_left_c)

