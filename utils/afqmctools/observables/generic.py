# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

def observable_1body(rmd_ij_batch:np.ndarray, Oji:np.ndarray):
    r"""Compute the expectation value of a 1-body operator.

    The rdms and Oij must have a relative transpose between them.
    
    Specifically, 
    
    .. math:: \langle \hat{O} \rangle = \sum_{ij} \langle \hat{c}_i^\dagger \hat{c}_j \rangle  O_{ij}

    and 

    .. math:: \langle \hat{c}_i^\dagger \hat{c}_j \rangle = \rho_{ji}
     
    then, 

    .. math:: \langle \hat{O} \rangle = Tr(\rho O.T) \text{ (after swapping dummy indices i<->j)}
    
    **By convention, we apply the transpose to the observable**
    
    Parameters
    ----------
    rmd_ij_batch : np.ndarray
        Batched reduced matrix elements of shape (nbatch,norb*norb).
    Oji : np.ndarray
        transpose of a 1-body operator with shape (norb*norb).
    
    Returns
    -------
    results : np.ndarray
        Expectation value of Oij for each rdm in the batch with shape (nbatch,).
    """
    return rmd_ij_batch @ Oji # can be 'dotted' with a CI vector

