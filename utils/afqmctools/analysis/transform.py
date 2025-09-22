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
import numba as nb

jit = nb.njit

def hermitize_factory(walker_type) -> callable:
    r"""
    Factory function to generate a hermitian transform function.

    The transform makes a matrix hermitian by averaging with its conjugate transpose.
    .. math::
        M_{nij} = \\frac{1}{2} (M_{nij} + M_{nji}^{*})
    
    where 'n' is the sample dimension, and 'i' and 'j' are the matrix dimensions.

    Notes
    -----
    Only works for 2D matrices, where the first dimension is the sample dimension, and the
    second dimension is the square of a the matrix dimension.
    """

    if walker_type == 'closed':
        def hermitize(data):
            M_squared = data.shape[-1]
            M = np.sqrt(M_squared)
            if M != int(M):
                raise ValueError(f"Flattened Matrix dimension {M_squared} is not a perfect square")
            else:
                M = int(M)
            transposed_indices = np.ravel_multi_index(np.unravel_index(np.arange(M_squared), (M,M)), (M,M), order='F')
            return 0.5 * (data + data[:,transposed_indices].conj())
        return hermitize
    elif walker_type == 'collinear':
        def hermitize(data):
            nspin = 2
            M_squared = data.shape[-1]/nspin
            if not np.allclose(M_squared,int(M_squared)): 
                raise ValueError(f"Flattened Matrix dimension {M_squared} is not divisible by number of spins {nspin}")
            else:
                M_squared = int(M_squared)

            M = np.sqrt(M_squared)
            
            if not M == int(M):
                raise ValueError(f"Flattened Matrix dimension {M_squared} is not a perfect square")
            else:
                M = int(M)
            new_data = np.zeros((data.shape[0], nspin, M_squared), dtype=data.dtype)
            for s in range(nspin):
                transposed_indices = np.ravel_multi_index(np.unravel_index(np.arange(M_squared), (M,M)), (M,M), order='F')
                # select the s-th spin channel with data[:,s*M_squared:(s+1)*M_squared]; and transpose by indexing with transposed_indices;
                new_data[:,s,:] = 0.5 * (data[:,s*M_squared:(s+1)*M_squared] + data[:,s*M_squared:(s+1)*M_squared][:,transposed_indices].conj())
            return new_data
        return hermitize
    elif walker_type == 'noncollinear':
        raise NotImplementedError("Non-collinear walker type not implemented")
    else:
        raise ValueError(f"Unknown walker type: {walker_type}")

def eval_one_body_obs_factory(one_body_operator=None,walker_type=None):
    """
    Factory function to generate a one-body energy transform function.

    The transform computes a one body observable from the one-body operator and the density matrix.
    
    .. math::
        O_{1} = \\sum_{pq} \\rho_{pq} O_{pq}
    
    where 'p' and 'q' are the matrix dimensions, and 'h' is the one-body operator.

    For closed shell, a factor of 2 is applied to the result to account for the two spin channels.
    for collinear, both spin channels are evaluated separately, and the result is summed.
    For non-collinear, the spin channels are evaluated separately, and the result is summed.
    """

    # needed for jit below
    one_body_operator = one_body_operator.astype(np.complex128)

    if walker_type == 'closed':
        @jit(cache=True)
        def transform(data):
            return 2*(data @ one_body_operator.flatten())[:,np.newaxis]
        return transform
    elif walker_type == 'collinear':
        @jit(cache=True)
        def transform(data):
            spin_up_contribution = data[:,0] @ one_body_operator.flatten()
            spin_down_contribution = data[:,1] @ one_body_operator.flatten()
            return (spin_up_contribution + spin_down_contribution )[:,np.newaxis]
        return transform
    elif walker_type == 'noncollinear':
        @jit(cache=True)
        def transform(data):
            return (data @ one_body_operator.flatten())[:,np.newaxis]
        return transform
    else:
        raise ValueError(f"Unknown walker type: {walker_type}")


def eval_two_body_obs_factory(cholesky=None,two_body_operator=None,walker_type=None):
    """
    Factory function to generate a two-body energy transform function.

    The transform computes a two body observable from the two-body operator and the density matrix.
    
    .. math::
        O_{2} = \\frac{1}{2} \\sum_{pqrs} \\rho_{pqrs} ( pq | rs )
    
    where 'p' and 'q' are the matrix dimensions, and 'h' is the two-body operator.

    For closed shell, a factor of 2 is applied to the result to account for the two spin channels.
    for collinear, both spin channels are evaluated separately, and the result is summed.
    For non-collinear, the spin channels are evaluated separately, and the result is summed.
    """
    if two_body_operator is None and cholesky is None:
        raise ValueError("Two-body integrals not provided; please provide them in Chemist's convention via the 'eri' argument "
                         " or in Cholesky format via the 'cholesky' argument")
    # needed for jit below
    two_body_operator = two_body_operator.astype(np.complex128)

    nbasis_4 = two_body_operator.shape[-1]
    nbasis = np.power(nbasis_4, 0.25)
    if not nbasis == int(nbasis):
        raise ValueError(f"Flattened Matrix dimension {nbasis_4} is not a 4th root")
    else:
        nbasis = int(nbasis)

    if two_body_operator is not None:
        if walker_type == 'closed':
            raise NotImplementedError("Closed walker type not implemented")
        elif walker_type == 'collinear':
            num_spin_sectors = 3
            @jit
            def transform(data):
                data = data.reshape((-1,num_spin_sectors,nbasis_4))
                # compute the two-body energy for each spin channel
                O2_aaaa = np.dot(data[:,0], two_body_operator)
                O2_aabb = np.dot(data[:,1], two_body_operator)
                O2_bbaa = O2_aabb
                O2_bbbb = np.dot(data[:,2], two_body_operator)
                return 0.5*(O2_aaaa + O2_bbbb + O2_aabb + O2_bbaa)[:,np.newaxis]
            return transform
        elif walker_type == 'noncollinear':
            raise NotImplementedError("Non-collinear walker type not implemented")
        else:
            raise ValueError(f"Unknown walker type:{ walker_type}")
    elif cholesky is not None:
        raise NotImplementedError("Cholesky not implemented; contact the developers")

