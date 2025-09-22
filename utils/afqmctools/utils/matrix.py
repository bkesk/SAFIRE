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
import scipy.sparse as sps

def force_herm(M, method='upper_triangular'):
    """Force a matrix to be Hermitian.
    
    Parameters
    ----------
    M : array_like or sparse matrix
        Input matrix to be forced to Hermitian.
    method : str, optional
        Method to use for forcing Hermitian. Default is 'upper_triangular'. See notes.
    
    Returns
    -------
    M_herm : array_like or sparse matrix
        Hermitian matrix.


    Notes
    -----
    The following methods are available:
    - 'upper_triangular' or 'triu': Set the lower triangular part of the matrix to zero and add the
      conjugate transpose of the upper triangular part. This effectively ignores the lower triangular
      part of the matrix.
    """
    if method in ('upper_triangular','triu'):
        print("forcing Hermiticity by taking upper triangular part. This ignores the lower triangular part.")
        if sps.issparse(M):
            M = sps.triu(M, 1)
        else:
            M = np.triu(M, 1)
        M = M + M.conj().T
    elif method in ('average','avg'):
        print("forcing Hermiticity by taking the average of M and its conjugate transpose.")
        M = 0.5 * (M + M.conj().T)
    else:
        raise ValueError("Unknown method: {}".format(method))
    
    return M


def is_hermitian(M, tol=1e-10):
    """
    Check if a matrix (NumPy array or SciPy sparse array) is Hermitian.

    Parameters
    ----------
    M (np.ndarray or sps.spmatrix): The matrix to check.
    tol (float): Tolerance for numerical equality.

    Returns
    -------
    bool: True if the matrix is Hermitian, False otherwise.
    """
    if sps.issparse(M):
        # Check if the matrix is equal to its conjugate transpose
        return (M - M.conj().T).nnz == 0
    else:
        # Check if the matrix is equal to its conjugate transpose
        return np.allclose(M, M.conj().T, atol=tol)
    
