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
import h5py as h5
import sys

if False and sys.version_info >= (3, 12):
    from itertools import batched
else:
    # remove once Python 3.12 is the official minimum supported Python version
    from itertools import islice
    def batched(iterable, n, *, strict=False):
        if n < 1:
            raise ValueError('n must be at least one')
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            if strict and len(batch) != n:
                raise ValueError('batched(): incomplete batch')
            yield batch

def batched_np_array(data:np.array, batchsize=1):
    """
    Generator to batch data into chunks of size blocksize.
    
    Parameters
    ----------
    iterable : iterable
        The data to be batched.
    batchsize : int
        The size of each batch.
    
    Returns
    -------
    generator
        A list of batches, each of size batchsize.
    """
    if batchsize == 1:
        return data
    else:
        nblocks = data.shape[0] // batchsize + (data.shape[0] % batchsize > 0)
        for i in range(nblocks):
            yield data[i*batchsize:(i+1)*batchsize]


def get_metadata(fstat, path='Metadata'):
    """Extract estimator metadata from h5 file.

    Parameters
    ----------
    filename : string
        output file containing density matrix (*.h5 file).
    path : string
        Path to metadata in filename.

    Returns
    -------
    meta : dict
        Estimator metadata.
    """
    if not path.endswith('Metadata'):
        path = path.rstrip('/') + '/Metadata'

    meta = dict()
    with h5.File(fstat, 'r') as fp:
        for k, v in fp[path].items():
            meta[k] = v[()]
    return meta

def _reblock_backend(data:np.array, blocksize=1):
    """
    Reblock data along first axis (backend).

    Parameters
    ----------
    data : np.ndarray
        Data to be reblocked. Leading dimension is assumed to be the 
        number of samples. This is the dimension that will be averaged over.
    blocksize : int
        Size of blocks to average over. The default is 1.

    Returns
    -------
    np.array
        Reblocked data. The first axis is the number of new blocks. 
        The remaining axes are unchanged.
    """
    if blocksize == 1:
        return data
    else:
        return data.reshape(-1, blocksize, *data.shape[1:]).mean(axis=1)

def reblock(data:np.array, blocksize):
    """
    Reblock data along first axis.

    Parameters
    ----------
    data : np.array
        The data to be reblocked.
    blocksize : int
        The size of each block. Recommended that the leading dimension of the data is evenly divisible by the blocksize.
        If not, the last block will be smaller than the others.

    Returns
    -------
    np.array
        Reblocked data. The first axis is the number of new blocks. 
        The remaining axes are unchanged.

    Notes
    -----
    This function executes faster when the data is evenly divisible by the blocksize;
    if the data is not evenly divisible, the output will be correct, but the function will be slower.

    .. Warning::
        Will run slower if the data is not evenly divisible by the blocksize.
        Recommended to ensure that the data is evenly divisible by the blocksize.

    """
    if data.shape[0] % blocksize != 0:
        n_full_blocks = data.shape[0] // blocksize
        incomplete_block_size = data.shape[0] % blocksize
        data_last = _reblock_backend(data[-(incomplete_block_size):], incomplete_block_size) 
        data_first = _reblock_backend(data[:n_full_blocks*blocksize], blocksize)
        return np.concatenate((data_first, data_last), axis=0)
    else:
        return _reblock_backend(data, blocksize)


def _Neq_from_Teq(Teq,taus=None,delta_tau=None):
    """
    Compute the number of equilibration *blocks* (Neq) from
       the equilibration *time* (Teq)
    """
    if taus is None and delta_tau is None:
        raise ValueError(
            "_Neq_from_Teq: must specify either delta_tau"
            " or a list of tau points to compute Neq"
        )

    if delta_tau is None:
        delta_tau = taus[1] - taus[0]
    return np.ceil(Teq/delta_tau)


def get_bp_taus(fstat):
    sym_md = get_metadata(fstat)
    bp_md = get_metadata(fstat, 'Observables/BackPropagated/Metadata')
    dt = sym_md['Timestep']
    nsteps = bp_md['BackPropSteps']
    taus = nsteps*dt
    return taus

