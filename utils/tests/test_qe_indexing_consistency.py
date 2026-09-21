# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.

import numpy as np

from afqmctools.observables.rhonk import get_centered_gvecs
from afqmctools.utils.qe_utils import _sparse_reciprocal_2dense


def test_centered_gvecs_sparse_dense_indexing_consistency():
    fft_grid = np.array([4, 4, 4], dtype=np.int32)
    target_g = np.array([1, -1, 0], dtype=np.int32)

    miller_inds = np.array([target_g], dtype=np.int32)
    reciprocal = np.array([1.0 + 0.0j], dtype=np.complex128)

    dense_3d = _sparse_reciprocal_2dense(miller_inds, reciprocal, fft_grid)
    dense_flat = dense_3d.reshape(-1)

    gvecs_common = get_centered_gvecs(tuple(fft_grid))
    match = np.all(gvecs_common == target_g, axis=1)

    assert np.count_nonzero(match) == 1
    assert np.count_nonzero(np.abs(dense_flat) > 0.0) == 1
    assert np.all(dense_flat[match] == 1.0 + 0.0j)
    assert np.all(dense_flat[~match] == 0.0 + 0.0j)
