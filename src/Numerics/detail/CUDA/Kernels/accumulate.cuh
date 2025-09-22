////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef KERNELS_ACCUMULATE_H
#define KERNELS_ACCUMULATE_H

#include <cassert>
#include <complex>
#include "Numerics/detail/define.hpp"

namespace kernels
{

// y[n][i] += alpha * sum_j A[n][j][i]  for dim==0
//         += alpha * sum_j A[n][i][j]  for dim==1
template<class T1, class T2>
void accumulate_impl(int dim, int nrow, int ncol, std::complex<T1> const alpha, 
                std::complex<T1> const* A, int lda, long Astride,
                std::complex<T2>* y, int incy, long ystride, int batchSize);

template<class T1, class T2>
void accumulate_impl(int dim, int nrow, int ncol, T1 const alpha, T1 const* A, int lda, long Astride,
                std::complex<T2>* y, int incy, long ystride, int batchSize);

} // namespace kernels

#endif
