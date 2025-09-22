////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef KERNELS_TERM_BY_TERM_MAT_MAT_OPERATIONS_H
#define KERNELS_TERM_BY_TERM_MAT_MAT_OPERATIONS_H

#include <cassert>
#include <complex>
#include "Numerics/detail/define.hpp"

namespace kernels
{

template<class T1, class T2>
void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
                      T1 const alpha, T1 const* A, int lda, long Astride, 
                      std::complex<T2>* B, int ldb, long Bstride, int batchSize);

template<class T1, class T2>
void term_by_term_mat_mat_strided(ma::TENSOR_OPERATIONS op, int nrow, int ncol,
                      std::complex<T1> const alpha, std::complex<T1> const* A, int lda, long Astride,
                      std::complex<T2>* B, int ldb, long Bstride, int batchSize);

} // namespace kernels

#endif
