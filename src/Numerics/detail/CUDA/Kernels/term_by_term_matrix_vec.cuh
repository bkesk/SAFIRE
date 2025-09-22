////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef KERNELS_TERM_BY_TERM_OPERATIONS_H
#define KERNELS_TERM_BY_TERM_OPERATIONS_H

#include <cassert>
#include <complex>
#include "Numerics/detail/define.hpp"

namespace kernels
{

template<class T1, class T2>
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<T1>* A, int lda, T2 const alpha, T2 const* x, int incx);

template<class T1, class T2>
void term_by_term_mat_vec(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<T1>* A, int lda,
                      std::complex<T2> const alpha, std::complex<T2> const* x, int incx);

template<class T1, class T2>
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<T1>* A, int lda, int Astride, 
                      T2 const alpha, T2 const* x, int incx, int Xstride, int batchSize);

template<class T1, class T2>
void term_by_term_mat_vec_strided(ma::TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                      std::complex<T1>* A, int lda, int Astride,
                      std::complex<T2> const alpha, std::complex<T2> const* x, int incx,
                      int Xstride, int batchSize);

} // namespace kernels

#endif
