////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef FILL_N_KERNELS_HPP
#define FILL_N_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{

template<typename T>
void fill_n(T* first, long N, long stride, T const value);

template<typename T>
void fill_n(T* first, long N, T const value);

template<typename T>
void fill2D_n(long N, long M, T* A, long lda, T const value);

template<typename T1, typename T2, typename T3>
void fill_if_zero_impl(int nrow, int ncol, T1 const* key, int incx,
                T2 const alpha, T3* A, int lda, long stride, int batchSize);

template<typename T1, typename T2, typename T3>
void fill_if_zero_impl(int nrow, int ncol, std::complex<T1> const* key, int incx,
                T2 const alpha, T3* A, int lda, long stride, int batchSize);

template<typename T1, typename T2, typename T3>
void fill_if_non_zero_impl(int nrow, int ncol, T1 const* key, int incx,
                T2 const alpha, T3* A, int lda, long stride, int batchSize);

} // namespace kernels

#endif
