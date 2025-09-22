////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef UNINITIALIZED_COPY_N_KERNELS_HPP
#define UNINITIALIZED_COPY_N_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void uninitialized_copy_n(int N, double const* first, int incx, double* array, int incy);
void uninitialized_copy_n(int N, std::complex<double> const* first, int incx, std::complex<double>* array, int incy);
void uninitialized_copy_n(int N, int const* first, int incx, int* array, int incy);

// long
void uninitialized_copy_n(long N, double const* first, long incx, double* array, long incy);
void uninitialized_copy_n(long N, std::complex<double> const* first, long incx, std::complex<double>* array, long incy);
void uninitialized_copy_n(long N, int const* first, long incx, int* array, long incy);

} // namespace kernels

#endif
