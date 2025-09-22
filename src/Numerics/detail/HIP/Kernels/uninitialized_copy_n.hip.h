////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
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
