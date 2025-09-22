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

#ifndef ADIAGAPY_KERNELS_HPP
#define ADIAGAPY_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void adiagApy(int N, float const alpha, float const* A, int lda, float* y, int incy);
void adiagApy(int N,
              std::complex<float> const alpha,
              std::complex<float> const* A,
              int lda,
              std::complex<float>* y,
              int incy);
void adiagApy(int N, double const alpha, double const* A, int lda, double* y, int incy);
void adiagApy(int N,
              std::complex<double> const alpha,
              std::complex<double> const* A,
              int lda,
              std::complex<double>* y,
              int incy);

} // namespace kernels

#endif
