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

#ifndef SUM_KERNELS_HPP
#define SUM_KERNELS_HPP

#include <complex>

namespace kernels
{
double sum(int n, double const* x, int incx);
std::complex<double> sum(int n, std::complex<double> const* x, int incx);
float sum(int n, float const* x, int incx);
std::complex<float> sum(int n, std::complex<float> const* x, int incx);

double sum(int m, int n, double const* x, int lda);
std::complex<double> sum(int m, int n, std::complex<double> const* x, int lda);
float sum(int m, int n, float const* x, int lda);
std::complex<float> sum(int m, int n, std::complex<float> const* x, int lda);

} // namespace kernels

#endif
