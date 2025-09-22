////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
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
