////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef ACAXPBB_KERNELS_HPP
#define ACAXPBB_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void acAxpbB(int m,
             int n,
             double const alpha,
             double const* A,
             int lda,
             double const* x,
             int incx,
             double const beta,
             double* B,
             int ldb);

void acAxpbB(int m,
             int n,
             float const alpha,
             float const* A,
             int lda,
             float const* x,
             int incx,
             float const beta,
             float* B,
             int ldb);

void acAxpbB(int m,
             int n,
             std::complex<double> const alpha,
             std::complex<double> const* A,
             int lda,
             std::complex<double> const* x,
             int incx,
             std::complex<double> const beta,
             std::complex<double>* B,
             int ldb);

void acAxpbB(int m,
             int n,
             std::complex<float> const alpha,
             std::complex<float> const* A,
             int lda,
             std::complex<float> const* x,
             int incx,
             std::complex<float> const beta,
             std::complex<float>* B,
             int ldb);

} // namespace kernels

#endif
