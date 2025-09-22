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

#ifndef BATCHEDDOT_KERNELS_HPP
#define BATCHEDDOT_KERNELS_HPP

#include <complex>

namespace kernels
{
// y[i] = beta * y[i] + sum_k alpha * A[k,i] * B[k,i]
void batchedDot(int m,
                int n,
                double const alpha,
                double const* A,
                int lda,
                double const* B,
                int ldb,
                double const beta,
                double* y,
                int incy);
void batchedDot(int m,
                int n,
                float const alpha,
                float const* A,
                int lda,
                float const* B,
                int ldb,
                float const beta,
                float* y,
                int incy);
void batchedDot(int m,
                int n,
                std::complex<double> const alpha,
                std::complex<double> const* A,
                int lda,
                std::complex<double> const* B,
                int ldb,
                std::complex<double> const beta,
                std::complex<double>* y,
                int incy);
void batchedDot(int m,
                int n,
                std::complex<float> const alpha,
                std::complex<float> const* A,
                int lda,
                std::complex<float> const* B,
                int ldb,
                std::complex<float> const beta,
                std::complex<float>* y,
                int incy);

} // namespace kernels

#endif
