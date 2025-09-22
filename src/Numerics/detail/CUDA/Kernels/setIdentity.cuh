////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef SETIDENTITY_KERNELS_HPP
#define SETIDENTITY_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void set_identity(int m, int n, double* A, int lda);
void set_identity(int m, int n, float* A, int lda);
void set_identity(int m, int n, std::complex<double>* A, int lda);
void set_identity(int m, int n, std::complex<float>* A, int lda);

void set_identity_strided(int nbatch, int stride, int m, int n, double* A, int lda);
void set_identity_strided(int nbatch, int stride, int m, int n, float* A, int lda);
void set_identity_strided(int nbatch, int stride, int m, int n, std::complex<double>* A, int lda);
void set_identity_strided(int nbatch, int stride, int m, int n, std::complex<float>* A, int lda);

} // namespace kernels

#endif
