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
