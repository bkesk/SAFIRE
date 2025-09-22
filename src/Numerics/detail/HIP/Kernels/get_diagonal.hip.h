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

#ifndef GET_DIAGONAL_KERNELS_HPP
#define GET_DIAGONAL_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void get_diagonal_strided(int nk,
                          int ni,
                          std::complex<double> const* B,
                          int ldb,
                          int stride,
                          std::complex<double>* A,
                          int lda);
void get_diagonal_strided(int nk,
                          int ni,
                          std::complex<float> const* B,
                          int ldb,
                          int stride,
                          std::complex<float>* A,
                          int lda);


} // namespace kernels

#endif
