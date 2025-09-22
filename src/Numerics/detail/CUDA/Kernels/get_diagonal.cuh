////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
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
