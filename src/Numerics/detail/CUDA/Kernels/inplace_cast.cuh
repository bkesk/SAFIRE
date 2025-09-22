////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef INPLACE_CAST_KERNELS_HPP
#define INPLACE_CAST_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void inplace_cast(unsigned long n, std::complex<float>* A, std::complex<double>* B);
void inplace_cast(unsigned long n, std::complex<double>* A, std::complex<float>* B);
void inplace_cast(long n, std::complex<float>* A, std::complex<double>* B);
void inplace_cast(long n, std::complex<double>* A, std::complex<float>* B);

} // namespace kernels

#endif
