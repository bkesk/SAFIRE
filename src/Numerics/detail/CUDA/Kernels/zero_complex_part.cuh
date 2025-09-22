////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef ZERO_COMPLEX_PART_KERNELS_HPP
#define ZERO_COMPLEX_PART_KERNELS_HPP

#include <complex>

namespace kernels
{
void zero_complex_part(int n, std::complex<double>* x);
void zero_complex_part(int n, std::complex<float>* x);
void zero_complex_part(int n, double* x);
void zero_complex_part(int n, float* x);

} // namespace kernels

#endif
