////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef AXTY_KERNELS_HPP
#define AXTY_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void axty(int n, float alpha, float const* x, float* y);
void axty(int n, double alpha, double const* x, double* y);
void axty(int n, std::complex<float> alpha, std::complex<float> const* x, std::complex<float>* y);
void axty(int n, std::complex<double> alpha, std::complex<double> const* x, std::complex<double>* y);

} // namespace kernels

#endif
