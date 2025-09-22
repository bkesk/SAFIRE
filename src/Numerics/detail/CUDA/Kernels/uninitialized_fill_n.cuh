////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef UNINITIALIZED_FILL_N_KERNELS_HPP
#define UNINITIALIZED_FILL_N_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void uninitialized_fill_n(bool* first, int N, bool const value);
void uninitialized_fill_n(int* first, int N, int const value);
void uninitialized_fill_n(float* first, int N, float const value);
void uninitialized_fill_n(double* first, int N, double const value);
void uninitialized_fill_n(std::complex<float>* first, int N, std::complex<float> const value);
void uninitialized_fill_n(std::complex<double>* first, int N, std::complex<double> const value);
//void uninitialized_fill_n(double2 * first, int N,  double2 const value);

void uninitialized_fill_n(bool* first, long N, bool const value);
void uninitialized_fill_n(int* first, long N, int const value);
void uninitialized_fill_n(float* first, long N, float const value);
void uninitialized_fill_n(double* first, long N, double const value);
void uninitialized_fill_n(std::complex<float>* first, long N, std::complex<float> const value);
void uninitialized_fill_n(std::complex<double>* first, long N, std::complex<double> const value);
//void uninitialized_fill_n(double2 * first, long N,  double2 const value);

} // namespace kernels

#endif
