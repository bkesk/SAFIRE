////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef PRINT_KERNELS_HPP
#define PRINT_KERNELS_HPP

#include <cassert>
#include <complex>

namespace kernels
{
void print(std::string str, std::complex<double> const* p, int n);
void print(std::string str, double const* p, int n);
void print(std::string str, int const* p, int n);
void print(std::string str, size_t const* p, int n);
void print(std::string str, long const* p, int n);


} // namespace kernels

#endif
