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

#ifndef COPY_N_CAST_KERNELS_HPP
#define COPY_N_CAST_KERNELS_HPP

#include <stdexcept>
#include <cassert>
#include <complex>
#include <iostream>

namespace kernels
{
void copy_n_cast(double const* A, int n, float* B);
void copy_n_cast(float const* A, int n, double* B);
void copy_n_cast(std::complex<double> const* A, int n, std::complex<float>* B);
void copy_n_cast(std::complex<float> const* A, int n, std::complex<double>* B);
inline void copy_n_cast(std::complex<float> const* A, int n, std::complex<float>* B)
{
  std::cerr << " Should not be calling copy_n_cast<T,T>. \n" << std::endl;
  throw std::runtime_error("Calling cast_n_copy(float const*,n,float*)");
}
inline void copy_n_cast(std::complex<double> const* A, int n, std::complex<double>* B)
{
  std::cerr << " Should not be calling copy_n_cast<T,T>. \n" << std::endl;
  throw std::runtime_error("Calling cast_n_copy(double const*,n,double*)");
}

} // namespace kernels

#endif
