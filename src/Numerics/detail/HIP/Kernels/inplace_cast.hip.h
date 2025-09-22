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
