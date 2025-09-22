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
