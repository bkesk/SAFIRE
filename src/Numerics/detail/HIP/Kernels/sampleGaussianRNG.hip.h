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

#ifndef SAMPLEGAUSSIANRNG_H
#define SAMPLEGAUSSIANRNG_H

#include <cassert>
#include <complex>
#include "rocrand/rocrand.h"

namespace kernels
{
void sampleGaussianRNG(double* V, int n, rocrand_generator& gen);
void sampleGaussianRNG(float* V, int n, rocrand_generator& gen);
void sampleGaussianRNG(std::complex<double>* V, int n, rocrand_generator& gen);
void sampleGaussianRNG(std::complex<float>* V, int n, rocrand_generator& gen);

} // namespace kernels

#endif
