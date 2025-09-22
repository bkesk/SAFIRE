////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#ifndef SAMPLEGAUSSIANRNG_H
#define SAMPLEGAUSSIANRNG_H

#include <cassert>
#include <complex>
#include "curand.h"

namespace kernels
{
void sampleGaussianRNG(double* V, int n, curandGenerator_t& gen);
void sampleGaussianRNG(float* V, int n, curandGenerator_t& gen);
void sampleGaussianRNG(std::complex<double>* V, int n, curandGenerator_t& gen);
void sampleGaussianRNG(std::complex<float>* V, int n, curandGenerator_t& gen);

void sampleUniformRNG(double* V, int n, curandGenerator_t& gen);
void sampleUniformRNG(float* V, int n, curandGenerator_t& gen);
void sampleUniformRNG(std::complex<double>* V, int n, curandGenerator_t& gen);
void sampleUniformRNG(std::complex<float>* V, int n, curandGenerator_t& gen);
} // namespace kernels

#endif
