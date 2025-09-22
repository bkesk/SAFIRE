////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#include <cassert>
#include <complex>
#include <cuda.h>
#include "curand.h"
#include <cuda_runtime.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#include "Numerics/detail/CUDA/Kernels/zero_complex_part.cuh"
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{

void sampleGaussianRNG(double* V, int n, curandGenerator_t& gen)
{
  qmc_cuda::curand_check(curandGenerateNormalDouble(gen, V, n, 0.0, 1.0), "curandGenerateNormalDouble");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

// Convert to double if really necessary
void sampleGaussianRNG(float* V, int n, curandGenerator_t& gen)
{
  qmc_cuda::curand_check(curandGenerateNormal(gen, V, n, float(0.0), float(1.0)), "curandGenerateNormal");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void sampleGaussianRNG(std::complex<double>* V, int n, curandGenerator_t& gen)
{
  qmc_cuda::curand_check(curandGenerateNormalDouble(gen, reinterpret_cast<double*>(V), 2 * n, 0.0, 1.0),
                         "curandGenerateNormalDouble");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
  // hack hack hack!!!
  kernels::zero_complex_part(n, V);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void sampleGaussianRNG(std::complex<float>* V, int n, curandGenerator_t& gen)
{
  qmc_cuda::curand_check(curandGenerateNormal(gen, reinterpret_cast<float*>(V), 2 * n, float(0.0), float(1.0)),
                         "curandGenerateNormal");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
  // hack hack hack!!!
  kernels::zero_complex_part(n, V);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void sampleUniformRNG(double* V, int n, curandGenerator_t& gen)
{
  qmc_cuda::curand_check(curandGenerateUniformDouble(gen, V, n), "curandGenerateUniformDouble");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

// Convert to double if really necessary
void sampleUniformRNG(float* V, int n, curandGenerator_t& gen)
{
  qmc_cuda::curand_check(curandGenerateUniform(gen, V, n), "curandGenerateUniform");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void sampleUniformRNG(std::complex<double>* V, int n, curandGenerator_t& gen)
{
  qmc_cuda::curand_check(curandGenerateUniformDouble(gen, reinterpret_cast<double*>(V), 2 * n),
                         "curandGenerateUniformDouble");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
  // hack hack hack!!!
  kernels::zero_complex_part(n, V);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void sampleUniformRNG(std::complex<float>* V, int n, curandGenerator_t& gen)
{
  qmc_cuda::curand_check(curandGenerateUniform(gen, reinterpret_cast<float*>(V), 2 * n),
                         "curandGenerateUniform");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
  // hack hack hack!!!
  kernels::zero_complex_part(n, V);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

} // namespace kernels


