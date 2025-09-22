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
#include <thrust/complex.h>
#include <cuda_runtime.h>
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{
template<typename T>
__global__ void kernel_zero_complex_part(int n, thrust::complex<T>* x)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n)
    x[i] = thrust::complex<T>(x[i].real(), 0.0);
}

void zero_complex_part(int n, std::complex<double>* x)
{
  int block_dim = 256;
  int grid_dim  = (n + block_dim - 1) / block_dim;
  kernel_zero_complex_part<<<grid_dim, block_dim>>>(n, reinterpret_cast<thrust::complex<double>*>(x));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void zero_complex_part(int n, std::complex<float>* x)
{
  int block_dim = 256;
  int grid_dim  = (n + block_dim - 1) / block_dim;
  kernel_zero_complex_part<<<grid_dim, block_dim>>>(n, reinterpret_cast<thrust::complex<float>*>(x));
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

void zero_complex_part(int n, double* x) { return; }

void zero_complex_part(int n, float* x) { return; }

} // namespace kernels
