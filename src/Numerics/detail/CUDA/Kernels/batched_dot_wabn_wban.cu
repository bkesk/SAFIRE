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
#include <thrust/system/cuda/detail/core/util.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"
#if __CUDA_ARCH__ < 600
#include "Numerics/detail/CUDA/Kernels/cuda_workaround_legacy_hardware.cuh"
#endif

namespace kernels
{
// Tab [nbatch][nwalk][nocc][nocc][nchol]
template<typename T, typename T2>
__global__ void kernel_batched_dot_wabn_wban(int nbatch,
                                             int nwalk,
                                             int nocc,
                                             int nchol,
                                             thrust::complex<T2> const* alpha,
                                             thrust::complex<T2> const* Tab,
                                             thrust::complex<T>* y,
                                             int incy)
{
  int batch = blockIdx.y;
  int w     = blockIdx.z;
  if (batch >= nbatch or w >= nwalk or blockIdx.x >= nocc * nocc)
    return;
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T>, DOT_BLOCK_SIZE> cache;
  int nocc2 		 = nocc * nocc;
  int a                  = blockIdx.x / nocc;
  int b                  = blockIdx.x % nocc;
  int i                  = threadIdx.x;
  thrust::complex<T> alp = static_cast<thrust::complex<T>>(alpha[batch]);
  thrust::complex<T2> const* A_(Tab + 2 * batch * nwalk * nocc2 * nchol + ((w * nocc + a) * nocc + b) * nchol);
  thrust::complex<T2> const* B_(Tab + (2 * batch + 1) * nwalk * nocc2 * nchol + ((w * nocc + b) * nocc + a) * nchol);
  cache[threadIdx.x] = thrust::complex<T>(0.0);
  while (i < nchol)
  {
    cache[threadIdx.x] += static_cast<thrust::complex<T>>(A_[i] * B_[i]);
    i += blockDim.x;
  }
  __syncthreads(); // required because later on the current thread is accessing
                   // data written by another thread
  i = DOT_BLOCK_SIZE / 2;
  while (i > 0)
  {
    if (threadIdx.x < i)
      cache[threadIdx.x] += cache[threadIdx.x + i];
    __syncthreads();
    i /= 2; //not sure bitwise operations are actually faster
  }
  //if( threadIdx.x == 0 ) *(y+w*incy) = (*(y+w*incy)) + alp * cache[ 0 ];
  if (threadIdx.x == 0)
  {
    T re   = (alp * cache[0]).real();
    T im   = (alp * cache[0]).imag();
    T* re_ = reinterpret_cast<T*>(y + w * incy);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
  }
}

template<typename T, typename T2>
__global__ void kernel_batched_dot_wanb_wbna(int nbatch,
                                             int nwalk,
                                             int nocc,
                                             int nchol,
                                             thrust::complex<T2> const* alpha,
                                             thrust::complex<T2> const* Tab,
                                             thrust::complex<T>* y,
                                             int incy)
{
  int batch = blockIdx.y;
  int w     = blockIdx.z;
  if (batch >= nbatch or w >= nwalk or blockIdx.x >= nocc * nocc)
    return;
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T>, DOT_BLOCK_SIZE> cache;
  int nocc2              = nocc * nocc;
  int a                  = blockIdx.x / nocc;
  int b                  = blockIdx.x % nocc;
  int i                  = threadIdx.x;
  thrust::complex<T> alp = static_cast<thrust::complex<T>>(alpha[batch]);
  thrust::complex<T2> const* A_(Tab + 2 * batch * nwalk * nocc2 * nchol + ((w * nocc + a) * nocc) * nchol + b);
  thrust::complex<T2> const* B_(Tab + (2 * batch + 1) * nwalk * nocc2 * nchol + ((w * nocc + b) * nocc) * nchol + a);
  cache[threadIdx.x] = thrust::complex<T>(0.0);
  while (i < nchol)
  {
    cache[threadIdx.x] += static_cast<thrust::complex<T>>(A_[i * nocc] * B_[i * nocc]);
    i += blockDim.x;
  }
  __syncthreads(); // required because later on the current thread is accessing
                   // data written by another thread
  i = DOT_BLOCK_SIZE / 2;
  while (i > 0)
  {
    if (threadIdx.x < i)
      cache[threadIdx.x] += cache[threadIdx.x + i];
    __syncthreads();
    i /= 2; //not sure bitwise operations are actually faster
  }
  //if( threadIdx.x == 0 ) *(y+w*incy) = (*(y+w*incy)) + alp * cache[ 0 ];
  if (threadIdx.x == 0)
  {
    T re   = (alp * cache[0]).real();
    T im   = (alp * cache[0]).imag();
    T* re_ = reinterpret_cast<T*>(y + w * incy);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
  }
}

template<typename T1, typename T2>
void batched_dot_wabn_wban(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<T1> const* alpha,
                           std::complex<T1> const* Tab,
                           std::complex<T2>* y,
                           int incy)
{
  int n_ = nocc * nocc;
  dim3 grid_dim(n_, nbatch, nwalk);
  kernel_batched_dot_wabn_wban<<<grid_dim, DOT_BLOCK_SIZE>>>(nbatch, nwalk, nocc, nchol,
                                                             reinterpret_cast<thrust::complex<T1> const*>(alpha),
                                                             reinterpret_cast<thrust::complex<T1> const*>(Tab),
                                                             reinterpret_cast<thrust::complex<T2>*>(y), incy);
  qmc_cuda::cuda_check(cudaGetLastError(), "batched_dot_wabn_wban");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "batched_dot_wabn_wban");
}

// anb/bna
template<typename T1, typename T2>
void batched_dot_wanb_wbna(int nbatch,
                           int nwalk,
                           int nocc,
                           int nchol,
                           std::complex<T1> const* alpha,
                           std::complex<T1> const* Tab,
                           std::complex<T2>* y,
                           int incy)
{
  int n_ = nocc * nocc;
  dim3 grid_dim(n_, nbatch, nwalk);
  kernel_batched_dot_wanb_wbna<<<grid_dim, DOT_BLOCK_SIZE>>>(nbatch, nwalk, nocc, nchol,
                                                             reinterpret_cast<thrust::complex<T1> const*>(alpha),
                                                             reinterpret_cast<thrust::complex<T1> const*>(Tab),
                                                             reinterpret_cast<thrust::complex<T2>*>(y), incy);
  qmc_cuda::cuda_check(cudaGetLastError(), "batched_dot_wanb_wbna");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "batched_dot_wanb_wbna");
}

template void batched_dot_wabn_wban(int,int,int,int,std::complex<double> const*,
	std::complex<double> const*, std::complex<double> *, int);
template void batched_dot_wabn_wban(int,int,int,int,std::complex<double> const*,
	std::complex<double> const*, std::complex<float> *, int);
template void batched_dot_wabn_wban(int,int,int,int,std::complex<float> const*,
	std::complex<float> const*, std::complex<float> *, int);
template void batched_dot_wabn_wban(int,int,int,int,std::complex<float> const*,
	std::complex<float> const*, std::complex<double> *, int);

template void batched_dot_wanb_wbna(int,int,int,int,std::complex<double> const*,
	std::complex<double> const*, std::complex<double> *, int);
template void batched_dot_wanb_wbna(int,int,int,int,std::complex<double> const*,
	std::complex<double> const*, std::complex<float> *, int);
template void batched_dot_wanb_wbna(int,int,int,int,std::complex<float> const*,
	std::complex<float> const*, std::complex<float> *, int);
template void batched_dot_wanb_wbna(int,int,int,int,std::complex<float> const*,
	std::complex<float> const*, std::complex<double> *, int);

} // namespace kernels
