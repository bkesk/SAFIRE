////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the __SFQMC_LICENSE_TYPE__
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2025 SAFIRE Developers
//
////////////////////////////////////////////////////////////////////////////////


#define BOOST_NO_AUTO_PTR
#include <cassert>
#include <complex>
#include <type_traits>
#include <thrust/complex.h>
#define ENABLE_CUDA 1
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{

template<typename T> 
__global__ void kernel_fill_n(long N, T* x, long incx, T const a)
{
  long N0(16 * blockDim.x * blockIdx.x);
  long N_(min(long(16 * blockDim.x), N - N0));
  x += incx * N0;
  for (long ip = threadIdx.x; ip < N_; ip += blockDim.x)
  {
    x[ip * incx] = a;
  }
}

template<typename T>
__global__ void kernel_fill2D_n(long N, long M, T* y, long lda, T const a)
{
  if(blockIdx.y < N) {
    long M0(16 * blockDim.x * blockIdx.x);
    long M_(min(long(16 * blockDim.x), M - M0));
    y += blockIdx.y*lda + M0;
    for (long jp = threadIdx.x; jp < M_; jp += blockDim.x)
    {
      y[jp] = a;
    }
  }
}

// seems wasteful, but not sure how else to do it!
template<typename T1, typename T2, typename T3>
__global__ void kernel_fill_if_zero_impl(int N, int M, T1 const* key, int incx, T2 const alpha,
                        T3* A, long lda, long stride, int nb)
{
  if(blockIdx.y < N and blockIdx.z < nb) {
    T3 a = static_cast<T3>(alpha);
    if(key[blockIdx.z*incx] != T1(0)) return; 
    long M0(16 * blockDim.x * blockIdx.x);
    long M_(min(long(16 * blockDim.x), M - M0));
    A += blockIdx.z*stride + blockIdx.y*lda + M0;
    for (long jp = threadIdx.x; jp < M_; jp += blockDim.x)
    {
      A[jp] = a;
    }
  }
}

template<typename T1, typename T2, typename T3>
__global__ void kernel_fill_if_zero_impl(int N, int M, thrust::complex<T1> const* key, int incx, T2 const alpha,
                        T3* A, long lda, long stride, int nb)
{
  if(blockIdx.y < N and blockIdx.z < nb) {
    T3 a = static_cast<T3>(alpha);
    if(key[blockIdx.z*incx] != abs(thrust::complex<T1>(0.0))) return; 
    long M0(16 * blockDim.x * blockIdx.x);
    long M_(min(long(16 * blockDim.x), M - M0));
    A += blockIdx.z*stride + blockIdx.y*lda + M0;
    for (long jp = threadIdx.x; jp < M_; jp += blockDim.x)
    {
      A[jp] = a;
    }
  }
}

template<typename T1, typename T2, typename T3>
__global__ void kernel_fill_if_non_zero_impl(int N, int M, T1 const* key, int incx, T2 const alpha,
                        T3* A, long lda, long stride, int nb)
{ 
  if(blockIdx.y < N and blockIdx.z < nb) {
    T3 a = static_cast<T3>(alpha);
    if(key[blockIdx.z*incx] == T1(0.0)) return;
    long M0(16 * blockDim.x * blockIdx.x); 
    long M_(min(long(16 * blockDim.x), M - M0));
    A += blockIdx.z*stride + blockIdx.y*lda + M0;
    for (long jp = threadIdx.x; jp < M_; jp += blockDim.x)
    { 
      A[jp] = a;
    }
  }
}

template<typename T>
void fill_n(T* first, long N, long incx, T const value)
{
  long nthr(512);
  long N_(16 * nthr);
  long nblk((N + N_ - 1) / N_);
  kernel_fill_n<<<nblk, nthr>>>(N, first, incx, value);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<typename T>
void fill_n(T* first, long N, T const value)
{ 
  long nthr(512);
  long N_(16 * nthr);
  long nblk((N + N_ - 1) / N_);
  kernel_fill_n<<<nblk, nthr>>>(N, first, 1l, value);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<typename T>
void fill2D_n(long N, long M, T* A, long lda, T const value)
{
  if(N==0 or M==0) { return; }
  else if(N==1) { fill_n(A,M,value); }
  else if(M==lda) { fill_n(A,N*M,value); } 
  else
  {	
    // blocking over columns over, looping over rows, keeping it simple!
    long nthr(512);
    long M_(16 * nthr);
    long mblk((M + M_ - 1) / M_);
    long N_(0);
    while( N_ < N )
    {
      long Ni = std::min(N-N_, long(MAXIMUM_GRID_DIM_YZ));
      dim3 grid_dim(mblk, Ni, 1);   
      kernel_fill2D_n<<<grid_dim, nthr>>>(Ni, M, A+lda*N_, lda, value);
      N_ += Ni; 	
    }
    qmc_cuda::cuda_check(cudaGetLastError());
    qmc_cuda::cuda_check(cudaDeviceSynchronize());
  }
}

template<typename T1, typename T2, typename T3>
void fill_if_zero_impl(int nrow, int ncol, T1 const* key, int incx,
                T2 const alpha, T3* A, int lda, long stride, int nb)
{
  // this will fail if nrow or nb > MAXIMUM_GRID_DIM_YZ! FIX!
  long nthr(512);
  long M_(16 * nthr);
  long nblk((ncol + M_ - 1) / M_);
  dim3 grid_dim(nblk, nrow, nb);
  kernel_fill_if_zero_impl<<<grid_dim, nthr>>>(nrow, ncol, key, incx, alpha, A, lda, stride, nb);
  qmc_cuda::cuda_check(cudaGetLastError(), "kernels::fill_if_zero_impl");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "kernels::fill_if_zero_impl");
}

template<typename T1, typename T2, typename T3>
void fill_if_zero_impl(int nrow, int ncol, std::complex<T1> const* key, int incx,
                T2 const alpha, T3* A, int lda, long stride, int nb)
{
  // this will fail if nrow or nb > MAXIMUM_GRID_DIM_YZ! FIX!
  long nthr(512);
  long M_(16 * nthr);
  long nblk((ncol + M_ - 1) / M_);
  dim3 grid_dim(nblk, nrow, nb);
  kernel_fill_if_zero_impl<<<grid_dim, nthr>>>(nrow, ncol, 
                    reinterpret_cast<thrust::complex<T1> const*>(key), 
                    incx, alpha, A, lda, stride, nb);
  qmc_cuda::cuda_check(cudaGetLastError(), "kernels::fill_if_zero_impl");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "kernels::fill_if_zero_impl");
}

template<typename T1, typename T2, typename T3>
void fill_if_non_zero_impl(int nrow, int ncol, T1 const* key, int incx,
                T2 const alpha, T3* A, int lda, long stride, int nb)
{ 
  // this will fail if nrow or nb > MAXIMUM_GRID_DIM_YZ! FIX!
  long nthr(512);
  long M_(16 * nthr);
  long nblk((ncol + M_ - 1) / M_);
  dim3 grid_dim(nblk, nrow, nb);
  kernel_fill_if_non_zero_impl<<<grid_dim, nthr>>>(nrow, ncol, key, incx, alpha, A, lda, stride, nb);
  qmc_cuda::cuda_check(cudaGetLastError(), "kernels::fill_if_non_zero_impl");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "kernels::fill_if_non_zero_impl");
}

// template specializations
template void fill_n(char* first, long N, long stride, char const value);
template void fill_n(int* first, long N, long stride, int const value);
template void fill_n(unsigned int* first, long N, long stride, unsigned int const value);
template void fill_n(long* first, long N, long stride, long const value);
template void fill_n(unsigned long* first, long N, long stride, unsigned long const value);
template void fill_n(float* first, long N, long stride, float const value);
template void fill_n(double* first, long N, long stride, double const value);
template void fill_n(std::complex<float>* first, long N, long stride, std::complex<float> const value);
template void fill_n(std::complex<double>* first, long N, long stride, std::complex<double> const value);

template void fill_n(char* first, long N, char const value);
template void fill_n(int* first, long N, int const value);
template void fill_n(unsigned int* first, long N, unsigned int const value);
template void fill_n(long* first, long N, long const value);
template void fill_n(unsigned long* first, long N, unsigned long const value);
template void fill_n(float* first, long N, float const value);
template void fill_n(double* first, long N, double const value);
template void fill_n(std::complex<float>* first, long N, std::complex<float> const value);
template void fill_n(std::complex<double>* first, long N, std::complex<double> const value);

template void fill2D_n(long N, long M, int* A, long lda, int const value);
template void fill2D_n(long N, long M, long* A, long lda, long const value);
template void fill2D_n(long N, long M, unsigned int* A, long lda, unsigned int const value);
template void fill2D_n(long N, long M, unsigned long* A, long lda, unsigned long const value);
template void fill2D_n(long N, long M, float* A, long lda, float const value);
template void fill2D_n(long N, long M, double* A, long lda, double const value);
template void fill2D_n(long N, long M, std::complex<double>* A, long lda, std::complex<double> const value);
template void fill2D_n(long N, long M, std::complex<float>* A, long lda, std::complex<float> const value);

template void fill_if_zero_impl(int,int,std::complex<double> const*,int,std::complex<double> const, std::complex<double>*,int,long,int);
template void fill_if_zero_impl(int,int,std::complex<double> const*,int,std::complex<float> const, std::complex<float>*,int,long,int);
template void fill_if_zero_impl(int,int,std::complex<float> const*,int,std::complex<float> const, std::complex<float>*,int,long,int);
template void fill_if_zero_impl(int,int,std::complex<float> const*,int,std::complex<double> const, std::complex<double>*,int,long,int);

template void fill_if_zero_impl(int,int,double const*,int,std::complex<double> const, std::complex<double>*,int,long,int);
template void fill_if_zero_impl(int,int,double const*,int,std::complex<float> const, std::complex<float>*,int,long,int);
template void fill_if_zero_impl(int,int,float const*,int,std::complex<float> const, std::complex<float>*,int,long,int);
template void fill_if_zero_impl(int,int,float const*,int,std::complex<double> const, std::complex<double>*,int,long,int);

template void fill_if_non_zero_impl(int,int,int const*,int,std::complex<double> const, std::complex<double>*,int,long,int);
template void fill_if_non_zero_impl(int,int,int const*,int,std::complex<float> const, std::complex<float>*,int,long,int);


} // namespace kernels
