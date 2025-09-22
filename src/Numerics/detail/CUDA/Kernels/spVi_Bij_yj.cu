//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include <complex>
#include <thrust/complex.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#include "Memory/CUDA/cuda_utilities.h"
#include <thrust/system/cuda/detail/core/util.h>

namespace kernels
{

// sloppy!!!
template<typename I1, typename T1, typename T2, typename T3>
__global__ void kernel_spVi_Bij_yj(int nj, int nnz, I1 const* index, T1 const* values,
                 thrust::complex<T2> const* B, int ldb, thrust::complex<T3>* y, int incy)
{
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T3>, MAX_THREADS_PER_DIM> cache;
  int i              = threadIdx.x;
  int j              = blockIdx.x;
  cache[threadIdx.x] = thrust::complex<T3>(0.0);
  I1 ldb_ = static_cast<I1>(ldb);
  while( i<nnz ) {
    cache[threadIdx.x] += static_cast<thrust::complex<T3>>(B[index[i] * ldb_ + I1(j)]) * 
                          static_cast<thrust::complex<T3>>(values[i]);  
    i += blockDim.x;
  }

  __syncthreads(); // required because later on the current thread is accessing
                   // data written by another thread
  i = MAX_THREADS_PER_DIM / 2;
  while (i > 0)
  {
    if (threadIdx.x < i)
      cache[threadIdx.x] += cache[threadIdx.x + i];
    __syncthreads();
    i /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.x == 0)
    y[ j * incy ] += cache[0]; 
}

template<typename I1, typename T1, typename T2, typename T3>
__global__ void kernel_spVi_Bij_yj(int nj, int nnz, I1 const* index, thrust::complex<T1> const* values,
                 thrust::complex<T2> const* B, int ldb, thrust::complex<T3>* y, int incy)
{ 
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T3>, MAX_THREADS_PER_DIM> cache; 
  int i              = threadIdx.x;
  int j              = blockIdx.x;
  cache[threadIdx.x] = thrust::complex<T3>(0.0);
  I1 ldb_ = static_cast<I1>(ldb);
  while( i<nnz ) {
    cache[threadIdx.x] += static_cast<thrust::complex<T3>>(B[index[i] * ldb_ + I1(j)]) *
                          static_cast<thrust::complex<T3>>(values[i]);
    i += blockDim.x;
  }
  
  __syncthreads(); // required because later on the current thread is accessing
                   // data written by another thread
  i = MAX_THREADS_PER_DIM / 2;
  while (i > 0)
  { 
    if (threadIdx.x < i) 
      cache[threadIdx.x] += cache[threadIdx.x + i];
    __syncthreads();
    i /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.x == 0)
    y[ j * incy ] += cache[0];
}

template<typename I1, typename T1, typename T2, typename T3>
void spVi_Bij_yj(int nj, int nnz, I1 const* index, T1 const* values,
                 std::complex<T2> const* B, int ldb, std::complex<T3>* y, int incy)
{
  kernel_spVi_Bij_yj<<<nj, MAX_THREADS_PER_DIM>>>(nj,nnz,index,values,
                            reinterpret_cast<thrust::complex<T2> const*>(B),ldb,
                            reinterpret_cast<thrust::complex<T3>*>(y),incy);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<typename I1, typename T1, typename T2, typename T3>
void spVi_Bij_yj(int nj, int nnz, I1 const* index, std::complex<T1> const* values,
                 std::complex<T2> const* B, int ldb, std::complex<T3>* y, int incy)
{
  kernel_spVi_Bij_yj<<<nj, MAX_THREADS_PER_DIM>>>(nj,nnz,index,
                            reinterpret_cast<thrust::complex<T1> const*>(values),
                            reinterpret_cast<thrust::complex<T2> const*>(B),ldb,
                            reinterpret_cast<thrust::complex<T3>*>(y),incy);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

// template instantiations
template void spVi_Bij_yj(int, int, int const*, double const*, 
              std::complex<double> const*, int, std::complex<double>*, int);   
template void spVi_Bij_yj(int, int, int const*, float const*, 
              std::complex<float> const*, int, std::complex<double>*, int);   
template void spVi_Bij_yj(int, int, int const*, std::complex<double> const*, 
              std::complex<double> const*, int, std::complex<double>*, int);   
template void spVi_Bij_yj(int, int, int const*, std::complex<float> const*, 
              std::complex<float> const*, int, std::complex<double>*, int);   


} // kernels
