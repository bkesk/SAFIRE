//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include <complex>
#include <thrust/complex.h>
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{

template<typename T>
__global__ void kernel_add_scalar(int N, int M, T a, T* A ,int lda)
{
  int i0 ( blockIdx.x*4*blockDim.x + threadIdx.x );
  int j0 ( blockIdx.y*4*blockDim.y + threadIdx.y );
  int const iN ( min(i0+4*blockDim.x, N) );
  int const jN ( min(j0+4*blockDim.y, M) );

  // A[i,j] += a 
  for (int i = i0; i < iN; i += blockDim.x) 
    for (int j = j0; j < jN; j += blockDim.y)
      A[i*lda + j] += a;
}

template<typename T>
__global__ void kernel_add_scalar(int N, int M, thrust::complex<T> a, 
                                  thrust::complex<T>* A ,int lda)
{
  int i0 ( blockIdx.x*4*blockDim.x + threadIdx.x );
  int j0 ( blockIdx.y*4*blockDim.y + threadIdx.y );
  int const iN ( min(i0+4*blockDim.x, N) );
  int const jN ( min(j0+4*blockDim.y, M) );

  // A[i,j] += a 
  for (int i = i0; i < iN; i += blockDim.x)
    for (int j = j0; j < jN; j += blockDim.y)
      A[i*lda + j] += a;
}

template<class T>
void add_scalar(int N, int M, T a, T* A ,int lda)
{
  // each block works on a sub matrix of size (4*8, 4*8)
  dim3 block_dim(8, 8, 1);
  // number of blocks needed in each dimension
  int xgrid_dim = ( N + 31 ) / 32;
  int ygrid_dim = ( M + 31 ) / 32;
  dim3 grid_dim(xgrid_dim, ygrid_dim, 1);
  kernel_add_scalar<<<grid_dim, block_dim>>>(N, M, a, A, lda);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template<class T>
void add_scalar(int N, int M, std::complex<T> a, std::complex<T>* A ,int lda)
{
  // each block works on a sub matrix of size (4*8, 4*8)
  dim3 block_dim(8, 8, 1);
  // number of blocks needed in each dimension
  int xgrid_dim = ( N + 31 ) / 32;
  int ygrid_dim = ( M + 31 ) / 32;
  dim3 grid_dim(xgrid_dim, ygrid_dim, 1);
  kernel_add_scalar<<<grid_dim, block_dim>>>(N, M, 
                    static_cast<thrust::complex<T>>(a), 
                    reinterpret_cast<thrust::complex<T>*>(A), lda);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

// template instantiations
template void add_scalar(int N, int M, double a, double* A ,int lda);
template void add_scalar(int N, int M, float a, float* A ,int lda);
template void add_scalar(int N, int M, std::complex<double> a, std::complex<double>* A ,int lda);
template void add_scalar(int N, int M, std::complex<float>a, std::complex<float>* A ,int lda);


} // kernels
