//////////////////////////////////////////////////////////////////////
// File created by:
// Miguel A. Morales
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

namespace kernels
{

// for i in [0,N), y[incy*i] = y[incy*i] + alpha sum_j^{0,M} opA(A)[i,j] * opB(B)[i,j]
template<typename T, typename Q1, typename Q2>
__global__ void kernel_dot(int ncycles, char TA, char TB, int N, int M,
                           thrust::complex<T> const alpha,
                           thrust::complex<Q1> const* A, int lda,
                           thrust::complex<Q2> const* B, int ldb,
                           thrust::complex<T>* y, int incy)
{
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T>, DOT_BLOCK_SIZE> cache;
  int i              = blockIdx.y;
  int j              = blockIdx.x*ncycles*blockDim.x + threadIdx.x;
  int jM             = min( (blockIdx.x+1)*ncycles*blockDim.x, M);
  cache[threadIdx.x] = thrust::complex<T>(0.0, 0.0);
  using T1 = typename thrust::complex<T>;

  // 
  bool cA(TA == 'H' || TA == 'C');
  bool cB(TB == 'H' || TB == 'C');
  bool tA(TA == 'H' || TA == 'T');
  bool tB(TB == 'H' || TB == 'T');

  auto an(A); 
  auto bn(B); 
  int sA(0); // stride of A
  int sB(0); // stride of B
  if (tA) {
    sA = lda*blockDim.x; 
    an += (j * lda + i);
  } else {
    sA = blockDim.x;   
    an += (i * lda + j);  
  }
  if (tB) {
    sB = ldb*blockDim.x;
    bn += (j * ldb + i);
  } else {
    sB = blockDim.x;
    bn += (i * ldb + j);
  }

  if (cA && cB)
  { 
    for (; j < jM; j+=blockDim.x, an+=sA, bn+=sB) 
      cache[threadIdx.x] += conj(static_cast<T1>(*an)) * conj(static_cast<T1>(*bn));
  }
  else if (cA && not cB)
  { 
    for (; j < jM; j+=blockDim.x, an+=sA, bn+=sB) 
      cache[threadIdx.x] += conj(static_cast<T1>(*an)) * static_cast<T1>(*bn);
  }
  else if (not cA && cB)
  { 
    for (; j < jM; j+=blockDim.x, an+=sA, bn+=sB) 
      cache[threadIdx.x] += static_cast<T1>(*an) * conj(static_cast<T1>(*bn));
  }
  else
  { 
    for (; j < jM; j+=blockDim.x, an+=sA, bn+=sB) 
      cache[threadIdx.x] += static_cast<T1>(*an) * static_cast<T1>(*bn);
  }

  __syncthreads(); // required because later on the current thread is accessing
                   // data written by another thread
  j = DOT_BLOCK_SIZE / 2;
  while (j > 0)
  {
    if (threadIdx.x < j)
      cache[threadIdx.x] += cache[threadIdx.x + j];
    __syncthreads();
    j /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.x == 0)
  {
    T re   = (alpha * cache[0]).real();
    T im   = (alpha * cache[0]).imag();
    T* re_ = reinterpret_cast<T*>(y + i * incy);
    atomicAdd(re_, re);
    atomicAdd(re_ + 1, im);
  }
}

// for i in [0,N), y[incy*i] = y[incy*i] + alpha sum_j^{0,M} opA(A)[i,j] * opB(B)[i,j]
template<class T, class Q1, class Q2>
void strided_batched_dot(char TA, char TB, int N, int M, 
                std::complex<T> const alpha, std::complex<Q1> const* A, int lda,
                std::complex<Q2> const* B, int ldb, 
                std::complex<T> *y, int incy)
{
   assert(TA == 'N' or TA == 'T' or TA == 'H' or TA == 'C');
   assert(TB == 'N' or TB == 'T' or TB == 'H' or TB == 'C');
  int ncycles = 4;
  int terms_per_block = ncycles * DOT_BLOCK_SIZE;
  // number of blocks needed in each dimension
  int mblocks = ( M + terms_per_block - 1 ) / terms_per_block;
  dim3 grid_dim(mblocks, N, 1);
  kernel_dot<<<grid_dim, DOT_BLOCK_SIZE>>>(ncycles, TA,TB,N,M, 
                                    static_cast<thrust::complex<T> const>(alpha),
                                    reinterpret_cast<thrust::complex<Q1> const*>(A), lda,
                                    reinterpret_cast<thrust::complex<Q2> const*>(B), ldb,
                                    reinterpret_cast<thrust::complex<T>*>(y), incy);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

// C[b,i] = C[b,i] + sum_j alpha * op(A[b])[i,j] * op(B[b])[i,j]
template<class T, class Q1, class Q2>
void strided_batched_dot(char TA, char TB, int nbatch, int N, int M,
                std::complex<T> const alpha, std::complex<Q1> const* A, int lda, long Astride,
                std::complex<Q2> const* B, int ldb, long Bstride,
                std::complex<T> *C, int ldc, long Cstride)
{
   assert(TA == 'N' or TA == 'T' or TA == 'H' or TA == 'C');
   assert(TB == 'N' or TB == 'T' or TB == 'H' or TB == 'C');
  int ncycles = 4;
  int terms_per_block = ncycles * DOT_BLOCK_SIZE;
  // number of blocks needed in each dimension
  int mblocks = ( M + terms_per_block - 1 ) / terms_per_block;
  dim3 grid_dim(mblocks, N, 1);
  for( int b=0; b<nbatch; ++b) {
    kernel_dot<<<grid_dim, DOT_BLOCK_SIZE>>>(ncycles, TA,TB,N,M,
                            static_cast<thrust::complex<T> const>(alpha),
                            reinterpret_cast<thrust::complex<Q1> const*>(A+b*Astride), lda,
                            reinterpret_cast<thrust::complex<Q2> const*>(B+b*Bstride), ldb,
                            reinterpret_cast<thrust::complex<T>*>(C+b*Cstride), ldc);
  }
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

// template specializations
template void strided_batched_dot(char TA, char TB, int N, int M,
                std::complex<double> const alpha, std::complex<double> const* A, int lda,
                std::complex<double> const* B, int ldb, 
                std::complex<double> *y, int incy);
template void strided_batched_dot(char TA, char TB, int N, int M,
                std::complex<double> const alpha, std::complex<float> const* A, int lda,
                std::complex<float> const* B, int ldb, 
                std::complex<double> *y, int incy);
template void strided_batched_dot(char TA, char TB, int nbatch, int N, int M,
                std::complex<double> const alpha, std::complex<double> const* A, int lda, long,
                std::complex<double> const* B, int ldb, long,
                std::complex<double> *C, int, long);
template void strided_batched_dot(char TA, char TB, int nbatch, int N, int M, 
                std::complex<double> const alpha, std::complex<float> const* A, int lda, long,
                std::complex<float> const* B, int ldb, long,
                std::complex<double> *C, int, long );
template void strided_batched_dot(char TA, char TB, int nbatch, int N, int M,
                std::complex<float> const alpha, std::complex<float> const* A, int lda, long,
                std::complex<float> const* B, int ldb, long,
                std::complex<float> *C, int, long);

} // namespace kernels
