
#include <cassert>
#include <complex>
#include <cuda.h>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{

template<typename T>
__global__ void kernel_conjugate_impl(int N, int M, 
                                      thrust::complex<T>* A, int lda, long stride, int nb)
{
  if( blockIdx.z < nb and blockIdx.y < N) {
    long M0(4 * blockDim.x * blockIdx.x);
    long M_(min(long(4 * blockDim.x), M - M0));
    A += blockIdx.z * stride + blockIdx.y * lda + M0;
    int ip = threadIdx.x;
    while(ip < M_) {
      A[ip] = conj(A[ip]);
      ip += blockDim.x;  
    }
  }
}

template<typename T>
void complex_conjugate_impl(int N, int M, std::complex<T>* A, int lda, long stride, int nb)
{
  int M_(4 * DEFAULT_BLOCK_SIZE);
  int nblk((M + M_ - 1) / M_);
  dim3 grid_dim(nblk, N, nb);
  kernel_conjugate_impl<<<grid_dim, DEFAULT_BLOCK_SIZE>>>(N, M,
                        reinterpret_cast<thrust::complex<T>*>(A), lda, stride, nb);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

template void complex_conjugate_impl(int,int,std::complex<double>*,int,long,int);
template void complex_conjugate_impl(int,int,std::complex<float>*,int,long,int);

} // namespace kernels
