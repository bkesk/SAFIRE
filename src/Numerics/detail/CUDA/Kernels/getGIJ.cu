
#include <complex>
#include <thrust/complex.h>
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#include "Memory/CUDA/cuda_utilities.h"
#include <thrust/system/cuda/detail/core/util.h>

namespace kernels
{

template<typename T, typename Q1, typename Q2>
__global__ void kernel_getGIJ(int nw, int nIJ, int nspin, int M, int nel_a, int nel_b,
        thrust::complex<Q1> const* Aup, int ldau, thrust::complex<Q1> const* Adn, int ldad,
        thrust::complex<Q2> const* B, int ldb, long strideB, thrust::complex<T>* C, int ldc, size_t const* n2IJ)
{
  __shared__ thrust::cuda_cub::core::uninitialized_array<thrust::complex<T>, DOT_BLOCK_SIZE> cache;
  int n              = blockIdx.x;
  int iw             = blockIdx.y; 
  int a              = threadIdx.x;
  cache[threadIdx.x] = thrust::complex<T>(0.0);

  int In = int(n2IJ[n]/size_t(M));
  int Jn = int(n2IJ[n]%size_t(M));
  int NEL = nel_a + ( nspin==2 ? nel_b : 0 );
  //  C[n][w] = sum_a A[I[n]][a] * G[w][a][J[n]]
  if( nspin == 2 and In >= M ) { 
    // beta
    auto A_(Adn + (In-M)*ldad + a);
    auto B_(B + iw*strideB + ( nel_a + a )*ldb + Jn);
    for(; a<nel_b; a+=blockDim.x, A_+=blockDim.x, B_+=blockDim.x*ldb)
      cache[threadIdx.x] += static_cast<thrust::complex<T>>(*A_) * 
                            static_cast<thrust::complex<T>>(*B_);
  } else {
    // alpha
    auto A_(Aup + In*ldau + a);
    auto B_(B + iw*strideB + a*ldb + Jn);
    for(; a<nel_a; a+=blockDim.x, A_+=blockDim.x, B_+=blockDim.x*ldb)
      cache[threadIdx.x] += static_cast<thrust::complex<T>>(*A_) * 
                            static_cast<thrust::complex<T>>(*B_);
  }

  __syncthreads(); // required because later on the current thread is accessing
                   // data written by another thread
  a = DOT_BLOCK_SIZE / 2;
  while (a > 0)
  {
    if (threadIdx.x < a)
      cache[threadIdx.x] += cache[threadIdx.x + a];
    __syncthreads();
    a /= 2; //not sure bitwise operations are actually faster
  }
  if (threadIdx.x == 0)
    C[ n*ldc + iw ] = cache[0];
}

//  C[n][w] = sum_a A[I[n]][a] * G[w][a][J[n]]
template<typename T, typename Q1, typename Q2>
void getGIJ_impl(int nw, int nIJ, int nspin, int M, int nel_a, int nel_b,
        std::complex<Q1> const* Aup, int ldau, std::complex<Q1> const* Adn, int ldad, 
        std::complex<Q2> const* B, int ldb, long strideB, 
        std::complex<T>* C, int ldc, size_t const* n2IJ)
{
  dim3 grid_dim(nIJ, nw, 1);
  kernel_getGIJ<<<grid_dim, DOT_BLOCK_SIZE>>>(nw, nIJ, nspin, M, nel_a, nel_b,
                            reinterpret_cast<thrust::complex<Q1> const*>(Aup),ldau,
                            reinterpret_cast<thrust::complex<Q1> const*>(Adn),ldad,
                            reinterpret_cast<thrust::complex<Q2> const*>(B),ldb,strideB,
                            reinterpret_cast<thrust::complex<T>*>(C),ldc,
                            n2IJ);
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}


// template instantiations
template void getGIJ_impl(int, int, int, int, int, int, 
    std::complex<double> const*, int, std::complex<double> const*,int, 
    std::complex<double> const*, int, long, std::complex<double> *, int, size_t const*);
template void getGIJ_impl(int, int, int, int, int, int,
    std::complex<double> const*, int, std::complex<double> const*, int, 
    std::complex<double> const*, int, long, std::complex<float> *, int, size_t const*);
template void getGIJ_impl(int, int, int, int, int, int,
    std::complex<double> const*, int, std::complex<double> const*, int, 
    std::complex<float> const*, int, long, std::complex<double> *, int, size_t const*);
template void getGIJ_impl(int, int, int, int, int, int,
    std::complex<double> const*, int, std::complex<double> const*, int, 
    std::complex<float> const*, int, long, std::complex<float> *, int, size_t const*);
template void getGIJ_impl(int, int, int, int, int, int,
    std::complex<float> const*, int, std::complex<float> const*, int, 
    std::complex<float> const*, int, long, std::complex<double> *, int, size_t const*);
template void getGIJ_impl(int, int, int, int, int, int,
    std::complex<float> const*, int, std::complex<float> const*, int, 
    std::complex<double> const*, int, long, std::complex<float> *, int, size_t const*);
template void getGIJ_impl(int, int, int, int, int, int,
    std::complex<float> const*, int, std::complex<float> const*, int, 
    std::complex<float> const*, int, long, std::complex<float> *, int, size_t const*);
template void getGIJ_impl(int, int, int, int, int, int,
    std::complex<float> const*, int, std::complex<float> const*, int,
    std::complex<double> const*, int, long, std::complex<double> *, int, size_t const*);


} // kernels
