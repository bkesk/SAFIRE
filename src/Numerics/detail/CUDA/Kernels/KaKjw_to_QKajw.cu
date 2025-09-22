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
#include "Numerics/detail/CUDA/Kernels/cuda_settings.h"
#define ENABLE_CUDA 1
#include "Memory/CUDA/cuda_utilities.h"

namespace kernels
{
// very sloppy, needs improvement!!!!
// A[nocc_tot][nmo_tot][nwalk]
// B[Q][K][nocc_max][nmo_max][nwalk]
template<typename T, typename T2>
__global__ void kernel_KaKjw_to_QKajw(int nwalk,
                                      int nkpts,
                                      int npol,
                                      int nmo_max,
                                      int nmo_tot,
                                      int nocc_max,
                                      int* nmo,
                                      int* nmo0,
                                      int* nocc,
                                      int* nocc0,
                                      int* QKtok2,
                                      T const* A,
                                      T2* B)
{
  int Q   = blockIdx.x;
  int K   = blockIdx.y;
  int pol = blockIdx.z;
  if (Q >= nkpts || K >= nkpts || pol > npol)
    return;
  int QK  = QKtok2[Q * nkpts + K];
  int na0 = nocc0[K];
  int nj0 = nmo0[QK];
  int na  = nocc[K];
  int nj  = nmo[QK];

  T const* A_(A + (na0 * npol * nmo_tot + nj0) * nwalk);
  T2* B_(B + ((Q * nkpts + K) * nocc_max) * npol * nmo_max * nwalk);

  if (threadIdx.x >= nj)
    return;
  if (threadIdx.y >= nwalk)
    return;

  for (int a = 0, a0 = pol * nmo_max * nwalk, a1 = pol * nmo_tot * nwalk; a < na;
       a++, a0 += npol * nmo_max * nwalk, a1 += npol * nmo_tot * nwalk)
    for (int j = threadIdx.x; j < nj; j += blockDim.x)
      for (int n = threadIdx.y; n < nwalk; n += blockDim.y)
        B_[a0 + j * nwalk + n] = static_cast<T2>(A_[a1 + j * nwalk + n]);
}

template<typename T, typename T2>
__global__ void kernel_KaKjw_to_QKajw(int nwalk,
                                      int nkpts,
                                      int npol,
                                      int nmo_max,
                                      int nmo_tot,
                                      int nocc_max,
                                      int* nmo,
                                      int* nmo0,
                                      int* nocc,
                                      int* nocc0,
                                      int* QKtok2,
                                      thrust::complex<T> const* A,
                                      thrust::complex<T2>* B)
{
  int Q   = blockIdx.x;
  int K   = blockIdx.y;
  int pol = blockIdx.z;
  if (Q >= nkpts || K >= nkpts || pol > npol)
    return;
  int QK  = QKtok2[Q * nkpts + K];
  int na0 = nocc0[K];
  int nj0 = nmo0[QK];
  int na  = nocc[K];
  int nj  = nmo[QK];

  thrust::complex<T> const* A_(A + (na0 * npol * nmo_tot + nj0) * nwalk);
  thrust::complex<T2>* B_(B + ((Q * nkpts + K) * nocc_max) * npol * nmo_max * nwalk);

  if (threadIdx.x >= nj)
    return;
  if (threadIdx.y >= nwalk)
    return;

  for (int a = 0, a0 = pol * nmo_max * nwalk, a1 = pol * nmo_tot * nwalk; a < na;
       a++, a0 += npol * nmo_max * nwalk, a1 += npol * nmo_tot * nwalk)
  {
    for (int j = threadIdx.x; j < nj; j += blockDim.x)
      for (int n = threadIdx.y; n < nwalk; n += blockDim.y)
        B_[a0 + j * nwalk + n] = static_cast<thrust::complex<T2>>(A_[a1 + j * nwalk + n]);
  }
}

template<typename T1, typename T2>
void KaKjw_to_QKajw(int nwalk,
                    int nkpts,
                    int npol,
                    int nmo_max,
                    int nmo_tot,
                    int nocc_max,
                    int* nmo,
                    int* nmo0,
                    int* nocc,
                    int* nocc0,
                    int* QKtok2,
                    T1 const* A,
                    T2* B)
{
  int xblock_dim = 16;
  int yblock_dim = std::min(nwalk, 32);
  dim3 block_dim(xblock_dim, yblock_dim, 1);
  dim3 grid_dim(nkpts, nkpts, npol);
  kernel_KaKjw_to_QKajw<<<grid_dim, block_dim>>>(nwalk, nkpts, npol, nmo_max, nmo_tot, nocc_max, nmo, nmo0, nocc, nocc0,
                                                 QKtok2, A, B);
  qmc_cuda::cuda_check(cudaGetLastError(), "KaKjw_to_QKajw");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "KaKjw_to_QKajw");
}

template<typename T1, typename T2>
void KaKjw_to_QKajw(int nwalk,
                    int nkpts,
                    int npol,
                    int nmo_max,
                    int nmo_tot,
                    int nocc_max,
                    int* nmo,
                    int* nmo0,
                    int* nocc,
                    int* nocc0,
                    int* QKtok2,
                    std::complex<T1> const* A,
                    std::complex<T2>* B)
{
  int xblock_dim = 16;
  int yblock_dim = std::min(nwalk, 32);
  dim3 block_dim(xblock_dim, yblock_dim, 1);
  dim3 grid_dim(nkpts, nkpts, npol);
  kernel_KaKjw_to_QKajw<<<grid_dim, block_dim>>>(nwalk, nkpts, npol, nmo_max, nmo_tot, nocc_max, nmo, nmo0, nocc, nocc0,
                                                 QKtok2, reinterpret_cast<thrust::complex<T1> const*>(A),
                                                 reinterpret_cast<thrust::complex<T2>*>(B));
  qmc_cuda::cuda_check(cudaGetLastError(), "KaKjw_to_QKajw");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "KaKjw_to_QKajw");
}

template void KaKjw_to_QKajw(int,int,int,int,int,int,int*,int*,int*,int*,int*,
			     float const*, float *);
template void KaKjw_to_QKajw(int,int,int,int,int,int,int*,int*,int*,int*,int*,
			     double const*, float *);
template void KaKjw_to_QKajw(int,int,int,int,int,int,int*,int*,int*,int*,int*,
			     double const*, double *);
template void KaKjw_to_QKajw(int,int,int,int,int,int,int*,int*,int*,int*,int*,
			     float const*, double *);

template void KaKjw_to_QKajw(int,int,int,int,int,int,int*,int*,int*,int*,int*,
			     std::complex<float> const*, std::complex<float> *);
template void KaKjw_to_QKajw(int,int,int,int,int,int,int*,int*,int*,int*,int*,
			     std::complex<double> const*, std::complex<float> *);
template void KaKjw_to_QKajw(int,int,int,int,int,int,int*,int*,int*,int*,int*,
			     std::complex<double> const*, std::complex<double> *);
template void KaKjw_to_QKajw(int,int,int,int,int,int,int*,int*,int*,int*,int*,
			     std::complex<float> const*, std::complex<double> *);

} // namespace kernels
