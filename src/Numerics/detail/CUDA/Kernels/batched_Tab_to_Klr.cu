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
// Tab [nbatch][nwalk][nocc][nocc][nchol]
// Klr [nwalk][nchol]
template<typename T>
__global__ void kernel_batched_Tab_to_Klr(int nterms,
                                          int nwalk,
                                          int nocc,
                                          int nchol_max,
                                          int ncholQ,
                                          int* kdiag,
                                          thrust::complex<T> const* Tab,
                                          thrust::complex<T>* Kl, int ldkl,
                                          thrust::complex<T>* Kr, int ldkr)
{
  int w = blockIdx.x;
  if (blockIdx.y == 0)
  {
    for (int k = 0; k < nterms; k++)
    {
      int batch = kdiag[k];
      if (w < nwalk)
      {
        for (int a = 0; a < nocc; a++)
        {
          thrust::complex<T> const* Tba_(Tab + batch * nwalk * nocc * nocc * nchol_max +
                                         ((w * nocc + a) * nocc + a) * nchol_max);
          thrust::complex<T>* Kr_(Kr + w * ldkr);
          int c = threadIdx.x;
          while (c < ncholQ)
          {
            Kr_[c] += Tba_[c];
            c += blockDim.x;
          }
        }
      }
    }
  }
  else if (blockIdx.y == 1)
  {
    for (int k = 0; k < nterms; k++)
    {
      int batch = kdiag[k];
      if (w < nwalk)
      {
        for (int a = 0; a < nocc; a++)
        {
          thrust::complex<T> const* Tab_(Tab + (batch + 1) * nwalk * nocc * nocc * nchol_max +
                                         ((w * nocc + a) * nocc + a) * nchol_max);
          thrust::complex<T>* Kl_(Kl + w * ldkl);
          int c = threadIdx.x;
          while (c < ncholQ)
          {
            Kl_[c] += Tab_[c];
            c += blockDim.x;
          }
        }
      }
    }
  }
}

template<typename T>
__global__ void kernel_batched_Tanb_to_Klr(int nterms,
                                           int nwalk,
                                           int nocc,
                                           int nchol_max,
                                           int ncholQ,
                                           int* kdiag,
                                           thrust::complex<T> const* Tab,
                                           thrust::complex<T>* Kl, int ldkl,
                                           thrust::complex<T>* Kr, int ldkr)
{
  int w = blockIdx.x;
  if (blockIdx.y == 0)
  {
    for (int k = 0; k < nterms; k++)
    {
      int batch = kdiag[k];
      if (w < nwalk)
      {
        for (int a = 0; a < nocc; a++)
        {
          thrust::complex<T> const* Tba_(Tab + batch * nwalk * nocc * nocc * nchol_max +
                                         ((w * nocc + a) * nocc) * nchol_max + a);
          thrust::complex<T>* Kr_(Kr + w * ldkr);
          int c = threadIdx.x;
          while (c < ncholQ)
          {
            Kr_[c] += Tba_[c * nocc];
            c += blockDim.x;
          }
        }
      }
    }
  }
  else if (blockIdx.y == 1)
  {
    for (int k = 0; k < nterms; k++)
    {
      int batch = kdiag[k];
      if (w < nwalk)
      {
        for (int a = 0; a < nocc; a++)
        {
          thrust::complex<T> const* Tab_(Tab + (batch + 1) * nwalk * nocc * nocc * nchol_max +
                                         ((w * nocc + a) * nocc) * nchol_max + a);
          thrust::complex<T>* Kl_(Kl + w * ldkl);
          int c = threadIdx.x;
          while (c < ncholQ)
          {
            Kl_[c] += Tab_[c * nocc];
            c += blockDim.x;
          }
        }
      }
    }
  }
}

void batched_Tab_to_Klr(int nterms,
                        int nwalk,
                        int nocc,
                        int nchol_max,
                        int ncholQ,
                        int* kdiag,
                        std::complex<double> const* Tab,
                        std::complex<double>* Kl, int ldkl,
                        std::complex<double>* Kr, int ldkr)
{
  dim3 grid_dim(nwalk, 2, 1);
  int nthr = std::min(256, ncholQ); // is this needed?
  kernel_batched_Tab_to_Klr<<<grid_dim, nthr>>>(nterms, nwalk, nocc, nchol_max, ncholQ, kdiag,
                                                reinterpret_cast<thrust::complex<double> const*>(Tab),
                                                reinterpret_cast<thrust::complex<double>*>(Kl),ldkl,
                                                reinterpret_cast<thrust::complex<double>*>(Kr),ldkr);
  qmc_cuda::cuda_check(cudaGetLastError(), "batched_Tab_to_Klr");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "batched_Tab_to_Klr");
}

void batched_Tab_to_Klr(int nterms,
                        int nwalk,
                        int nocc,
                        int nchol_max,
                        int ncholQ,
                        int* kdiag,
                        std::complex<float> const* Tab,
                        std::complex<float>* Kl, int ldkl,
                        std::complex<float>* Kr, int ldkr)
{
  dim3 grid_dim(nwalk, 2, 1);
  int nthr = std::min(256, ncholQ); // is this needed?
  kernel_batched_Tab_to_Klr<<<grid_dim, nthr>>>(nterms, nwalk, nocc, nchol_max, ncholQ, kdiag,
                                                reinterpret_cast<thrust::complex<float> const*>(Tab),
                                                reinterpret_cast<thrust::complex<float>*>(Kl),ldkl,
                                                reinterpret_cast<thrust::complex<float>*>(Kr),ldkr);
  qmc_cuda::cuda_check(cudaGetLastError(), "batched_Tab_to_Klr");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "batched_Tab_to_Klr");
}

void batched_Tanb_to_Klr(int nterms,
                         int nwalk,
                         int nocc,
                         int nchol_max,
                         int ncholQ,
                         int* kdiag,
                         std::complex<double> const* Tab,
                         std::complex<double>* Kl, int ldkl,
                         std::complex<double>* Kr, int ldkr)
{
  dim3 grid_dim(nwalk, 2, 1);
  int nthr = std::min(256, ncholQ); // is this needed?
  kernel_batched_Tanb_to_Klr<<<grid_dim, nthr>>>(nterms, nwalk, nocc, nchol_max, ncholQ, kdiag,
                                                 reinterpret_cast<thrust::complex<double> const*>(Tab),
                                                 reinterpret_cast<thrust::complex<double>*>(Kl),ldkl,
                                                 reinterpret_cast<thrust::complex<double>*>(Kr),ldkr);
  qmc_cuda::cuda_check(cudaGetLastError(), "batched_Tanb_to_Klr");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "batched_Tanb_to_Klr");
}

void batched_Tanb_to_Klr(int nterms,
                         int nwalk,
                         int nocc,
                         int nchol_max,
                         int ncholQ,
                         int* kdiag,
                         std::complex<float> const* Tab,
                         std::complex<float>* Kl, int ldkl,
                         std::complex<float>* Kr, int ldkr)
{
  dim3 grid_dim(nwalk, 2, 1);
  int nthr = std::min(256, ncholQ); // is this needed?
  kernel_batched_Tanb_to_Klr<<<grid_dim, nthr>>>(nterms, nwalk, nocc, nchol_max, ncholQ, kdiag,
                                                 reinterpret_cast<thrust::complex<float> const*>(Tab),
                                                 reinterpret_cast<thrust::complex<float>*>(Kl),ldkl,
                                                 reinterpret_cast<thrust::complex<float>*>(Kr),ldkr);
  qmc_cuda::cuda_check(cudaGetLastError(), "batched_Tanb_to_Klr");
  qmc_cuda::cuda_check(cudaDeviceSynchronize(), "batched_Tanb_to_Klr");
}

} // namespace kernels
