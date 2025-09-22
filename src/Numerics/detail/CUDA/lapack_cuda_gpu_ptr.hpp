////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#ifndef LAPACK_GPU_HPP
#define LAPACK_GPU_HPP

#include <cassert>
#include "AFQMC/Utilities/type_conversion.hpp"
#include "Memory/custom_pointers.hpp"
#include "Memory/buffer_managers.h"
#include "Memory/arch.hpp"
#include "Numerics/detail/CUDA/cublas_wrapper.hpp"
#include "Numerics/detail/CUDA/cusolver_wrapper.hpp"
#include "Numerics/detail/CUDA/Kernels/setIdentity.cuh"

namespace ma
{
using sfqmc::afqmc::remove_complex;
using qmc_cuda::cuda_check;
using qmc_cuda::cusolver_check;

template<typename T, typename R, typename I>
inline static void hevr(char JOBZ,
                        char RANGE,
                        char UPLO,
                        int N,
                        device::device_pointer<T> A,
                        int LDA,
                        T VL,
                        T VU,
                        int IL,
                        int IU,
                        T ABSTOL,
                        int& M,
                        device::device_pointer<T> W,
                        device::device_pointer<T> Z,
                        int LDZ,
                        device::device_pointer<I> ISUPPZ,
                        device::device_pointer<T> WORK,
                        int& LWORK,
                        device::device_pointer<R> RWORK,
                        int& LRWORK,
                        device::device_pointer<I> IWORK,
                        int& LIWORK,
                        int& INFO,
			device_cuda_backend)
{
  throw std::runtime_error("Error: hevr not implemented in gpu.");
}

// getrf_bufferSize
template<typename T>
inline static void getrf_bufferSize(const int n, const int m, device::device_pointer<T> a, int lda, int& lwork, device_cuda_backend)
{
  cusolver::cusolver_getrf_bufferSize(arch::global_cusolverDn_handle, n, m, raw_pointer_cast(a), lda, &lwork);
}

template<typename T, typename R, typename I>
inline static void getrf(const int n,
                         const int m,
                         device::device_pointer<T>&& a,
                         int lda,
                         device::device_pointer<I>&& piv,
                         int& st,
                         device::device_pointer<R> work, 
			 device_cuda_backend)
{
  cusolverStatus_t status = cusolver::cusolver_getrf(arch::global_cusolverDn_handle, n, m, raw_pointer_cast(a), lda,
                                                     raw_pointer_cast(work), raw_pointer_cast(piv), raw_pointer_cast(piv) + n);
  arch::memcopy(&st, raw_pointer_cast(piv) + n, sizeof(int), arch::memcopyD2H);
  if (CUSOLVER_STATUS_SUCCESS != status)
  {
    std::cerr << " cublas_getrf status, info: " << status << " " << st << std::endl;
    std::cerr.flush();
    throw std::runtime_error("Error: cublas_getrf returned error code.");
  }
}

// getrfBatched
template<typename T, typename I>
inline static void getrfBatched_v1(const int n,
                                device::device_pointer<T>* a,
                                int lda,
                                device::device_pointer<I> piv,
                                device::device_pointer<I> info,
                                int batchSize)
{
  T** A_d;
  T** A_h;
  A_h = new T*[batchSize];
  for (int i = 0; i < batchSize; i++)
    A_h[i] = raw_pointer_cast(a[i]);
  arch::malloc((void**)&A_d, batchSize * sizeof(*A_h));
  arch::memcopy(A_d, A_h, batchSize * sizeof(*A_h), arch::memcopyH2D);
  cublasStatus_t status = cublas::cublas_getrfBatched(arch::global_cublas_handle, n, A_d, lda, raw_pointer_cast(piv),
                                                      raw_pointer_cast(info), batchSize);
  if (CUBLAS_STATUS_SUCCESS != status)
    throw std::runtime_error("Error: cublas_getrf returned error code.");
  arch::free(A_d);
  delete[] A_h;
}

template<typename T, typename I>
inline static void getrfBatched_v2(const int n,
                                device::device_pointer<T>* a,
                                int lda,
                                device::device_pointer<I> piv,
                                device::device_pointer<I> info,
                                int batchSize)
{

  using sfqmc::afqmc::DeviceBufferManager;
  DeviceBufferManager buffer_manager;

  using qmc_cuda::global_cuda_streams;
  if (global_cuda_streams.size() < batchSize)
  {
    int n0 = global_cuda_streams.size();
    for (int b = n0; b < batchSize; b++)
    {
      global_cuda_streams.emplace_back(cudaStream_t{});
      cudaStreamCreateWithFlags(&(global_cuda_streams.back()), cudaStreamNonBlocking);
    }
  }

  int lwork=-1;
  cusolver_check(cusolver::cusolver_getrf_bufferSize(arch::global_cusolverDn_handle, n, n, 
				raw_pointer_cast(a[0]), lda, &lwork), "getrfBatched_v2");
  auto alloc{buffer_manager.get_generator().template get_allocator<T>()};
  auto ptr = alloc.allocate(lwork*batchSize);

  cudaStream_t s0;
  cusolver_check(cusolverDnGetStream(arch::global_cusolverDn_handle,&s0),"getrfBatched_v2");

  for(int b=0; b<batchSize; ++b) {
    cusolver_check(cusolverDnSetStream(arch::global_cusolverDn_handle,global_cuda_streams[b]),
                                        "getrfBatched_v2");
    cusolver_check(cusolver::cusolver_getrf(arch::global_cusolverDn_handle, n, n, 
	    raw_pointer_cast(a[b]), lda, raw_pointer_cast(ptr)+b*lwork, 
	    raw_pointer_cast(piv)+b*n, raw_pointer_cast(info)+b, false), "getrfBatched_v2");
  }

  qmc_cuda::cuda_check(cudaDeviceSynchronize());
  qmc_cuda::cuda_check(cudaGetLastError());
  cusolver_check(cusolverDnSetStream(arch::global_cusolverDn_handle,s0),"getrfBatched_v2");
  alloc.deallocate(ptr, lwork*batchSize);
}

template<typename T, typename I>
inline static void getrfBatched(const int n,
                                device::device_pointer<T>* a,
                                int lda,
                                device::device_pointer<I> piv,
                                device::device_pointer<I> info,
                                int batchSize,
				device_cuda_backend)
{
  // some heuristics here
  // in the future, test both approaches and keep the fastest one
  if( batchSize<10 ) { 
    if(n<128)
      getrfBatched_v1(n,a,lda,piv,info,batchSize);
    else
      getrfBatched_v2(n,a,lda,piv,info,batchSize);
  } else if( batchSize<20 ) { 
    if(n<256)
      getrfBatched_v1(n,a,lda,piv,info,batchSize);
    else
      getrfBatched_v2(n,a,lda,piv,info,batchSize);
  } else if( batchSize<50 ) { 
    if(n<432)
      getrfBatched_v1(n,a,lda,piv,info,batchSize);
    else
      getrfBatched_v2(n,a,lda,piv,info,batchSize);
  } else if( batchSize<250 ) { 
    if(n<768)
      getrfBatched_v1(n,a,lda,piv,info,batchSize);
    else
      getrfBatched_v2(n,a,lda,piv,info,batchSize);
  } else { 
    if(n<1024)
      getrfBatched_v1(n,a,lda,piv,info,batchSize);
    else
      getrfBatched_v2(n,a,lda,piv,info,batchSize);
  }
}

// getri_bufferSize
template<typename T>
inline static void getri_bufferSize(int n, device::device_pointer<T> a, int lda, int& lwork, device_cuda_backend)
{
  // gpu uses getrs to invert matrix, which requires n*n workspace
  lwork = n * n;
}

// write separate query function to avoid hack!!!
template<typename T, typename R, typename I>
inline static void getri(int n,
                         device::device_pointer<T> a,
                         int lda,
                         device::device_pointer<I> piv,
                         device::device_pointer<R> work,
                         int n1,
                         int& status,
			 device_cuda_backend)
{
  if (n1 < n * n)
    throw std::runtime_error("Error: getri<GPU_MEMORY_POINTER_TYPE> required buffer space of n*n.");
  if (lda != n)
    throw std::runtime_error("Error: getri<GPU_MEMORY_POINTER_TYPE> required lda = 1.");

  int* info;
  arch::malloc((void**)&info, sizeof(int), "lapack_cuda_gpu_ptr::getri");

  kernels::set_identity(n, n, raw_pointer_cast(work), n);
  if (CUSOLVER_STATUS_SUCCESS !=
      cusolver::cusolver_getrs(arch::global_cusolverDn_handle, CUBLAS_OP_N, n, n, raw_pointer_cast(a), lda, raw_pointer_cast(piv),
                               raw_pointer_cast(work), n, info))
    throw std::runtime_error("Error: cusolver_getrs returned error code.");
  arch::memcopy(raw_pointer_cast(a), raw_pointer_cast(work), n * n * sizeof(T), arch::memcopyD2D);
  arch::memcopy(&status, info, sizeof(int), arch::memcopyD2H);
  arch::free(info);
}

// getriBatched
template<typename T, typename I>
inline static void getriBatched(int n,
                                device::device_pointer<T>* a,
                                int lda,
                                device::device_pointer<I> piv,
                                device::device_pointer<T>* ainv,
                                int ldc,
                                device::device_pointer<I> info,
                                int batchSize,
				device_cuda_backend)
{
  T **A_d, **C_d;
  T **A_h, **C_h;
  A_h = new T*[batchSize];
  C_h = new T*[batchSize];
  for (int i = 0; i < batchSize; i++)
  {
    A_h[i] = raw_pointer_cast(a[i]);
    C_h[i] = raw_pointer_cast(ainv[i]);
  }
  arch::malloc((void**)&A_d, batchSize * sizeof(*A_h));
  arch::malloc((void**)&C_d, batchSize * sizeof(*C_h));
  arch::memcopy(A_d, A_h, batchSize * sizeof(*A_h), arch::memcopyH2D);
  arch::memcopy(C_d, C_h, batchSize * sizeof(*C_h), arch::memcopyH2D);
  cublasStatus_t status = cublas::cublas_getriBatched(arch::global_cublas_handle, n, A_d, lda, raw_pointer_cast(piv), C_d,
                                                      ldc, raw_pointer_cast(info), batchSize);
  if (CUBLAS_STATUS_SUCCESS != status)
    throw std::runtime_error("Error: cublas_getri returned error code.");
  arch::free(A_d);
  arch::free(C_d);
  delete[] A_h;
  delete[] C_h;
}

// matinveBatched
template<typename T1, typename T2, typename I>
inline static void matinvBatched(int n,
                                 device::device_pointer<T1>* a,
                                 int lda,
                                 device::device_pointer<T2>* ainv,
                                 int lda_inv,
                                 device::device_pointer<I> info,
                                 int batchSize,
				 device_cuda_backend)
{
  T1 **A_d, **A_h;
  T2 **C_d, **C_h;
  A_h = new T1*[batchSize];
  C_h = new T2*[batchSize];
  for (int i = 0; i < batchSize; i++)
  {
    A_h[i] = raw_pointer_cast(a[i]);
    C_h[i] = raw_pointer_cast(ainv[i]);
  }
  arch::malloc((void**)&A_d, batchSize * sizeof(*A_h));
  arch::malloc((void**)&C_d, batchSize * sizeof(*C_h));
  arch::memcopy(A_d, A_h, batchSize * sizeof(*A_h), arch::memcopyH2D);
  arch::memcopy(C_d, C_h, batchSize * sizeof(*C_h), arch::memcopyH2D);
  cublasStatus_t status = cublas::cublas_matinvBatched(arch::global_cublas_handle, n, A_d, lda, C_d, lda_inv,
                                                       raw_pointer_cast(info), batchSize);
  if (CUBLAS_STATUS_SUCCESS != status)
    throw std::runtime_error("Error: cublas_matinv returned error code.");
  arch::free(A_d);
  arch::free(C_d);
  delete[] A_h;
  delete[] C_h;
}

// geqrf
template<typename T>
inline static void geqrf_bufferSize(int m, int n, device::device_pointer<T> a, int lda, int& lwork, device_cuda_backend)
{
  if (CUSOLVER_STATUS_SUCCESS !=
      cusolver::cusolver_geqrf_bufferSize(arch::global_cusolverDn_handle, m, n, raw_pointer_cast(a), lda, &lwork))
    throw std::runtime_error("Error: cusolver_geqrf_bufferSize returned error code.");
}

template<typename T>
inline static void geqrf(int M,
                         int N,
                         device::device_pointer<T> A,
                         const int LDA,
                         device::device_pointer<T> TAU,
                         device::device_pointer<T> WORK,
                         int LWORK,
                         int& INFO,
			 device_cuda_backend)
{
  // allocating here for now
  int* piv;
  arch::malloc((void**)&piv, sizeof(int), "lapack_cuda_gpu_ptr::geqrf");

  cusolverStatus_t status = cusolver::cusolver_geqrf(arch::global_cusolverDn_handle, M, N, raw_pointer_cast(A), LDA,
                                                     raw_pointer_cast(TAU), raw_pointer_cast(WORK), LWORK, piv);
  arch::memcopy(&INFO, piv, sizeof(int), arch::memcopyD2H);
  if (CUSOLVER_STATUS_SUCCESS != status)
  {
    std::cerr << " cublas_geqrf status, info: " << status << " " << INFO << std::endl;
    std::cerr.flush();
    throw std::runtime_error("Error: cublas_geqrf returned error code.");
  }
  arch::free(piv);
}

// gelqf
template<typename T>
inline static void gelqf_bufferSize(int m, int n, device::device_pointer<T> a, int lda, int& lwork, device_cuda_backend)
{
  lwork = 0;
}

template<typename T>
inline static void gelqf(int M,
                         int N,
                         device::device_pointer<T> A,
                         const int LDA,
                         device::device_pointer<T> TAU,
                         device::device_pointer<T> WORK,
                         int LWORK,
                         int& INFO,
			 device_cuda_backend)
{
  throw std::runtime_error("Error: gelqf not implemented in CUDA backend. \n");
}

// gqr
template<typename T>
static void gqr_bufferSize(int m, int n, int k, device::device_pointer<T> a, int lda, int& lwork, device_cuda_backend)
{
  if (CUSOLVER_STATUS_SUCCESS !=
      cusolver::cusolver_gqr_bufferSize(arch::global_cusolverDn_handle, m, n, k, raw_pointer_cast(a), lda, &lwork))
    throw std::runtime_error("Error: cusolver_gqr_bufferSize returned error code.");
}

template<typename T>
void static gqr(int M,
                int N,
                int K,
                device::device_pointer<T> A,
                const int LDA,
                device::device_pointer<T> TAU,
                device::device_pointer<T> WORK,
                int LWORK,
                int& INFO,
		device_cuda_backend)
{
  // allocating here for now
  int* piv;
  arch::malloc((void**)&piv, sizeof(int), "lapack_cuda_gpu_ptr::gqr");

  cusolverStatus_t status = cusolver::cusolver_gqr(arch::global_cusolverDn_handle, M, N, K, raw_pointer_cast(A), LDA,
                                                   raw_pointer_cast(TAU), raw_pointer_cast(WORK), LWORK, piv);
  arch::memcopy(&INFO, piv, sizeof(int), arch::memcopyD2H);
  if (CUSOLVER_STATUS_SUCCESS != status)
  {
    std::cerr << " cublas_gqr status, info: " << status << " " << INFO << std::endl;
    std::cerr.flush();
    throw std::runtime_error("Error: cublas_gqr returned error code.");
  }
  arch::free(piv);
}

template<typename T, typename I>
void static gqrStrided(int M,
                       int N,
                       int K,
                       device::device_pointer<T> A,
                       const int LDA,
                       const int Astride,
                       device::device_pointer<T> TAU,
                       const int Tstride,
                       device::device_pointer<T> WORK,
                       int LWORK,
                       device::device_pointer<I> info,
                       int batchSize,
		       device_cuda_backend)
{
  cusolverStatus_t status =
      cusolver::cusolver_gqr_strided(arch::global_cusolverDn_handle, M, N, K, raw_pointer_cast(A), LDA, Astride,
                                     raw_pointer_cast(TAU), Tstride, raw_pointer_cast(WORK), LWORK, raw_pointer_cast(info), batchSize);
  if (CUSOLVER_STATUS_SUCCESS != status)
  {
    std::cerr << " cublas_gqr_strided status: " << status << std::endl;
    std::cerr.flush();
    throw std::runtime_error("Error: cublas_gqr_strided returned error code.");
  }
}

// glq
template<typename T>
static void glq_bufferSize(int m, int n, int k, device::device_pointer<T> a, int lda, int& lwork, device_cuda_backend)
{
  lwork = 0;
}

template<typename T>
void static glq(int M,
                int N,
                int K,
                device::device_pointer<T> A,
                const int LDA,
                device::device_pointer<T> TAU,
                device::device_pointer<T> WORK,
                int LWORK,
                int& INFO,
		device_cuda_backend)
{
  throw std::runtime_error("Error: glq not implemented in CUDA backend. \n");
}

// batched geqrf
template<typename T, typename I>
inline static void geqrfBatched(int M,
                                int N,
                                device::device_pointer<T>* A,
                                const int LDA,
                                device::device_pointer<T>* TAU,
                                device::device_pointer<I> info,
                                int batchSize,
				device_cuda_backend)
{
// cublas_geqrfBatched has TERRIBLE performance!
// using serial version until things are improved!
#if 0
  T** B_h = new T*[2 * batchSize];
  T** A_h(B_h);
  T** T_h(B_h + batchSize);
  for (int i = 0; i < batchSize; i++)
    A_h[i] = raw_pointer_cast(A[i]);
  for (int i = 0; i < batchSize; i++)
    T_h[i] = raw_pointer_cast(TAU[i]);
  T** B_d;
  std::vector<int> inf(batchSize);
  arch::malloc((void**)&B_d, 2 * batchSize * sizeof(*B_h));
  arch::memcopy(B_d, B_h, 2 * batchSize * sizeof(*B_h), arch::memcopyH2D);
  T** A_d(B_d);
  T** T_d(B_d + batchSize);
  cublasStatus_t status = cublas::cublas_geqrfBatched(arch::global_cublas_handle, M, N, 
                A_d, LDA, T_d, raw_pointer_cast(inf.data()), batchSize);
  if (CUBLAS_STATUS_SUCCESS != status)
    throw std::runtime_error("Error: cublas_geqrfBatched returned error code.");
  arch::free(B_d);
  delete[] B_h;
#else

  using sfqmc::afqmc::DeviceBufferManager;
  DeviceBufferManager buffer_manager;

  using qmc_cuda::global_cuda_streams;
  if (global_cuda_streams.size() < batchSize)
  {
    int n0 = global_cuda_streams.size();
    for (int n = n0; n < batchSize; n++)
    {
      global_cuda_streams.emplace_back(cudaStream_t{});
      cudaStreamCreateWithFlags(&(global_cuda_streams.back()), cudaStreamNonBlocking);
    }
  }

  int lwork=-1;
  cusolver_check(cusolver::cusolver_geqrf_bufferSize(arch::global_cusolverDn_handle, M, N,
                                raw_pointer_cast(A[0]), LDA, &lwork), "geqrfBatched");
  auto alloc{buffer_manager.get_generator().template get_allocator<T>()};
  auto ptr = alloc.allocate(lwork*batchSize);

  // get current stream
  cudaStream_t s0;
  cusolver_check(cusolverDnGetStream(arch::global_cusolverDn_handle,&s0),"geqrfBatched");

  for(int n=0; n<batchSize; ++n) {
    cusolver_check(cusolverDnSetStream(arch::global_cusolverDn_handle,global_cuda_streams[n]),
                                        "geqrfBatched");
    cusolver_check(cusolver::cusolver_geqrf(arch::global_cusolverDn_handle, M, N,
            raw_pointer_cast(A[n]), LDA, raw_pointer_cast(TAU[n]),
            raw_pointer_cast(ptr)+n*lwork, lwork, raw_pointer_cast(info)+n, false), "geqrfBatched");
  }

  // sync, restore original stream and release buffer memory
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
  qmc_cuda::cuda_check(cudaGetLastError());
  cusolver_check(cusolverDnSetStream(arch::global_cusolverDn_handle,s0),"geqrfBatched");
  alloc.deallocate(ptr, lwork*batchSize);

#endif
}

template<typename T, typename I>
inline static void geqrfStrided(int M,
                                int N,
                                device::device_pointer<T> A,
                                const int LDA,
                                const int Astride,
                                device::device_pointer<T> TAU,
                                const int Tstride,
                                device::device_pointer<I> info,
                                int batchSize,
				device_cuda_backend)
{
// cublas_geqrfBatched has TERRIBLE performance!
// using serial version until things are improved!
#if 0
  std::vector<int> inf(batchSize);
  T** A_h = new T*[batchSize];
  T** T_h = new T*[batchSize];
  for (int i = 0; i < batchSize; i++)
    A_h[i] = raw_pointer_cast(A) + i * Astride;
  for (int i = 0; i < batchSize; i++)
    T_h[i] = raw_pointer_cast(TAU) + i * Tstride;
  T **A_d, **T_d;
  arch::malloc((void**)&A_d, batchSize * sizeof(*A_h));
  arch::memcopy(A_d, A_h, batchSize * sizeof(*A_h), arch::memcopyH2D);
  arch::malloc((void**)&T_d, batchSize * sizeof(*T_h));
  arch::memcopy(T_d, T_h, batchSize * sizeof(*T_h), arch::memcopyH2D);
  cublas_check(cublas::cublas_geqrfBatched(arch::global_cublas_handle,
            M, N, A_d, LDA, T_d, raw_pointer_cast(inf.data()), batchSize), "geqrfStrided");
  for (int i = 0; i < batchSize; i++)
    RUNTIME_CHECK(inf[i] == 0, "");
  arch::free(A_d);
  delete[] A_h;
  arch::free(T_d);
  delete[] T_h;
#else

  using sfqmc::afqmc::DeviceBufferManager;
  DeviceBufferManager buffer_manager;

  using qmc_cuda::global_cuda_streams;
  if (global_cuda_streams.size() < batchSize)
  { 
    int n0 = global_cuda_streams.size();
    for (int n = n0; n < batchSize; n++)
    { 
      global_cuda_streams.emplace_back(cudaStream_t{});
      cudaStreamCreateWithFlags(&(global_cuda_streams.back()), cudaStreamNonBlocking);
    }
  }

  int lwork=-1; 
  cusolver_check(cusolver::cusolver_geqrf_bufferSize(arch::global_cusolverDn_handle, M, N, 
                                raw_pointer_cast(A), LDA, &lwork), "geqrfStrided");
  auto alloc{buffer_manager.get_generator().template get_allocator<T>()};
  auto ptr = alloc.allocate(lwork*batchSize);  

  // get current stream
  cudaStream_t s0;
  cusolver_check(cusolverDnGetStream(arch::global_cusolverDn_handle,&s0),"geqrfStrided");

  for(int n=0; n<batchSize; ++n) {
    cusolver_check(cusolverDnSetStream(arch::global_cusolverDn_handle,global_cuda_streams[n]),
                                       "geqrfStrided");
    cusolver_check(cusolver::cusolver_geqrf(arch::global_cusolverDn_handle, M, N, 
            raw_pointer_cast(A)+n*Astride, LDA, raw_pointer_cast(TAU)+n*Tstride, 
            raw_pointer_cast(ptr)+n*lwork, lwork, raw_pointer_cast(info)+n, false),"geqrfStrided");
  }

  // sync, restore original stream and release buffer memory
  qmc_cuda::cuda_check(cudaDeviceSynchronize());  
  qmc_cuda::cuda_check(cudaGetLastError());
  cusolver_check(cusolverDnSetStream(arch::global_cusolverDn_handle,s0),"geqrfStrided");
  alloc.deallocate(ptr, lwork*batchSize);
  
#endif
}

// gesvd_bufferSize
template<typename T>
inline static void gesvd_bufferSize(const int m, const int n, device::device_pointer<T> a, int& lwork, device_cuda_backend)
{
  cusolver::cusolver_gesvd_bufferSize(arch::global_cusolverDn_handle, m, n, raw_pointer_cast(a), &lwork);
}

template<typename T, typename R>
inline static void gesvd(char jobU,
                         char jobVT,
                         const int m,
                         const int n,
                         device::device_pointer<T>&& A,
                         int lda,
                         device::device_pointer<R>&& S,
                         device::device_pointer<T>&& U,
                         int ldu,
                         device::device_pointer<T>&& VT,
                         int ldvt,
                         device::device_pointer<T>&& W,
                         int lw,
                         int& st,
			 device_cuda_backend)
{
  int* devSt;
  arch::malloc((void**)&devSt, sizeof(int));
  cusolverStatus_t status =
      cusolver::cusolver_gesvd(arch::global_cusolverDn_handle, jobU, jobVT, m, n, raw_pointer_cast(A), lda, raw_pointer_cast(S),
                               raw_pointer_cast(U), ldu, raw_pointer_cast(VT), ldvt, raw_pointer_cast(W), lw, devSt);
  arch::memcopy(&st, devSt, sizeof(int), arch::memcopyD2H);
  if (CUSOLVER_STATUS_SUCCESS != status)
  {
    std::cerr << " cublas_gesvd status, info: " << status << " " << st << std::endl;
    std::cerr.flush();
    throw std::runtime_error("Error: cublas_gesvd returned error code.");
  }
  arch::free(devSt);
}

template<typename T, typename R>
inline static void gesvd(char jobU,
                         char jobVT,
                         const int m,
                         const int n,
                         device::device_pointer<T>&& A,
                         int lda,
                         device::device_pointer<R>&& S,
                         device::device_pointer<T>&& U,
                         int ldu,
                         device::device_pointer<T>&& VT,
                         int ldvt,
                         device::device_pointer<T>&& W,
                         int lw,
                         device::device_pointer<R>&& RW,
                         int& st,
			 device_cuda_backend)
{
  int* devSt;
  arch::malloc((void**)&devSt, sizeof(int));
  cusolverStatus_t status =
      cusolver::cusolver_gesvd(arch::global_cusolverDn_handle, jobU, jobVT, m, n, raw_pointer_cast(A), lda, raw_pointer_cast(S),
                               raw_pointer_cast(U), ldu, raw_pointer_cast(VT), ldvt, raw_pointer_cast(W), lw, devSt);
  arch::memcopy(&st, devSt, sizeof(int), arch::memcopyD2H);
  if (CUSOLVER_STATUS_SUCCESS != status)
  {
    std::cerr << " cublas_gesvd status, info: " << status << " " << st << std::endl;
    std::cerr.flush();
    throw std::runtime_error("Error: cublas_gesvd returned error code.");
  }
  arch::free(devSt);
}


} // namespace ma

#endif
