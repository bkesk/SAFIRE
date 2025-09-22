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

#ifndef BLAS_CUDA_GPU_PTR_HPP
#define BLAS_CUDA_GPU_PTR_HPP

#include <type_traits>
#include <cassert>
#include <vector>
//#include "Memory/CUDA/cuda_gpu_pointer.hpp"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "Memory/device_pointers.hpp"
#include "Memory/arch.hpp"
#include "Numerics/detail/CUDA/cublas_wrapper.hpp"
#include "Memory/custom_pointers.hpp"
#include "Memory/buffer_managers.h"
//#include "Numerics/detail/CUDA/cublasXt_wrapper.hpp"
// hand coded kernels for blas extensions
#include "Numerics/detail/CUDA/Kernels/adotpby.cuh"
#include "Numerics/detail/CUDA/Kernels/setIdentity.cuh"
#include "Numerics/detail/CUDA/Kernels/axty.cuh"
#include "Numerics/detail/CUDA/Kernels/sum.cuh"
#include "Numerics/detail/CUDA/Kernels/adiagApy.cuh"
#include "Numerics/detail/CUDA/Kernels/acAxpbB.cuh"
#include "Numerics/detail/CUDA/Kernels/zero_complex_part.cuh"
#include "Numerics/detail/CUDA/Kernels/axpyBatched.cuh"
#include "Numerics/detail/CUDA/Kernels/get_diagonal.cuh"
#include "Numerics/detail/CUDA/Kernels/copy_select.cuh"
#include "Numerics/detail/CUDA/Kernels/copy_n_cast.cuh"
#include "Numerics/detail/CUDA/Kernels/add_scalar.cuh"
#include "Numerics/detail/CUDA/Kernels/accumulate.cuh"
#include "Numerics/detail/CUDA/Kernels/fill_n.cuh"
#include "Numerics/detail/CUDA/Kernels/term_by_term_matrix_vec.cuh"
#include "Numerics/detail/CUDA/Kernels/term_by_term_matrix_mat.cuh"

// Currently available:
// Lvl-1: dot, axpy, scal
// Lvl-2: gemv
// Lvl-3: gemm

namespace ma
{
// copy/copy2D Specializations
// No tag dispatching here!
// copy is the ONE routine we want to allow with mixed host/device pointers!
template<typename T, typename Q>
inline static void copy(int n, device::device_pointer<Q> x, int incx, device::device_pointer<T> y, int incy)
{
  static_assert(std::is_same<typename std::decay<Q>::type, T>::value, "Wrong dispatch.\n");
  if (CUBLAS_STATUS_SUCCESS !=
      cublas::cublas_copy(arch::global_cublas_handle, n, raw_pointer_cast(x), incx, raw_pointer_cast(y), incy))
    throw std::runtime_error("Error: cublas_copy returned error code.");
}

template<typename T, typename Q>
inline static void copy(int n, T const* x, int incx, device::device_pointer<Q> y, int incy)
{
  static_assert(std::is_same<typename std::decay<Q>::type, T>::value, "Wrong dispatch.\n");
  arch::memcopy2D(raw_pointer_cast(y), sizeof(Q) * incy, x, sizeof(T) * incx, sizeof(T), n, arch::memcopyH2D,
                  "lapack_cuda_gpu_ptr::copy");
}

template<typename T, typename Q>
inline static void copy(int n, device::device_pointer<Q> x, int incx, T* y, int incy)
{
  static_assert(std::is_same<typename std::decay<Q>::type, T>::value, "Wrong dispatch.\n");
  RUNTIME_CHECK(sizeof(Q) == sizeof(T), "");
  arch::memcopy2D(y, sizeof(T) * incy, raw_pointer_cast(x), sizeof(Q) * incx, sizeof(T), n, arch::memcopyD2H,
                  "lapack_cuda_gpu_ptr::copy");
}

template<typename T, typename T2>
inline static void copy2D(int N, int M, device::device_pointer<T> src, int lda, device::device_pointer<T2> dst, int ldb)
{
  static_assert(std::is_same<typename std::decay<T>::type, T2>::value, "Wrong dispatch.\n");
  arch::memcopy2D(raw_pointer_cast(dst), sizeof(T2) * ldb, raw_pointer_cast(src), sizeof(T) * lda, M * sizeof(T), N,
                  arch::memcopyD2D, "blas_cuda_gpu_ptr::copy2D");
}

template<typename T, typename T2>
inline static void copy2D(int N, int M, T const* src, int lda, device::device_pointer<T2> dst, int ldb)
{
  static_assert(std::is_same<typename std::decay<T>::type, T2>::value, "Wrong dispatch.\n");
  arch::memcopy2D(raw_pointer_cast(dst), sizeof(T2) * ldb, src, sizeof(T) * lda, M * sizeof(T), N, arch::memcopyH2D,
                  "blas_cuda_gpu_ptr::copy2D");
}

template<typename T, typename T2>
inline static void copy2D(int N, int M, device::device_pointer<T> src, int lda, T2* dst, int ldb)
{
  static_assert(std::is_same<typename std::decay<T>::type, T2>::value, "Wrong dispatch.\n");
  arch::memcopy2D(dst, sizeof(T2) * ldb, raw_pointer_cast(src), sizeof(T) * lda, M * sizeof(T), N, arch::memcopyD2H,
                  "blas_cuda_gpu_ptr::copy2D");
}

// scal Specializations
template<typename T, typename Q>
inline static void scal(int n, Q alpha, device::device_pointer<T> x, int incx, device_cuda_backend)
{
  static_assert(std::is_convertible<typename std::decay<Q>::type, T>::value, "Wrong dispatch.\n");
  if (CUBLAS_STATUS_SUCCESS != cublas::cublas_scal(arch::global_cublas_handle, n, T(alpha), raw_pointer_cast(x), incx))
    throw std::runtime_error("Error: cublas_scal returned error code.");
}

// dot Specializations
template<typename T, typename Q>
inline static auto dot(int const n, device::device_pointer<Q> x, int const incx, device::device_pointer<T> y, int const incy, device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q>::type, typename std::decay<T>::type>::value, "Wrong dispatch.\n");
  return cublas::cublas_dot(arch::global_cublas_handle, n, raw_pointer_cast(x), incx, raw_pointer_cast(y), incy);
}

// axpy Specializations
template<typename T, typename Q>
inline static void axpy(int n, T const a, device::device_pointer<Q> x, int incx, device::device_pointer<T> y, int incy, device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q>::type, T>::value, "Wrong dispatch.\n");
  if (CUBLAS_STATUS_SUCCESS !=
      cublas::cublas_axpy(arch::global_cublas_handle, n, a, raw_pointer_cast(x), incx, raw_pointer_cast(y), incy))
    throw std::runtime_error("Error: cublas_axpy returned error code.");
}

// GEMV Specializations
template<typename T, typename T2, typename Q1, typename Q2>
inline static void gemv(char Atrans,
                        int M,
                        int N,
                        T2 alpha,
                        device::device_pointer<Q1> A,
                        int lda,
                        device::device_pointer<Q2> x,
                        int incx,
                        T2 beta,
                        device::device_pointer<T> y,
                        int incy,
			device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, T2>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T>::value, "Wrong dispatch.\n");
  if (CUBLAS_STATUS_SUCCESS !=
      cublas::cublas_gemv(arch::global_cublas_handle, Atrans, M, N, alpha, raw_pointer_cast(A), lda, raw_pointer_cast(x), incx, beta,
                          raw_pointer_cast(y), incy))
    throw std::runtime_error("Error: cublas_gemv returned error code.");
}

// GEMM Specializations
// why is this not working with T const????
template<typename T, typename T2, typename Q1, typename Q2>
inline static void gemm(char Atrans,
                        char Btrans,
                        int M,
                        int N,
                        int K,
                        T2 alpha,
                        device::device_pointer<Q1> A,
                        int lda,
                        device::device_pointer<Q2> B,
                        int ldb,
                        T2 beta,
                        device::device_pointer<T> C,
                        int ldc,
			device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, T>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T2>::value, "Wrong dispatch.\n");
  if (CUBLAS_STATUS_SUCCESS !=
      cublas::cublas_gemm(arch::global_cublas_handle, Atrans, Btrans, M, N, K, alpha, raw_pointer_cast(A), lda, raw_pointer_cast(B),
                          ldb, beta, raw_pointer_cast(C), ldc))
    throw std::runtime_error("Error: cublas_gemm returned error code.");
}

// Blas Extensions
// geam
template<typename T, typename Q1, typename Q2>
inline static void geam(char Atrans,
                        char Btrans,
                        int M,
                        int N,
                        T const alpha,
                        device::device_pointer<Q1> A,
                        int lda,
                        T const beta,
                        device::device_pointer<Q2> B,
                        int ldb,
                        device::device_pointer<T> C,
                        int ldc,
			device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, T>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T>::value, "Wrong dispatch.\n");
  if (CUBLAS_STATUS_SUCCESS !=
      cublas::cublas_geam(arch::global_cublas_handle, Atrans, Btrans, M, N, alpha, raw_pointer_cast(A), lda, beta,
                          raw_pointer_cast(B), ldb, raw_pointer_cast(C), ldc))
    throw std::runtime_error("Error: cublas_geam returned error code.");
}

/*
template<typename T>
//inline static void set1D(int n, T const alpha, ptr x, int incx)
inline static void set1D(int n, T const alpha, device::device_pointer<T> x, int incx)
{
  // No set funcion in cuda!!! Avoiding kernels for now
  //std::vector<T> buff(n,alpha);
  //if(CUBLAS_STATUS_SUCCESS != cublasSetVector(n,sizeof(T),buff.data(),1,raw_pointer_cast(x),incx))
  T alpha_(alpha);
  if (CUBLAS_STATUS_SUCCESS != cublasSetVector(n, sizeof(T), std::addressof(alpha), 1, raw_pointer_cast(x), incx))
    throw std::runtime_error("Error: cublasSetVector returned error code.");
}
*/

// dot extension
template<typename T, typename T1, typename T2, typename Q1, typename Q2>
inline static void adotpby(int const n,
                           T1 const alpha,
                           device::device_pointer<Q1> x,
                           int const incx,
                           device::device_pointer<Q2> y,
                           int const incy,
                           T2 const beta,
                           device::device_pointer<T> result,
			   device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, T1>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T1>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<T2>::type, T>::value, "Wrong dispatch.\n");
  kernels::adotpby(n, alpha, raw_pointer_cast(x), incx, raw_pointer_cast(y), incy, beta, raw_pointer_cast(result));
}

// dot extension
template<typename T, typename T1, typename T2, typename Q1, typename Q2>
inline static void strided_adotpby(int nk,
                                   int const n,
                                   T1 const alpha,
                                   device::device_pointer<Q1> A,
                                   int const lda,
                                   device::device_pointer<Q2> B,
                                   int const ldb,
                                   T2 const beta,
                                   device::device_pointer<T> y,
                                   int inc,
				   device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, T1>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T1>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<T2>::type, T>::value, "Wrong dispatch.\n");
  kernels::strided_adotpby(nk, n, alpha, raw_pointer_cast(A), lda, raw_pointer_cast(B), ldb, beta, raw_pointer_cast(y), inc);
}

// axty
template<typename T, typename Q>
inline static void axty(int n, T const alpha, device::device_pointer<Q> x, int incx, device::device_pointer<T> y, int incy, device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q>::type, T>::value, "Wrong dispatch.\n");
  if (incx != 1 || incy != 1)
    throw std::runtime_error("Error: axty with inc != 1 not implemented.");
  kernels::axty(n, alpha, raw_pointer_cast(x), raw_pointer_cast(y));
}

// acAxpbB
template<typename T, typename Q1, typename Q2>
inline static void acAxpbB(int m,
                           int n,
                           T const alpha,
                           device::device_pointer<Q1> A,
                           int lda,
                           device::device_pointer<Q2> x,
                           int incx,
                           T const beta,
                           device::device_pointer<T> B,
                           int ldb,
			   device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, T>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T>::value, "Wrong dispatch.\n");
  kernels::acAxpbB(m, n, alpha, raw_pointer_cast(A), lda, raw_pointer_cast(x), incx, beta, raw_pointer_cast(B), ldb);
}

// adiagApy
template<typename T, typename Q1>
inline static void adiagApy(int n, T const alpha, device::device_pointer<Q1> A, int lda, device::device_pointer<T> y, int incy, device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, T>::value, "Wrong dispatch.\n");
  kernels::adiagApy(n, alpha, raw_pointer_cast(A), lda, raw_pointer_cast(y), incy);
}

template<typename T>
inline static void zero_complex_part(int n, device::device_pointer<T> x, device_cuda_backend)
{
  kernels::zero_complex_part(n, raw_pointer_cast(x));
}

template<typename T>
inline static auto sum(int n, device::device_pointer<T> x, int incx, device_cuda_backend)
{
  return kernels::sum(n, raw_pointer_cast(x), incx);
}

template<typename T>
inline static auto sum(int m, int n, device::device_pointer<T> A, int lda, device_cuda_backend)
{
  return kernels::sum(m, n, raw_pointer_cast(A), lda);
}

template<typename T>
void set_identity(int m, int n, device::device_pointer<T> A, int lda, device_cuda_backend)
{
  kernels::set_identity(m, n, raw_pointer_cast(A), lda);
}

template<typename T>
void set_identity_strided(int nbatch, int stride, int m, int n, device::device_pointer<T> A, int lda, device_cuda_backend)
{
  kernels::set_identity_strided(nbatch, stride, m, n, raw_pointer_cast(A), lda);
}

template<typename T, typename Q1, typename Q2>
inline static void gemmStridedBatched(char Atrans,
                                      char Btrans,
                                      int M,
                                      int N,
                                      int K,
                                      T const alpha,
                                      device::device_pointer<Q1> A,
                                      int lda,
                                      int strideA,
                                      device::device_pointer<Q2> B,
                                      int ldb,
                                      int strideB,
                                      T const beta,
                                      device::device_pointer<T> C,
                                      int ldc,
                                      int strideC,
                                      int batchSize,
				      device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, T>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T>::value, "Wrong dispatch.\n");
  cublas::cublas_gemmStridedBatched(arch::global_cublas_handle, Atrans, Btrans, M, N, K, alpha, raw_pointer_cast(A), lda,
                                    strideA, raw_pointer_cast(B), ldb, strideB, beta, raw_pointer_cast(C), ldc, strideC, batchSize);
}

template<typename T, typename Q1, typename Q2>
inline static void gemmStridedBatched(char Atrans,
                                      char Btrans,
                                      int M,
                                      int N,
                                      int K,
                                      T const alpha,
                                      device::device_pointer<Q1> A,
                                      int lda,
                                      int strideA,
                                      device::device_pointer<Q2> B,
                                      int ldb,
                                      int strideB,
                                      T const beta,
                                      device::device_pointer<std::complex<T>> C,
                                      int ldc,
                                      int strideC,
                                      int batchSize,
                                      device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, std::complex<T>>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T>::value, "Wrong dispatch.\n");
  cublas::cublas_gemmStridedBatched(arch::global_cublas_handle, Atrans, Btrans, M, N, K, alpha, raw_pointer_cast(A), lda,
                                    strideA, raw_pointer_cast(B), ldb, strideB, beta, raw_pointer_cast(C), ldc, strideC, batchSize);
}

template<typename T,
         typename Q1,
         typename Q2,
         typename = typename std::enable_if_t<std::is_same<typename std::decay<Q1>::type, T>::value>,
         typename = typename std::enable_if_t<std::is_same<typename std::decay<Q2>::type, T>::value>>
inline static void gemmBatched(char Atrans,
                               char Btrans,
                               int M,
                               int N,
                               int K,
                               T const alpha,
                               device::device_pointer<Q1>* A,
                               int lda,
                               device::device_pointer<Q2>* B,
                               int ldb,
                               T const beta,
                               device::device_pointer<T>* C,
                               int ldc,
                               int batchSize,
			       device_cuda_backend)
{
  static_assert(std::is_same<typename std::decay<Q1>::type, T>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T>::value, "Wrong dispatch.\n");
  // replace with single call to arch::malloc and arch::memcopy
  T **A_d, **B_d, **C_d;
  Q1** A_h;
  Q2** B_h;
  T** C_h;
  A_h = new Q1*[batchSize];
  B_h = new Q2*[batchSize];
  C_h = new T*[batchSize];
  for (int i = 0; i < batchSize; i++)
  {
    A_h[i] = raw_pointer_cast(A[i]);
    B_h[i] = raw_pointer_cast(B[i]);
    C_h[i] = raw_pointer_cast(C[i]);
  }
  arch::malloc((void**)&A_d, batchSize * sizeof(*A_h));
  arch::malloc((void**)&B_d, batchSize * sizeof(*B_h));
  arch::malloc((void**)&C_d, batchSize * sizeof(*C_h));
  arch::memcopy(A_d, A_h, batchSize * sizeof(*A_h), arch::memcopyH2D);
  arch::memcopy(B_d, B_h, batchSize * sizeof(*B_h), arch::memcopyH2D);
  arch::memcopy(C_d, C_h, batchSize * sizeof(*C_h), arch::memcopyH2D);
  cublas::cublas_gemmBatched(arch::global_cublas_handle, Atrans, Btrans, M, N, K, alpha, A_d, lda, B_d, ldb, beta,
                             C_d, ldc, batchSize);
  arch::free(A_d);
  arch::free(B_d);
  arch::free(C_d);
  delete[] A_h;
  delete[] B_h;
  delete[] C_h;
}

template<typename T,
         typename Q1,
         typename Q2,
         typename T2,
         typename = typename std::enable_if_t<std::is_same<typename std::decay<Q1>::type, T2>::value>,
         typename = typename std::enable_if_t<std::is_same<typename std::decay<Q2>::type, T>::value>,
         typename = typename std::enable_if_t<std::is_same<std::complex<T>, T2>::value>>
inline static void gemmBatched(char Atrans,
                               char Btrans,
                               int M,
                               int N,
                               int K,
                               T const alpha,
                               device::device_pointer<Q1>* A,
                               int lda,
                               device::device_pointer<Q2>* B,
                               int ldb,
                               T const beta,
                               device::device_pointer<T2>* C,
                               int ldc,
                               int batchSize,
			       device_cuda_backend)
{
  // check that remove_complex<T2> == T ???
  static_assert(std::is_same<typename std::decay<Q1>::type, T2>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T>::value, "Wrong dispatch.\n");
  RUNTIME_CHECK(Atrans == 'N' || Atrans == 'n', "");
  // replace with single call to arch::malloc and arch::memcopy
  T2** A_d;
  T** B_d;
  T2** C_d;
  Q1** A_h;
  Q2** B_h;
  T2** C_h;
  A_h = new Q1*[batchSize];
  B_h = new Q2*[batchSize];
  C_h = new T2*[batchSize];
  for (int i = 0; i < batchSize; i++)
  {
    A_h[i] = raw_pointer_cast(A[i]);
    B_h[i] = raw_pointer_cast(B[i]);
    C_h[i] = raw_pointer_cast(C[i]);
  }
  arch::malloc((void**)&A_d, batchSize * sizeof(*A_h));
  arch::malloc((void**)&B_d, batchSize * sizeof(*B_h));
  arch::malloc((void**)&C_d, batchSize * sizeof(*C_h));
  arch::memcopy(A_d, A_h, batchSize * sizeof(*A_h), arch::memcopyH2D);
  arch::memcopy(B_d, B_h, batchSize * sizeof(*B_h), arch::memcopyH2D);
  arch::memcopy(C_d, C_h, batchSize * sizeof(*C_h), arch::memcopyH2D);
  cublas::cublas_gemmBatched(arch::global_cublas_handle, Atrans, Btrans, M, N, K, alpha, A_d, lda, B_d, ldb, beta,
                             C_d, ldc, batchSize);
  arch::free(A_d);
  arch::free(B_d);
  arch::free(C_d);
  delete[] A_h;
  delete[] B_h;
  delete[] C_h;
}

template<typename T1, typename T2, typename T3>
inline static void axpyBatched(int n,
                               T1* x,
                               device::device_pointer<T2>* a,
                               int inca,
                               device::device_pointer<T3>* b,
                               int incb,
                               int batchSize,
			       device_cuda_backend)
{
  T2 const** a_ = new T2 const*[batchSize];
  T3** b_       = new T3*[batchSize];
  for (int i = 0; i < batchSize; i++)
  {
    a_[i] = raw_pointer_cast(a[i]);
    b_[i] = raw_pointer_cast(b[i]);
  }
  kernels::axpy_batched_gpu(n, x, a_, inca, b_, incb, batchSize);
  delete[] a_;
  delete[] b_;
}

template<typename T1, typename T2, typename T3>
inline static void sumGwBatched(int n,
                                T1* x,
                                device::device_pointer<T2>* a,
                                int inca,
                                device::device_pointer<T3>* b,
                                int incb,
                                int b0,
                                int nw,
                                int batchSize,
				device_cuda_backend)
{
  T2 const** a_ = new T2 const*[batchSize];
  T3** b_       = new T3*[batchSize];
  for (int i = 0; i < batchSize; i++)
  {
    a_[i] = raw_pointer_cast(a[i]);
    b_[i] = raw_pointer_cast(b[i]);
  }
  kernels::sumGw_batched_gpu(n, x, a_, inca, b_, incb, b0, nw, batchSize);
  delete[] a_;
  delete[] b_;
}

template<typename T, typename T2>
inline static void get_diagonal_strided(int nk,
                                        int ni,
                                        device::device_pointer<T> A,
                                        int lda,
                                        int stride,
                                        device::device_pointer<T2> B,
                                        int ldb,
					device_cuda_backend)
{
  kernels::get_diagonal_strided(nk, ni, raw_pointer_cast(A), lda, stride, raw_pointer_cast(B), ldb);
}

// expand = true,
//   B[ index[n] ][:] = beta B[ index[n] ][:] + alpha A[n][:]
// expand = false,
//   B[n][:] = beta B[n][:] + alpha A[ index[n] ][:]
template<class T1,
         class T2,
         class T3,
         class T4,
         class index_t
        >
inline static void copy_select_impl(int N, int M, T1 alpha, device::device_pointer<T2> A, int lda,
                                long Astride, T3 beta, device::device_pointer<T4> B, int ldb,
                                long Bstride, device::device_pointer<index_t> index, int nbatch, bool expand, device_cuda_backend)
{
  kernels::copy_select(N, M, static_cast<T2>(alpha), raw_pointer_cast(A), lda, Astride, 
                             static_cast<T4>(beta), raw_pointer_cast(B), ldb, Bstride, 
                             raw_pointer_cast(index), nbatch, expand);
}

// A(i,j) += a;
template<class T1, class T2>
inline static void add_scalar_impl(int N, int M, T1 a, device::device_pointer<T2> A ,int lda, device_cuda_backend)
{
  T2 a_ = static_cast<T1>(a);
  kernels::add_scalar(N,M,a_,raw_pointer_cast(A),lda);
}

template<class T1, class T2, class T3>
void accumulate_impl(int dim, int nrow, int ncol, T1 alpha, device::device_pointer<T2> A, int lda, 
            long Astride, device::device_pointer<T3> y, int incy, long ystride, int batchSize, device_cuda_backend)
{
  kernels::accumulate_impl(dim, nrow, ncol, alpha, raw_pointer_cast(A), lda, Astride,
                           raw_pointer_cast(y), incy, ystride, batchSize );
}

/*
 * Performs the generic operation: (limited to matrices for now)
 * A[i,j] = A[i,j] op x[...], 
 *   where op is {+,-,*,/} and x[...] depends on dim (0:i, 1:j, ...}
 */
template<typename T1, typename T2, typename T3>
void term_by_term_matrix_vector(ma::TENSOR_OPERATIONS op,
                                int dim,
                                int nrow,
                                int ncol,
                                device::device_pointer<T1> A,
                                int lda,
                                T2 alpha,
                                device::device_pointer<T3> x,
                                int incx,
				device_cuda_backend)
{
  RUNTIME_CHECK(dim == 0 || dim == 1, "");
  kernels::term_by_term_mat_vec(op,dim, nrow, ncol, raw_pointer_cast(A), lda,
                alpha, raw_pointer_cast(x), incx);
}

template<typename T1, typename T2, typename T3>
void term_by_term_matrix_vector_strided(ma::TENSOR_OPERATIONS op,
                                int dim,
                                int nrow,
                                int ncol,
                                device::device_pointer<T1> A,
                                int lda,
                                int Astride,
                                T2 alpha,
                                device::device_pointer<T3> x,
                                int incx,
                                int Xstride,
                                int batchSize,
				device_cuda_backend)
{
  RUNTIME_CHECK(dim == 0 || dim == 1, "");
  kernels::term_by_term_mat_vec_strided(op,dim, nrow, ncol, raw_pointer_cast(A), lda, Astride,
                alpha, raw_pointer_cast(x), incx, Xstride, batchSize);
}

template<typename T1, typename T2, typename T3>
void term_by_term_matrix_matrix_strided(ma::TENSOR_OPERATIONS op,
                                int nrow,
                                int ncol,
                                T1 alpha,
                                device::device_pointer<T2> A,
                                int lda,
                                long Astride,
                                device::device_pointer<T3> B,
                                int ldb,
                                long Bstride,
                                int batchSize,
				device_cuda_backend)
{
  kernels::term_by_term_mat_mat_strided(op, nrow, ncol, alpha, raw_pointer_cast(A), lda, Astride,
                raw_pointer_cast(B), ldb, Bstride, batchSize);
}

template<typename T1, typename T2, typename T3>
void fill_if_zero_impl(int nrow, int ncol, device::device_pointer<T1> key, int incx, 
            T2 alpha, device::device_pointer<T3> A, int lda, long stride, int batchSize, device_cuda_backend)
{
  kernels::fill_if_zero_impl(nrow, ncol, raw_pointer_cast(key), incx, alpha, 
                             raw_pointer_cast(A), lda, stride, batchSize);
}

template<typename T1, typename T2, typename T3>
void fill_if_non_zero_impl(int nrow, int ncol, device::device_pointer<T1> key, int incx,
            T2 alpha, device::device_pointer<T3> A, int lda, long stride, int batchSize, device_cuda_backend)
{
  kernels::fill_if_non_zero_impl(nrow, ncol, raw_pointer_cast(key), incx, alpha,
                             raw_pointer_cast(A), lda, stride, batchSize);
}

template<class T>
inline static void complex_conjugate_impl(int N, int M, 
                            device::device_pointer<T> A ,int lda, long stride, int nbatch, device_cuda_backend)
{
  kernels::complex_conjugate_impl(N, M, raw_pointer_cast(A), lda, stride, nbatch);
}

template<typename T1, typename T2>
void copy_n_cast_impl(int N, int M, device::device_pointer<T1> A, int lda, long Astride,
                      device::device_pointer<T2> B, int ldb, long Bstride, int nbatch, device_cuda_backend)
{
  kernels::copy_n_cast_impl(N,M,raw_pointer_cast(A),lda,Astride,raw_pointer_cast(B),ldb,Bstride,nbatch);
}

template<typename T1, typename T2>
void copy_n_cast_impl(int N, int M, T1* A, int lda, long Astride,
                      device::device_pointer<T2> B, int ldb, long Bstride, int nbatch, device_cuda_backend)
{
  if(nbatch==0 or N==0 or M==0) return;
  using sfqmc::afqmc::HostBufferManager;
  using hostptr_t = typename std::allocator_traits<HostBufferManager::template allocator_t<T2>>::pointer;
  HostBufferManager host_buffer_manager;
  auto host_alloc{host_buffer_manager.get_generator().template get_allocator<T2>()};
  // allocate memory on host
  auto host_ptr = host_alloc.allocate(N*M*nbatch);
  // copy A to contiguous memory on host
  copy_n_cast_impl(N,M,A,lda,Astride,host_ptr,M,long(N*M),nbatch,cpu_backend{});

  if(ldb==M and Bstride == long(N*M)) {
    // already contiguous, copy directly
    arch::memcopy(raw_pointer_cast(B), host_ptr, N*M*nbatch, arch::memcopyH2D, "blas_cuda_gpu_ptr::copy_n_cast_impl");
  } else { 
    // allocate contiguous memory on GPU 
    using sfqmc::afqmc::DeviceBufferManager;
    using devptr_t = typename std::allocator_traits<DeviceBufferManager::template allocator_t<T2>>::pointer;
    DeviceBufferManager dev_buffer_manager;
    auto dev_alloc{dev_buffer_manager.get_generator().template get_allocator<T2>()};
    // allocate memory on GPU
    auto dev_ptr = dev_alloc.allocate(N*M*nbatch);
    arch::memcopy(raw_pointer_cast(dev_ptr), host_ptr, N*M*nbatch, arch::memcopyH2D, "blas_cuda_gpu_ptr::copy_n_cast_impl");
    // copy from contiguous to non-contiguous
    kernels::copy_n_cast_impl(N,M,raw_pointer_cast(dev_ptr),M,long(N*M),raw_pointer_cast(B),ldb,Bstride,nbatch);
    dev_alloc.deallocate(dev_ptr, N*M*nbatch);
  }
  host_alloc.deallocate(host_ptr, N*M*nbatch);
}

template<typename T1, typename T2>
void copy_n_cast_impl(int N, int M, device::device_pointer<T1> A, int lda, long Astride,
                      T2* B, int ldb, long Bstride, int nbatch, device_cuda_backend)
{
  if(nbatch==0 or N==0 or M==0) return;
  using sfqmc::afqmc::HostBufferManager;
  using hostptr_t = typename std::allocator_traits<HostBufferManager::template allocator_t<T1>>::pointer;
  HostBufferManager host_buffer_manager;
  auto host_alloc{host_buffer_manager.get_generator().template get_allocator<T1>()};
  // allocate memory on host
  auto host_ptr = host_alloc.allocate(N*M*nbatch);

  if(lda==M and Astride == long(N*M)) {
    // already contiguous, copy directly
    arch::memcopy(host_ptr, raw_pointer_cast(A), N*M*nbatch, arch::memcopyD2H, "blas_cuda_gpu_ptr::copy_n_cast_impl");
  } else {
    // allocate contiguous memory on GPU 
    using sfqmc::afqmc::DeviceBufferManager;
    using devptr_t = typename std::allocator_traits<DeviceBufferManager::template allocator_t<T1>>::pointer;
    DeviceBufferManager dev_buffer_manager;
    auto dev_alloc{dev_buffer_manager.get_generator().template get_allocator<T1>()};
    // allocate memory on GPU
    auto dev_ptr = dev_alloc.allocate(N*M*nbatch);
    // copy from non-contiguous to contiguous
    kernels::copy_n_cast_impl(N,M,raw_pointer_cast(A),lda,Astride,raw_pointer_cast(dev_ptr),M,long(N*M),nbatch);
    arch::memcopy(host_ptr,raw_pointer_cast(dev_ptr), N*M*nbatch, arch::memcopyD2H, "blas_cuda_gpu_ptr::copy_n_cast_impl");
    dev_alloc.deallocate(dev_ptr, N*M*nbatch);
  }

  // copy from contiguous memory to B
  copy_n_cast_impl(N,M,host_ptr,M,long(N*M),B,ldb,Bstride,nbatch,cpu_backend{});

  host_alloc.deallocate(host_ptr, N*M*nbatch);
}


} // namespace ma

#endif
