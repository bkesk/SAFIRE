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

#ifndef BLAS_EXTENSIONS_CPU_H
#define BLAS_EXTENSIONS_CPU_H

// generic header for blas routines
#include "Numerics/detail/dispatch.hpp"
#include "Numerics/detail/CPU/blas_cpu.hpp"

#if defined(HAVE_MKL)
inline CBLAS_TRANSPOSE cblas_operation(char Op)
{
  if (Op == 'N')
    return CblasNoTrans;
  else if (Op == 'T')
    return CblasTrans;
  else if (Op == 'H' || Op == 'C')
    return CblasConjTrans;
  else
    throw std::runtime_error("unknown cblas_peration option");
}
#endif


namespace ma
{

// C = alpha*op(A) + beta*op(B)
// unoptimized implementation
// assumes fortran ordering
template<typename T>
inline static void geam(char Atrans,
                        char Btrans,
                        int m,
                        int n,
                        T const alpha,
                        T const* A,
                        int lda,
                        T const beta,
                        T const* B,
                        int ldb,
                        T* C,
                        int ldc, 
			cpu_backend)
{
  if (n == 0 || m == 0)
    return;
  RUNTIME_CHECK(ldc >= m, "");
  // Special cases
  // S1. set C to zero
  if (alpha == T(0) && beta == T(0))
  {
    T zero(0);
    for (int j = 0; j < n; j++)
      for (int i = 0; i < m; i++)
        C[i + j * ldc] = zero;
    return;
  }
  // S2.
  if (alpha == T(1) && beta == T(0))
  {
    if (std::distance<T const*>(A, C) > 0)
    {
      if (Atrans == 'N' || Atrans == 'n')
        RUNTIME_CHECK(std::distance<T const*>(A, C) >= n * lda - lda + m, "");
      else
        RUNTIME_CHECK(std::distance<T const*>(A, C) >= m * lda - lda + n, "");
    }
    else
    {
      RUNTIME_CHECK(std::distance<T const*>(C, A) >= n * ldc - ldc + m, "");
    }
    if (Atrans == 'N' || Atrans == 'n')
    {
      RUNTIME_CHECK(lda >= m, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = A[i + j * lda];
    }
    else if (Atrans == 'T' || Atrans == 't')
    {
      RUNTIME_CHECK(lda >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = A[j + i * lda];
    }
    else if (Atrans == 'C' || Atrans == 'c')
    {
      RUNTIME_CHECK(lda >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = ma::conj(A[j + i * lda]);
    }
    else
    {
      throw std::runtime_error("Error: Invalid Atrans in geam.");
    }
    return;
  }
  if (alpha == T(0) && beta == T(1))
  {
    if (std::distance<T const*>(B, C) > 0)
    {
      if (Btrans == 'N' || Btrans == 'n')
        RUNTIME_CHECK(std::distance<T const*>(B, C) >= n * ldb - ldb + m, "");
      else
        RUNTIME_CHECK(std::distance<T const*>(B, C) >= m * ldb - ldb + n, "");
    }
    else
    {
      RUNTIME_CHECK(std::distance<T const*>(C, B) >= n * ldc - ldc + m, "");
    }
    if (Btrans == 'N' || Btrans == 'n')
    {
      RUNTIME_CHECK(ldb >= m, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = B[i + j * ldb];
    }
    else if (Btrans == 'T' || Btrans == 't')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = B[j + i * ldb];
    }
    else if (Btrans == 'C' || Btrans == 'c')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = ma::conj(B[j + i * ldb]);
    }
    else
    {
      throw std::runtime_error("Error: Invalid Btrans in geam.");
    }
    return;
  }
  // Check for in-place modes
  // I1. A==C, lda==ldc, Atrans=='N'
  if (std::distance<T const*>(A, C) == 0)
  {
    if ((lda != ldc) || ((Atrans != 'N') && (Atrans != 'n')))
      throw std::runtime_error("Error: In-place mode requires Op(A)='N' and lda=ldc.");
    if (std::distance<T const*>(B, C) > 0)
    {
      if (Btrans == 'N' || Btrans == 'n')
        RUNTIME_CHECK(std::distance<T const*>(B, C) >= n * ldb - ldb + m, "");
      else
        RUNTIME_CHECK(std::distance<T const*>(B, C) >= m * ldb - ldb + n, "");
    }
    else
    {
      RUNTIME_CHECK(std::distance<T const*>(C, B) >= n * ldc - ldc + m, "");
    }
    if (Btrans == 'N' || Btrans == 'n')
    {
      RUNTIME_CHECK(ldb >= m, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = alpha * C[i + j * ldc] + beta * B[i + j * ldb];
    }
    else if (Btrans == 'T' || Btrans == 't')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = alpha * C[i + j * ldc] + beta * B[j + i * ldb];
    }
    else if (Btrans == 'C' || Btrans == 'c')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = alpha * C[i + j * ldc] + beta * ma::conj(B[j + i * ldb]);
    }
    else
    {
      throw std::runtime_error("Error: Invalid Btrans in geam.");
    }
    return;
  }
  // I2.  B==C, ldb==ldc, Btrans=='N'
  if (std::distance<T const*>(B, C) == 0)
  {
    if ((ldb != ldc) || ((Btrans != 'N') && (Btrans != 'n')))
      throw std::runtime_error("Error: In-place mode requires Op(B)='N' and ldb=ldc.");
    if (std::distance<T const*>(A, C) > 0)
    {
      if (Atrans == 'N' || Atrans == 'n')
        RUNTIME_CHECK(std::distance<T const*>(A, C) >= n * lda - lda + m, "");
      else
        RUNTIME_CHECK(std::distance<T const*>(A, C) >= m * lda - lda + n, "");
    }
    else
    {
      RUNTIME_CHECK(std::distance<T const*>(C, A) >= n * ldc - ldc + m, "");
    }
    if (Atrans == 'N' || Atrans == 'n')
    {
      RUNTIME_CHECK(lda >= m, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = beta * C[i + j * ldc] + alpha * A[i + j * lda];
    }
    else if (Atrans == 'T' || Atrans == 't')
    {
      RUNTIME_CHECK(lda >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = beta * C[i + j * ldc] + alpha * A[j + i * lda];
    }
    else if (Atrans == 'C' || Atrans == 'c')
    {
      RUNTIME_CHECK(lda >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = beta * C[i + j * ldc] + alpha * ma::conj(A[j + i * lda]);
    }
    else
    {
      throw std::runtime_error("Error: Invalid Atrans in geam.");
    }
    return;
  }
  // check that C does not overlap A or B
  if (std::distance<T const*>(A, C) > 0)
  {
    if (Atrans == 'N' || Atrans == 'n')
      RUNTIME_CHECK(std::distance<T const*>(A, C) >= n * lda - lda + m, "");
    else
      RUNTIME_CHECK(std::distance<T const*>(A, C) >= m * lda - lda + n, "");
  }
  else
  {
    RUNTIME_CHECK(std::distance<T const*>(C, A) >= n * ldc - ldc + m, "");
  }
  if (std::distance<T const*>(B, C) > 0)
  {
    if (Btrans == 'N' || Btrans == 'n')
      RUNTIME_CHECK(std::distance<T const*>(B, C) >= n * ldb - ldb + m, "");
    else
      RUNTIME_CHECK(std::distance<T const*>(B, C) >= m * ldb - ldb + n, "");
  }
  else
  {
    RUNTIME_CHECK(std::distance<T const*>(C, B) >= n * ldc - ldc + m, "");
  }
  // Generic case
  if (Atrans == 'N' || Atrans == 'n')
  {
    RUNTIME_CHECK(lda >= m, "");
    if (Btrans == 'N' || Btrans == 'n')
    {
      RUNTIME_CHECK(ldb >= m, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = alpha * A[i + j * lda] + beta * B[i + j * ldb];
    }
    else if (Btrans == 'T' || Btrans == 't')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = alpha * A[i + j * lda] + beta * B[j + i * ldb];
    }
    else if (Btrans == 'C' || Btrans == 'c')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = alpha * A[i + j * lda] + beta * ma::conj(B[j + i * ldb]);
    }
    else
    {
      throw std::runtime_error("Error: Invalid Btrans in geam.");
    }
  }
  else if (Atrans == 'T' || Atrans == 't')
  {
    RUNTIME_CHECK(lda >= n, "");
    if (Btrans == 'N' || Btrans == 'n')
    {
      RUNTIME_CHECK(ldb >= m, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = alpha * A[j + i * lda] + beta * B[i + j * ldb];
    }
    else if (Btrans == 'T' || Btrans == 't')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          C[i + j * ldc] = alpha * A[j + i * lda] + beta * B[j + i * ldb];
    }
    else if (Btrans == 'C' || Btrans == 'c')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          C[i + j * ldc] = alpha * A[j + i * lda] + beta * ma::conj(B[j + i * ldb]);
    }
    else
    {
      throw std::runtime_error("Error: Invalid Btrans in geam.");
    }
  }
  else if (Atrans == 'C' || Atrans == 'c')
  {
    RUNTIME_CHECK(lda >= n, "");
    if (Btrans == 'N' || Btrans == 'n')
    {
      RUNTIME_CHECK(ldb >= m, "");
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          C[i + j * ldc] = alpha * ma::conj(A[j + i * lda]) + beta * B[i + j * ldb];
    }
    else if (Btrans == 'T' || Btrans == 't')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          C[i + j * ldc] = alpha * ma::conj(A[j + i * lda]) + beta * B[j + i * ldb];
    }
    else if (Btrans == 'C' || Btrans == 'c')
    {
      RUNTIME_CHECK(ldb >= n, "");
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          C[i + j * ldc] = alpha * ma::conj(A[j + i * lda]) + beta * ma::conj(B[j + i * ldb]);
    }
    else
    {
      throw std::runtime_error("Error: Invalid Btrans in geam.");
    }
  }
  else
  {
    throw std::runtime_error("Error: Invalid Atrans in geam.");
  }
}

template<typename T>
inline static void zero_complex_part([[maybe_unused]] int n, [[maybe_unused]] T* x, cpu_backend)
{}

template<typename T>
inline static void zero_complex_part(int n, std::complex<T>* x, cpu_backend)
{
  for (int i = 0; i < n; ++i, ++x)
    *x = std::complex<T>(real(*x), 0.0);
}

template<typename T>
inline static void set1D(int n, T const alpha, T* x, int incx, cpu_backend)
{
  for (int i = 0; i < n; i++, x += incx)
    *x = alpha;
}

// y = beta*y + alpha * dot(a,b)
// Move to kernels
template<typename T, typename Q>
inline static void adotpby(int n,
                           T const alpha,
                           const T* restrict a,
                           int incx,
                           const T* restrict b,
                           int incy,
                           Q const beta,
                           Q* result, 
			   cpu_backend)
{
  T res = T(0);
  for (int i = 0, ia = 0, ib = 0; i < n; ++i, ia += incx, ib += incy)
    res += a[ia] * b[ib];
  *result = beta * (*result) + static_cast<Q>(alpha * res);
}

// y[n*inc] = beta*y[n*inc] + alpha * dot(a[n*lda],b[n*lda])
template<typename T, typename Q>
inline static void strided_adotpby(int nb,
                                   int n,
                                   T const alpha,
                                   const T* restrict a,
                                   int lda,
                                   const T* restrict b,
                                   int ldb,
                                   Q const beta,
                                   Q* result,
                                   int inc, cpu_backend)
{
  for (int k = 0; k < nb; k++)
    adotpby(n, alpha, a + k * lda, 1, b + k * ldb, 1, beta, result + k * inc, cpu_backend{});
}

// y[i] = y[i] * alpha * x[i], with appropriate strides
template<typename T>
inline static void axty(int n, T const alpha, T const* x, int incx, T* y, int incy, cpu_backend)
{
  for (int i = 0; i < n; i++)
    y[i * incy] *= alpha * x[i * incx];
}

// implements z[i][j] = beta * z[i][j] + alpha * conj(y[i][j]) * x[i]
template<typename T>
inline static void acAxpbB(int m,
                           int n,
                           T const alpha,
                           T const* A,
                           int lda,
                           T const* x,
                           int incx,
                           T const beta,
                           T* B,
                           int ldb, 
			   cpu_backend)
{
  for (int j = 0; j < n; j++)
  {
    auto Bj = B + j * ldb;
    auto Aj = A + j * lda;
    for (int i = 0; i < m; i++)
      Bj[i] = beta * Bj[i] + alpha * ma::conj(Aj[i]) * x[i * incx];
  }
}


// y[i] = y[i] + alpha*A[i][i]
template<typename T>
inline static void adiagApy(int n, T const alpha, T const* A, int lda, T* y, int incy, cpu_backend)
{
  for (int i = 0; i < n; i++)
    y[i * incy] += alpha * A[i * lda + i];
}

template<typename T>
inline static T sum(int n, T const* x, int incx, cpu_backend)
{
  T res(0);
  for (int i = 0; i < n; i++)
    res += x[i * incx];
  return res;
}

// assume Fortran ordering like all blas calls
template<typename T>
inline static T sum(int m, int n, T const* A, int lda, cpu_backend)
{
  T res(0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      res += A[i * lda + j];
  return res;
}

// assume Fortran ordering like all blas calls
template<typename T>
void set_identity(int m, int n, T* A, int lda, cpu_backend)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      A[i * lda + j] = ((i == j) ? 1 : 0);
}

template<typename T>
void set_identity_strided(int nbatch, int stride, int m, int n, T* A, int lda, cpu_backend)
{
  for (int b = 0, b0 = 0; b < nbatch; b++, b0 += stride)
    set_identity(m, n, A + b0, lda, cpu_backend{});
}

#ifdef HAVE_MKL

inline static void gemm_batch(const CBLAS_LAYOUT Layout,
                              const CBLAS_TRANSPOSE* transa_array,
                              const CBLAS_TRANSPOSE* transb_array,
                              const int* m_array,
                              const int* n_array,
                              const int* k_array,
                              const float* alpha_array,
                              const void** a_array,
                              const int* lda_array,
                              const void** b_array,
                              const int* ldb_array,
                              const void* beta_array,
                              void** c_array,
                              const int* ldc_array,
                              const int group_count,
                              const int* group_size)
{
  cblas_sgemm_batch(Layout, transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array,
                    b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

inline static void gemm_batch(const CBLAS_LAYOUT Layout,
                              const CBLAS_TRANSPOSE* transa_array,
                              const CBLAS_TRANSPOSE* transb_array,
                              const int* m_array,
                              const int* n_array,
                              const int* k_array,
                              const double* alpha_array,
                              const void** a_array,
                              const int* lda_array,
                              const void** b_array,
                              const int* ldb_array,
                              const void* beta_array,
                              void** c_array,
                              const int* ldc_array,
                              const int group_count,
                              const int* group_size)
{
  cblas_dgemm_batch(Layout, transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array,
                    b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

inline static void gemm_batch(const CBLAS_LAYOUT Layout,
                              const CBLAS_TRANSPOSE* transa_array,
                              const CBLAS_TRANSPOSE* transb_array,
                              const int* m_array,
                              const int* n_array,
                              const int* k_array,
                              const std::complex<float>* alpha_array,
                              const void** a_array,
                              const int* lda_array,
                              const void** b_array,
                              const int* ldb_array,
                              const void* beta_array,
                              void** c_array,
                              const int* ldc_array,
                              const int group_count,
                              const int* group_size)
{
  cblas_cgemm_batch(Layout, transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array,
                    b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

inline static void gemm_batch(const CBLAS_LAYOUT Layout,
                              const CBLAS_TRANSPOSE* transa_array,
                              const CBLAS_TRANSPOSE* transb_array,
                              const int* m_array,
                              const int* n_array,
                              const int* k_array,
                              const std::complex<double>* alpha_array,
                              const void** a_array,
                              const int* lda_array,
                              const void** b_array,
                              const int* ldb_array,
                              const void* beta_array,
                              void** c_array,
                              const int* ldc_array,
                              const int group_count,
                              const int* group_size)
{
  cblas_zgemm_batch(Layout, transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array,
                    b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

#endif

template<typename T>
inline static void gemmStridedBatched(char Atrans,
                                      char Btrans,
                                      int M,
                                      int N,
                                      int K,
                                      T alpha,
                                      const T* A,
                                      int lda,
                                      int strideA,
                                      const T* restrict B,
                                      int ldb,
                                      int strideB,
                                      T beta,
                                      T* restrict C,
                                      int ldc,
                                      int strideC,
                                      int batchSize,
				      cpu_backend)
{
#ifdef HAVE_MKL
  // MKL has batched gemm, but with pointer interface. Translate here
  std::vector<const void*> Aptrs(batchSize);
  std::vector<const void*> Bptrs(batchSize);
  std::vector<void*> Cptrs(batchSize);

  for (int i = 0; i < batchSize; i++)
  {
    Aptrs[i] = static_cast<const void*>(A + i * strideA);
    Bptrs[i] = static_cast<const void*>(B + i * strideB);
    Cptrs[i] = static_cast<void*>(C + i * strideC);
  }
  CBLAS_TRANSPOSE opA(cblas_operation(Atrans));
  CBLAS_TRANSPOSE opB(cblas_operation(Btrans));

  // Expect arrays of size group_count.
  // This is 1 in strided case, so passing pointers to everything
  gemm_batch(CblasColMajor, &opA, &opB, &M, &N, &K, &alpha, Aptrs.data(), &lda, Bptrs.data(), &ldb, &beta, Cptrs.data(),
             &ldc, 1, &batchSize);
#else
  // No batched gemm, :-( gemm loop
  for (int i = 0; i < batchSize; i++)
    ma::cpu::gemm(Atrans, Btrans, M, N, K, alpha, A + i * strideA, lda, B + i * strideB, ldb, beta, C + i * strideC, ldc);
#endif
}

template<typename T>
inline static void gemmStridedBatched(char Atrans,
                                      char Btrans,
                                      int M,
                                      int N,
                                      int K,
                                      T alpha,
                                      const std::complex<T>* A,
                                      int lda,
                                      int strideA,
                                      const T* restrict B,
                                      int ldb,
                                      int strideB,
                                      T beta,
                                      std::complex<T>* restrict C,
                                      int ldc,
                                      int strideC,
                                      int batchSize,
                                      cpu_backend)
{
  RUNTIME_CHECK(Atrans == 'n' || Atrans == 'N', "");
#ifdef HAVE_MKL
  // MKL has batched gemm, but with pointer interface. Translate here
  std::vector<const void*> Aptrs(batchSize);
  std::vector<const void*> Bptrs(batchSize);
  std::vector<void*> Cptrs(batchSize);

  for (int i = 0; i < batchSize; i++)
  {
    Aptrs[i] = static_cast<const void*>(A + i * strideA);
    Bptrs[i] = static_cast<const void*>(B + i * strideB);
    Cptrs[i] = static_cast<void*>(C + i * strideC);
  }
  CBLAS_TRANSPOSE opA(cblas_operation(Atrans));
  CBLAS_TRANSPOSE opB(cblas_operation(Btrans));

  // Expect arrays of size group_count.
  // This is 1 in strided case, so passing pointers to everything
  int M2 = 2*M;
  int lda2 = 2*lda;
  int ldc2 = 2*ldc;
  gemm_batch(CblasColMajor, &opA, &opB, &M2, &N, &K, &alpha, Aptrs.data(), &lda2, Bptrs.data(), &ldb, &beta, Cptrs.data(),
             &ldc2, 1, &batchSize);
#else
  // No batched gemm, :-( gemm loop
  for (int i = 0; i < batchSize; i++)
    ma::cpu::gemm(Atrans, Btrans, M, N, K, alpha, A + i * strideA, 2*lda, B + i * strideB, ldb, beta, C + i * strideC, 2*ldc);
#endif
}

template<typename T,
         typename Q1,
         typename Q2,
         typename = typename std::enable_if_t<std::is_same<typename std::decay<Q1>::type, T>::value>,
         typename = typename std::enable_if_t<std::is_same<typename std::decay<Q2>::type, T>::value>
	> 	
inline static void gemmBatched(char Atrans,
                               char Btrans,
                               int M,
                               int N,
                               int K,
                               T alpha,
                               Q1** A,
                               int lda,
                               Q2** B,
                               int ldb,
                               T beta,
                               T** C,
                               int ldc,
                               int batchSize,
			       cpu_backend)
{
#ifdef HAVE_MKL
  CBLAS_TRANSPOSE opA(cblas_operation(Atrans));
  CBLAS_TRANSPOSE opB(cblas_operation(Btrans));
  gemm_batch(CblasColMajor, &opA, &opB, &M, &N, &K, &alpha, (const void**)A, &lda, (const void**)B, &ldb, &beta,
             (void**)C, &ldc, 1, &batchSize);
#else
  // No batched gemm, :-( gemm loop
  for (int i = 0; i < batchSize; i++)
    ma::cpu::gemm(Atrans, Btrans, M, N, K, alpha, A[i], lda, B[i], ldb, beta, C[i], ldc);
#endif
}

//  template<typename T, typename Q>
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
                               T alpha,
                               Q1** A,
                               int lda,
                               Q2** B,
                               int ldb,
                               T beta,
                               T2** C,
                               int ldc,
                               int batchSize, 
			       cpu_backend)
{
#ifdef HAVE_MKL
  RUNTIME_CHECK(Atrans == 'n' || Atrans == 'N', "");
  CBLAS_TRANSPOSE opA(cblas_operation(Atrans));
  CBLAS_TRANSPOSE opB(cblas_operation(Btrans));
  int M_(2 * M);
  int lda_(2 * lda);
  int ldc_(2 * ldc);
  gemm_batch(CblasColMajor, &opA, &opB, &M_, &N, &K, &alpha, (const void**)A, &lda_, (const void**)B, &ldb, &beta,
             (void**)C, &ldc_, 1, &batchSize);
#else
  // No batched gemm, :-( gemm loop
  for (int i = 0; i < batchSize; i++)
    ma::cpu::gemm(Atrans, Btrans, M, N, K, alpha, A[i], lda, B[i], ldb, beta, C[i], ldc);
#endif
}

template<typename T>
inline static void axpyBatched(int n, T* x, T* const* a, int inca, T** b, int incb, int batchSize, cpu_backend)
{
  for (int i = 0; i < batchSize; i++)
    ma::cpu::axpy(n, x[i], a[i], inca, b[i], incb);
}

template<typename T>
inline static void sumGwBatched(int n, T* x, T* const* a, int inca, T** b, int incb, [[maybe_unused]] int b0, [[maybe_unused]] int nw, int batchSize, cpu_backend)
{
  // since this is not parallel, no need to coordinate over iw
  for (int i = 0; i < batchSize; i++)
    ma::cpu::axpy(n, x[i], a[i], inca, b[i], incb);
}

// A[k][i] = B[k][i][i]
template<typename T>
inline static void get_diagonal_strided(int nk, int ni, T const* B, int ldb, int stride, T* A, int lda, cpu_backend)
{
  for (int k = 0; k < nk; ++k)
  {
    auto Bk = B + k * stride;
    auto Ak = A + k * lda;
    for (int i = 0; i < ni; ++i)
      Ak[i] += Bk[i * ldb + i];
  }
}

// expand = true
//   B[ index[n] ][:] = beta B[ index[n] ][:] + alpha A[n][:]
// expand = false 
//   B[n][:] = beta B[n][:] + alpha A[ index[n] ][:]
template<class T1,
         class T2,
         class T3,
         class T4,
         class index_t
        >
inline static void copy_select_impl(int N, int M, T1 alpha, T2 const* A, int lda,
                                int Astride, T3 beta, T4* B, int ldb, int Bstride,
                                index_t const* index, int nbatch, bool expand, cpu_backend)
{
  T4 a_ = static_cast<T4>(alpha);
  T4 b_ = static_cast<T4>(beta);
  if( expand ) {
    for(int ib=0; ib<nbatch; ib++) 
    {
      for (int i = 0; i < N; ++i)
      {
        auto id = index[i];
        auto A_ = A + ib*Astride + i  * lda;
        auto B_ = B + ib*Bstride + id * ldb;
        for (int j = 0; j < M; ++j)
          B_[j] = b_ * B_[j] + a_ * static_cast<T4>(A_[j]); 
      }
    }
  } else {
    for(int ib=0; ib<nbatch; ib++) 
    {
      for (int i = 0; i < N; ++i)
      {
        auto id = index[i];
        auto A_ = A + ib*Astride + id  * lda;
        auto B_ = B + ib*Bstride + i * ldb;
        for (int j = 0; j < M; ++j)
          B_[j] = b_ * B_[j] + a_ * static_cast<T4>(A_[j]);
      }
    }
  }
}

// A[n](i,j) = std::conj(A[n](i,j)) ;
template<class T>
inline static void complex_conjugate_impl(int N, int M, std::complex<T>* A ,int lda, long stride, int nbatch, cpu_backend)
{ 
  for(int b=0; b<nbatch; b++, A+=stride) 
  {
    auto A_ = A;
    for(int i=0; i<N; i++, A_+=lda)
      for(int j=0; j<M; j++)
        A_[j] = std::conj(A_[j]);
  }
}

// A(i,j) += a;
template<class T1, class T2>
inline static void add_scalar_impl(int N, int M, T1 a, T2 *A ,int lda, cpu_backend)
{
  T2 a_ = static_cast<T2>(a);
  for(int i=0; i<N; i++, A+=lda)
    for(int j=0; j<M; j++)
      A[j]+=a_;
} 

template<class T1, class T2>
void accumulate_impl(int dim, int nrow, int ncol, T1 const alpha, T1 const* A, int lda,
            long Astride, T2* y, int incy, long ystride, int batchSize, cpu_backend)
{
  if(dim==0) {
    // y[n][i] += alpha * sum_j A[n][j][i]  for dim==0
    for(int ib=0; ib<batchSize; ib++, y+=ystride, A+=Astride)
    {
      auto A_ = A;
      for (int j = 0; j < nrow; ++j, A_+=lda)
        for (int i = 0; i < ncol; ++i)
          y[i*incy] += static_cast<T2>( alpha * A_[i] ); 
    }
  } else if( dim == 1 ) {
    // y[n][i] += alpha * sum_j A[n][i][j]  for dim==1
    for(int ib=0; ib<batchSize; ib++, y+=ystride, A+=Astride)
    { 
      auto A_ = A;
      for (int i = 0; i < nrow; ++i, A_+=lda) { 
        T2 r(0.0);
        for (int j = 0; j < ncol; ++j) 
          r += static_cast<T2>( A_[j] );
        y[i*incy] += static_cast<T2>( alpha ) * r ;
      }  
    }  
  } 
}

/*
 * Performs the generic operation: (limited to matrices for now)
 * A[i,j] = A[i,j] op x[...], 
 *   where op is {+,-,*,/} and x[...] depends on dim (0:i, 1:j, ...}
 */
template<typename T, typename T2>
void term_by_term_matrix_vector(TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                T* A, int lda, T2 const alpha, T2 const* x, int incx, cpu_backend)
{
  RUNTIME_CHECK(dim == 0 || dim == 1, "");
  T a_ = static_cast<T>(alpha);
  if (op == TOp_PLUS)
  {
    if (dim == 0)
    {
      // A[i,j] += x[i]
      for (int i = 0; i < nrow; i++, A += lda, x += incx)
        for (int j = 0; j < ncol; j++)
          A[j] += a_ * static_cast<T>(*x);
    }
    else if (dim == 1)
    {
      // A[i,j] += x[j]
      for (int i = 0; i < nrow; i++, A += lda)
        for (int j = 0; j < ncol; j++)
          A[j] += a_ * static_cast<T>(x[j * incx]);
    }
  }
  else if (op == TOp_MINUS)
  {
    if (dim == 0)
    {
      // A[i,j] += x[i]
      for (int i = 0; i < nrow; i++, A += lda, x += incx)
        for (int j = 0; j < ncol; j++)
          A[j] -= a_ * static_cast<T>(*x);
    }
    else if (dim == 1)
    {
      // A[i,j] += x[j]
      for (int i = 0; i < nrow; i++, A += lda)
        for (int j = 0; j < ncol; j++)
          A[j] -= a_ * static_cast<T>(x[j * incx]);
    }
  }
  else if (op == TOp_MUL)
  {
    if (dim == 0)
    {
      // A[i,j] += x[i]
      for (int i = 0; i < nrow; i++, A += lda, x += incx)
        for (int j = 0; j < ncol; j++)
          A[j] *= (a_ * static_cast<T>(*x));
    }
    else if (dim == 1)
    {
      // A[i,j] += x[j]
      for (int i = 0; i < nrow; i++, A += lda)
        for (int j = 0; j < ncol; j++)
          A[j] *= (a_ * static_cast<T>(x[j * incx]));
    }
  }
  else if (op == TOp_DIV)
  {
    if (dim == 0)
    {
      // A[i,j] += x[i]
      for (int i = 0; i < nrow; i++, A += lda, x += incx)
        for (int j = 0; j < ncol; j++)
          A[j] /= (a_ * static_cast<T>(*x));
    }
    else if (dim == 1)
    {
      // A[i,j] += x[j]
      for (int i = 0; i < nrow; i++, A += lda)
        for (int j = 0; j < ncol; j++)
          A[j] /= (a_ * static_cast<T>(x[j * incx]));
    }
  }
  else
  {
    APP_ABORT(" Error: Unknown operation in term_by_term_matrix_vector. \n");
  }
}

template<typename T, typename T2>
void term_by_term_matrix_vector_strided(TENSOR_OPERATIONS op, int dim, int nrow, int ncol,
                T* A, int lda, int Astride, T2 const alpha, T2 const* x, int incx,
                int Xstride, int batchSize, cpu_backend)
{
  for(int n=0; n<batchSize; ++n)
    term_by_term_matrix_vector(op,dim,nrow,ncol,A+n*Astride,lda,alpha,x+n*Xstride,incx,cpu_backend{});
}

/*
 * B[i,j] = B[i,j] op A[i,j], where op is {+,-,*,/} 
 */
template<typename T1, typename T2, typename T3>
void term_by_term_matrix_matrix_strided(TENSOR_OPERATIONS op, int nrow, int ncol,
                T1 const alpha, T2 const* A, int lda, long Astride, 
                T3* B, int ldb, long Bstride, int batchSize, cpu_backend)
{
  T3 a_ = static_cast<T3>(alpha);
  if (op == TOp_PLUS)
  {
    // B[i,j] += alpha * A[i,j] 
    for(int ib=0; ib<batchSize; ++ib, A+=Astride, B+=Bstride) {
      auto A_ = A;
      auto B_ = B;
      for (int i = 0; i < nrow; i++, A_ += lda, B_ += ldb)
        for (int j = 0; j < ncol; j++)
          B_[j] += a_ * static_cast<T3>(A_[j]);
    }
  }
  else if (op == TOp_MINUS)
  {  
    // B[i,j] -= alpha * A[i,j] 
    for(int ib=0; ib<batchSize; ++ib, A+=Astride, B+=Bstride) {
      auto A_ = A;
      auto B_ = B;
      for (int i = 0; i < nrow; i++, A_ += lda, B_ += ldb)
        for (int j = 0; j < ncol; j++)
          B_[j] -= a_ * static_cast<T3>(A_[j]);
    }
  } 
  else if (op == TOp_MUL)
  {
    // B[i,j] *= alpha * A[i,j] 
    for(int ib=0; ib<batchSize; ++ib, A+=Astride, B+=Bstride) {
      auto A_ = A;
      auto B_ = B;
      for (int i = 0; i < nrow; i++, A_ += lda, B_ += ldb)
        for (int j = 0; j < ncol; j++)
          B_[j] *= a_ * static_cast<T3>(A_[j]);
    }
  } 
  else if (op == TOp_DIV)
  {
    // B[i,j] /= (alpha * A[i,j]) 
    for(int ib=0; ib<batchSize; ++ib, A+=Astride, B+=Bstride) {
      auto A_ = A;
      auto B_ = B;
      for (int i = 0; i < nrow; i++, A_ += lda, B_ += ldb)
        for (int j = 0; j < ncol; j++)
          B_[j] /= (a_ * static_cast<T3>(A_[j]));
    }
  } 
}

/*
 */
template<typename T1, typename T2, typename T3>
void fill_if_zero_impl(int nrow, int ncol, T1 const* key, int incx,
                T2 const alpha, T3* A, int lda, long stride, int batchSize, cpu_backend)
{
  T3 alp = static_cast<T3>(alpha);
  for(int n=0; n<batchSize; n++) {
    if(key[n*incx] == std::abs(T1(0))) {
      for (long i = 0; i < nrow; ++i) {
        auto A_ = A + n*stride + i*lda;
        for (long j = 0; j < ncol; ++j)
          A_[j] = alp;
      } 
    }
  }
}

template<typename T1, typename T2, typename T3>
void fill_if_non_zero_impl(int nrow, int ncol, T1 const* key, int incx,
                T2 const alpha, T3* A, int lda, long stride, int batchSize, cpu_backend)
{
  T3 alp = static_cast<T3>(alpha);
  for(int n=0; n<batchSize; n++) {
    if(key[n*incx] != std::abs(T1(0))) {
      for (long i = 0; i < nrow; ++i) {
        auto A_ = A + n*stride + i*lda;
        for (long j = 0; j < ncol; ++j)
          A_[j] = alp;
      }
    }
  }
}

template<typename T1, typename T2>
void copy_n_cast_impl(int N, int M, T1 const* A, int lda, long Astride,
                      T2* B, int ldb, long Bstride, int nbatch, cpu_backend)
{
  for(int n=0; n<nbatch; n++, A+=Astride, B+=Bstride) {
    auto A_ = A;
    auto B_ = B;
    for (long i = 0; i < N; ++i, A_+=lda, B_+=ldb) {
      for (long j = 0; j < M; ++j)
        B_[j] = static_cast<T2>(A_[j]);
    }
  }
}

} // namespace ma

#endif
