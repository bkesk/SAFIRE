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

#ifndef BLAS_CPU_H
#define BLAS_CPU_H

// generic header for blas routines
#include "Numerics/detail/CPU/Blasf.h"
#include "config.0.h"

namespace ma
{
namespace cpu
{

/*
inline static void axpy(int n, double x, const double* a, double* b) 
{ 
  daxpy(n, x, a, INCX, b, INCY); 
}
*/

inline static void axpy(int n, double x, const double* a, int incx, double* b, int incy)
{
  daxpy(n, x, a, incx, b, incy);
}

//inline static void axpy(int n, const double* a, double* b) { daxpy(n, done, a, INCX, b, INCY); }

inline static void axpy(int n, float x, const float* a, int incx, float* b, int incy) { saxpy(n, x, a, incx, b, incy); }

inline static void axpy(int n,
                        const std::complex<float> x,
                        const std::complex<float>* a,
                        int incx,
                        std::complex<float>* b,
                        int incy)
{
  caxpy(n, x, a, incx, b, incy);
}

inline static void axpy(int n,
                        const std::complex<double> x,
                        const std::complex<double>* a,
                        int incx,
                        std::complex<double>* b,
                        int incy)
{
  zaxpy(n, x, a, incx, b, incy);
}

inline static void axpy(int n, const float x, const float* a, int incx, double* b, int incy)
{
  for (int i = 0; i < n; ++i, a += incx, b += incy)
    (*b) += static_cast<double>(x * (*a));
}

inline static void axpy(int n, const double x, const double* a, int incx, float* b, int incy)
{
  for (int i = 0; i < n; ++i, a += incx, b += incy)
    (*b) += static_cast<float>(x * (*a));
}

inline static void axpy(int n,
                        const std::complex<float> x,
                        const std::complex<float>* a,
                        int incx,
                        std::complex<double>* b,
                        int incy)
{
  for (int i = 0; i < n; ++i, a += incx, b += incy)
    (*b) += static_cast<std::complex<double>>(x * (*a));
}

inline static void axpy(int n,
                        const std::complex<double> x,
                        const std::complex<double>* a,
                        int incx,
                        std::complex<float>* b,
                        int incy)
{
  for (int i = 0; i < n; ++i, a += incx, b += incy)
    (*b) += static_cast<std::complex<float>>(x * (*a));
}

inline static double norm2(int n, const double* a, int incx = 1) { return dnrm2(n, a, incx); }

inline static double norm2(int n, const std::complex<double>* a, int incx = 1) { return dznrm2(n, a, incx); }

inline static float norm2(int n, const float* a, int incx = 1) { return snrm2(n, a, incx); }

inline static void scal(int n, float alpha, float* x, int incx = 1) { sscal(n, alpha, x, incx); }

inline static void scal(int n, std::complex<float> alpha, std::complex<float>* x, int incx = 1)
{
  cscal(n, alpha, x, incx);
}

inline static void scal(int n, double alpha, double* x, int incx = 1) { dscal(n, alpha, x, incx); }

inline static void scal(int n, std::complex<double> alpha, std::complex<double>* x, int incx = 1)
{
  zscal(n, alpha, x, incx);
}

inline static void scal(int n, double alpha, std::complex<double>* x, int incx = 1) { zdscal(n, alpha, x, incx); }

inline static void scal(int n, float alpha, std::complex<float>* x, int incx = 1) { csscal(n, alpha, x, incx); }

inline static void gemv(char trans_in,
                        int n,
                        int m,
                        double alpha,
                        const double* restrict amat,
                        int lda,
                        const double* x,
                        int incx,
                        double beta,
                        double* y,
                        int incy)
{
  dgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
}

inline static void gemv(char trans_in,
                        int n,
                        int m,
                        float alpha,
                        const float* restrict amat,
                        int lda,
                        const float* x,
                        int incx,
                        float beta,
                        float* y,
                        int incy)
{
  sgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
}

inline static void gemv(char trans_in,
                        int n,
                        int m,
                        const std::complex<double>& alpha,
                        const std::complex<double>* restrict amat,
                        int lda,
                        const std::complex<double>* restrict x,
                        int incx,
                        const std::complex<double>& beta,
                        std::complex<double>* y,
                        int incy)
{
  zgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
}

inline static void gemv(char trans_in,
                        int n,
                        int m,
                        const std::complex<float>& alpha,
                        const std::complex<float>* restrict amat,
                        int lda,
                        const std::complex<float>* restrict x,
                        int incx,
                        const std::complex<float>& beta,
                        std::complex<float>* y,
                        int incy)
{
  cgemv(trans_in, n, m, alpha, amat, lda, x, incx, beta, y, incy);
}

#if defined(HAVE_MKL)
inline static void gemv(char trans_in,
                        int n,
                        int m,
                        const double& alpha,
                        const double* restrict amat,
                        int lda,
                        const std::complex<double>* restrict x,
                        int incx,
                        const double& beta,
                        std::complex<double>* y,
                        int incy)
{
  dzgemv(trans_in, n, m, std::complex<double>(alpha), amat, lda, x, incx, std::complex<double>(beta), y, incy);
}

inline static void gemv(char trans_in,
                        int n,
                        int m,
                        const float& alpha,
                        const float* restrict amat,
                        int lda,
                        const std::complex<float>* restrict x,
                        int incx,
                        const float& beta,
                        std::complex<float>* y,
                        int incy)
{
  scgemv(trans_in, n, m, std::complex<float>(alpha), amat, lda, x, incx, std::complex<float>(beta), y, incy);
}
#else
inline static void gemv(char trans_in,
                        int n,
                        int m,
                        const double& alpha,
                        const double* restrict amat,
                        int lda,
                        const std::complex<double>* restrict x,
                        int incx,
                        const double& beta,
                        std::complex<double>* y,
                        int incy)
{
  // A * x  --(Fortran)--> A^T * x --> X * A, where X is the interpretation of x as a 2 row matrix
  // A^T * x  --(Fortran)--> A * x --> X * A^T, where X is the interpretation of x as a 2 row matrix
  if (trans_in == 'n' || trans_in == 'N')
    dgemm('N', 'T', 2, n, m, alpha, reinterpret_cast<double const*>(x), 2 * incx, amat, lda, beta,
          reinterpret_cast<double*>(y), 2 * incy);
  else if (trans_in == 't' || trans_in == 'T')
    dgemm('N', 'N', 2, m, n, alpha, reinterpret_cast<double const*>(x), 2 * incx, amat, lda, beta,
          reinterpret_cast<double*>(y), 2 * incy);
  else
  {
    print_stacktrace throw std::runtime_error("Error: Incorrect trans_in. \n");
  }
}

inline static void gemv(char trans_in,
                        int n,
                        int m,
                        const float& alpha,
                        const float* restrict amat,
                        int lda,
                        const std::complex<float>* restrict x,
                        int incx,
                        const float& beta,
                        std::complex<float>* y,
                        int incy)
{
  // A * x  --(Fortran)--> A^T * x --> X * A, where X is the interpretation of x as a 2 row matrix
  // A^T * x  --(Fortran)--> A * x --> X * A^T, where X is the interpretation of x as a 2 row matrix
  if (trans_in == 'n' || trans_in == 'N')
    sgemm('N', 'T', 2, n, m, alpha, reinterpret_cast<float const*>(x), 2 * incx, amat, lda, beta,
          reinterpret_cast<float*>(y), 2 * incy);
  else if (trans_in == 't' || trans_in == 'T')
    sgemm('N', 'N', 2, m, n, alpha, reinterpret_cast<float const*>(x), 2 * incx, amat, lda, beta,
          reinterpret_cast<float*>(y), 2 * incy);
  else
  {
    print_stacktrace throw std::runtime_error("Error: Incorrect trans_in. \n");
  }
}
#endif

inline static void gemm(char Atrans,
                        char Btrans,
                        int M,
                        int N,
                        int K,
                        double alpha,
                        const double* A,
                        int lda,
                        const double* restrict B,
                        int ldb,
                        double beta,
                        double* restrict C,
                        int ldc)
{
  dgemm(Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline static void gemm(char Atrans,
                        char Btrans,
                        int M,
                        int N,
                        int K,
                        float alpha,
                        const float* A,
                        int lda,
                        const float* restrict B,
                        int ldb,
                        float beta,
                        float* restrict C,
                        int ldc)
{
  sgemm(Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline static void gemm(char Atrans,
                        char Btrans,
                        int M,
                        int N,
                        int K,
                        std::complex<double> alpha,
                        const std::complex<double>* A,
                        int lda,
                        const std::complex<double>* restrict B,
                        int ldb,
                        std::complex<double> beta,
                        std::complex<double>* restrict C,
                        int ldc)
{
  zgemm(Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline static void gemm(char Atrans,
                        char Btrans,
                        int M,
                        int N,
                        int K,
                        std::complex<float> alpha,
                        const std::complex<float>* A,
                        int lda,
                        const std::complex<float>* restrict B,
                        int ldb,
                        std::complex<float> beta,
                        std::complex<float>* restrict C,
                        int ldc)
{
  cgemm(Atrans, Btrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline static void gemm(char Atrans,
                        char Btrans,
                        int M,
                        int N,
                        int K,
                        double alpha,
                        const std::complex<double>* A,
                        int lda,
                        const double* restrict B,
                        int ldb,
                        double beta,
                        std::complex<double>* restrict C,
                        int ldc)
{
  RUNTIME_CHECK(Atrans == 'n' || Atrans == 'N', "");
  dgemm(Atrans, Btrans, 2 * M, N, K, alpha, reinterpret_cast<double const*>(A), 2 * lda, B, ldb, beta,
        reinterpret_cast<double*>(C), 2 * ldc);
}

inline static void gemm(char Atrans,
                        char Btrans,
                        int M,
                        int N,
                        int K,
                        float alpha,
                        const std::complex<float>* A,
                        int lda,
                        const float* restrict B,
                        int ldb,
                        float beta,
                        std::complex<float>* restrict C,
                        int ldc)
{
  RUNTIME_CHECK(Atrans == 'n' || Atrans == 'N', "");
  sgemm(Atrans, Btrans, 2 * M, N, K, alpha, reinterpret_cast<float const*>(A), 2 * lda, B, ldb, beta,
        reinterpret_cast<float*>(C), 2 * ldc);
}


template<typename T>
inline static T dot(int n, const T* restrict a, const T* restrict b)
{
  T res = T(0);
  for (int i = 0; i < n; ++i)
    res += a[i] * b[i];
  return res;
}

template<typename T>
inline static std::complex<T> dot(int n, const std::complex<T>* restrict a, const T* restrict b)
{
  std::complex<T> res = T(0);
  for (int i = 0; i < n; ++i)
    res += a[i] * b[i];
  return res;
}

template<typename T>
inline static std::complex<T> dot(int n, const std::complex<T>* restrict a, const std::complex<T>* restrict b)
{
  std::complex<T> res = 0.0;
  for (int i = 0; i < n; ++i)
    res += a[i] * b[i];
  return res;
}

template<typename T>
inline static std::complex<T> dot(int n, const T* restrict a, const std::complex<T>* restrict b)
{
  std::complex<T> res = 0.0;
  for (int i = 0; i < n; ++i)
    res += a[i] * b[i];
  return res;
}

template<typename T>
inline static T dot(int n, const T* restrict a, int incx, const T* restrict b, int incy)
{
  T res = T(0);
  for (int i = 0, ia = 0, ib = 0; i < n; ++i, ia += incx, ib += incy)
    res += a[ia] * b[ib];
  return res;
}

template<typename T>
inline static std::complex<T> dot(int n, const std::complex<T>* restrict a, int incx, const T* restrict b, int incy)
{
  std::complex<T> res = T(0);
  for (int i = 0, ia = 0, ib = 0; i < n; ++i, ia += incx, ib += incy)
    res += a[ia] * b[ib];
  return res;
}

template<typename T>
inline static std::complex<T> dot(int n, const T* restrict a, int incx, const std::complex<T>* restrict b, int incy)
{
  std::complex<T> res = T(0);
  for (int i = 0, ia = 0, ib = 0; i < n; ++i, ia += incx, ib += incy)
    res += a[ia] * b[ib];
  return res;
}

template<typename T>
std::complex<T> dot(int n, const std::complex<T>* a, int incx, const std::complex<T>* b, int incy)
{
  std::complex<T> res = T(0);
  for (int i = 0, ia = 0, ib = 0; i < n; ++i, ia += incx, ib += incy)
    res += a[ia] * b[ib];
  return res;
}

template<typename T>
inline static void copy(int n, const T* restrict a, T* restrict b)
{
  memcpy(b, a, sizeof(T) * n);
}

/** copy using memcpy(target,source,size)
   * @param target starting address of the targe
   * @param source starting address of the source
   * @param number of elements to copy
   */
template<typename T>
inline static void copy(T* restrict target, const T* restrict source, int n)
{
  memcpy(target, source, sizeof(T) * n);
}

template<typename T>
inline static void copy(int n, const std::complex<T>* restrict a, T* restrict b)
{
  for (int i = 0; i < n; ++i)
    b[i] = a[i].real();
}

template<typename T>
inline static void copy(int n, const T* restrict a, std::complex<T>* restrict b)
{
  for (int i = 0; i < n; ++i)
    b[i] = a[i];
}

template<typename T>
inline static void copy(int n, const T* restrict x, int incx, T* restrict y, int incy)
{
  const int xmax = incx * n;
  for (int ic = 0, jc = 0; ic < xmax; ic += incx, jc += incy)
    y[jc] = x[ic];
}

template<typename T>
inline static void copy2D(int N, int M, T const* src, int lda, T* dst, int ldb)
{
  for (int i = 0; i < N; ++i, src += lda, dst += ldb)
    copy(M, src, 1, dst, 1);
}

inline static void ger(int m,
                       int n,
                       double alpha,
                       const double* x,
                       int incx,
                       const double* y,
                       int incy,
                       double* a,
                       int lda)
{
  dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

inline static void ger(int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda)
{
  sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

inline static void ger(int m,
                       int n,
                       const std::complex<double>& alpha,
                       const std::complex<double>* x,
                       int incx,
                       const std::complex<double>* y,
                       int incy,
                       std::complex<double>* a,
                       int lda)
{
  zgeru(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

inline static void ger(int m,
                       int n,
                       const std::complex<float>& alpha,
                       const std::complex<float>* x,
                       int incx,
                       const std::complex<float>* y,
                       int incy,
                       std::complex<float>* a,
                       int lda)
{
  cgeru(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

} // namespace cpu
} // namespace ma
#endif
