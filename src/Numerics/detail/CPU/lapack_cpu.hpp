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

#ifndef NUMERICS_DETAIL_CPU_LAPACK_CPU_H
#define NUMERICS_DETAIL_CPU_LAPACK_CPU_H

// generic header for blas routines
#include "Numerics/detail/CPU/Blasf.h"
#include "Numerics/detail/utilities.hpp"

//#ifdef ENABLE_CUDA
//#define __NO_QR__
//#endif

namespace ma
{
namespace cpu
{

inline void gesvd(char jobu,
                         char jobvt,
                         int m,
                         int n,
                         float* a,
                         int lda,
                         float* s,
                         float* u,
                         int ldu,
                         float* vt,
                         int ldvt,
                         float* work,
                         int lwork,
                         int& info)
{
  sgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
}

inline void gesvd(char jobu,
                         char jobvt,
                         int m,
                         int n,
                         double* a,
                         int lda,
                         double* s,
                         double* u,
                         int ldu,
                         double* vt,
                         int ldvt,
                         double* work,
                         int lwork,
                         int& info)
{
  dgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
}

inline void gesvd(char jobu,
                         char jobvt,
                         int m,
                         int n,
                         std::complex<float>* a,
                         int lda,
                         float* s,
                         std::complex<float>* u,
                         int ldu,
                         std::complex<float>* vt,
                         int ldvt,
                         std::complex<float>* work,
                         int lwork,
                         float* rwork,
                         int& info)
{
  cgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
}

inline void gesvd(char jobu,
                         char jobvt,
                         int m,
                         int n,
                         std::complex<double>* a,
                         int lda,
                         double* s,
                         std::complex<double>* u,
                         int ldu,
                         std::complex<double>* vt,
                         int ldvt,
                         std::complex<double>* work,
                         int lwork,
                         double* rwork,
                         int& info)
{
  zgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
}

template<typename T>
inline void gesvd_bufferSize(const int m, const int n, T* a, int& lwork)
{
  T work, S;
  int status;
  lwork = -1;
  gesvd('A', 'A', m, n, a, m, nullptr, nullptr, m, nullptr, m, &work, lwork, status);
  lwork = int(work);
}

template<typename T>
inline void gesvd_bufferSize(const int m, const int n, std::complex<T>* a, int& lwork)
{
  std::complex<T> work;
  T rwork;
  int status;
  lwork = -1;
  gesvd('A', 'A', m, n, a, m, nullptr, nullptr, m, nullptr, m, &work, lwork, &rwork, status);
  lwork = int(real(work));
}

inline void hevr(char& JOBZ,
                        char& RANGE,
                        char& UPLO,
                        int& N,
                        float* A,
                        int& LDA,
                        float& VL,
                        float& VU,
                        int& IL,
                        int& IU,
                        float& ABSTOL,
                        int& M,
                        float* W,
                        float* Z,
                        int& LDZ,
                        int* ISUPPZ,
                        [[maybe_unused]] float* WORK,
                        int& LWORK,
                        float* RWORK,
                        int& LRWORK,
                        int* IWORK,
                        int& LIWORK,
                        int& INFO)
{
  bool query = (LWORK == -1) or (LRWORK == -1) or (LIWORK == -1);
  if (query)
  {
    LWORK  = -1;
    LRWORK = -1;
    LIWORK = -1;
  }
  ssyevr(JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, ISUPPZ, RWORK, LRWORK, IWORK, LIWORK,
         INFO);
  if (query)
  {
    LWORK  = int(0);
    LRWORK = int(RWORK[0]);
    LIWORK = int(IWORK[0]);
  }
}

inline void hevr(char& JOBZ,
                        char& RANGE,
                        char& UPLO,
                        int& N,
                        double* A,
                        int& LDA,
                        double& VL,
                        double& VU,
                        int& IL,
                        int& IU,
                        double& ABSTOL,
                        int& M,
                        double* W,
                        double* Z,
                        int& LDZ,
                        int* ISUPPZ,
                        [[maybe_unused]] double* WORK,
                        int& LWORK,
                        double* RWORK,
                        int& LRWORK,
                        int* IWORK,
                        int& LIWORK,
                        int& INFO)
{
  bool query = (LWORK == -1) or (LRWORK == -1) or (LIWORK == -1);
  if (query)
  {
    LWORK  = -1;
    LRWORK = -1;
    LIWORK = -1;
  }
  dsyevr(JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, ISUPPZ, RWORK, LRWORK, IWORK, LIWORK,
         INFO);
  if (query)
  {
    LWORK  = int(0);
    LRWORK = int(RWORK[0]);
    LIWORK = int(IWORK[0]);
  }
}

inline void hevr(char& JOBZ,
                        char& RANGE,
                        char& UPLO,
                        int& N,
                        std::complex<float>* A,
                        int& LDA,
                        float& VL,
                        float& VU,
                        int& IL,
                        int& IU,
                        float& ABSTOL,
                        int& M,
                        float* W,
                        std::complex<float>* Z,
                        int& LDZ,
                        int* ISUPPZ,
                        std::complex<float>* WORK,
                        int& LWORK,
                        float* RWORK,
                        int& LRWORK,
                        int* IWORK,
                        int& LIWORK,
                        int& INFO)
{
  bool query = (LWORK == -1) or (LRWORK == -1) or (LIWORK == -1);
  if (query)
  {
    LWORK  = -1;
    LRWORK = -1;
    LIWORK = -1;
  }
  cheevr(JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, ISUPPZ, WORK, LWORK, RWORK, LRWORK, IWORK,
         LIWORK, INFO);
  if (query)
  {
    LWORK  = int(real(WORK[0]));
    LRWORK = int(RWORK[0]);
    LIWORK = int(IWORK[0]);
  }
}

inline void hevr(char& JOBZ,
                        char& RANGE,
                        char& UPLO,
                        int& N,
                        std::complex<double>* A,
                        int& LDA,
                        double& VL,
                        double& VU,
                        int& IL,
                        int& IU,
                        double& ABSTOL,
                        int& M,
                        double* W,
                        std::complex<double>* Z,
                        int& LDZ,
                        int* ISUPPZ,
                        std::complex<double>* WORK,
                        int& LWORK,
                        double* RWORK,
                        int& LRWORK,
                        int* IWORK,
                        int& LIWORK,
                        int& INFO)
{
  bool query = (LWORK == -1) or (LRWORK == -1) or (LIWORK == -1);
  if (query)
  {
    LWORK  = -1;
    LRWORK = -1;
    LIWORK = -1;
  }
  zheevr(JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, ISUPPZ, WORK, LWORK, RWORK, LRWORK, IWORK,
         LIWORK, INFO);
  if (query)
  {
    LWORK  = int(real(WORK[0]));
    LRWORK = int(RWORK[0]);
    LIWORK = int(IWORK[0]);
  }
}

inline void gvx(int ITYPE,
                       char& JOBZ,
                       char& RANGE,
                       char& UPLO,
                       int& N,
                       float* A,
                       int& LDA,
                       float* B,
                       int& LDB,
                       float& VL,
                       float& VU,
                       int& IL,
                       int& IU,
                       float& ABSTOL,
                       int& M,
                       float* W,
                       float* Z,
                       int& LDZ,
                       float* WORK,
                       int& LWORK,
                       [[maybe_unused]] float* RWORK,
                       int* IWORK,
                       int* IFAIL,
                       int& INFO)
{
  bool query = (LWORK == -1);
  if (query)
  {
    LWORK = -1;
  }
  ssygvx(ITYPE, JOBZ, RANGE, UPLO, N, A, LDA, B, LDB, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, LWORK, IWORK, IFAIL,
         INFO);
  if (query)
  {
    LWORK = int(WORK[0]);
  }
}

inline void gvx(int ITYPE,
                       char& JOBZ,
                       char& RANGE,
                       char& UPLO,
                       int& N,
                       double* A,
                       int& LDA,
                       double* B,
                       int& LDB,
                       double& VL,
                       double& VU,
                       int& IL,
                       int& IU,
                       double& ABSTOL,
                       int& M,
                       double* W,
                       double* Z,
                       int& LDZ,
                       double* WORK,
                       int& LWORK,
                       [[maybe_unused]] double* RWORK,
                       int* IWORK,
                       int* IFAIL,
                       int& INFO)
{
  bool query = (LWORK == -1);
  if (query)
  {
    LWORK = -1;
  }
  dsygvx(ITYPE, JOBZ, RANGE, UPLO, N, A, LDA, B, LDB, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, LWORK, IWORK, IFAIL,
         INFO);
  if (query)
  {
    LWORK = int(WORK[0]);
  }
}

inline void gvx(int ITYPE,
                       char& JOBZ,
                       char& RANGE,
                       char& UPLO,
                       int& N,
                       std::complex<float>* A,
                       int& LDA,
                       std::complex<float>* B,
                       int& LDB,
                       float& VL,
                       float& VU,
                       int& IL,
                       int& IU,
                       float& ABSTOL,
                       int& M,
                       float* W,
                       std::complex<float>* Z,
                       int& LDZ,
                       std::complex<float>* WORK,
                       int& LWORK,
                       float* RWORK,
                       int* IWORK,
                       int* IFAIL,
                       int& INFO)
{
  bool query = (LWORK == -1);
  if (query)
  {
    LWORK = -1;
  }
  chegvx(ITYPE, JOBZ, RANGE, UPLO, N, A, LDA, B, LDB, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, LWORK, RWORK, IWORK,
         IFAIL, INFO);
  if (query)
  {
    LWORK = int(real(WORK[0]));
  }
}

inline void gvx(int ITYPE,
                       char& JOBZ,
                       char& RANGE,
                       char& UPLO,
                       int& N,
                       std::complex<double>* A,
                       int& LDA,
                       std::complex<double>* B,
                       int& LDB,
                       double& VL,
                       double& VU,
                       int& IL,
                       int& IU,
                       double& ABSTOL,
                       int& M,
                       double* W,
                       std::complex<double>* Z,
                       int& LDZ,
                       std::complex<double>* WORK,
                       int& LWORK,
                       double* RWORK,
                       int* IWORK,
                       int* IFAIL,
                       int& INFO)
{
  bool query = (LWORK == -1);
  if (query)
  {
    LWORK = -1;
  }
  zhegvx(ITYPE, JOBZ, RANGE, UPLO, N, A, LDA, B, LDB, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, LWORK, RWORK, IWORK,
         IFAIL, INFO);
  if (query)
  {
    LWORK = int(real(WORK[0]));
  }
}

inline void getrf_bufferSize([[maybe_unused]] const int n, [[maybe_unused]] const int m, [[maybe_unused]] float* a, [[maybe_unused]] const int lda, [[maybe_unused]] int& lwork) { lwork = 0; }
inline void getrf_bufferSize([[maybe_unused]] const int n, [[maybe_unused]] const int m, [[maybe_unused]] double* a, [[maybe_unused]] const int lda, [[maybe_unused]] int& lwork) { lwork = 0; }
inline void getrf_bufferSize([[maybe_unused]] const int n, [[maybe_unused]] const int m, [[maybe_unused]] std::complex<float>* a, [[maybe_unused]] const int lda, [[maybe_unused]] int& lwork)
{
  lwork = 0;
}
inline void getrf_bufferSize([[maybe_unused]]  const int n, [[maybe_unused]] const int m, [[maybe_unused]] std::complex<double>* a, [[maybe_unused]] const int lda, [[maybe_unused]] int& lwork)
{
  lwork = 0;
}

inline void getrf(const int& n, const int& m, double* a, const int& n0, int* piv, int& st, [[maybe_unused]] double* work = nullptr)
{
  dgetrf(n, m, a, n0, piv, st);
}

inline void getrf(const int& n, const int& m, float* a, const int& n0, int* piv, int& st, [[maybe_unused]] float* work = nullptr)
{
  sgetrf(n, m, a, n0, piv, st);
}

inline void getrf(const int& n,
                  const int& m,
                  std::complex<double>* a,
                  const int& n0,
                  int* piv,
                  int& st,
                  [[maybe_unused]] std::complex<double>* work = nullptr)
{
  zgetrf(n, m, a, n0, piv, st);
}

inline void getrf(const int& n,
                  const int& m,
                  std::complex<float>* a,
                  const int& n0,
                  int* piv,
                  int& st,
                  [[maybe_unused]] std::complex<float>* work = nullptr)
{
  cgetrf(n, m, a, n0, piv, st);
}

inline void getrfBatched(const int n, float** a, int lda, int* piv, int* info, int batchSize)
{
  for (int i = 0; i < batchSize; i++)
    getrf(n, n, a[i], lda, piv + i * n, *(info + i));
}

inline void getrfBatched(const int n, double** a, int lda, int* piv, int* info, int batchSize)
{
  for (int i = 0; i < batchSize; i++)
    getrf(n, n, a[i], lda, piv + i * n, *(info + i));
}

inline void getrfBatched(const int n, std::complex<double>** a, int lda, int* piv, int* info, int batchSize)
{
  for (int i = 0; i < batchSize; i++)
    getrf(n, n, a[i], lda, piv + i * n, *(info + i));
}

inline void getrfBatched(const int n, std::complex<float>** a, int lda, int* piv, int* info, int batchSize)
{
  for (int i = 0; i < batchSize; i++)
    getrf(n, n, a[i], lda, piv + i * n, *(info + i));
}

inline void getri_bufferSize(int n, [[maybe_unused]]  float const* a, int lda, int& lwork)
{
  float work;
  int status;
  lwork = -1;
  sgetri(n, nullptr, lda, nullptr, &work, lwork, status);
  lwork = int(work);
}

inline void getri_bufferSize(int n, [[maybe_unused]] double const* a, int lda, int& lwork)
{
  double work;
  int status;
  lwork = -1;
  dgetri(n, nullptr, lda, nullptr, &work, lwork, status);
  lwork = int(work);
}

inline void getri_bufferSize(int n, [[maybe_unused]] std::complex<float> const* a, int lda, int& lwork)
{
  std::complex<float> work;
  int status;
  lwork = -1;
  cgetri(n, nullptr, lda, nullptr, &work, lwork, status);
  lwork = int(real(work));
}

inline void getri_bufferSize(int n, [[maybe_unused]] std::complex<double> const* a, int lda, int& lwork)
{
  std::complex<double> work;
  int status;
  lwork = -1;
  zgetri(n, nullptr, lda, nullptr, &work, lwork, status);
  lwork = int(real(work));
}


inline void getri(int n, float* restrict a, int n0, int const* restrict piv, float* restrict work, int n1, int& status)
{
  sgetri(n, a, n0, piv, work, n1, status);
}

inline void getri(int n,
                  double* restrict a,
                  int n0,
                  int const* restrict piv,
                  double* restrict work,
                  int n1,
                  int& status)
{
  dgetri(n, a, n0, piv, work, n1, status);
}

inline void getri(int n,
                  std::complex<float>* restrict a,
                  int n0,
                  int const* restrict piv,
                  std::complex<float>* restrict work,
                  int n1,
                  int& status)
{
  cgetri(n, a, n0, piv, work, n1, status);
}

inline void getri(int n,
                  std::complex<double>* restrict a,
                  int n0,
                  int const* restrict piv,
                  std::complex<double>* restrict work,
                  int n1,
                  int& status)
{
  zgetri(n, a, n0, piv, work, n1, status);
}

template<typename T>
inline void getriBatched(int n, T** a, int lda, int* piv, T** ainv, int ldc, int* info, int batchSize)
{
  for (int i = 0; i < batchSize; i++)
  {
    getri(n, a[i], lda, piv + i * n, ainv[i], n * n, *(info + i));
    for (int j = 0; j < n; j++)
      std::copy_n(a[i] + j * lda, n, ainv[i] + j * ldc);
  }
}

inline void geqrf(int M,
                  int N,
                  std::complex<double>* A,
                  const int LDA,
                  std::complex<double>* TAU,
                  std::complex<double>* WORK,
                  int LWORK,
                  int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  zgeqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void geqrf(int M, int N, double* A, const int LDA, double* TAU, double* WORK, int LWORK, int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  dgeqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void geqrf(int M,
                  int N,
                  std::complex<float>* A,
                  const int LDA,
                  std::complex<float>* TAU,
                  std::complex<float>* WORK,
                  int LWORK,
                  int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  cgeqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void geqrf(int M, int N, float* A, const int LDA, float* TAU, float* WORK, int LWORK, int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  sgeqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

template<typename T>
inline void geqrf_bufferSize(int m, int n, [[maybe_unused]] T* a, int lda, int& lwork)
{
  typename std::decay<T>::type work;
  int status;
  lwork = -1;
  geqrf(m, n, nullptr, lda, nullptr, &work, lwork, status);
  lwork = int(real(work));
}

inline void gelqf(int M,
                  int N,
                  std::complex<double>* A,
                  const int LDA,
                  std::complex<double>* TAU,
                  std::complex<double>* WORK,
                  int LWORK,
                  int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  zgelqf(M, N, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void gelqf(int M, int N, double* A, const int LDA, double* TAU, double* WORK, int LWORK, int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  dgelqf(M, N, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void gelqf(int M,
                  int N,
                  std::complex<float>* A,
                  const int LDA,
                  std::complex<float>* TAU,
                  std::complex<float>* WORK,
                  int LWORK,
                  int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  cgelqf(M, N, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void gelqf(int M, int N, float* A, const int LDA, float* TAU, float* WORK, int LWORK, int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  sgelqf(M, N, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

template<typename T>
inline void gelqf_bufferSize(int m, int n, [[maybe_unused]] T* a, int lda, int& lwork)
{
  typename std::decay<T>::type work;
  int status;
  lwork = -1;
  gelqf(m, n, nullptr, lda, nullptr, &work, lwork, status);
  lwork = int(real(work));
}

inline void gqr(int M,
                int N,
                int K,
                std::complex<double>* A,
                const int LDA,
                std::complex<double>* TAU,
                std::complex<double>* WORK,
                int LWORK,
                int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  zungqr(M, N, K, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void gqr(int M, int N, int K, double* A, const int LDA, double* TAU, double* WORK, int LWORK, int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  dorgqr(M, N, K, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void gqr(int M,
                int N,
                int K,
                std::complex<float>* A,
                const int LDA,
                std::complex<float>* TAU,
                std::complex<float>* WORK,
                int LWORK,
                int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  cungqr(M, N, K, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void gqr(int M, int N, int K, float* A, const int LDA, float* TAU, float* WORK, int LWORK, int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  sorgqr(M, N, K, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

template<typename T>
inline void gqr_bufferSize(int m, int n, int k, [[maybe_unused]] T* a, int lda, int& lwork)
{
  typename std::decay<T>::type work;
  int status;
  lwork = -1;
  gqr(m, n, k, nullptr, lda, nullptr, &work, lwork, status);
  lwork = int(real(work));
}

inline void glq(int M,
                int N,
                int K,
                std::complex<double>* A,
                const int LDA,
                std::complex<double>* TAU,
                std::complex<double>* WORK,
                int LWORK,
                int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  zunglq(M, N, K, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void glq(int M, int N, int K, double* A, const int LDA, double* TAU, double* WORK, int LWORK, int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  dorglq(M, N, K, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void glq(int M,
                int N,
                int K,
                std::complex<float>* A,
                const int LDA,
                std::complex<float>* TAU,
                std::complex<float>* WORK,
                int LWORK,
                int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  cunglq(M, N, K, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

inline void glq(int M, int N, int K, float* A, const int LDA, float* TAU, float* WORK, int const LWORK, int& INFO)
{
#ifdef __NO_QR__
  INFO    = 0;
  WORK[0] = 0;
#else
  sorglq(M, N, K, A, LDA, TAU, WORK, LWORK, INFO);
#endif
}

template<typename T>
inline void glq_bufferSize(int m, int n, int k, [[maybe_unused]] T* a, int lda, int& lwork)
{
  typename std::decay<T>::type work;
  int status;
  lwork = -1;
  glq(m, n, k, nullptr, lda, nullptr, &work, lwork, status);
  lwork = int(real(work));
}

template<typename T1, typename T2>
inline void matinvBatched(int n, T1* const* a, int lda, T2** ainv, int lda_inv, int* info, int batchSize)
{
  std::vector<int> piv(n);
  int lwork(-1);
  getri_bufferSize(n, ainv[0], lda_inv, lwork);
  std::vector<T2> work(lwork);
  for (int i = 0; i < batchSize; i++)
  {
    for (int p = 0; p < n; p++)
      for (int q = 0; q < n; q++)
        ainv[i][p * lda_inv + q] = a[i][p * lda + q];
    getrf(n, n, ainv[i], lda, piv.data(), *(info + i));
    getri(n, ainv[i], lda, piv.data(), work.data(), lwork, *(info + i));
  }
}

} // namespace cpu
} // namespace ma

#endif 
