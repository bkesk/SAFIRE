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

#ifndef NUMERICS_HELPERS_HPP
#define NUMERICS_HELPERS_HPP

#include <cassert>
#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
#include "Memory/custom_pointers.hpp"
#include "Numerics/device_kernels.hpp"
#endif

// MAM: Move this to detail!

namespace ma
{
template<class T>
inline T determinant_from_getrf(int n, T* M, int lda, int* pivot, T LogOverlapFactor)
{
  T res(0.0);
  T sg(1.0);
  for (int i = 0, ip = 1; i != n; i++, ip++)
  {
    if (real(M[i * lda + i]) < 0.0)
    {
      res += std::log(-static_cast<T>(M[i * lda + i]));
      sg *= -1.0;
    }
    else
      res += std::log(static_cast<T>(M[i * lda + i]));
    if (pivot[i] != ip)
      sg *= -1.0;
  }
  return sg * std::exp(res - LogOverlapFactor);
}

template<class T>
inline void determinant_from_getrf(int n, T* M, int lda, int* pivot, T LogOverlapFactor, T* res)
{
  *res = T(0.0);
  T sg(1.0);
  for (int i = 0, ip = 1; i != n; i++, ip++)
  {
    if (real(M[i * lda + i]) < 0.0)
    {
      *res += std::log(-static_cast<T>(M[i * lda + i]));
      sg *= -1.0;
    }
    else
      *res += std::log(static_cast<T>(M[i * lda + i]));
    if (pivot[i] != ip)
      sg *= -1.0;
  }
  *res = sg * std::exp(*res - LogOverlapFactor);
}

template<class T>
inline void strided_determinant_from_getrf(int n,
                                           T* M,
                                           int lda,
                                           int Mstride,
                                           int* pivot,
                                           int pstride,
                                           T LogOverlapFactor,
                                           T* res,
                                           int incx,
                                           int nbatch)
{
  for (int b = 0; b < nbatch; b++)
    determinant_from_getrf(n, M + b * Mstride, lda, pivot + b * pstride, LogOverlapFactor, 
            res + b*incx);
}

template<class T>
inline void batched_determinant_from_getrf(int n,
                                           T** M,
                                           int lda,
                                           int* pivot,
                                           int pstride,
                                           T LogOverlapFactor,
                                           T* res,
                                           int incx,
                                           int nbatch)
{
  for (int b = 0; b < nbatch; b++)
    determinant_from_getrf(n, M[b], lda, pivot + b * pstride, LogOverlapFactor, res + b*incx);
}

template<class T>
T determinant_from_geqrf(int n, T* M, int lda, T* buff, T LogOverlapFactor)
{
  T res(0.0);
  for (int i = 0; i < n; i++)
  {
    if (real(M[i * lda + i]) < 0.0)
      buff[i] = T(-1.0);
    else
      buff[i] = T(1.0);
    res += std::log(buff[i] * M[i * lda + i]);
  }
  return res - LogOverlapFactor;
}

// specializations for complex
template<class T>
inline std::complex<T> determinant_from_getrf(int n,
                                              std::complex<T>* M,
                                              int lda,
                                              int* pivot,
                                              std::complex<T> LogOverlapFactor)
{
  std::complex<T> res(0.0, 0.0);
  for (int i = 0, ip = 1; i != n; i++, ip++)
  {
    if (pivot[i] == ip)
    {
      res += std::log(+static_cast<std::complex<T>>(M[i * lda + i]));
    }
    else
    {
      res += std::log(-static_cast<std::complex<T>>(M[i * lda + i]));
    }
  }
  return std::exp(res - LogOverlapFactor);
}

template<class T>
inline void determinant_from_getrf(int n,
                                   std::complex<T>* M,
                                   int lda,
                                   int* pivot,
                                   std::complex<T> LogOverlapFactor,
                                   std::complex<T>* res)
{
  *res = std::complex<T>(0.0, 0.0);
  for (int i = 0, ip = 1; i != n; i++, ip++)
  {
    if (pivot[i] == ip)
    {
      *res += std::log(+static_cast<std::complex<T>>(M[i * lda + i]));
    }
    else
    {
      *res += std::log(-static_cast<std::complex<T>>(M[i * lda + i]));
    }
  }
  *res = std::exp(*res - LogOverlapFactor);
}

template<class T>
inline void determinant_from_geqrf(int n, T* M, int lda, T* buff)
{
  for (int i = 0; i < n; i++)
  {
    if (real(M[i * lda + i]) < 0)
      buff[i] = T(-1.0);
    else
      buff[i] = T(1.0);
  }
}

template<class T>
inline void scale_columns(int n, int m, T* A, int lda, T* scl)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      A[i * lda + j] *= scl[j];
}


#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
// using thrust for now to avoid kernels!!!
template<class T>
inline void determinant_from_getrf(int n,
                                   device::device_pointer<T> A,
                                   int lda,
                                   device::device_pointer<int> piv,
                                   T LogOverlapFactor,
                                   T* res)
{
  kernels::determinant_from_getrf_gpu(n, raw_pointer_cast(A), lda, raw_pointer_cast(piv), LogOverlapFactor, res);
}

template<class T>
inline void strided_determinant_from_getrf(int n,
                                           device::device_pointer<T> A,
                                           int lda,
                                           int Mstride,
                                           device::device_pointer<int> piv,
                                           int pstride,
                                           T LogOverlapFactor,
                                           T* res,
                                           int incx, 
                                           int nbatch)
{
  kernels::strided_determinant_from_getrf_gpu(n, raw_pointer_cast(A), lda, Mstride, raw_pointer_cast(piv), pstride,
                                              LogOverlapFactor, res, incx, nbatch);
}

template<class T>
inline void batched_determinant_from_getrf(int n,
                                           device::device_pointer<T>* A,
                                           int lda,
                                           device::device_pointer<int> piv,
                                           int pstride,
                                           T LogOverlapFactor,
                                           T* res,
                                           int incx, 
                                           int nbatch)
{
  T** A_h = new T*[nbatch];
  for (int i = 0; i < nbatch; i++)
    A_h[i] = raw_pointer_cast(A[i]);
  T** A_d;
  arch::malloc((void**)&A_d, nbatch * sizeof(*A_h));
  arch::memcopy(A_d, A_h, nbatch * sizeof(*A_h), arch::memcopyH2D);
  kernels::batched_determinant_from_getrf_gpu(n, A_d, lda, raw_pointer_cast(piv), pstride, LogOverlapFactor, res, incx, nbatch);
  arch::free(A_d);
  delete[] A_h;
}

template<class T>
inline T determinant_from_getrf(int n, device::device_pointer<T> A, int lda, device::device_pointer<int> piv, T LogOverlapFactor)
{
  return kernels::determinant_from_getrf_gpu(n, raw_pointer_cast(A), lda, raw_pointer_cast(piv), LogOverlapFactor);
}

template<class T>
T determinant_from_geqrf(int n, device::device_pointer<T> M, int lda, device::device_pointer<T> buff, T LogOverlapFactor)
{
  return kernels::determinant_from_geqrf_gpu(n, raw_pointer_cast(M), lda, raw_pointer_cast(buff), LogOverlapFactor);
}

template<class T>
inline void determinant_from_geqrf(int n,device::device_pointer<T> M, int lda, device::device_pointer<T> buff)
{
  kernels::determinant_from_geqrf_gpu(n, raw_pointer_cast(M), lda, raw_pointer_cast(buff));
}

template<class T>
inline void scale_columns(int n, int m, device::device_pointer<T> A, int lda, device::device_pointer<T> scl)
{
  kernels::scale_columns(n, m, raw_pointer_cast(A), lda, raw_pointer_cast(scl));
}

template<class ptrA, class ptrB>
inline void scale_columns(int n, int m, ptrA A, int lda, ptrB scl)
{
  print_stacktrace;
  throw std::runtime_error("Error: Calling ma::scale_columns atch all.");
}
#endif

} // namespace ma 

#endif
