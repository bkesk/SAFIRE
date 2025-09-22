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

#ifndef MA_DETAIL_CPU_BLAS_HPP
#define MA_DETAIL_CPU_BLAS_HPP

#include "config.0.h"
#include "Numerics/detail/dispatch.hpp"
#include "Numerics/detail/CPU/blas_cpu.hpp"

namespace ma
{

template<typename T1, typename T2, typename T3>
void axpy(int n, T1 a, T2 x, int incx, T3 y, int incy, cpu_backend)
{
  // check_dispatch<T2,T3>(cpu_backend);
  ma::cpu::axpy(n, a, raw_pointer_cast(x), incx, raw_pointer_cast(y), incy);
}

template<typename T1, typename T2>
void scal(int n, T1 a, T2 x, int incx, cpu_backend)
{ 
  ma::cpu::scal(n, a, raw_pointer_cast(x), incx);
}

template<typename T1, typename T2>
auto dot(int n, T1 x, int incx, T2 y, int incy, cpu_backend)
-> decltype( ma::cpu::dot(n, raw_pointer_cast(x), incx, raw_pointer_cast(y), incy) )
{ 
  return ma::cpu::dot(n, raw_pointer_cast(x), incx, raw_pointer_cast(y), incy);
}

template<typename T>
auto norm2(int n, T x, int incx, cpu_backend)
-> decltype( ma::cpu::norm2(n, raw_pointer_cast(x), incx) )
{
  return ma::cpu::norm2(n, raw_pointer_cast(x), incx);
}

// No tag dispatching here!
// copy is the ONE routine we want to allow with mixed host/device pointers!
// still needs pointer_dispatch in the upper level
template<typename T1, typename T2>
void copy(int n, T1* x, int incx, T2* y, int incy)
{
  ma::cpu::copy(n, raw_pointer_cast(x), incx, raw_pointer_cast(y), incy);
}

template<typename T1, typename T2>
void copy2D(int n, int m, T1* A, int lda, T2* B, int ldb)
{
  ma::cpu::copy2D(n, m, raw_pointer_cast(A), lda, raw_pointer_cast(B), ldb);
}

template<typename T1, typename T2, typename T3, typename T4>
void ger(int m, int n, T1 alpha, T2 x, int incx, T3 y, int incy,
                                 T4 A, int lda, cpu_backend)
{
  ma::cpu::ger(m, n, alpha, raw_pointer_cast(x), incx,
                            raw_pointer_cast(y), incy,
                            raw_pointer_cast(A), lda);
}


template<typename T1, typename T2, typename T3, typename T4, typename T5> 
void gemv(char trans_in, int n, int m, T1 alpha, T2 A, int lda, T3 x, int incx,
				       T4 beta, T5 y, int incy, cpu_backend)          
{
  ma::cpu::gemv(trans_in, n, m, alpha, raw_pointer_cast(A), lda, 
			      raw_pointer_cast(x), incx, 
		       beta,  raw_pointer_cast(y), incy);
}


template<typename T1, typename T2, typename T3, typename T4, typename T5>
void gemm(char Atrans, char Btrans, int M, int N, int K, 
				T1 alpha, T2 A, int lda, T3 B, int ldb,
                                T4 beta, T5 C, int ldc, cpu_backend)
{
  ma::cpu::gemm(Atrans, Btrans, M, N, K, alpha, raw_pointer_cast(A), lda,
                              			raw_pointer_cast(B), ldb,
                       			 beta,  raw_pointer_cast(C), ldc);
}

} // namespace ma

#endif
