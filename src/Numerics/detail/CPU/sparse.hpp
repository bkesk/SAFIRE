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

#ifndef NUMERICS_DETAIL_CPU_SPARSE_HPP
#define NUMERICS_DETAIL_CPU_SPARSE_HPP

#include "config.0.h"
#include "Numerics/detail/dispatch.hpp"
#include "Numerics/detail/CPU/sparse_cpu.hpp"

namespace ma
{

template<typename T1, typename T2, typename T3, typename T4,
         typename T5, typename T6, typename T7, typename T8>
void csrmv(char transa, int M, int K, T1 alpha, const char* matdescra,
           T2 A, T3 indx, T4 pntrb, T5 pntre, T6 x, T7 beta, T8 y, cpu_backend)
{ 
  ma::cpu::csrmv(transa,M,K,alpha,matdescra,raw_pointer_cast(A),
        raw_pointer_cast(indx),raw_pointer_cast(pntrb),raw_pointer_cast(pntre),
        raw_pointer_cast(x),beta,raw_pointer_cast(y));
}   

template<typename T1, typename T2, typename T3, typename T4,
	 typename T5, typename T6, typename T7, typename T8>
void csrmm(char transa, int M, int N, int K, T1 alpha, const char* matdescra, 
	   T2 A, T3 indx, T4 pntrb, T5 pntre, T6 B, int ldb, int strideB, 
	   T7 beta, T8 C, int ldc, int strideC, int nbatch, cpu_backend)
{
  auto B_ = raw_pointer_cast(B);
  auto C_ = raw_pointer_cast(C);
  for(int n=0; n<nbatch; ++n, B_+=strideB, C_+=strideC)
    ma::cpu::csrmm_impl(transa,M,N,K,alpha,matdescra,raw_pointer_cast(A),
	raw_pointer_cast(indx),raw_pointer_cast(pntrb),raw_pointer_cast(pntre),
	raw_pointer_cast(B_),ldb,beta,raw_pointer_cast(C_),ldc);
}


} // namespace ma

#endif
